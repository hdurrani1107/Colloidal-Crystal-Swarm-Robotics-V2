##############################################################################
# engine.py
#
# Engine for 2D Olfati-Saber flocking algorithm with goal beacons
# Complete overhaul based on original 3d_lattice_flock.py algorithm
# Implements 3 distinct lattice flocks with virtual leaders at centroids
#
# Author: Humzah Durrani
# AI Disclosure: AI was used for feature development, handling and debugging
##############################################################################

##########################
# Importing Libraries
##########################
import numpy as np
from olfati_saber_flock.goal_beacon import GoalBeaconSystem

#####################################
# Core Math Functions - Olfati-Saber
#####################################

def sigma_1(z):
    """Smooths version of identity function that avoids singularities at 0"""
    if np.isscalar(z):
        return z / np.sqrt(1 + z ** 2)
    else:
        norm_z = np.linalg.norm(z, axis=-1, keepdims=True)
        safe_norm = np.where(norm_z < 1e-6, 1e-6, norm_z)
        return z / np.sqrt(1 + safe_norm ** 2)

def sigma_norm(z, eps=0.1):
    """Used to compute distances with smooth behavior"""
    if np.isscalar(z):
        return (np.sqrt(1 + eps * z ** 2) - 1) / eps
    else:
        norm_z = np.linalg.norm(z, axis=-1, keepdims=True)
        return (np.sqrt(1 + eps * norm_z ** 2) - 1) / eps

def sigma_grad(z, eps=0.1):
    """Gradient of sigma norm, to compute normalized direction vectors for force field"""
    if np.isscalar(z):
        return z / np.sqrt(1 + eps * z ** 2)
    else:
        norm_z = np.linalg.norm(z, axis=-1, keepdims=True)
        safe_norm = np.where(norm_z < 1e-6, 1e-6, norm_z)
        return z / np.sqrt(1 + eps * safe_norm ** 2)

def phi(z, A=5, B=5):
    """Base potential function"""
    C = np.abs(A - B) / np.sqrt(4 * A * B)
    return ((A + B) * sigma_1(z + C) + (A - B)) / 2

def bump_function(z, H=0.2):
    """Smooths transition between 1 and 0 when z crosses threshold"""
    Ph = np.zeros_like(z)
    mask1 = z <= 1
    mask2 = z < H
    mask3 = z < 0
    
    Ph[mask1] = (1 + np.cos(np.pi * (z[mask1] - H) / (1 - H))) / 2
    Ph[mask2] = 1
    Ph[mask3] = 0
    return Ph

def phi_alpha(z, R=12, D=10, eps=0.1):
    """Function that uses bump function to restrict the range of interaction"""
    r_alpha = sigma_norm(R, eps)
    d_alpha = sigma_norm(D, eps)
    return bump_function(z / r_alpha) * phi(z - d_alpha)

def influence(q_i, q_js, R=12, eps=0.1):
    """Function for influence coefficients based on distance (velocity matching)"""
    r_alpha = sigma_norm(R, eps)
    if q_js.ndim == 1:
        distance = sigma_norm(np.linalg.norm(q_js - q_i), eps)
    else:
        distances = np.array([sigma_norm(np.linalg.norm(q_j - q_i), eps) for q_j in q_js])
        distance = distances
    return bump_function(distance / r_alpha)

##########################
# Helper Functions
##########################

def get_adj_mat(nodes, r, eps=0.1):
    """Function that indicates whether two agents are within interaction range"""
    n = len(nodes)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = sigma_norm(np.linalg.norm(nodes[i, :2] - nodes[j, :2]), eps)
                adj[i, j] = dist <= sigma_norm(r, eps)
    return adj

def local_dir(q_i, q_js, eps=0.1):
    """Function to calculate direction vectors (agents to neighbors)"""
    if q_js.ndim == 1:
        return sigma_grad(q_js - q_i, eps)
    else:
        return np.array([sigma_grad(q_j - q_i, eps) for q_j in q_js])

def obs_rep(obstacles, agent_p, R_obs=10):
    """Function to calculate repulsive forces from agents to obstacles"""
    u_obs = np.zeros(2)
    for obs in obstacles:
        d_vec = agent_p - obs[0]  # obs is (position, radius) tuple
        d_norm = np.linalg.norm(d_vec)
        if d_norm < R_obs:
            if d_norm < 1e-3:
                repulsion = 1e6
            else:
                repulsion = 100 / (d_norm**2 + 1e-3)
            u_obs += repulsion * (d_vec / (d_norm + 1e-3))
    return u_obs

##########################
# Multi-Agent Setup
##########################

def init_three_flocks(number, bounds, spacing=6.0):
    """Initialize three separate lattice flocks in different areas of the space"""
    agents_per_flock = number // 3
    remainder = number % 3
    
    all_agents = []
    flock_sizes = [agents_per_flock + (1 if i < remainder else 0) for i in range(3)]
    
    # Define starting positions closer to center but non-overlapping
    center_x, center_y = bounds[1] // 2, bounds[1] // 2
    offset = 40  # Distance from center
    
    flock_centers = [
        (center_x - offset, center_y - offset//2),    # Left
        (center_x + offset, center_y - offset//2),    # Right
        (center_x, center_y + offset)                 # Top
    ]
    
    for flock_id, (center, size) in enumerate(zip(flock_centers, flock_sizes)):
        flock_agents = init_proper_hex_grid(size, center, spacing)
        all_agents.extend(flock_agents)
    
    return np.array(all_agents)

def init_proper_hex_grid(n, center, spacing):
    """Initialize agents in proper hexagonal grid pattern (like LJ-Swarm)"""
    agents = []
    cols = int(np.sqrt(n))
    rows = int(np.ceil(n / cols))

    dx = spacing
    dy = spacing * np.sqrt(3) / 2  # height between rows in hex pattern

    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            # Offset every other row to create the hexagonal pattern
            x_offset = 0.5 * dx if row % 2 else 0.0
            x = col * dx + x_offset 
            y = row * dy
            agents.append([x, y, 0.0, 0.0])  # [x, y, vx, vy]
            count += 1

    agents = np.array(agents)

    # Center the lattice around the specified center
    if len(agents) > 0:
        min_xy = np.min(agents[:, :2], axis=0)
        max_xy = np.max(agents[:, :2], axis=0)
        grid_center = (min_xy + max_xy) / 2
        shift = np.array(center) - grid_center
        agents[:, :2] += shift

    return agents.tolist()

class multi_agent:
    def __init__(self, number, sample_time, bounds, obstacles=None):
        self.dt = sample_time
        self.bounds = bounds
        self.obstacles = obstacles if obstacles else []
        self.goal_beacons = None  # Will be set up later if needed
        
        # Initialize three separate lattice flocks
        self.agents = init_three_flocks(number, bounds, spacing=6.0)
        
        # Olfati-Saber algorithm parameters from original paper
        self.eps = 0.1  # Smoothing factor for sigma_norm
        self.A = 5      # Shapes phi function
        self.B = 5      # Shapes phi function  
        self.C = np.abs(self.A - self.B) / np.sqrt(4 * self.A * self.B)  # Normalization constant
        self.H = 0.2    # Bump function threshold
        self.R = 12     # Range
        self.D = 10     # Distance
        
        # Control gain parameters from the paper
        self.c1_alpha = 3  # Inter-agent interactions
        self.c2_alpha = 2 * np.sqrt(self.c1_alpha)  # Velocity alignment
        self.c1_gamma = 5  # Global control (leader attraction)
        self.c2_gamma = 0.2 * np.sqrt(self.c1_gamma)  # Leader velocity damping
        
        # Initialize three flocks with virtual leaders
        self.setup_three_flocks(number)
        
        # Performance optimization
        self.r_alpha = sigma_norm(self.R, self.eps)
        self.d_alpha = sigma_norm(self.D, self.eps)
    
    def setup_three_flocks(self, number):
        """Setup three distinct flocks with virtual leaders (centroids)"""
        agents_per_flock = number // 3
        remainder = number % 3
        
        # Create flock assignments
        start_idx = 0
        self.flocks = {}
        
        for flock_id in range(3):
            flock_size = agents_per_flock + (1 if flock_id < remainder else 0)
            self.flocks[flock_id] = list(range(start_idx, start_idx + flock_size))
            start_idx += flock_size
        
        # Virtual leaders are centroids of each flock (computed dynamically)
        self.virtual_leaders = {}
        self.update_virtual_leaders()
        
        # Create pre-computed flock membership array for fast lookup
        self.agent_flock_ids = np.zeros(number, dtype=int)
        for flock_id, agent_indices in self.flocks.items():
            for agent_idx in agent_indices:
                self.agent_flock_ids[agent_idx] = flock_id
        
        print(f"Setup 3 flocks: {[len(agents) for agents in self.flocks.values()]} agents")
        print(f"Virtual leaders (centroids) will be computed dynamically")
    
    def update_virtual_leaders(self):
        """Update virtual leader positions (centroids of each flock)"""
        for flock_id, agent_indices in self.flocks.items():
            if len(agent_indices) > 0:
                flock_positions = self.agents[agent_indices, :2]
                centroid = np.mean(flock_positions, axis=0)
                self.virtual_leaders[flock_id] = centroid
            else:
                self.virtual_leaders[flock_id] = np.array([0.0, 0.0])
    
    def get_closest_beacon_for_flock(self, flock_id):
        """Find the closest active goal beacon to a flock's virtual leader"""
        if not self.goal_beacons or len(self.goal_beacons.beacons) == 0:
            return None
            
        leader_pos = self.virtual_leaders[flock_id]
        closest_beacon = None
        min_distance = float('inf')
        
        for beacon in self.goal_beacons.beacons:
            if beacon.is_active and (beacon.owner_flock_id is None or beacon.owner_flock_id == flock_id):
                distance = np.linalg.norm(leader_pos - beacon.position)
                if distance < min_distance:
                    min_distance = distance
                    closest_beacon = beacon
                    
        return closest_beacon
    
    def get_agent_flock(self, agent_idx):
        """Get which flock an agent belongs to"""
        return self.agent_flock_ids[agent_idx]
    
    def compute_gamma_forces(self):
        """Compute gamma forces (virtual leader attraction) for each flock"""
        n = len(self.agents)
        forces = np.zeros((n, 2))
        
        # Update virtual leader positions
        self.update_virtual_leaders()
        
        for flock_id, agent_indices in self.flocks.items():
            # Get closest beacon for this flock
            closest_beacon = self.get_closest_beacon_for_flock(flock_id)
            
            if closest_beacon is not None:
                # Virtual leader moves toward beacon
                gamma_pos = closest_beacon.position
            else:
                # No beacon available - virtual leader stays at centroid
                gamma_pos = self.virtual_leaders[flock_id]
            
            # Apply gamma force to all agents in this flock
            for agent_idx in agent_indices:
                agent_p = self.agents[agent_idx, :2]
                agent_v = self.agents[agent_idx, 2:]
                
                # Gamma force toward virtual leader position
                u_gamma = -self.c1_gamma * sigma_1(agent_p - gamma_pos) - self.c2_gamma * agent_v
                forces[agent_idx] = u_gamma
        
        return forces

    def setup_goal_beacons(self, beacon_config):
        self.goal_beacons = GoalBeaconSystem(
            bounds=self.bounds,
            beacon_radius=beacon_config['beacon_radius'],
            spawn_interval=beacon_config['spawn_interval'],
            base_lifetime=beacon_config['base_lifetime'],
            velocity_damping=beacon_config['velocity_damping'],
            logger=beacon_config.get('logger', None)
        )
        self.goal_beacons._owner = self
    
    def compute_flocking_forces(self):
        """Compute Olfati-Saber alpha forces for lattice formation within each flock"""
        n = len(self.agents)
        forces = np.zeros((n, 2))
        
        # Get adjacency matrix based on original Olfati-Saber algorithm
        adj_mat = get_adj_mat(self.agents, self.R, self.eps)
        
        # Apply same-flock constraint
        for i in range(n):
            for j in range(n):
                if i != j and adj_mat[i, j]:
                    # Only maintain adjacency if agents are in the same flock
                    if self.agent_flock_ids[i] != self.agent_flock_ids[j]:
                        adj_mat[i, j] = False
        
        for i in range(n):
            agent_p = self.agents[i, :2]
            agent_v = self.agents[i, 2:]
            
            # Initialize control input
            u_alpha = np.zeros(2)
            
            # Identify and process neighbors (same flock only)
            neighbor_indices = np.where(adj_mat[i])[0]
            
            if len(neighbor_indices) > 0:
                neighbor_p = self.agents[neighbor_indices, :2]
                neighbor_v = self.agents[neighbor_indices, 2:]
                
                # Compute direction vectors
                directions = local_dir(agent_p, neighbor_p, self.eps)
                
                # Interaction with neighbors (spacing control)
                phi_values = np.array([phi_alpha(sigma_norm(np.linalg.norm(neighbor_p[j] - agent_p), self.eps), 
                                                self.R, self.D, self.eps) for j in range(len(neighbor_p))])
                u1 = self.c1_alpha * np.sum(phi_values[:, np.newaxis] * directions, axis=0)
                
                # Velocity alignment with neighbors
                n_influence = influence(agent_p, neighbor_p, self.R, self.eps)
                if np.isscalar(n_influence):
                    n_influence = np.array([n_influence])
                u2 = self.c2_alpha * np.sum(n_influence[:, np.newaxis] * (neighbor_v - agent_v), axis=0)
                
                # Total alpha influence
                u_alpha = u1 + u2
            
            forces[i] = u_alpha
        
        return forces
    
    def compute_inter_flock_repulsion(self):
        """Compute repulsion between agents from different flocks"""
        n = len(self.agents)
        forces = np.zeros((n, 2))
        repulsion_range = 25.0  # Range for inter-flock repulsion
        repulsion_strength = 10.0  # Strength of repulsion
        
        for i in range(n):
            agent_p = self.agents[i, :2]
            agent_flock = self.agent_flock_ids[i]
            total_repulsion = np.zeros(2)
            
            for j in range(n):
                if i != j and self.agent_flock_ids[j] != agent_flock:
                    other_p = self.agents[j, :2]
                    distance_vec = agent_p - other_p
                    distance = np.linalg.norm(distance_vec)
                    
                    if distance < repulsion_range and distance > 1e-6:
                        # Repulsion force inversely proportional to distance
                        repulsion_force = repulsion_strength / (distance + 1e-3)
                        direction = distance_vec / distance
                        total_repulsion += repulsion_force * direction
            
            forces[i] = total_repulsion
        
        return forces
    
    def compute_obstacle_forces(self):
        """Compute repulsive forces from obstacles"""
        n = len(self.agents)
        forces = np.zeros((n, 2))
        
        for i in range(n):
            agent_p = self.agents[i, :2]
            u_obs = obs_rep(self.obstacles, agent_p)
            forces[i] = 20 * u_obs  # Scale obstacle repulsion 
        
        return forces
    
    def update(self, external_forces, temp=None, frame=0):
        """Update agent positions and velocities using Olfati-Saber algorithm"""
        n = len(self.agents)
        
        # Update goal beacons first
        if self.goal_beacons:
            self.goal_beacons.frame += 1
            self.goal_beacons.update()
            
            # Process agent-beacon interactions
            for i in range(n):
                agent_p = self.agents[i, :2]
                agent_v = self.agents[i, 2:]
                self.goal_beacons.process_agent(i, agent_p, agent_v, is_leader=False)
        
        # Compute all forces according to Olfati-Saber algorithm
        F_alpha = self.compute_flocking_forces()    # Alpha forces (lattice formation)
        F_gamma = self.compute_gamma_forces()       # Gamma forces (leader attraction)
        F_inter = self.compute_inter_flock_repulsion()  # Inter-flock repulsion
        F_obs = self.compute_obstacle_forces()      # Obstacle avoidance
        
        # Total control input
        total_forces = F_alpha + F_gamma + F_inter + F_obs + external_forces
        
        # Update agent states
        for i in range(n):
            # Velocity update with acceleration limit
            acceleration = total_forces[i]
            max_accel = 20.0
            accel_magnitude = np.linalg.norm(acceleration)
            if accel_magnitude > max_accel:
                acceleration = (acceleration / accel_magnitude) * max_accel
            
            new_velocity = self.agents[i, 2:] + acceleration * self.dt
            
            # Velocity limits
            max_velocity = 12.0
            velocity_magnitude = np.linalg.norm(new_velocity)
            if velocity_magnitude > max_velocity:
                new_velocity = (new_velocity / velocity_magnitude) * max_velocity
            
            # Apply velocity damping if agent is trapped in beacon
            if self.goal_beacons and self.goal_beacons.is_agent_trapped(i):
                beacon_pos, beacon_radius = self.goal_beacons.get_beacon_info_for_agent(i)
                if beacon_pos is not None:
                    damping_factor = 0.9  # Strong damping inside beacon
                    new_velocity *= damping_factor
            
            self.agents[i, 2:] = new_velocity
            
            # Position update
            new_position = self.agents[i, :2] + new_velocity * self.dt
            
            # Apply beacon boundary constraints
            if self.goal_beacons and self.goal_beacons.is_agent_trapped(i):
                beacon_pos, beacon_radius = self.goal_beacons.get_beacon_info_for_agent(i)
                if beacon_pos is not None:
                    dist_to_center = np.linalg.norm(new_position - beacon_pos)
                    if dist_to_center > beacon_radius:
                        direction = (new_position - beacon_pos) / dist_to_center
                        new_position = beacon_pos + direction * beacon_radius
            
            self.agents[i, :2] = new_position
            
            # Boundary conditions (bounce off walls)
            x, y = self.agents[i, :2]
            
            if x < self.bounds[0]:
                self.agents[i, 0] = self.bounds[0]
                self.agents[i, 2] *= -0.8
            elif x > self.bounds[1]:
                self.agents[i, 0] = self.bounds[1]
                self.agents[i, 2] *= -0.8
                
            if y < self.bounds[0]:
                self.agents[i, 1] = self.bounds[0]
                self.agents[i, 3] *= -0.8
            elif y > self.bounds[1]:
                self.agents[i, 1] = self.bounds[1]
                self.agents[i, 3] *= -0.8