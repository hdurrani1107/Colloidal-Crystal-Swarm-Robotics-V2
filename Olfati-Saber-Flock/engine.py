#######################################################################
# engine.py
#
# Engine for 2D Olfati-Saber flocking algorithm with goal beacons
# Adapted from 3d_lattice_flock.py for 2D simulation
#
# Author: Humzah Durrani
#######################################################################

##########################
# Importing Libraries
##########################
import numpy as np
from goal_beacon import GoalBeaconSystem

##########################
# Core Math Functions - Olfati-Saber
##########################

def sigma_1(z):
    """Smooth version of identity function that avoids singularities at 0"""
    norm_z = np.linalg.norm(z, axis=-1, keepdims=True)
    # Handle scalar case
    if z.ndim == 1:
        norm_z = np.linalg.norm(z)
        if norm_z < 1e-6:
            return z
        return z / np.sqrt(1 + norm_z ** 2)
    # Handle array case
    safe_norm = np.where(norm_z < 1e-6, 1e-6, norm_z)
    return z / np.sqrt(1 + safe_norm ** 2)

def sigma_norm(z, eps=0.1):
    """Used to compute distances with smooth behavior"""
    norm_z = np.linalg.norm(z, axis=-1, keepdims=True)
    return (np.sqrt(1 + eps * norm_z ** 2) - 1) / eps

def sigma_grad(z, eps=0.1):
    """Gradient of sigma norm, to compute normalized direction vectors for force field"""
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
    Ph[z <= 1] = (1 + np.cos(np.pi * (z[z <= 1] - H) / (1 - H))) / 2
    Ph[z < H] = 1
    Ph[z < 0] = 0
    return Ph

def phi_alpha(z, R=12, D=10, eps=0.1):
    """Function that uses bump function to restrict the range of interaction"""
    r_alpha = sigma_norm(np.array([[R]]), eps)[0, 0]  # Convert to scalar
    d_alpha = sigma_norm(np.array([[D]]), eps)[0, 0]  # Convert to scalar
    return bump_function(z / r_alpha) * phi(z - d_alpha)

def influence(q_i, q_js, R=12, eps=0.1):
    """Function for influence coefficients based on distance (velocity matching)"""
    r_alpha = sigma_norm(np.array([[R]]), eps)[0, 0]
    distances = sigma_norm(q_js - q_i, eps).flatten()
    return bump_function(distances / r_alpha)

##########################
# Multi-Agent Setup
##########################

def init_grid(n, bounds, spacing, center=(100, 100)):
    """Initialize agents in hexagonal grid pattern (2D)"""
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

    # Center the lattice
    min_xy = np.min(agents[:, :2], axis=0)
    max_xy = np.max(agents[:, :2], axis=0)
    grid_center = (min_xy + max_xy) / 2
    shift = np.array(center) - grid_center
    agents[:, :2] += shift

    # Keep only agents within bounds
    agents = agents[(agents[:, 0] >= bounds[0]) & (agents[:, 0] <= bounds[1]) &
                    (agents[:, 1] >= bounds[0]) & (agents[:, 1] <= bounds[1])]

    return np.array(agents)

def init_random_positions(n, bounds, min_spacing=5.0, max_attempts=10000):
    """Initialize agents at random positions with minimum spacing"""
    agents = []
    attempts = 0
    min_dist_sq = min_spacing ** 2

    while len(agents) < n and attempts < max_attempts:
        x = np.random.uniform(bounds[0] + 10, bounds[1] - 10)
        y = np.random.uniform(bounds[0] + 10, bounds[1] - 10)
        new_pos = np.array([x, y])

        # Check for overlap
        if all(np.sum((new_pos - np.array(pos[:2]))**2) >= min_dist_sq for pos in agents):
            # Random initial velocity
            vx = np.random.uniform(-5, 5)
            vy = np.random.uniform(-5, 5)
            agents.append([x, y, vx, vy])

        attempts += 1

    if len(agents) < n:
        raise RuntimeError(f"Could not place {n} agents without overlap after {max_attempts} attempts.")

    return np.array(agents)

class multi_agent:
    def __init__(self, number, sample_time, bounds, obstacles=None):
        self.dt = sample_time
        self.bounds = bounds
        self.obstacles = obstacles if obstacles else []
        self.goal_beacons = None  # Will be set up later if needed
        
        # Initialize agents using random positions for more natural flocking
        #self.agents = init_random_positions(number, bounds, min_spacing=8.0)
        self.agents = init_grid(number, bounds, spacing=6)
        
        # Initialize three flocks with leaders
        self.setup_three_flocks(number)
        
        # Agent parameters for Olfati-Saber algorithm
        self.R = 12  # Interaction range
        self.D = 6  # Desired distance
        self.eps = 0.1  # Smoothing parameter
        
        # Control gains
        self.c1_alpha = 5  # Inter-agent interaction gain
        self.c2_alpha = 2 * np.sqrt(self.c1_alpha)   # Velocity alignment gain
        
        # Stabilization parameters
        self.global_drag = 0.2       # viscous drag
        self.accel_max   = 20.0      # clamp per-agent acceleration
        self.inter_flock_R = 10.0    # cross-flock separation radius
        self.k_inter       = 6.0     # cross-flock repulsion gain
    
    def setup_three_flocks(self, number):
        """Setup three flocks with leaders"""
        agents_per_flock = number // 3
        self.flocks = {
            0: list(range(0, agents_per_flock)),  # Flock 1: agents 0-49
            1: list(range(agents_per_flock, 2 * agents_per_flock)),  # Flock 2: agents 50-99  
            2: list(range(2 * agents_per_flock, number))  # Flock 3: agents 100-149
        }
        
        # Assign leaders (first agent in each flock)
        self.leaders = {0: 0, 1: agents_per_flock, 2: 2 * agents_per_flock}
        
        # Leaders will target closest goal beacons instead of fixed goals
        # No need for independent leader goals anymore
        
        print(f"Setup 3 flocks: {len(self.flocks[0])}, {len(self.flocks[1])}, {len(self.flocks[2])} agents")
        print(f"Leaders: Flock 1 agent {self.leaders[0]}, Flock 2 agent {self.leaders[1]}, Flock 3 agent {self.leaders[2]}")
    
    def get_closest_beacon(self, leader_pos):
        """Find the closest active goal beacon to a leader position"""
        if not self.goal_beacons or len(self.goal_beacons.beacons) == 0:
            return None
            
        closest_beacon = None
        min_distance = float('inf')
        
        for beacon in self.goal_beacons.beacons:
            if beacon.is_active:
                distance = np.linalg.norm(leader_pos - beacon.position)
                if distance < min_distance:
                    min_distance = distance
                    closest_beacon = beacon
                    
        return closest_beacon
    
    def get_agent_flock(self, agent_idx):
        """Get which flock an agent belongs to"""
        for flock_id, agent_list in self.flocks.items():
            if agent_idx in agent_list:
                return flock_id
        return 0  # Default to flock 0
    
    def is_leader(self, agent_idx):
        """Check if agent is a leader"""
        return agent_idx in self.leaders.values()
    
    def compute_leader_forces(self):
        """Compute stabilized leader forces with PD control + tether + clamp"""
        n = len(self.agents)
        F = np.zeros((n, 2))
        P = self.agents[:, :2]
        V = self.agents[:, 2:]

        k_beacon = 1.5   # attraction
        k_damp   = 1.0   # leader velocity damping
        k_tether = 0.3   # leader to flock centroid
        fmax     = 10.0  # clamp

        for flock_id, leader_idx in self.leaders.items():
            if leader_idx >= n: 
                continue
            pL, vL = P[leader_idx], V[leader_idx]
            f = np.zeros(2)

            b = self.get_closest_beacon(pL)
            if b is not None and b.is_active:
                d = b.position - pL
                dist = np.linalg.norm(d) + 1e-6
                f += k_beacon * (d / dist) - k_damp * vL  # normalized PD

            # centroid tether
            flock_idxs = np.array(self.flocks[flock_id], dtype=int)
            centroid = P[flock_idxs].mean(axis=0)
            f += k_tether * (centroid - pL)

            # clamp
            m = np.linalg.norm(f)
            if m > fmax:
                f *= (fmax / m)

            F[leader_idx] = f
        return F

    def setup_goal_beacons(self, beacon_config):
        """Set up the goal beacon system"""
        from goal_beacon import GoalBeaconSystem
        self.goal_beacons = GoalBeaconSystem(
            bounds=self.bounds,
            beacon_radius=beacon_config['beacon_radius'],
            spawn_interval=beacon_config['spawn_interval'],
            base_lifetime=beacon_config['base_lifetime'],
            velocity_damping=beacon_config['velocity_damping'],
            logger=beacon_config.get('logger', None)
        )
        self.goal_beacons._owner = self
    
    def get_adjacency_matrix(self):
        """Get adjacency matrix for agents within interaction range and same flock"""
        n = len(self.agents)
        adj = np.zeros((n, n), dtype=bool)
        positions = self.agents[:, :2]
        r_alpha = sigma_norm(np.array([[self.R]]), self.eps)[0, 0]
        
        for i in range(n):
            fi = self.get_agent_flock(i)
            for j in range(n):
                if i == j or self.get_agent_flock(j) != fi:
                    continue
                sij = sigma_norm(positions[j] - positions[i], self.eps)
                adj[i, j] = (sij <= r_alpha)
        return adj
    
    def compute_flocking_forces(self):
        """Compute Olfati-Saber flocking forces for all agents"""
        n = len(self.agents)
        forces = np.zeros((n, 2))
        positions = self.agents[:, :2]
        velocities = self.agents[:, 2:]
        
        # Get adjacency matrix
        adj_mat = self.get_adjacency_matrix()
        
        for i in range(n):
            agent_p = positions[i]
            agent_v = velocities[i]
            
            # Find neighbors
            neighbor_indices = adj_mat[i]
            
            if np.sum(neighbor_indices) > 0:
                neighbor_positions = positions[neighbor_indices]
                neighbor_velocities = velocities[neighbor_indices]
                
                # Compute relative positions
                relative_pos = neighbor_positions - agent_p  # Shape: (n_neighbors, 2)
                
                # Compute interaction forces (attraction/repulsion)
                distances = sigma_norm(relative_pos, self.eps).flatten()  # Shape: (n_neighbors,)
                phi_values = phi_alpha(distances, self.R, self.D, self.eps)  # Shape: (n_neighbors,)
                directions = sigma_grad(relative_pos, self.eps)  # Shape: (n_neighbors, 2)
                
                # Sum over neighbors for position-based interaction
                u1 = self.c1_alpha * np.sum(phi_values[:, np.newaxis] * directions, axis=0)  # spacing
                
                # Velocity alignment
                influence_weights = influence(agent_p, neighbor_positions, self.R, self.eps)  # Shape: (n_neighbors,)
                velocity_diff = neighbor_velocities - agent_v  # Shape: (n_neighbors, 2)
                u2 = self.c2_alpha * np.sum(influence_weights[:, np.newaxis] * velocity_diff, axis=0)  # alignment
                
                forces[i] = u1 + u2
        
        return forces
    
    def compute_interflock_repulsion(self):
        """Compute purely repulsive inter-flock forces"""
        n = len(self.agents)
        F = np.zeros((n, 2))
        P = self.agents[:, :2]
        R = self.inter_flock_R
        k = self.k_inter
        
        for i in range(n):
            fi = self.get_agent_flock(i)
            for j in range(n):
                if i == j or self.get_agent_flock(j) == fi:
                    continue
                d = P[j] - P[i]
                dist = np.linalg.norm(d) + 1e-6
                if dist < R:
                    w = (R - dist) / R
                    F[i] += -k * w * (d / dist)
        return F
    
    def compute_obstacle_forces(self):
        """Compute repulsive forces from obstacles"""
        n = len(self.agents)
        forces = np.zeros((n, 2))
        positions = self.agents[:, :2]
        
        for i, pos in enumerate(positions):
            total_obs_force = np.zeros(2)
            
            for obs_pos, obs_radius in self.obstacles:
                # Vector from obstacle to agent
                d_vec = pos - obs_pos
                distance = np.linalg.norm(d_vec)
                
                # Repulsive force if too close
                if distance < obs_radius + 15:  # Add safety margin
                    if distance < 1e-3:
                        distance = 1e-3
                    
                    # Strong repulsion
                    repulsion_strength = 100.0 / (distance ** 2 + 1e-3)
                    direction = d_vec / distance
                    total_obs_force += repulsion_strength * direction
            
            forces[i] = total_obs_force
        
        return forces
    
    def update(self, external_forces, temp=None, frame=0):
        """Update agent positions and velocities"""
        # No need to update leader goals - they target beacons directly
        
        # Update goal beacons first
        if self.goal_beacons:
            self.goal_beacons.frame += 1
            self.goal_beacons.update()
            
            # Process agent-beacon interactions
            positions = self.agents[:, :2]
            velocities = self.agents[:, 2:]
            for i, (pos, vel) in enumerate(zip(positions, velocities)):
                is_leader = self.is_leader(i)
                self.goal_beacons.process_agent(i, pos, vel, is_leader=is_leader)
        
        n = len(self.agents)
        positions = self.agents[:, :2]
        velocities = self.agents[:, 2:]
        
        # Compute all forces
        F_flock  = self.compute_flocking_forces()
        F_obst   = self.compute_obstacle_forces()
        F_inter  = self.compute_interflock_repulsion()
        F_leader = self.compute_leader_forces()

        # No dual beacon forces - leaders get forces only from compute_leader_forces()
        # F_beacon removed to eliminate redundant beacon attraction
        
        # Total forces with global drag
        total_forces = F_flock + F_obst + F_inter + F_leader + external_forces - self.global_drag * self.agents[:, 2:]
        
        # Per-agent acceleration clamp
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True) + 1e-9
        scale = np.minimum(1.0, self.accel_max / norms)
        total_forces *= scale
        
        # Update velocities and positions
        for i in range(n):
            # Velocity update
            new_velocity = velocities[i] + total_forces[i] * self.dt

            #Velocity Scaled from Simulation Comparability
            # Different velocity caps for leaders vs followers to maintain cohesion
            if self.is_leader(i):
                max_velocity = 8.0  # Slower leaders maintain flock cohesion
            else:
                max_velocity = 10.0  # Followers can move faster to catch up
            
            velocity_magnitude = np.linalg.norm(new_velocity)
            if velocity_magnitude > max_velocity:
                new_velocity = (new_velocity / velocity_magnitude) * max_velocity
            
            # Apply velocity damping if agent is trapped in beacon
            if self.goal_beacons and self.goal_beacons.is_agent_trapped(i):
                beacon_pos, beacon_radius = self.goal_beacons.get_beacon_info_for_agent(i)
                if beacon_pos is not None:
                    # Strong velocity damping inside beacon
                    damping_factor = 0.9  # Scale down velocity rapidly
                    new_velocity *= damping_factor
            
            self.agents[i, 2:] = new_velocity
            
            
            # Position update
            new_position = positions[i] + new_velocity * self.dt
            
            # Apply beacon boundary constraints
            if self.goal_beacons and self.goal_beacons.is_agent_trapped(i):
                beacon_pos, beacon_radius = self.goal_beacons.get_beacon_info_for_agent(i)
                if beacon_pos is not None:
                    # Check if new position would leave the beacon
                    dist_to_center = np.linalg.norm(new_position - beacon_pos)
                    if dist_to_center > beacon_radius:
                        # Keep agent inside the boundary
                        direction = (new_position - beacon_pos) / dist_to_center
                        new_position = beacon_pos + direction * beacon_radius
            
            self.agents[i, :2] = new_position
            
            # Apply hard boundary conditions
            x, y = self.agents[i, :2]
            
            # Bounce off walls
            if x < self.bounds[0]:
                self.agents[i, 0] = self.bounds[0]
                self.agents[i, 2] *= -0.8  # Some energy loss
            elif x > self.bounds[1]:
                self.agents[i, 0] = self.bounds[1]
                self.agents[i, 2] *= -0.8
                
            if y < self.bounds[0]:
                self.agents[i, 1] = self.bounds[0]
                self.agents[i, 3] *= -0.8
            elif y > self.bounds[1]:
                self.agents[i, 1] = self.bounds[1]
                self.agents[i, 3] *= -0.8