#######################################################################
# swarm-engine.py
#
# Engine behind swarm behavior  
#
# Author: Humzah Durrani
#######################################################################

##########################
# Importing Libraries
##########################
import numpy as np
from goal import InvisibleGoal

##########################
# Multi-Agent Setup
##########################

def sigma_1(z):
    return z / np.sqrt(1 + z ** 2)

def epsilon_temp(T, epsilon, alpha):
    return epsilon * np.exp(-alpha * T)

def init_melt(agents, init_temp, mass=1, kB=1):
    #Starting off with initial noise sampled from Maxwell Boltzmann Distribution
    N = len(agents)
    std_dev = np.sqrt(kB * init_temp / mass)
    agents[:, 2:] = np.random.normal(0, std_dev, size=(N, 2))

def init_grid(n, bounds, spacing, center=(10, 10)):
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
            agents.append([x, y, 0.0, 0.0])
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

def init_grid_random_no_overlap(n, bounds, spacing, max_attempts=10000):
    agents = []
    attempts = 0
    min_dist_sq = spacing ** 2  # compare squared distances for speed

    while len(agents) < n and attempts < max_attempts:
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])
        new_pos = np.array([x, y])

        # Check for overlap
        if all(np.sum((new_pos - np.array(pos[:2]))**2) >= min_dist_sq for pos in agents):
            agents.append([x, y, 0.0, 0.0])  # x, y, vx=0, vy=0

        attempts += 1

    if len(agents) < n:
        raise RuntimeError(f"Could not place {n} agents without overlap after {max_attempts} attempts.")

    return np.array(agents)

class multi_agent:

    def __init__(self, number, sigma, sampletime, bounds, obstacles):
        self.dt = sampletime
        self.bounds = bounds
        self.obstacles = obstacles
        self.goal = None  # Will be set up later if needed
        spacing = (2**(1/6)) * sigma
        self.agents = init_grid(number, bounds, spacing)
        #self.agents = init_grid_random_no_overlap(number, bounds, spacing)
    
    def setup_goal(self, gamma_pos, goal_config):
        """Set up the discovery-based goal system"""
        self.goal = InvisibleGoal(
            gamma_pos=gamma_pos,
            detection_radius=goal_config['detection_radius'],
            trap_strength=goal_config['trap_strength'],
            trap_radius=goal_config.get('trap_radius'),
            max_capacity=goal_config.get('max_capacity', 10)
        )
    
    def compute_neighbor_count(self, i, radius):
        pos_i = self.agents[i, :2]
        count = 0
        for j in range(len(self.agents)):
            if i == j:
                continue
            pos_j = self.agents[j, :2]
            if np.linalg.norm(pos_i - pos_j) < radius:
                count += 1
        return count

    def compute_forces(self, sigma, epsilon, distance, temp, c1_gamma, c2_gamma, gamma_pos, alpha=0.5):
        n = len(self.agents)
        forces = np.zeros((n, 2))
        #epsilon_T = epsilon_temp(temp, epsilon, alpha)

        for i in range(n):
            pos_i = self.agents[i, :2]
            vel_i = self.agents[i, 2:]
            total_force = np.zeros(2)

            for j in range(n):
                if i == j:
                    continue
                pos_j = self.agents[j, :2]
                offset = pos_i - pos_j
                
                #APPLYING MINIMUM IMAGE CONVENTION to ensure particle of the lowest distance is chosen
                #Includes particles on the other side of the PBC
                # Box size need multiple boxes
                #I did this wrong too
                #box_length = self.bounds[1] - self.bounds[0]

                # Apply MIC in x and y
                #for dim in range(2):
                #    if offset[dim] > 0.5 * box_length:
                #        offset[dim] -= box_length
                #    elif offset[dim] < -0.5 * box_length:
                #        offset[dim] += box_length


                dist = np.linalg.norm(offset)
                #This is an error: I don't know why I changed this but effectively
                #Once it hits the optimal distance it stops the lj-potential.
                #So all it wants to do is repel.
                #Updated: Converted it all the vector notation
                #if dist < distance and dist > 1e-3:
                
                ##############################
                # LJ Force Potential
                ##############################

                if dist > 1e-5:
                    inv_r = 1.0 / dist
                    inv_r6 = (sigma * inv_r) ** 6
                    inv_r12 = inv_r6 ** 2
                    lj_scalar = 24 * epsilon * (2 * inv_r12 - inv_r6) * inv_r
                    total_force += lj_scalar * (offset / dist)
                    #total_force += lj_force
            #if not is_liquid:


            ##############################
            # LJ Force Potential
            ##############################

            if self.obstacles:
                for obs_pos, obs_radius in self.obstacles:
                    obs_vec = pos_i - obs_pos
                    dist_to_obs = np.linalg.norm(obs_vec)
                    overlap = obs_radius + 1.5 * sigma - dist_to_obs  # soft repulsion zone

                    if overlap > 0:
                        repulsion_strength = 100  # tune this
                        total_force += (repulsion_strength * overlap / dist_to_obs) * (obs_vec)

            ##############################
            # Goal Broadcasting System (Discovery-Based)
            ##############################
            if self.goal:
                # Check if agent discovers the goal (enters detection radius)
                goal_force = self.goal.process_agent_discovery(i, pos_i, vel_i, c1_gamma, c2_gamma)
                total_force += goal_force
            # NO gamma force when goal system is active - pure exploration via LJ + Langevin

            forces[i] = total_force

        return forces

    def update(self, forces, temp):
        n = len(self.agents)
        for i in range(n):
            #pos_i = self.agents[i, :2]
            v = self.agents[i, 2:]
            #wall_margin = 10.0
            #repulsion_strength = 1.0
            #x, y = self.agents[i, :2]
            #f = forces[i]

            ############################
            #Langevin Thermostat Velocity Update
            ############################

            mass = 1
            friction = 3
            kB = 1
            noise = np.random.normal(0, 1, size=2)

            c1 = np.exp(-friction * self.dt)
            c2 = np.sqrt((1 - c1**2) * kB * temp / mass)            
            v_new = v * c1 + ((forces[i] / mass) * self.dt) + (c2 * noise)

            #a = (2 - friction * self.dt) / (2 + friction * self.dt)
            #b = np.sqrt(kB * temp * (self.dt / 2))
            # = (2 * self.dt) / (2 + friction * self.dt)
            #v_new = (a * v) + (b * np.sqrt(1)) + ((self.dt/2)(forces[i]))

            self.agents[i, 2:] = v_new
            self.agents[i, :2] += v_new * self.dt

            #for ETH ZURICH Verlet
            #self.agents[i, :2] = self.agents[i, :2] + (v_new * c)

            #Broken piece of code but does the pbc
            #box_length = self.bounds[1] - self.bounds[0]
            #self.agents[i, :2] = (self.agents[i, :2] - self.bounds[0]) % box_length + self.bounds[0]
            
            ############################
            #Hard Edge Boundary
            ############################

            x, y = self.agents[i, :2]

            if x < self.bounds[0]:
                self.agents[i, 0] = self.bounds[0]
                self.agents[i, 2] *= -1
            elif x > self.bounds[1]:
                self.agents[i, 0] = self.bounds[1]
                self.agents[i, 2] *= -1

            if y < self.bounds[0]:
                self.agents[i, 1] = self.bounds[0]
                self.agents[i, 3] *= -1
            elif y > self.bounds[1]:
                self.agents[i, 1] = self.bounds[1]
                self.agents[i, 3] *= -1 

            #PBC
            #for dim in [0, 1]:
            #    if self.agents[i, dim] < self.bounds[0]:
            #        self.agents[i, dim] = self.bounds[1] - (self.bounds[0] - self.agents[i, dim]) % (self.bounds[1] - self.bounds[0])
            #    elif self.agents[i, dim] > self.bounds[1]:
            #        self.agents[i, dim] = self.bounds[0] + (self.agents[i, dim] - self.bounds[1]) % (self.bounds[1] - self.bounds[0])
