#######################################################################
# swarm-engine.py
#
# Engine behind swarm behavior  
#
# Author: Humzah Durrani
# AI Disclosure: Only used for debugging and vectorized force compute
#######################################################################

##########################
# Importing Libraries
##########################
import numpy as np
from lj_swarm.cooling_zone import CoolingZoneSystem

##########################
# Sigmoid Function
##########################

def sigma_1(z):
    return z / np.sqrt(1 + z ** 2)


############################
# Temperature based Epsilon
############################

def epsilon_temp(T, epsilon, alpha):
    #Unused
    return epsilon * np.exp(-alpha * T)

############################
# Melt Initial Condition
############################

def init_melt(agents, init_temp, mass=1, kB=1):
    #Starting off with initial noise sampled from Maxwell Boltzmann Distribution
    N = len(agents)
    std_dev = np.sqrt(kB * init_temp / mass)
    agents[:, 2:] = np.random.normal(0, std_dev, size=(N, 2))

######################################
# Ideal Hex-Grid Initial Condition
#######################################

def init_grid(n, bounds, spacing, center=(100, 100)):
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

#######################################
# Random No-Overlap Initial Condition
########################################

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

############################
# Multi-Agent Class
############################
class multi_agent:

    def __init__(self, number, sigma, sampletime, bounds, obstacles):
        #Initial LJ-Swarm Params
        self.dt = sampletime
        self.bounds = bounds
        self.obstacles = obstacles
        self.cooling_zones = None  # Will be set up later if needed
        spacing = (2**(1/6)) * sigma
        self.agents = init_grid(number, bounds, spacing)
        #self.agents = init_grid_random_no_overlap(number, bounds, spacing)
        
        # Individual agent temperatures (initially set to system temperature)
        self.agent_temperatures = np.full(number, 150.0)  # Will be updated with actual initial temp
    
    ############################
    # Initial Agent Temperature
    ############################
    def initialize_agent_temperatures(self, initial_temp):
        self.agent_temperatures.fill(initial_temp)
    
    ############################
    # Cooling zone setup
    ############################
    def setup_cooling_zones(self, zone_config):
        #Cooling Zone Initialization
        self.cooling_zones = CoolingZoneSystem(
            bounds=self.bounds,
            zone_radius=zone_config['zone_radius'],
            spawn_interval=zone_config['spawn_interval'],
            base_lifetime=zone_config['base_lifetime'],
            zone_temperature=zone_config['zone_temperature'],
            max_concurrent_zones=zone_config.get('max_concurrent_zones', 3),
            radius_std=zone_config.get('radius_std', 2.0),
            logger=zone_config.get('logger', None)
        )
    
    ############################
    # Agent Neighbor Count
    ############################

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
    
    #################################################
    # Update Agent Temperature based on Cooling Zone
    #################################################

    def update_agent_temperatures(self, system_temp):
        cooling_rate = 0.05
        
        for i in range(len(self.agents)):
            if self.cooling_zones and self.cooling_zones.is_agent_trapped(i):
                # Agent is in cooling zone - gradually cool toward zone temperature
                zone_pos, zone_radius = self.cooling_zones.get_zone_info_for_agent(i)
                if zone_pos is not None:
                    # Get zone temperature from the zone that trapped this agent
                    for zone in self.cooling_zones.zones:
                        if i in zone.trapped_agents:
                            target_temp = zone.zone_temperature
                            break
                    else:
                        target_temp = system_temp * 0.1  # fallback
                    
                    # Exponential approach to target temperature
                    temp_diff = target_temp - self.agent_temperatures[i]
                    self.agent_temperatures[i] += cooling_rate * temp_diff
            else:
                # Agent is free - gradually warm toward system temperature
                temp_diff = system_temp - self.agent_temperatures[i]
                self.agent_temperatures[i] += cooling_rate * temp_diff

    ###############################################
    # Compute Forces (Vectorized) (Experimental)
    ###############################################
    
    def compute_forces(self, sigma, epsilon, distance, temp, c1_gamma, c2_gamma, alpha=0.5):
        n = len(self.agents)
        positions = self.agents[:, :2]
        
        ##################################
        # Vectorized LJ Force Calculation
        ###################################
        
        # Compute all pairwise distance vectors at once
        pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # Shape: (n, n, 2)
        distances = np.linalg.norm(pos_diff, axis=2)  # Shape: (n, n)
        
        # Create mask: avoid self-interaction, apply distance cutoffs
        mask = (distances > 1e-5) & (distances < 3*sigma) & ~np.eye(n, dtype=bool)
        
        # Vectorized LJ potential calculation with safe division
        safe_distances = np.where(distances > 1e-5, distances, 1.0)  # Avoid division by zero
        inv_r = 1.0 / safe_distances
        inv_r6 = (sigma * inv_r) ** 6
        inv_r12 = inv_r6 ** 2
        lj_scalar = 24 * epsilon * (2 * inv_r12 - inv_r6) * inv_r
        
        # Apply mask and compute force directions with safe division
        lj_scalar = np.where(mask, lj_scalar, 0)
        directions = pos_diff / safe_distances[:, :, np.newaxis]  # Safe division
        directions = np.where(mask[:, :, np.newaxis], directions, 0)  # Apply mask
        
        # Sum forces for each agent (broadcasting lj_scalar across direction vectors)
        forces = np.sum(lj_scalar[:, :, np.newaxis] * directions, axis=1)

        ################################
        # Obstacle Forces (still loop-based for now)
        #################################
        
        #Toggled off for now
        if self.obstacles:
            for i in range(n):
                pos_i = positions[i]
                for obs_pos, obs_radius in self.obstacles:
                    obs_vec = pos_i - obs_pos
                    dist_to_obs = np.linalg.norm(obs_vec)
                    overlap = obs_radius + 1.5 * sigma - dist_to_obs  # soft repulsion zone

                    if overlap > 0:
                        repulsion_strength = 100  # tune this
                        forces[i] += (repulsion_strength * overlap / dist_to_obs) * (obs_vec)

        
        ##############################
        # Cooling Zone System
        ##############################

        if self.cooling_zones:
            for i in range(n):
                pos_i = positions[i]
                vel_i = self.agents[i, 2:]
                # Process agent interaction with cooling zones
                self.cooling_zones.process_agent(i, pos_i, vel_i)

        return forces

    ############################
    # Original Compute Forces
    ############################
    def compute_forces_original(self, sigma, epsilon, distance, temp, c1_gamma, c2_gamma, alpha=0.5):
        #Kept In-case above failed
        n = len(self.agents)
        forces = np.zeros((n, 2))

        #Nested For-Loop very slow

        for i in range(n):
            pos_i = self.agents[i, :2]
            vel_i = self.agents[i, 2:]
            total_force = np.zeros(2)

            for j in range(n):
                if i == j:
                    continue
                pos_j = self.agents[j, :2]
                offset = pos_i - pos_j
                dist = np.linalg.norm(offset)

                if dist > 1e-5 and dist < (3*sigma):
                    inv_r = 1.0 / dist
                    inv_r6 = (sigma * inv_r) ** 6
                    inv_r12 = inv_r6 ** 2
                    lj_scalar = 24 * epsilon * (2 * inv_r12 - inv_r6) * inv_r
                    total_force += lj_scalar * (offset / dist)

            # Obstacle forces
            if self.obstacles:
                for obs_pos, obs_radius in self.obstacles:
                    obs_vec = pos_i - obs_pos
                    dist_to_obs = np.linalg.norm(obs_vec)
                    overlap = obs_radius + 1.5 * sigma - dist_to_obs

                    if overlap > 0:
                        repulsion_strength = 100
                        total_force += (repulsion_strength * overlap / dist_to_obs) * (obs_vec)

            # Cooling zone processing
            if self.cooling_zones:
                self.cooling_zones.process_agent(i, pos_i, vel_i)

            forces[i] = total_force

        return forces

    ###########################
    # Update everything
    ############################

    def update(self, forces, temp):
        # Update cooling zones first
        if self.cooling_zones:
            self.cooling_zones.frame += 1
            self.cooling_zones.update()
            
        # Update individual agent temperatures
        self.update_agent_temperatures(temp)
        
        n = len(self.agents)
        for i in range(n):
            #pos_i = self.agents[i, :2]
            v = self.agents[i, 2:]
            #wall_margin = 10.0
            #repulsion_strength = 1.0
            #x, y = self.agents[i, :2]
            #f = forces[i]

            #####################################
            #Langevin Thermostat Velocity Update
            #####################################

            mass = 1
            friction = 3
            kB = 1
            noise = np.random.normal(0, 1, size=2)

            # Use individual agent temperature
            agent_temp = self.agent_temperatures[i]
            
            c1 = np.exp(-friction * self.dt)
            c2 = np.sqrt((1 - c1**2) * kB * agent_temp / mass)            
            v_new = v * c1 + ((forces[i] / mass) * self.dt) + (c2 * noise)

            #a = (2 - friction * self.dt) / (2 + friction * self.dt)
            #b = np.sqrt(kB * temp * (self.dt / 2))
            # = (2 * self.dt) / (2 + friction * self.dt)
            #v_new = (a * v) + (b * np.sqrt(1)) + ((self.dt/2)(forces[i]))

            self.agents[i, 2:] = v_new
            new_pos = self.agents[i, :2] + v_new * self.dt
            
            # Apply cooling zone boundary constraints
            if self.cooling_zones and self.cooling_zones.is_agent_trapped(i):
                zone_pos, zone_radius = self.cooling_zones.get_zone_info_for_agent(i)
                if zone_pos is not None:
                    # Check if new position would leave the zone
                    dist_to_center = np.linalg.norm(new_pos - zone_pos)
                    if dist_to_center > zone_radius:
                        # Keep agent inside the boundary
                        direction = (new_pos - zone_pos) / dist_to_center
                        new_pos = zone_pos + direction * zone_radius
            
            self.agents[i, :2] = new_pos

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
