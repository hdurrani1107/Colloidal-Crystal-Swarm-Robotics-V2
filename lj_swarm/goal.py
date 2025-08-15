#######################################################################
# goal.py
#
# 
# THIS IS ENTIRELY REDUNDANT CODE. THIS WAS THE ORIGINAL MECHANICAL
# WHERE THE GOALS WERE INVISIBLE
# 
# Invisible Goal System with Detection and Trap Mechanics
#
# Author: Humzah Durrani
#######################################################################

########################
# Importing Libraries
########################

import numpy as np

########################
# Sigmoid Function
########################

def sigma_1(z): 
    return z / np.sqrt(1 + z ** 2) 

########################
# Invisible Goal Class
########################
class InvisibleGoal:
    def __init__(self, gamma_pos, detection_radius, trap_strength, trap_radius=None, max_capacity=10):
        #Goal Params
        self.position = gamma_pos
        self.detection_radius = detection_radius
        self.trap_radius = trap_radius if trap_radius is not None else detection_radius * 0.6
        self.trap_strength = trap_strength
        self.max_capacity = max_capacity
        # State tracking
        self.trapped_agents = set()
        self.discovering_agents = set()
        self.is_discovered = False
        self.is_broadcasting = False
        self.is_full = False
    
    ########################
    # Goal Interaction When Discovered
    ########################
    
    def process_agent_discovery(self, agent_idx, agent_pos, agent_vel, c1_gamma, c2_gamma):
        #Applies force vector when agent discovers goal
        distance_to_goal = np.linalg.norm(agent_pos - self.position)
        
        #Agent stumbles into detection radius
        if distance_to_goal <= self.detection_radius and not self.is_discovered:
            self.is_discovered = True
            self.is_broadcasting = True
            print(f"GOAL DISCOVERED by Agent {agent_idx}!")
        
        # Track agents currently in detection range
        if distance_to_goal <= self.detection_radius:
            self.discovering_agents.add(agent_idx)
        else:
            self.discovering_agents.discard(agent_idx)
            
        #Agent enters trap radius
        if distance_to_goal <= self.trap_radius:
            if agent_idx not in self.trapped_agents:
                self.trapped_agents.add(agent_idx)
                print(f"Agent {agent_idx} trapped in goal!")
                
                # Check if goal is full
                if len(self.trapped_agents) >= self.max_capacity:
                    self.is_full = True
                    self.is_broadcasting = False
                    print(f"Goal full ({self.max_capacity} agents) - Broadcasting OFF")
        
        # FORCE CALCULATION
        if agent_idx in self.trapped_agents:
            # Trapped agents get strong attractive force
            objective = self.position - agent_pos
            trap_force = self.trap_strength * c1_gamma * sigma_1(objective) - c2_gamma * agent_vel
            return trap_force
        elif self.is_broadcasting and self.is_discovered:
            # Non-trapped agents get normal gamma force when goal is broadcasting
            objective = self.position - agent_pos
            broadcast_force = c1_gamma * sigma_1(objective) - c2_gamma * agent_vel
            return broadcast_force
        else:
            # No goal force
            return np.zeros(2)
    
    ########################
    # Goal Visibility
    ########################
    
    def is_goal_visible(self):
        return self.is_discovered
    
    ########################
    # Trapped Agent Count
    ########################
    
    def get_trapped_count(self):
        return len(self.trapped_agents)
    
    ########################
    # Goal Status for CL
    ########################
    
    def get_status_string(self):
        #Returns Status to CL
        if not self.is_discovered:
            return "Hidden"
        elif self.is_broadcasting:
            return f"Broadcasting ({len(self.trapped_agents)}/{self.max_capacity})"
        elif self.is_full:
            return f"Full ({len(self.trapped_agents)}/{self.max_capacity})"
        else:
            return "Inactive"
    
    ########################
    # Resets Tracking Data
    ########################
    
    def reset_frame_data(self):
        self.discovering_agents.clear()