#######################################################################
# goal.py
#
# Invisible Goal System with Detection and Trap Mechanics
#
# Author: Humzah Durrani
#######################################################################

import numpy as np

def sigma_1(z):
    """Sigmoid function used in gamma force calculation"""
    return z / np.sqrt(1 + z ** 2)

class InvisibleGoal:
    def __init__(self, gamma_pos, detection_radius, trap_strength, trap_radius=None):
        self.position = gamma_pos
        self.detection_radius = detection_radius
        self.trap_radius = trap_radius if trap_radius is not None else detection_radius * 0.6
        self.trap_strength = trap_strength
        self.trapped_agents = set()
        self.detecting_agents = set()
        
    def process_agent(self, agent_idx, agent_pos, agent_vel, c1_gamma, c2_gamma):
        """
        Process an agent's interaction with the goal.
        Returns: (use_normal_gamma, enhanced_force_vector, is_detecting)
        """
        distance_to_goal = np.linalg.norm(agent_pos - self.position)
        
        # Check detection radius (goal becomes visible)
        is_detecting = distance_to_goal <= self.detection_radius
        if is_detecting:
            self.detecting_agents.add(agent_idx)
        else:
            self.detecting_agents.discard(agent_idx)
            
        # Check trap radius (strong attraction + no escape)
        is_trapped = distance_to_goal <= self.trap_radius
        if is_trapped:
            self.trapped_agents.add(agent_idx)
        
        # Once trapped, agent cannot escape (even if it moves outside trap_radius)
        if agent_idx in self.trapped_agents:
            # Enhanced gamma force with trap strength multiplier
            objective = self.position - agent_pos
            enhanced_gamma = self.trap_strength * c1_gamma * sigma_1(objective) - c2_gamma * agent_vel
            return False, enhanced_gamma, is_detecting
        else:
            # Use normal gamma force
            return True, np.zeros(2), is_detecting
    
    def is_goal_visible(self):
        """Returns True if any agent is detecting the goal"""
        return len(self.detecting_agents) > 0
    
    def get_trapped_count(self):
        """Returns number of trapped agents"""
        return len(self.trapped_agents)
    
    def reset_detection(self):
        """Reset detection tracking (call each frame)"""
        self.detecting_agents.clear()