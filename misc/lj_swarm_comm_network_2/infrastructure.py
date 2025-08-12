#######################################################################
# infrastructure.py
#
# Infrastructure Node System for Communication Networks
# Active nodes that attract agents for robust coverage
#
# Author: Humzah Durrani 
#######################################################################

##########################
# Importing Libraries
##########################

import numpy as np

##########################
# Infrastructure setup
##########################

def sigma_1(z):
    return z / np.sqrt(1 + z ** 2)

class InfrastructureNode:

    ##########################
    # Init Class
    ##########################
    
    def __init__(self, position, comm_radius, attraction_strength=15.0, 
                 crystallization_radius=None, max_agents=50):

        #Init starting values
        self.position = np.array(position)
        self.comm_radius = comm_radius
        self.attraction_strength = attraction_strength
        self.crystallization_radius = crystallization_radius if crystallization_radius else comm_radius * 0.7
        self.max_agents = max_agents
        
        # State tracking
        self.connected_agents = set()  # agents within crystallization radius
        self.in_range_agents = set()   # agents within communication radius
        self.is_active = True
        self.is_saturated = False


    ##########################
    # Agent-Node Control
    ##########################
        
    def process_agent_interaction(self, agent_idx, agent_pos, agent_vel, c1_gamma, c2_gamma):

        distance_to_node = np.linalg.norm(agent_pos - self.position)
        
        # Track agents in communication range
        if distance_to_node <= self.comm_radius:
            self.in_range_agents.add(agent_idx)
        else:
            self.in_range_agents.discard(agent_idx)
            
        # Track agents in crystallization range
        if distance_to_node <= self.crystallization_radius:
            if agent_idx not in self.connected_agents:
                self.connected_agents.add(agent_idx)
                #print(f"Agent {agent_idx} connected to infrastructure node at {self.position}")
                
                # Check if node is saturated
                if len(self.connected_agents) >= self.max_agents:
                    self.is_saturated = True
                    #print(f"Infrastructure node at {self.position} saturated ({self.max_agents} agents)")
        else:
            # Agent moved out of crystallization radius
            if agent_idx in self.connected_agents:
                self.connected_agents.discard(agent_idx)
                self.is_saturated = False  # No longer saturated
                
        # Calculate force based on distance and node state
        if distance_to_node <= self.comm_radius and self.is_active and not self.is_saturated:
            # Apply attraction force similar to goal force
            objective = self.position - agent_pos
            
            # Connected agents get stronger force to maintain crystallization
            if agent_idx in self.connected_agents:
                # Strong crystallization force to keep agents in formation
                force_strength = self.attraction_strength * 1.5
                crystallization_force = force_strength * c1_gamma * sigma_1(objective) - c2_gamma * agent_vel
                return crystallization_force
            else:
                # Regular attraction force for agents in communication range
                attraction_force = self.attraction_strength * c1_gamma * sigma_1(objective) - c2_gamma * agent_vel
                return attraction_force
        else:
            # No force applied
            return np.zeros(2)
        
    ##########################
    # Agents inside crystal
    ##########################
    
    def get_connected_count(self):
        return len(self.connected_agents)
    
    ##########################
    # Agents in Comms Range
    ##########################

    def get_in_range_count(self):
        return len(self.in_range_agents)
    
    ##################################
    # CL Line Visualization of Agents
    ##################################
    
    def get_status_string(self):
        if self.is_saturated:
            return f"Saturated ({len(self.connected_agents)}/{self.max_agents})"
        elif len(self.connected_agents) > 0:
            return f"Active ({len(self.connected_agents)}/{self.max_agents})"
        else:
            return "Idle"
    
    ##########################
    # Clear out frame data
    ##########################
    
    def reset_frame_data(self):
        self.in_range_agents.clear()


##############################
# Infrastructure Node Control
##############################

class InfrastructureManager:

    ##############################
    # Init Nodes
    ##############################

    def __init__(self):
        self.nodes = []
    
    ##############################
    # Add nodes into control
    ##############################

    def add_node(self, position, comm_radius, attraction_strength=15.0, 
                 crystallization_radius=None, max_agents=50):
        node = InfrastructureNode(position, comm_radius, attraction_strength, 
                                crystallization_radius, max_agents)
        self.nodes.append(node)
        return node
    
    ##############################
    # Add node config information
    ##############################

    def add_nodes_from_config(self, infrastructure_list, infrastructure_config):
        
        #Pull infrastructure data from config dictionary
        comm_radius = infrastructure_config.get('broadcast radius')
        attraction_strength = infrastructure_config.get('attraction_strength')
        max_agents = infrastructure_config.get('max_agents')
        
        for pos in infrastructure_list:
            self.add_node(pos, comm_radius, attraction_strength, max_agents=max_agents)
    
    ##############################
    # Compute Node Forces on system
    ##############################  
    
    def compute_infrastructure_forces(self, agent_idx, agent_pos, agent_vel, c1_gamma, c2_gamma):

        total_force = np.zeros(2)
        
        for node in self.nodes:
            force = node.process_agent_interaction(agent_idx, agent_pos, agent_vel, c1_gamma, c2_gamma)
            total_force += force
            
        return total_force
    

    ##############################
    # Checks agents across all nodes
    ##############################

    def get_total_connected_agents(self):
        return sum(node.get_connected_count() for node in self.nodes)
    
    
    ##############################
    # Coverage Stats
    ##############################
    
    def get_coverage_stats(self):
        stats = {
            'total_nodes': len(self.nodes),
            'active_nodes': sum(1 for node in self.nodes if node.get_connected_count() > 0),
            'saturated_nodes': sum(1 for node in self.nodes if node.is_saturated),
            'total_connected': self.get_total_connected_agents(),
            'avg_connections': self.get_total_connected_agents() / len(self.nodes) if self.nodes else 0
        }
        return stats
    
    ##############################
    # Reset Frame
    ##############################

    def reset_frame_data(self):
        for node in self.nodes:
            node.reset_frame_data()