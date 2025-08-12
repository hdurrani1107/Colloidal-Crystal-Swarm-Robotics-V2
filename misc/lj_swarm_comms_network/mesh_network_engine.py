#######################################################################
# mesh_network_engine.py
#
# Engine for swarm-based mesh network coverage using RSSI
#
# Author: Humzah Durrani 
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum

class NodeType(Enum):
    INFRASTRUCTURE = "infrastructure"
    MOBILE_AGENT = "mobile_agent"

@dataclass
class NetworkNode:
    """Fixed infrastructure communication nodes"""
    position: np.ndarray
    signal_strength: float  # RSSI transmission power
    coverage_radius: float
    node_id: int
    node_type: NodeType = NodeType.INFRASTRUCTURE
    is_active: bool = True

@dataclass
class DataPacket:
    """Data packet for network simulation"""
    packet_id: int
    source_id: int
    destination_id: int
    data: str
    hop_count: int = 0
    path: List[int] = None
    
    def __post_init__(self):
        if self.path is None:
            self.path = [self.source_id]

class MeshNetworkAgent:
    """Mobile agent with RSSI capabilities and LJ physics"""
    
    def __init__(self, agent_id: int, position: np.ndarray, 
                 sigma: float = 3.0, epsilon: float = 3.0, 
                 rssi_strength: float = 10.0, temp: float = 1.0):
        self.agent_id = agent_id
        self.position = position.copy().astype(np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.sigma = sigma
        self.epsilon = epsilon
        self.rssi_strength = rssi_strength
        self.temperature = temp
        self.target_temperature = temp
        
        # Network properties
        self.max_comm_range = rssi_strength * 2.0  # RSSI-based communication range
        self.connected_nodes: Set[int] = set()
        self.connected_agents: Set[int] = set()
        self.data_packets: List[DataPacket] = []
        self.coverage_quality = 0.0
        
        # State tracking
        self.is_deployed = False
        self.target_position: Optional[np.ndarray] = None
        self.coverage_priority = 0.0

def calculate_rssi(distance: float, tx_power: float, path_loss_exp: float = 2.0) -> float:
    """Calculate RSSI based on distance and transmission power"""
    if distance < 0.1:  # Avoid division by zero
        distance = 0.1
    return tx_power - 20 * path_loss_exp * np.log10(distance)

def rssi_to_quality(rssi: float, min_rssi: float = -80.0, max_rssi: float = -20.0) -> float:
    """Convert RSSI to connection quality (0-1)"""
    return max(0, min(1, (rssi - min_rssi) / (max_rssi - min_rssi)))

class MeshNetworkSimulation:
    """Main simulation engine for swarm mesh network"""
    
    def __init__(self, bounds: List[float], sample_time: float = 0.005):
        self.bounds = bounds
        self.dt = sample_time
        self.agents: List[MeshNetworkAgent] = []
        self.infrastructure_nodes: List[NetworkNode] = []
        self.obstacles: List[Tuple[np.ndarray, float]] = []
        self.deadzones: List[Tuple[np.ndarray, float]] = []  # Areas with poor signal
        self.noise_zones: List[Tuple[np.ndarray, float, float]] = []  # (pos, radius, noise_level)
        
        # Network metrics
        self.coverage_map = None
        self.connectivity_matrix = None
        self.data_packets: List[DataPacket] = []
        self.packet_counter = 0
        
        # Physics parameters
        self.friction = 3.0
        self.mass = 1.0
        self.kB = 1.0
        
    def add_infrastructure_node(self, position: np.ndarray, signal_strength: float, 
                              coverage_radius: float) -> int:
        """Add a fixed communication node"""
        node_id = len(self.infrastructure_nodes)
        node = NetworkNode(
            position=position.copy().astype(np.float64),
            signal_strength=signal_strength,
            coverage_radius=coverage_radius,
            node_id=node_id
        )
        self.infrastructure_nodes.append(node)
        return node_id
    
    def add_agent(self, position: np.ndarray, **kwargs) -> int:
        """Add a mobile mesh agent"""
        agent_id = len(self.agents)
        agent = MeshNetworkAgent(agent_id, position, **kwargs)
        self.agents.append(agent)
        return agent_id
    
    def add_obstacle(self, position: np.ndarray, radius: float):
        """Add physical obstacle"""
        self.obstacles.append((position.copy().astype(np.float64), radius))
    
    def add_deadzone(self, position: np.ndarray, radius: float):
        """Add communication deadzone"""
        self.deadzones.append((position.copy().astype(np.float64), radius))
    
    def add_noise_zone(self, position: np.ndarray, radius: float, noise_level: float):
        """Add noise interference zone"""
        self.noise_zones.append((position.copy().astype(np.float64), radius, noise_level))
    
    def compute_coverage_gaps(self, grid_resolution: int = 50) -> np.ndarray:
        """Compute coverage quality across the environment"""
        x = np.linspace(self.bounds[0], self.bounds[1], grid_resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], grid_resolution)
        X, Y = np.meshgrid(x, y)
        coverage = np.zeros_like(X)
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                point = np.array([X[i, j], Y[i, j]])
                max_quality = 0.0
                
                # Check coverage from infrastructure nodes
                for node in self.infrastructure_nodes:
                    if not node.is_active:
                        continue
                    dist = np.linalg.norm(point - node.position)
                    if dist <= node.coverage_radius:
                        rssi = calculate_rssi(dist, node.signal_strength)
                        quality = rssi_to_quality(rssi)
                        max_quality = max(max_quality, quality)
                
                # Check coverage from mobile agents
                for agent in self.agents:
                    dist = np.linalg.norm(point - agent.position)
                    if dist <= agent.max_comm_range:
                        rssi = calculate_rssi(dist, agent.rssi_strength)
                        quality = rssi_to_quality(rssi)
                        max_quality = max(max_quality, quality)
                
                # Apply deadzone effects
                for deadzone_pos, deadzone_radius in self.deadzones:
                    dist_to_dead = np.linalg.norm(point - deadzone_pos)
                    if dist_to_dead <= deadzone_radius:
                        max_quality *= max(0, 1 - (deadzone_radius - dist_to_dead) / deadzone_radius)
                
                coverage[i, j] = max_quality
        
        self.coverage_map = coverage
        return coverage
    
    def find_coverage_targets(self) -> List[np.ndarray]:
        """Find optimal positions for agents to improve coverage"""
        if self.coverage_map is None:
            self.compute_coverage_gaps()
        
        # Find low coverage areas
        low_coverage_threshold = 0.3
        targets = []
        
        grid_resolution = self.coverage_map.shape[0]
        x = np.linspace(self.bounds[0], self.bounds[1], grid_resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], grid_resolution)
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                if self.coverage_map[i, j] < low_coverage_threshold:
                    # Check if position is not in obstacle
                    pos = np.array([x[j], y[i]])
                    valid_position = True
                    
                    for obs_pos, obs_radius in self.obstacles:
                        if np.linalg.norm(pos - obs_pos) < obs_radius + 2.0:
                            valid_position = False
                            break
                    
                    if valid_position:
                        targets.append(pos)
        
        return targets
    
    def update_network_connectivity(self):
        """Update connectivity between all network elements"""
        # Reset connections
        for agent in self.agents:
            agent.connected_nodes.clear()
            agent.connected_agents.clear()
        
        # Agent to infrastructure connections
        for agent in self.agents:
            for node in self.infrastructure_nodes:
                if not node.is_active:
                    continue
                dist = np.linalg.norm(agent.position - node.position)
                rssi = calculate_rssi(dist, min(agent.rssi_strength, node.signal_strength))
                
                # Apply noise interference
                for noise_pos, noise_radius, noise_level in self.noise_zones:
                    noise_dist = np.linalg.norm(agent.position - noise_pos)
                    if noise_dist <= noise_radius:
                        rssi -= noise_level
                
                if rssi > -80.0:  # Minimum viable RSSI
                    agent.connected_nodes.add(node.node_id)
        
        # Agent to agent connections
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents):
                if i >= j:
                    continue
                
                dist = np.linalg.norm(agent1.position - agent2.position)
                rssi = calculate_rssi(dist, min(agent1.rssi_strength, agent2.rssi_strength))
                
                # Apply noise interference
                for noise_pos, noise_radius, noise_level in self.noise_zones:
                    mid_point = (agent1.position + agent2.position) / 2
                    noise_dist = np.linalg.norm(mid_point - noise_pos)
                    if noise_dist <= noise_radius:
                        rssi -= noise_level
                
                if rssi > -80.0:
                    agent1.connected_agents.add(agent2.agent_id)
                    agent2.connected_agents.add(agent1.agent_id)
    
    def compute_lj_forces(self) -> np.ndarray:
        """Compute Lennard-Jones forces between agents"""
        n = len(self.agents)
        forces = np.zeros((n, 2))
        
        for i in range(n):
            agent_i = self.agents[i]
            total_force = np.zeros(2)
            
            # Agent-agent LJ interactions
            for j in range(n):
                if i == j:
                    continue
                
                agent_j = self.agents[j]
                offset = agent_i.position - agent_j.position
                dist = np.linalg.norm(offset)
                
                if dist > 1e-5:
                    sigma = (agent_i.sigma + agent_j.sigma) / 2
                    epsilon = np.sqrt(agent_i.epsilon * agent_j.epsilon)
                    
                    inv_r = 1.0 / dist
                    inv_r6 = (sigma * inv_r) ** 6
                    inv_r12 = inv_r6 ** 2
                    lj_scalar = 24 * epsilon * (2 * inv_r12 - inv_r6) * inv_r
                    total_force += lj_scalar * (offset / dist)
            
            # Obstacle repulsion
            for obs_pos, obs_radius in self.obstacles:
                obs_vec = agent_i.position - obs_pos
                dist_to_obs = np.linalg.norm(obs_vec)
                overlap = obs_radius + 1.5 * agent_i.sigma - dist_to_obs
                
                if overlap > 0:
                    repulsion_strength = 100
                    total_force += (repulsion_strength * overlap / dist_to_obs) * obs_vec
            
            # Coverage-based attraction to target positions
            targets = self.find_coverage_targets()
            if targets and not agent_i.is_deployed:
                # Find closest uncovered target
                distances = [np.linalg.norm(agent_i.position - target) for target in targets]
                closest_idx = np.argmin(distances)
                target = targets[closest_idx]
                
                # Attractive force toward coverage gap
                coverage_vec = target - agent_i.position
                coverage_dist = np.linalg.norm(coverage_vec)
                
                if coverage_dist > 0.5:  # Not close enough yet
                    coverage_strength = 50 * agent_i.temperature  # Higher temp = faster movement
                    total_force += coverage_strength * (coverage_vec / coverage_dist)
                else:
                    agent_i.is_deployed = True
                    agent_i.target_temperature = 0.1  # Cool down when deployed
            
            forces[i] = total_force
        
        return forces
    
    def update_agents(self, forces: np.ndarray):
        """Update agent positions using Langevin dynamics"""
        for i, agent in enumerate(self.agents):
            # Temperature adaptation
            temp_diff = agent.target_temperature - agent.temperature
            agent.temperature += 0.01 * temp_diff  # Smooth temperature change
            
            # Langevin thermostat
            noise = np.random.normal(0, 1, size=2)
            c1 = np.exp(-self.friction * self.dt)
            c2 = np.sqrt((1 - c1**2) * self.kB * agent.temperature / self.mass)
            
            agent.velocity = (agent.velocity * c1 + 
                            (forces[i] / self.mass) * self.dt + 
                            c2 * noise)
            
            agent.position += agent.velocity * self.dt
            
            # Boundary conditions
            for dim in range(2):
                if agent.position[dim] < self.bounds[0]:
                    agent.position[dim] = self.bounds[0]
                    agent.velocity[dim] *= -1
                elif agent.position[dim] > self.bounds[1]:
                    agent.position[dim] = self.bounds[1]
                    agent.velocity[dim] *= -1
    
    def create_data_packet(self, source_id: int, destination_id: int, data: str) -> int:
        """Create a new data packet for transmission"""
        packet = DataPacket(
            packet_id=self.packet_counter,
            source_id=source_id,
            destination_id=destination_id,
            data=data
        )
        self.data_packets.append(packet)
        self.packet_counter += 1
        return packet.packet_id
    
    def route_packet(self, packet: DataPacket) -> bool:
        """Attempt to route a packet through the network"""
        # Simple routing algorithm - find path through connected nodes/agents
        current_pos = None
        destination_pos = None
        
        # Find current and destination positions
        if packet.source_id < len(self.infrastructure_nodes):
            current_pos = self.infrastructure_nodes[packet.source_id].position
        else:
            agent_id = packet.source_id - len(self.infrastructure_nodes)
            if agent_id < len(self.agents):
                current_pos = self.agents[agent_id].position
        
        if packet.destination_id < len(self.infrastructure_nodes):
            destination_pos = self.infrastructure_nodes[packet.destination_id].position
        else:
            agent_id = packet.destination_id - len(self.infrastructure_nodes)
            if agent_id < len(self.agents):
                destination_pos = self.agents[agent_id].position
        
        if current_pos is None or destination_pos is None:
            return False
        
        # For now, simple success if within communication range
        dist = np.linalg.norm(current_pos - destination_pos)
        return dist <= 20.0  # Simplified routing success criteria
    
    def step(self):
        """Single simulation step"""
        # Update network connectivity
        self.update_network_connectivity()
        
        # Compute and apply forces
        forces = self.compute_lj_forces()
        self.update_agents(forces)
        
        # Update coverage map periodically
        if np.random.random() < 0.1:  # Update every ~10 steps
            self.compute_coverage_gaps()
        
        # Process data packets
        packets_to_remove = []
        for packet in self.data_packets:
            if self.route_packet(packet):
                packets_to_remove.append(packet)
            else:
                packet.hop_count += 1
                if packet.hop_count > 50:  # Max hops exceeded
                    packets_to_remove.append(packet)
        
        for packet in packets_to_remove:
            self.data_packets.remove(packet)