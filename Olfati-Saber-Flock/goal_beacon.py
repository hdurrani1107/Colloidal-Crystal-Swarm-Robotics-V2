#######################################################################
# goal_beacon.py
#
# Goal Beacon System with Dynamic Spawning and Agent Trapping
# Adapted from cooling_zone.py for Olfati-Saber flocking simulation
#
# Author: Humzah Durrani
#######################################################################

import numpy as np
from typing import Optional
from metrics import MetricsLogger

class GoalBeacon:
    _next_id = 0
    def __init__(self, position, radius, base_lifetime, velocity_damping=0.95,
                 logger: Optional[MetricsLogger]=None):
        self.id = GoalBeacon._next_id; GoalBeacon._next_id += 1
        self.position = position
        self.radius = radius
        self.base_lifetime = base_lifetime
        self.velocity_damping = velocity_damping  # Factor to scale down velocities inside beacon
        self.trapped_agents = set()
        self.frames_remaining = None  # set when first agent enters
        self.is_active = True
        self.lifetime_started = False
        self.logger = logger
        self.birth_frame = None
        self.first_arrival_frame = None
        
    def add_agent(self, agent_idx):
        """Add an agent to this goal beacon"""
        if agent_idx not in self.trapped_agents:
            self.trapped_agents.add(agent_idx)
            print(f"Agent {agent_idx} entered goal beacon at {self.position}")
            
            # Start lifetime countdown if this is the first agent
            if not self.lifetime_started:
                self.lifetime_started = True
                print(f"Goal beacon lifetime countdown started!")
            
            self.update_lifetime()
    
    def update_lifetime(self):
        """Update the lifetime based on number of trapped agents, accounting for elapsed time."""
        num_agents = len(self.trapped_agents)
        if num_agents > 0:
            reduction_factor = 1.0 + (num_agents - 1) * 0.2  # Each extra agent reduces lifetime by 20%
            new_total = max(1, int(self.base_lifetime / reduction_factor))
            if self.lifetime_started and self.frames_remaining is not None:
                elapsed = self.base_lifetime - self.frames_remaining
                remaining = max(1, new_total - elapsed)
                self.frames_remaining = remaining
            else:
                self.frames_remaining = new_total
    
    def is_inside(self, position):
        """Check if a position is inside this goal beacon"""
        distance = np.linalg.norm(position - self.position)
        return distance <= self.radius
    
    def get_goal_force(self, agent_pos, agent_vel, c1_gamma=5.0, c2_gamma=2.0):
        """
        Calculate goal beacon force for an agent using Olfati-Saber gamma agent formulation
        Returns force towards beacon center with velocity damping
        """
        if not self.is_active:
            return np.zeros(2)
            
        # Direction from agent to beacon center
        direction_to_beacon = self.position - agent_pos
        distance = np.linalg.norm(direction_to_beacon)
        
        # Avoid division by zero
        if distance < 1e-6:
            return np.zeros(2)
        
        # Olfati-Saber gamma agent formulation adapted for 2D
        # u_gamma = -c1_gamma * sigma_1(agent_p - gamma_pos) - c2_gamma * agent_q
        def sigma_1(z):
            """Smooth version of identity function"""
            norm_z = np.linalg.norm(z)
            if norm_z < 1e-6:
                return z
            return z / np.sqrt(1 + norm_z ** 2)
        
        # Attraction to beacon position
        position_term = -c1_gamma * sigma_1(-direction_to_beacon)  # Note: negated for attraction
        
        # Velocity damping term
        velocity_term = -c2_gamma * agent_vel
        
        return position_term + velocity_term
    
    def update(self):
        """Update the goal beacon state"""
        if self.is_active and self.lifetime_started:
            # Only countdown if lifetime has started (first agent entered)
            self.frames_remaining -= 1
            if self.frames_remaining <= 0:
                self.is_active = False
                print(f"Goal beacon at {self.position} disappeared after reaching lifetime, freeing {len(self.trapped_agents)} agents")
                return False  # Beacon should be removed
        # Beacon stays active if no agents have entered yet or still has time left
        return True
    
    def get_trapped_agents(self):
        """Get set of trapped agent indices"""
        return self.trapped_agents.copy()

class GoalBeaconSystem:
    def __init__(self, bounds, beacon_radius=15.0, spawn_interval=500, base_lifetime=300, velocity_damping=0.95,
                 logger: Optional[MetricsLogger]=None):
        self.bounds = bounds
        self.beacon_radius = beacon_radius
        self.spawn_interval = spawn_interval
        self.base_lifetime = base_lifetime
        self.velocity_damping = velocity_damping
        self.beacons = []
        self.frames_since_last_spawn = 0
        self.agent_beacon_map = {}  # Maps agent_idx to beacon
        self.logger = logger
        self.frame = 0  # will be set by caller each tick
        self.spawn_beacon()
        
    def spawn_beacon(self):
        """Spawn a new goal beacon at a random location"""
        margin = self.beacon_radius + 5
        x = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
        y = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
        position = np.array([x, y])

        new_beacon = GoalBeacon(position, self.beacon_radius, self.base_lifetime,
                               self.velocity_damping, logger=self.logger)
        new_beacon.birth_frame = self.frame
        self.beacons.append(new_beacon)
        if self.logger:
            self.logger.log_event(
                frame=self.frame, t=None, event="beacon_spawn", beacon_id=new_beacon.id,
                beacon_pos_x=float(position[0]), beacon_pos_y=float(position[1]),
                beacon_radius=self.beacon_radius, trapped_count=0, extra=None
            )
        print(f"New goal beacon spawned at {position} with radius {self.beacon_radius}")
        print(f"Beacon bounds: x=[{self.bounds[0] + margin}, {self.bounds[1] - margin}], y=[{self.bounds[0] + margin}, {self.bounds[1] - margin}]")
        
    def process_agent(self, agent_idx, agent_pos, agent_vel):
        """Process an agent's interaction with goal beacons"""
        if agent_idx not in self.agent_beacon_map:
            for beacon in self.beacons:
                if beacon.is_active and beacon.is_inside(agent_pos):
                    beacon.add_agent(agent_idx)
                    self.agent_beacon_map[agent_idx] = beacon
                    if beacon.first_arrival_frame is None:
                        beacon.first_arrival_frame = self.frame
                        if self.logger:
                            self.logger.log_event(
                                frame=self.frame, t=None, event="first_arrival", beacon_id=beacon.id,
                                beacon_pos_x=float(beacon.position[0]), beacon_pos_y=float(beacon.position[1]),
                                beacon_radius=beacon.radius, trapped_count=len(beacon.trapped_agents), extra=None
                            )
                    print(f"DEBUG: Agent {agent_idx} entered beacon at {beacon.position}")
                    break
    
    def is_agent_trapped(self, agent_idx):
        """Check if an agent is trapped in a goal beacon"""
        if agent_idx in self.agent_beacon_map:
            beacon = self.agent_beacon_map[agent_idx]
            return beacon.is_active
        return False
    
    def get_beacon_info_for_agent(self, agent_idx):
        """Get beacon position and radius for a trapped agent"""
        if agent_idx in self.agent_beacon_map:
            beacon = self.agent_beacon_map[agent_idx]
            if beacon.is_active:
                return beacon.position, beacon.radius
        return None, None
    
    def get_goal_forces(self, agent_positions, agent_velocities):
        """
        Calculate goal beacon forces for all agents
        Returns array of forces for each agent
        """
        n_agents = len(agent_positions)
        forces = np.zeros((n_agents, 2))
        
        for i, (pos, vel) in enumerate(zip(agent_positions, agent_velocities)):
            total_force = np.zeros(2)
            
            # Each beacon contributes attraction force
            for beacon in self.beacons:
                if beacon.is_active:
                    beacon_force = beacon.get_goal_force(pos, vel)
                    total_force += beacon_force
            
            forces[i] = total_force
            
        return forces
    
    def update(self):
        """Update all goal beacons and spawn new ones"""
        active_beacons = []
        freed_agents = set()
        for beacon in self.beacons:
            if beacon.update():
                active_beacons.append(beacon)
            else:
                freed_agents.update(beacon.get_trapped_agents())
                if self.logger:
                    self.logger.log_event(
                        frame=self.frame, t=None, event="beacon_complete", beacon_id=beacon.id,
                        beacon_pos_x=float(beacon.position[0]), beacon_pos_y=float(beacon.position[1]),
                        beacon_radius=beacon.radius, trapped_count=len(beacon.trapped_agents),
                        extra={"lifetime_frames": (self.frame - (beacon.birth_frame or self.frame))}
                    )
        for agent_idx in freed_agents:
            if agent_idx in self.agent_beacon_map:
                del self.agent_beacon_map[agent_idx]

        beacons_disappeared = len(self.beacons) > len(active_beacons)
        self.beacons = active_beacons

        if beacons_disappeared and len(self.beacons) == 0:
            self.frames_since_last_spawn = 0

        self.frames_since_last_spawn += 1
        if len(self.beacons) == 0 and self.frames_since_last_spawn >= self.spawn_interval:
            self.spawn_beacon()
            self.frames_since_last_spawn = 0
    
    def get_active_beacons(self):
        """Get list of active goal beacons for visualization"""
        return [(beacon.position, beacon.radius, len(beacon.trapped_agents)) 
                for beacon in self.beacons if beacon.is_active]