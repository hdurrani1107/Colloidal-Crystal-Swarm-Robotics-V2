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
from olfati_saber_flock.metrics import MetricsLogger

class GoalBeacon:
    _next_id = 0
    def __init__(self, position, radius, base_lifetime, velocity_damping=0.985,
                 logger: Optional[MetricsLogger]=None):
        self.id = GoalBeacon._next_id; GoalBeacon._next_id += 1
        self.position = position
        self.radius = radius
        self.base_lifetime = base_lifetime
        self.velocity_damping = velocity_damping
        self.trapped_agents = set()
        self.frames_remaining = None
        self.is_active = True
        self.lifetime_started = False
        self.logger = logger
        self.birth_frame = None
        self.first_arrival_frame = None
        self.owner_flock_id = None  # REQUIRED reservation
        
    def add_agent(self, agent_idx, is_leader=False, flock_id=None):
        """Add an agent to this goal beacon"""
        if agent_idx in self.trapped_agents:
            return
        self.trapped_agents.add(agent_idx)

        # Reservation: first agent claims the beacon for their flock
        if self.owner_flock_id is None and flock_id is not None:
            self.owner_flock_id = flock_id
            print(f"Goal beacon at {self.position} reserved by flock {flock_id} (agent {agent_idx})")

        # Countdown starts when any agent enters (modified from LJ-Swarm logic)
        if not self.lifetime_started:
            self.lifetime_started = True
            print(f"Goal beacon lifetime countdown started by agent {agent_idx}!")

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
    
    def get_goal_force(self, agent_pos, agent_vel, agent_flock_id=None, c1_gamma=5.0, c2_gamma=2.0):
        """
        Calculate goal beacon force for an agent using Olfati-Saber gamma agent formulation
        No temperature dependency - just velocity dampening inside beacon
        """
        if not self.is_active:
            return np.zeros(2)
        # Only owner flock gets attraction (first-come-first-served reservation)
        if self.owner_flock_id is not None and agent_flock_id != self.owner_flock_id:
            return np.zeros(2)

        d = self.position - agent_pos
        dist = np.linalg.norm(d)
        if dist < 1e-6:
            return -c2_gamma * agent_vel
        
        # Olfati-Saber gamma force (attraction to beacon + velocity damping)
        from olfati_saber_flock.engine import sigma_1
        pos_term = -c1_gamma * sigma_1(agent_pos - self.position)
        vel_term = -c2_gamma * agent_vel
        return pos_term + vel_term
    
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
    def __init__(self, bounds, beacon_radius=15.0, spawn_interval=500, base_lifetime=300, velocity_damping=0.985,
                 max_concurrent_beacons=3, radius_std=2.0, logger: Optional[MetricsLogger]=None):
        self.bounds = bounds
        self.beacon_radius = beacon_radius  # Mean radius for normal distribution
        self.radius_std = radius_std        # Standard deviation for radius sampling
        self.spawn_interval = spawn_interval
        self.base_lifetime = base_lifetime
        self.velocity_damping = velocity_damping
        self.max_concurrent_beacons = max_concurrent_beacons
        self.beacons = []
        self.frames_since_last_spawn = 0
        self.agent_beacon_map = {}  # Maps agent_idx to beacon
        self.logger = logger
        self.frame = 0  # will be set by caller each tick
        self._owner = None  # Back-reference to engine
        
        # Spawn all max_concurrent_beacons at initialization
        for i in range(self.max_concurrent_beacons):
            success = self.spawn_beacon()
            if not success:
                print(f"Warning: Could only spawn {i} out of {self.max_concurrent_beacons} initial beacons")
                break
        
    def spawn_beacon(self):
        """Spawn a new goal beacon at a random location with random radius from normal distribution"""
        # Sample radius from normal distribution (matching LJ-Swarm logic)
        radius = max(5.0, np.random.normal(self.beacon_radius, self.radius_std))
        
        # Find non-overlapping position
        position = self._find_non_overlapping_position(radius)
        if position is None:
            print("Could not find non-overlapping position for new goal beacon")
            return False

        new_beacon = GoalBeacon(position, radius, self.base_lifetime,
                               self.velocity_damping, logger=self.logger)
        new_beacon.birth_frame = self.frame
        self.beacons.append(new_beacon)
        if self.logger:
            self.logger.log_event(
                frame=self.frame, t=None, event="beacon_spawn", beacon_id=new_beacon.id,
                beacon_pos_x=float(position[0]), beacon_pos_y=float(position[1]),
                beacon_radius=radius, trapped_count=0, extra=None
            )
        print(f"New goal beacon spawned at {position} with radius {radius:.2f}")
        return True
    
    def _find_non_overlapping_position(self, radius, max_attempts=100):
        """Find a position that doesn't overlap with existing beacons"""
        margin = radius + 5
        
        for attempt in range(max_attempts):
            x = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
            y = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
            position = np.array([x, y])
            
            # Check if this position overlaps with any existing beacon
            overlaps = False
            for existing_beacon in self.beacons:
                if existing_beacon.is_active:
                    distance = np.linalg.norm(position - existing_beacon.position)
                    # Ensure minimum separation of at least 25 units between beacon centers
                    min_distance = max(radius + existing_beacon.radius + 10, 25)
                    if distance < min_distance:
                        overlaps = True
                        break
            
            if not overlaps:
                return position
        
        return None  # Could not find non-overlapping position
        
    def process_agent(self, agent_idx, agent_pos, agent_vel, is_leader=False):
        """Process an agent's interaction with goal beacons"""
        flock_id = self._owner.get_agent_flock(agent_idx) if self._owner else None
        if agent_idx not in self.agent_beacon_map:
            for b in self.beacons:
                if b.is_active and b.is_inside(agent_pos):
                    b.add_agent(agent_idx, is_leader=is_leader, flock_id=flock_id)
                    self.agent_beacon_map[agent_idx] = b
                    if b.first_arrival_frame is None:
                        b.first_arrival_frame = self.frame
                        if self.logger:
                            self.logger.log_event(
                                frame=self.frame, t=None, event="first_arrival", beacon_id=b.id,
                                beacon_pos_x=float(b.position[0]), beacon_pos_y=float(b.position[1]),
                                beacon_radius=b.radius, trapped_count=len(b.trapped_agents), extra=None
                            )
                    leader_status = "leader" if is_leader else "follower"
                    print(f"DEBUG: Agent {agent_idx} ({leader_status}) entered beacon at {b.position}")
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
    
    def get_goal_force_for_agent(self, pos, vel, agent_flock_id=None):
        """Get goal beacon force for a single agent (used for leaders)"""
        total = np.zeros(2)
        for b in self.beacons:
            if b.is_active:
                total += b.get_goal_force(pos, vel, agent_flock_id=agent_flock_id)
        return total
    
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
        
        # Spawn new beacons if we have fewer than max_concurrent_beacons and enough time has passed
        if len(self.beacons) < self.max_concurrent_beacons and self.frames_since_last_spawn >= self.spawn_interval:
            success = self.spawn_beacon()
            if success:
                self.frames_since_last_spawn = 0
            # If spawn failed, don't reset timer - will try again next frame
    
    def get_active_beacons(self):
        """Get list of active goal beacons for visualization"""
        return [(b.position, b.radius, len(b.trapped_agents), b.owner_flock_id)
                for b in self.beacons if b.is_active]