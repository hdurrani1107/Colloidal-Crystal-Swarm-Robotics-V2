#######################################################################
# cooling_zone.py
#
# Cooling Zone System with Dynamic Spawning and Agent Trapping
#
# Author: Humzah Durrani
#######################################################################

import numpy as np

class CoolingZone:
    def __init__(self, position, radius, base_lifetime, zone_temperature=10.0):
        self.position = position
        self.radius = radius
        self.base_lifetime = base_lifetime
        self.zone_temperature = zone_temperature
        self.trapped_agents = set()
        self.frames_remaining = None  # Will be set when first agent enters
        self.is_active = True
        self.lifetime_started = False  # Track whether countdown has begun
        
    def add_agent(self, agent_idx):
        """Add an agent to this cooling zone"""
        if agent_idx not in self.trapped_agents:
            self.trapped_agents.add(agent_idx)
            print(f"Agent {agent_idx} entered cooling zone at {self.position}")
            
            # Start lifetime countdown if this is the first agent
            if not self.lifetime_started:
                self.lifetime_started = True
                print(f"Cooling zone lifetime countdown started!")
            
            self.update_lifetime()
    
    def update_lifetime(self):
        """Update the lifetime based on number of trapped agents"""
        num_agents = len(self.trapped_agents)
        if num_agents > 0:
            # More agents = faster disappearing
            reduction_factor = 1.0 + (num_agents - 1) * 0.2  # Each additional agent reduces lifetime by 20%
            self.frames_remaining = max(1, int(self.base_lifetime / reduction_factor))
    
    def is_inside(self, position):
        """Check if a position is inside this cooling zone"""
        distance = np.linalg.norm(position - self.position)
        return distance <= self.radius
    
    def update(self):
        """Update the cooling zone state"""
        if self.is_active and self.lifetime_started:
            # Only countdown if lifetime has started (first agent entered)
            self.frames_remaining -= 1
            if self.frames_remaining <= 0:
                self.is_active = False
                print(f"Cooling zone at {self.position} disappeared after reaching lifetime, freeing {len(self.trapped_agents)} agents")
                return False  # Zone should be removed
        # Zone stays active if no agents have entered yet or still has time left
        return True
    
    def get_trapped_agents(self):
        """Get set of trapped agent indices"""
        return self.trapped_agents.copy()

class CoolingZoneSystem:
    def __init__(self, bounds, zone_radius=15.0, spawn_interval=500, base_lifetime=300, zone_temperature=10.0):
        self.bounds = bounds
        self.zone_radius = zone_radius
        self.spawn_interval = spawn_interval
        self.base_lifetime = base_lifetime
        self.zone_temperature = zone_temperature
        self.zones = []
        self.frames_since_last_spawn = 0
        self.agent_zone_map = {}  # Maps agent_idx to zone
        
        # Spawn initial cooling zone
        self.spawn_zone()
        
    def spawn_zone(self):
        """Spawn a new cooling zone at a random location"""
        # Choose random position within bounds, avoiding edges
        margin = self.zone_radius + 5
        x = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
        y = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
        position = np.array([x, y])
        
        new_zone = CoolingZone(position, self.zone_radius, self.base_lifetime, self.zone_temperature)
        self.zones.append(new_zone)
        print(f"New cooling zone spawned at {position} with radius {self.zone_radius}")
        print(f"Zone bounds: x=[{self.bounds[0] + margin}, {self.bounds[1] - margin}], y=[{self.bounds[0] + margin}, {self.bounds[1] - margin}]")
        
    def process_agent(self, agent_idx, agent_pos, agent_vel):
        """Process an agent's interaction with cooling zones"""
        # Check if agent enters any active cooling zone
        if agent_idx not in self.agent_zone_map:
            for zone in self.zones:
                if zone.is_active and zone.is_inside(agent_pos):
                    zone.add_agent(agent_idx)
                    self.agent_zone_map[agent_idx] = zone
                    print(f"DEBUG: Agent {agent_idx} entered zone at {zone.position}")
                    break
    
    def is_agent_trapped(self, agent_idx):
        """Check if an agent is trapped in a cooling zone"""
        if agent_idx in self.agent_zone_map:
            zone = self.agent_zone_map[agent_idx]
            return zone.is_active
        return False
    
    def get_zone_info_for_agent(self, agent_idx):
        """Get zone position and radius for a trapped agent"""
        if agent_idx in self.agent_zone_map:
            zone = self.agent_zone_map[agent_idx]
            if zone.is_active:
                return zone.position, zone.radius
        return None, None
    
    def update(self):
        """Update all cooling zones and spawn new ones"""
        # Update existing zones and remove inactive ones
        active_zones = []
        freed_agents = set()
        
        for zone in self.zones:
            if zone.update():  # Returns True if zone is still active
                active_zones.append(zone)
            else:
                # Zone disappeared, free all its agents
                freed_agents.update(zone.get_trapped_agents())
        
        # Remove freed agents from agent_zone_map
        for agent_idx in freed_agents:
            if agent_idx in self.agent_zone_map:
                del self.agent_zone_map[agent_idx]
        
        # Check if any zones disappeared to reset spawn timer
        zones_disappeared = len(self.zones) > len(active_zones)
        self.zones = active_zones
        
        # Reset spawn timer when zones disappear
        if zones_disappeared and len(self.zones) == 0:
            self.frames_since_last_spawn = 0
        
        # Spawn new zone if needed
        self.frames_since_last_spawn += 1
        if len(self.zones) == 0 and self.frames_since_last_spawn >= self.spawn_interval:
            self.spawn_zone()
            self.frames_since_last_spawn = 0
    
    def get_active_zones(self):
        """Get list of active cooling zones for visualization"""
        return [(zone.position, zone.radius, len(zone.trapped_agents)) 
                for zone in self.zones if zone.is_active]