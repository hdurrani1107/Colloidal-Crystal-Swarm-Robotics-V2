#######################################################################
# cooling_zone.py
#
# Cooling Zone System with Dynamic Spawning and Agent Trapping
#
# Author: Humzah Durrani
# AI Disclosure: AI was used for debugging
#######################################################################

##########################
# Importing Libraries
##########################

import numpy as np
from typing import Optional
from lj_swarm.metrics import MetricsLogger


##########################
# Cooling Zone Class
##########################
class CoolingZone:
    _next_id = 0
    def __init__(self, position, radius, base_lifetime, zone_temperature=10.0,
                 logger: Optional[MetricsLogger]=None):
        #Cooling Zone Parameters
        self.id = CoolingZone._next_id; CoolingZone._next_id += 1
        self.position = position
        self.radius = radius
        self.base_lifetime = base_lifetime
        self.zone_temperature = zone_temperature
        self.trapped_agents = set()
        self.frames_remaining = None
        self.is_active = True
        self.lifetime_started = False
        self.logger = logger
        self.birth_frame = None
        self.first_arrival_frame = None
        
    ##########################
    # Adding Agents into zone
    ##########################
    
    def add_agent(self, agent_idx):
        if agent_idx not in self.trapped_agents:
            self.trapped_agents.add(agent_idx)
            print(f"Agent {agent_idx} entered cooling zone at {self.position}")
            
            # Start lifetime countdown if this is the first agent
            if not self.lifetime_started:
                self.lifetime_started = True
                print(f"Cooling zone lifetime countdown started!")
            
            self.update_lifetime()

    ##########################
    # Update Zone Lifetime
    ##########################

    def update_lifetime(self):
        #More agents decrease lifetime of zone
        num_agents = len(self.trapped_agents)
        if num_agents > 0:
            #20% Reduction in lifetime per agent
            reduction_factor = 1.0 + (num_agents - 1) * 0.2
            new_total = max(1, int(self.base_lifetime / reduction_factor))
            if self.lifetime_started and self.frames_remaining is not None:
                elapsed = self.base_lifetime - self.frames_remaining
                remaining = max(1, new_total - elapsed)
                self.frames_remaining = remaining
            else:
                self.frames_remaining = new_total

    ##############################
    # Cooling Zone Position Check
    ##############################

    def is_inside(self, position):
        #Checks if position is inside cooling zone
        distance = np.linalg.norm(position - self.position)
        return distance <= self.radius
    
    ##########################
    # Update Cooling Zone State
    ##########################
    
    def update(self):
        if self.is_active and self.lifetime_started:
            # Only countdown if lifetime has started (first agent entered)
            self.frames_remaining -= 1
            if self.frames_remaining <= 0:
                self.is_active = False
                print(f"Cooling zone at {self.position} disappeared after reaching lifetime, freeing {len(self.trapped_agents)} agents")
                #Zone Removal
                return False
        # Zone stays active otherwise
        return True
    
    ##########################
    # Get Trapped Agents
    ##########################
    
    def get_trapped_agents(self):
        #Cooling Zone Trapped Agent Check
        #Redundant
        return self.trapped_agents.copy()


##########################
# Cooling Zone System Class
##########################
class CoolingZoneSystem:
    def __init__(self, bounds, zone_radius=15.0, spawn_interval=500, base_lifetime=300, zone_temperature=10.0,
                 max_concurrent_zones=3, radius_std=2.0, logger: Optional[MetricsLogger]=None):
        #Cooling Zone System Params
        self.bounds = bounds
        self.zone_radius = zone_radius  # Mean radius for normal distribution
        self.radius_std = radius_std    # Standard deviation for radius sampling
        self.spawn_interval = spawn_interval
        self.base_lifetime = base_lifetime
        self.zone_temperature = zone_temperature
        self.max_concurrent_zones = max_concurrent_zones
        self.zones = []
        self.frames_since_last_spawn = 0
        self.agent_zone_map = {}  # Maps agent_idx to zone
        #Clean Up Logger and metrics
        self.logger = logger
        self.frame = 0  # will be set by caller each tick
        #self.completed_total = 0
        
        # Spawn all max_concurrent_zones at initialization
        for i in range(self.max_concurrent_zones):
            success = self.spawn_zone()
            if not success:
                print(f"Warning: Could only spawn {i} out of {self.max_concurrent_zones} initial zones")
                break
        
    ##########################
    # Spawn Cooling Zone
    ##########################
    
    def spawn_zone(self):
        # Sample radius from normal distribution
        radius = max(5.0, np.random.normal(self.zone_radius, self.radius_std))
        
        # Find non-overlapping position
        position = self._find_non_overlapping_position(radius)
        if position is None:
            print("Could not find non-overlapping position for new cooling zone")
            return False

        #Create new Zone
        new_zone = CoolingZone(position, radius, self.base_lifetime,
                               self.zone_temperature, logger=self.logger)
        new_zone.birth_frame = self.frame
        self.zones.append(new_zone)
        
        #Logger Redundant
        #Clean up logger and metrics
        if self.logger:
            self.logger.log_event(
                frame=self.frame, t=None, event="zone_spawn", zone_id=new_zone.id,
                zone_pos_x=float(position[0]), zone_pos_y=float(position[1]),
                zone_radius=radius, trapped_count=0, extra=None
            )
        print(f"New cooling zone spawned at {position} with radius {radius:.2f}")
        return True
    

    ####################################
    # Cooling Zone Overlap Spawn Check
    ####################################

    def _find_non_overlapping_position(self, radius, max_attempts=100):
        #Prevents Cooling Zone Spawns from overlaping
        margin = radius + 5
        
        for attempt in range(max_attempts):
            x = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
            y = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
            position = np.array([x, y])
            
            # Check if this position overlaps with any existing zone
            overlaps = False
            for existing_zone in self.zones:
                if existing_zone.is_active:
                    distance = np.linalg.norm(position - existing_zone.position)
                    # Ensure minimum separation of at least 25 units between zone centers
                    min_distance = max(radius + existing_zone.radius + 10, 25)
                    if distance < min_distance:
                        overlaps = True
                        break
            
            if not overlaps:
                return position
        
        return None  # Could not find non-overlapping position
    
    #################################
    # Cooling Zone Agent Interaction
    #################################
        
    def process_agent(self, agent_idx, agent_pos, agent_vel):
        #Classifies if Agent is inside active cooling zone
        if agent_idx not in self.agent_zone_map: #Checks if agent index is in zone map
            for zone in self.zones: 
                if zone.is_active and zone.is_inside(agent_pos):
                    zone.add_agent(agent_idx)
                    self.agent_zone_map[agent_idx] = zone
                    if zone.first_arrival_frame is None:
                        zone.first_arrival_frame = self.frame

                        #Logger Redundant
                        #Clean up Metrics
                        if self.logger:
                            self.logger.log_event(
                                frame=self.frame, t=None, event="first_arrival", zone_id=zone.id,
                                zone_pos_x=float(zone.position[0]), zone_pos_y=float(zone.position[1]),
                                zone_radius=zone.radius, trapped_count=len(zone.trapped_agents), extra=None
                            )
                    print(f"DEBUG: Agent {agent_idx} entered zone at {zone.position}")
                    break
    
    #################################
    # Trapped Agent Check
    #################################

    def is_agent_trapped(self, agent_idx):
        #Trapped Agent inside Cooling Zone
        if agent_idx in self.agent_zone_map:
            zone = self.agent_zone_map[agent_idx]
            return zone.is_active
        return False
    
    #################################################
    # Cooling Zone Information for Trapped Agents
    ################################################

    def get_zone_info_for_agent(self, agent_idx):
        #Check Agent in Zone Map
        if agent_idx in self.agent_zone_map:
            zone = self.agent_zone_map[agent_idx]
            #If Active Return Position and Radius
            if zone.is_active:
                return zone.position, zone.radius
        return None, None
    
    #################################
    # Update Cooling Zones
    #################################
    
    def update(self):
        active_zones = []
        freed_agents = set()
        for zone in self.zones:
            if zone.update():
                active_zones.append(zone)
            else:
                #self.completed_total += 1
                freed_agents.update(zone.get_trapped_agents())
                
                #Logger Redundant
                #Clean up Metrics
                if self.logger:
                    self.logger.log_event(
                        frame=self.frame, t=None, event="zone_complete", zone_id=zone.id,
                        zone_pos_x=float(zone.position[0]), zone_pos_y=float(zone.position[1]),
                        zone_radius=zone.radius, trapped_count=len(zone.trapped_agents),
                        extra={"lifetime_frames": (self.frame - (zone.birth_frame or self.frame))}
                    )
        
        #Freed Agents are cleared from Zone Map
        for agent_idx in freed_agents:
            if agent_idx in self.agent_zone_map:
                del self.agent_zone_map[agent_idx]

        #Checks cleared Zones
        zones_disappeared = len(self.zones) > len(active_zones)
        
        #Zones update
        self.zones = active_zones

        if zones_disappeared and len(self.zones) == 0:
            self.frames_since_last_spawn = 0

        self.frames_since_last_spawn += 1
        
        # Spawn new zones if we have fewer than max_concurrent_zones and enough time has passed
        if len(self.zones) < self.max_concurrent_zones and self.frames_since_last_spawn >= self.spawn_interval:
            success = self.spawn_zone()
            if success:
                self.frames_since_last_spawn = 0
            # If spawn failed, don't reset timer - will try again next frame
            #self.frame += 1
    
    #################################
    # Active Zone Check
    #################################

    def get_active_zones(self):
        #For Simulation UI
        return [(zone.position, zone.radius, len(zone.trapped_agents)) 
                for zone in self.zones if zone.is_active]