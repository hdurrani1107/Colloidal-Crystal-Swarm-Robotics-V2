#######################################################################
# custom_systems.py
#
# Custom Goal/Zone Systems for Validation Testing
# Allows injection of predetermined goal positions and radii
#
# Author: Humzah Durrani
#######################################################################

import sys
import os
import numpy as np
from typing import List, Optional

def create_custom_cooling_zone_system(bounds, goal_set, zone_radius=15.0, spawn_interval=500, 
                                     base_lifetime=300, zone_temperature=10.0, max_concurrent_zones=3, 
                                     radius_std=2.0, logger=None):
    """Factory function to create CustomCoolingZoneSystem with proper imports"""
    
    # Import LJ-Swarm modules locally
    script_dir = os.path.dirname(__file__)
    lj_path = os.path.join(script_dir, '..', 'LJ-Swarm')
    if lj_path not in sys.path:
        sys.path.insert(0, lj_path)
    
    from cooling_zone import CoolingZone, CoolingZoneSystem
    
    class CustomCoolingZoneSystem(CoolingZoneSystem):
        """Custom cooling zone system that uses predetermined goal positions and radii"""
        
        def __init__(self, bounds, goal_set, zone_radius=15.0, spawn_interval=500, 
                     base_lifetime=300, zone_temperature=10.0, max_concurrent_zones=3, 
                     radius_std=2.0, logger=None):
            
            # Store the predetermined goal set
            self.predetermined_goals = goal_set
            self.current_goal_index = 0
            
            # Initialize parent with modified parameters
            super().__init__(
                bounds=bounds,
                zone_radius=zone_radius,
                spawn_interval=spawn_interval,
                base_lifetime=base_lifetime,
                zone_temperature=zone_temperature,
                max_concurrent_zones=max_concurrent_zones,
                radius_std=radius_std,
                logger=logger
            )
            
            # Clear the automatically spawned zones and spawn our predetermined ones
            self.zones = []
            self.current_goal_index = 0
            
            # Spawn initial zones from predetermined set
            for i in range(min(self.max_concurrent_zones, len(self.predetermined_goals))):
                self._spawn_predetermined_zone()
        
        def _spawn_predetermined_zone(self):
            """Spawn a zone using the next predetermined goal data"""
            if self.current_goal_index >= len(self.predetermined_goals):
                return False  # No more predetermined goals
            
            goal_data = self.predetermined_goals[self.current_goal_index]
            position = np.array(goal_data.position)
            radius = goal_data.radius
            
            new_zone = CoolingZone(position, radius, self.base_lifetime,
                                  self.zone_temperature, logger=self.logger)
            new_zone.birth_frame = self.frame
            self.zones.append(new_zone)
            
            if self.logger:
                self.logger.log_event(
                    frame=self.frame, t=None, event="zone_spawn", zone_id=new_zone.id,
                    zone_pos_x=float(position[0]), zone_pos_y=float(position[1]),
                    zone_radius=radius, trapped_count=0, extra=None
                )
            
            print(f"Spawned predetermined cooling zone {self.current_goal_index + 1} at {position} with radius {radius:.2f}")
            self.current_goal_index += 1
            return True
        
        def spawn_zone(self):
            """Override to use predetermined goals"""
            return self._spawn_predetermined_zone()
        
        def update(self):
            """Update all cooling zones and spawn new predetermined ones"""
            active_zones = []
            freed_agents = set()
            
            for zone in self.zones:
                if zone.update():
                    active_zones.append(zone)
                else:
                    freed_agents.update(zone.get_trapped_agents())
                    if self.logger:
                        self.logger.log_event(
                            frame=self.frame, t=None, event="zone_complete", zone_id=zone.id,
                            zone_pos_x=float(zone.position[0]), zone_pos_y=float(zone.position[1]),
                            zone_radius=zone.radius, trapped_count=len(zone.trapped_agents),
                            extra={"lifetime_frames": (self.frame - (zone.birth_frame or self.frame))}
                        )
            
            for agent_idx in freed_agents:
                if agent_idx in self.agent_zone_map:
                    del self.agent_zone_map[agent_idx]

            zones_disappeared = len(self.zones) > len(active_zones)
            self.zones = active_zones

            if zones_disappeared and len(self.zones) == 0:
                self.frames_since_last_spawn = 0

            self.frames_since_last_spawn += 1
            
            # Spawn new zones if we have fewer than max_concurrent_zones and enough time has passed
            if len(self.zones) < self.max_concurrent_zones and self.frames_since_last_spawn >= self.spawn_interval:
                success = self._spawn_predetermined_zone()
                if success:
                    self.frames_since_last_spawn = 0
                # If spawn failed (no more predetermined goals), don't reset timer

    return CustomCoolingZoneSystem(bounds, goal_set, zone_radius, spawn_interval, 
                                   base_lifetime, zone_temperature, max_concurrent_zones, 
                                   radius_std, logger)


def create_custom_goal_beacon_system(bounds, goal_set, beacon_radius=15.0, spawn_interval=500, 
                                    base_lifetime=300, velocity_damping=0.985, max_concurrent_beacons=3, 
                                    radius_std=2.0, logger=None):
    """Factory function to create CustomGoalBeaconSystem with proper imports"""
    
    # Import Olfati-Saber modules locally
    script_dir = os.path.dirname(__file__)
    os_path = os.path.join(script_dir, '..', 'Olfati-Saber-Flock')
    if os_path not in sys.path:
        sys.path.insert(0, os_path)
    
    from goal_beacon import GoalBeacon, GoalBeaconSystem
    
    class CustomGoalBeaconSystem(GoalBeaconSystem):
        """Custom goal beacon system that uses predetermined goal positions and radii"""
        
        def __init__(self, bounds, goal_set, beacon_radius=15.0, spawn_interval=500, 
                     base_lifetime=300, velocity_damping=0.985, max_concurrent_beacons=3, 
                     radius_std=2.0, logger=None):
            
            # Store the predetermined goal set
            self.predetermined_goals = goal_set
            self.current_goal_index = 0
            
            # Initialize parent with modified parameters
            super().__init__(
                bounds=bounds,
                beacon_radius=beacon_radius,
                spawn_interval=spawn_interval,
                base_lifetime=base_lifetime,
                velocity_damping=velocity_damping,
                max_concurrent_beacons=max_concurrent_beacons,
                radius_std=radius_std,
                logger=logger
            )
            
            # Clear the automatically spawned beacons and spawn our predetermined ones
            self.beacons = []
            self.current_goal_index = 0
            
            # Spawn initial beacons from predetermined set
            for i in range(min(self.max_concurrent_beacons, len(self.predetermined_goals))):
                self._spawn_predetermined_beacon()
        
        def _spawn_predetermined_beacon(self):
            """Spawn a beacon using the next predetermined goal data"""
            if self.current_goal_index >= len(self.predetermined_goals):
                return False  # No more predetermined goals
            
            goal_data = self.predetermined_goals[self.current_goal_index]
            position = np.array(goal_data.position)
            radius = goal_data.radius
            
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
            
            print(f"Spawned predetermined goal beacon {self.current_goal_index + 1} at {position} with radius {radius:.2f}")
            self.current_goal_index += 1
            return True
        
        def spawn_beacon(self):
            """Override to use predetermined goals"""
            return self._spawn_predetermined_beacon()
        
        def _find_non_overlapping_position(self, radius, max_attempts=100):
            """Override - not needed since we use predetermined positions"""
            return None  # This method won't be called in our custom implementation
        
        def update(self):
            """Update all goal beacons and spawn new predetermined ones"""
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
                success = self._spawn_predetermined_beacon()
                if success:
                    self.frames_since_last_spawn = 0
                # If spawn failed (no more predetermined goals), don't reset timer

    return CustomGoalBeaconSystem(bounds, goal_set, beacon_radius, spawn_interval, 
                                 base_lifetime, velocity_damping, max_concurrent_beacons, 
                                 radius_std, logger)