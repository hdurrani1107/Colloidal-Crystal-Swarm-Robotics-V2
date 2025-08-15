#!/usr/bin/env python3

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class GoalData:
    position: Tuple[float, float]
    radius: float
    spawn_order: int

def test_lj_swarm_minimal():
    """Test LJ-Swarm with custom system creation"""
    print("=== Testing LJ-Swarm minimal setup ===")
    
    try:
        # Set up paths
        script_dir = os.path.dirname(__file__)
        lj_path = os.path.join(script_dir, '..', 'LJ-Swarm')
        sys.path.insert(0, lj_path)
        
        import engine
        from custom_systems import create_custom_cooling_zone_system
        
        # Create test goal set
        goal_set = [
            GoalData(position=(50, 50), radius=10.0, spawn_order=0),
            GoalData(position=(150, 150), radius=12.0, spawn_order=1)
        ]
        
        # Initialize simulation
        sim = engine.multi_agent(10, 3.0, 0.005, [0, 200], [])
        engine.init_melt(sim.agents, init_temp=150)
        sim.initialize_agent_temperatures(150)
        
        print("LJ-Swarm simulation initialized")
        
        # Create custom cooling zone system
        cooling_zone_config = {
            'zone_radius': 10.0,
            'radius_std': 1.0,
            'max_concurrent_zones': 2,
            'spawn_interval': 400,
            'base_lifetime': 1000,
            'zone_temperature': 0.0,
            'logger': None
        }
        
        sim.cooling_zones = create_custom_cooling_zone_system(
            bounds=[0, 200],
            goal_set=goal_set,
            **cooling_zone_config
        )
        
        print(f"Created cooling zone system with {len(sim.cooling_zones.zones)} initial zones")
        print("SUCCESS: LJ-Swarm minimal test passed")
        
        return True
        
    except Exception as e:
        print(f"ERROR: LJ-Swarm minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_olfati_saber_minimal():
    """Test Olfati-Saber with custom system creation"""
    print("\n=== Testing Olfati-Saber minimal setup ===")
    
    try:
        # Clear previous imports
        modules_to_clear = [mod for mod in sys.modules.keys() if 'engine' in mod]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Set up paths
        script_dir = os.path.dirname(__file__)
        os_path = os.path.join(script_dir, '..', 'Olfati-Saber-Flock')
        sys.path.insert(0, os_path)
        
        import engine
        from custom_systems import create_custom_goal_beacon_system
        
        # Create test goal set
        goal_set = [
            GoalData(position=(50, 50), radius=12.0, spawn_order=0),
            GoalData(position=(150, 150), radius=14.0, spawn_order=1)
        ]
        
        # Initialize simulation
        sim = engine.multi_agent(10, 0.005, [0, 200], [])
        
        print("Olfati-Saber simulation initialized")
        
        # Create custom goal beacon system
        goal_beacon_config = {
            'beacon_radius': 12.0,
            'radius_std': 2.0,
            'max_concurrent_beacons': 2,
            'spawn_interval': 400,
            'base_lifetime': 1000,
            'velocity_damping': 0.985,
            'logger': None
        }
        
        sim.goal_beacons = create_custom_goal_beacon_system(
            bounds=[0, 200],
            goal_set=goal_set,
            **goal_beacon_config
        )
        sim.goal_beacons._owner = sim
        
        print(f"Created goal beacon system with {len(sim.goal_beacons.beacons)} initial beacons")
        print("SUCCESS: Olfati-Saber minimal test passed")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Olfati-Saber minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing minimal validation framework setup...")
    
    lj_success = test_lj_swarm_minimal()
    os_success = test_olfati_saber_minimal()
    
    if lj_success and os_success:
        print("\nALL TESTS PASSED - Ready for full validation!")
    else:
        print("\nSOME TESTS FAILED - Check errors above")