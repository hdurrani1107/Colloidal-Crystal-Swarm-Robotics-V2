#!/usr/bin/env python3

import sys
import os

print("Testing import structure for validation framework...")

# Test 1: LJ-Swarm imports
print("\n=== Testing LJ-Swarm imports ===")
try:
    script_dir = os.path.dirname(__file__)
    lj_path = os.path.join(script_dir, '..', 'LJ-Swarm')
    print(f"LJ-Swarm path: {lj_path}")
    sys.path.insert(0, lj_path)
    
    import engine as lj_engine
    print("SUCCESS: LJ-Swarm engine imported successfully")
    
    import cooling_zone
    print("SUCCESS: LJ-Swarm cooling_zone imported successfully")
    
    # Test creating a simulation
    sim = lj_engine.multi_agent(10, 3.0, 0.005, [0, 200], [])
    print("SUCCESS: LJ-Swarm simulation created successfully")
    del sim
    
except Exception as e:
    print(f"ERROR: LJ-Swarm import failed: {e}")

# Clear sys.path for clean test
for path in [p for p in sys.path if 'LJ-Swarm' in p]:
    sys.path.remove(path)

# Clear modules
modules_to_clear = [mod for mod in sys.modules.keys() if 'engine' in mod or 'cooling_zone' in mod]
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]

# Test 2: Olfati-Saber imports
print("\n=== Testing Olfati-Saber imports ===")
try:
    os_path = os.path.join(script_dir, '..', 'Olfati-Saber-Flock')
    print(f"Olfati-Saber path: {os_path}")
    sys.path.insert(0, os_path)
    
    import engine as os_engine
    print("SUCCESS: Olfati-Saber engine imported successfully")
    
    import goal_beacon
    print("SUCCESS: Olfati-Saber goal_beacon imported successfully")
    
    # Test creating a simulation
    sim = os_engine.multi_agent(10, 0.005, [0, 200], [])
    print("SUCCESS: Olfati-Saber simulation created successfully")
    del sim
    
except Exception as e:
    print(f"ERROR: Olfati-Saber import failed: {e}")

# Test 3: Custom systems
print("\n=== Testing custom systems ===")
try:
    from custom_systems import create_custom_cooling_zone_system, create_custom_goal_beacon_system
    print("SUCCESS: Custom systems factory functions imported successfully")
    
except Exception as e:
    print(f"ERROR: Custom systems import failed: {e}")

print("\n=== Import test completed ===")