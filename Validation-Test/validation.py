#!/usr/bin/env python3
"""
Simple Validation Script for LJ-Swarm vs Olfati-Saber-Flock Comparison

This script creates a simple validation test that:
1. Generates a random sequence of 10 zone positions and radii
2. Runs LJ-Swarm simulation with these zones
3. Runs Olfati-Saber-Flock simulation with the same zones
4. Repeats for 10 iterations
5. Creates comparison graphs

Author: Claude Code Assistant
"""

# --- Set matplotlib backend before importing pyplot ---
import matplotlib
matplotlib.use("Agg")  # safe for headless

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disambiguated imports from packages
import lj_swarm.engine as lj_engine
from lj_swarm.cooling_zone import CoolingZone

import olfati_saber_flock.engine as os_engine
from olfati_saber_flock.goal_beacon import GoalBeacon

# Output directory setup
OUTPUT_DIR = REPO_ROOT / "output" / "simple-validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ZoneData:
    """Simple zone data structure"""
    def __init__(self, position: Tuple[float, float], radius: float):
        self.position = position
        self.radius = radius

class ValidationResults:
    """Store validation results"""
    def __init__(self):
        self.lj_total_times = []
        self.lj_avg_times = []
        self.os_total_times = []
        self.os_avg_times = []
        self.iteration_data = []

def generate_random_zones(num_zones=10, bounds=[0, 200], seed=None):
    """Generate random zone positions and radii"""
    if seed is not None:
        np.random.seed(seed)
    
    zones = []
    margin = 25
    
    for i in range(num_zones):
        # Random position with margin from boundaries
        x = np.random.uniform(bounds[0] + margin, bounds[1] - margin)
        y = np.random.uniform(bounds[0] + margin, bounds[1] - margin)
        
        # Random radius (mean=11, std=2 to match both systems)
        radius = max(5.0, np.random.normal(11.0, 2.0))
        
        zones.append(ZoneData((x, y), radius))
    
    return zones

def run_lj_simulation(zones: List[ZoneData], trial_num: int):
    """Run LJ-Swarm simulation with predetermined zones"""
    print(f"  Running LJ-Swarm trial {trial_num}...")
    
    try:
        # Simulation parameters
        agents = 150
        sigma = 3.0
        epsilon = 3.0
        sample_time = 0.005
        bounds = [0, 200]
        obstacles = []
        constant_temperature = 150
        
        # Initialize simulation
        sim = lj_engine.multi_agent(agents, sigma, sample_time, bounds, obstacles)
        
        # Create custom cooling zone system with predetermined zones
        sim.cooling_zones = CustomCoolingZoneSystem(zones)
        
        # Initialize agents to melted state
        lj_engine.init_melt(sim.agents, init_temp=constant_temperature)
        sim.initialize_agent_temperatures(constant_temperature)
        
        # Run simulation
        frame_count = 0
        completed_zones = 0
        target_zones = 10
        zone_completion_times = []
        start_time = time.time()
        max_frames = 1000000  # Safety limit
        
        while completed_zones < target_zones and frame_count < max_frames:
            zones_before = len(sim.cooling_zones.zones)
            
            # Run simulation step
            forces = sim.compute_forces(sigma, epsilon, (2**(1/6)) * sigma, constant_temperature, 10, 10, 0.5)
            sim.update(forces, constant_temperature)
            
            # Check for completed zones
            zones_after = len(sim.cooling_zones.zones)
            if zones_before > zones_after:
                zones_completed_this_step = zones_before - zones_after
                completed_zones += zones_completed_this_step
                current_time = frame_count * sample_time
                zone_completion_times.extend([current_time] * zones_completed_this_step)
            
            frame_count += 1
        
        total_time = frame_count * sample_time
        avg_time = total_time / target_zones if completed_zones >= target_zones else float('inf')
        
        print(f"    LJ-Swarm completed in {total_time:.2f}s, avg {avg_time:.2f}s per zone")
        return total_time, avg_time, completed_zones >= target_zones
        
    except Exception as e:
        print(f"    LJ-Swarm failed: {e}")
        return float('inf'), float('inf'), False

def run_olfati_simulation(zones: List[ZoneData], trial_num: int):
    """Run Olfati-Saber-Flock simulation with predetermined zones"""
    print(f"  Running Olfati-Saber trial {trial_num}...")
    
    try:
        # Simulation parameters
        agents = 150
        sample_time = 0.005
        bounds = [0, 200]
        obstacles = []
        
        # Initialize simulation
        sim = os_engine.multi_agent(agents, sample_time, bounds, obstacles)
        
        # Create custom goal beacon system with predetermined zones
        sim.goal_beacons = CustomGoalBeaconSystem(zones)
        sim.goal_beacons._owner = sim
        
        # Run simulation
        frame_count = 0
        completed_beacons = 0
        target_beacons = 10
        beacon_completion_times = []
        start_time = time.time()
        max_frames = 1000000  # Safety limit
        
        while completed_beacons < target_beacons and frame_count < max_frames:
            beacons_before = len(sim.goal_beacons.beacons)
            
            # Run simulation step
            external_forces = np.zeros((len(sim.agents), 2))
            sim.update(external_forces, temp=0, frame=frame_count)
            
            # Check for completed beacons
            beacons_after = len(sim.goal_beacons.beacons)
            if beacons_before > beacons_after:
                beacons_completed_this_step = beacons_before - beacons_after
                completed_beacons += beacons_completed_this_step
                current_time = frame_count * sample_time
                beacon_completion_times.extend([current_time] * beacons_completed_this_step)
            
            frame_count += 1
        
        total_time = frame_count * sample_time
        avg_time = total_time / target_beacons if completed_beacons >= target_beacons else float('inf')
        
        print(f"    Olfati-Saber completed in {total_time:.2f}s, avg {avg_time:.2f}s per beacon")
        return total_time, avg_time, completed_beacons >= target_beacons
        
    except Exception as e:
        print(f"    Olfati-Saber failed: {e}")
        return float('inf'), float('inf'), False

class CustomCoolingZoneSystem:
    """Simplified custom cooling zone system for predetermined zones"""
    
    def __init__(self, predetermined_zones: List[ZoneData]):
        self.predetermined_zones = predetermined_zones
        self.zones = []
        self.current_zone_index = 0
        self.agent_zone_map = {}
        self.frame = 0
        self.frames_since_last_spawn = 0
        self.spawn_interval = 400
        self.max_concurrent_zones = 3
        
        # Spawn initial zones
        for i in range(min(self.max_concurrent_zones, len(self.predetermined_zones))):
            self._spawn_next_zone()
    
    def _spawn_next_zone(self):
        """Spawn the next predetermined zone"""
        if self.current_zone_index >= len(self.predetermined_zones):
            return False
        
        zone_data = self.predetermined_zones[self.current_zone_index]
        
        new_zone = CoolingZone(
            np.array(zone_data.position), 
            zone_data.radius, 
            base_lifetime=1000,
            zone_temperature=0.0
        )
        new_zone.birth_frame = self.frame
        self.zones.append(new_zone)
        
        print(f"Spawned cooling zone {self.current_zone_index + 1} at {zone_data.position} with radius {zone_data.radius:.2f}")
        self.current_zone_index += 1
        return True
    
    def process_agent(self, agent_idx, agent_pos, agent_vel):
        """Process agent interaction with zones"""
        if agent_idx not in self.agent_zone_map:
            for zone in self.zones:
                if zone.is_active and zone.is_inside(agent_pos):
                    zone.add_agent(agent_idx)
                    self.agent_zone_map[agent_idx] = zone
                    break
    
    def is_agent_trapped(self, agent_idx):
        """Check if agent is trapped"""
        if agent_idx in self.agent_zone_map:
            zone = self.agent_zone_map[agent_idx]
            return zone.is_active
        return False
    
    def get_zone_info_for_agent(self, agent_idx):
        """Get zone info for agent"""
        if agent_idx in self.agent_zone_map:
            zone = self.agent_zone_map[agent_idx]
            if zone.is_active:
                return zone.position, zone.radius
        return None, None
    
    def update(self):
        """Update zones and spawn new ones"""
        active_zones = []
        freed_agents = set()
        
        for zone in self.zones:
            if zone.update():
                active_zones.append(zone)
            else:
                freed_agents.update(zone.get_trapped_agents())
        
        for agent_idx in freed_agents:
            if agent_idx in self.agent_zone_map:
                del self.agent_zone_map[agent_idx]
        
        zones_disappeared = len(self.zones) > len(active_zones)
        self.zones = active_zones
        
        if zones_disappeared and len(self.zones) == 0:
            self.frames_since_last_spawn = 0
        
        self.frames_since_last_spawn += 1
        
        # Spawn new zones
        if len(self.zones) < self.max_concurrent_zones and self.frames_since_last_spawn >= self.spawn_interval:
            success = self._spawn_next_zone()
            if success:
                self.frames_since_last_spawn = 0
        
        # Increment frame counter
        self.frame += 1
    
    def get_active_zones(self):
        """Get active zones for visualization"""
        return [(zone.position, zone.radius, len(zone.trapped_agents)) 
                for zone in self.zones if zone.is_active]

class CustomGoalBeaconSystem:
    """Simplified custom goal beacon system for predetermined zones"""
    
    def __init__(self, predetermined_zones: List[ZoneData]):
        self.predetermined_zones = predetermined_zones
        self.beacons = []
        self.current_beacon_index = 0
        self.agent_beacon_map = {}
        self.frame = 0
        self.frames_since_last_spawn = 0
        self.spawn_interval = 400
        self.max_concurrent_beacons = 3
        self._owner = None
        
        # Spawn initial beacons
        for i in range(min(self.max_concurrent_beacons, len(self.predetermined_zones))):
            self._spawn_next_beacon()
    
    def _spawn_next_beacon(self):
        """Spawn the next predetermined beacon"""
        if self.current_beacon_index >= len(self.predetermined_zones):
            return False
        
        zone_data = self.predetermined_zones[self.current_beacon_index]
        
        new_beacon = GoalBeacon(
            np.array(zone_data.position), 
            zone_data.radius, 
            base_lifetime=1000,
            velocity_damping=0.985
        )
        new_beacon.birth_frame = self.frame
        self.beacons.append(new_beacon)
        
        print(f"Spawned goal beacon {self.current_beacon_index + 1} at {zone_data.position} with radius {zone_data.radius:.2f}")
        self.current_beacon_index += 1
        return True
    
    def process_agent(self, agent_idx, agent_pos, agent_vel, is_leader=False):
        """Process agent interaction with beacons"""
        flock_id = self._owner.get_agent_flock(agent_idx) if self._owner else None
        if agent_idx not in self.agent_beacon_map:
            for beacon in self.beacons:
                if beacon.is_active and beacon.is_inside(agent_pos):
                    beacon.add_agent(agent_idx, is_leader=is_leader, flock_id=flock_id)
                    self.agent_beacon_map[agent_idx] = beacon
                    break
    
    def is_agent_trapped(self, agent_idx):
        """Check if agent is trapped"""
        if agent_idx in self.agent_beacon_map:
            beacon = self.agent_beacon_map[agent_idx]
            return beacon.is_active
        return False
    
    def get_beacon_info_for_agent(self, agent_idx):
        """Get beacon info for agent"""
        if agent_idx in self.agent_beacon_map:
            beacon = self.agent_beacon_map[agent_idx]
            if beacon.is_active:
                return beacon.position, beacon.radius
        return None, None
    
    def get_goal_forces(self, agent_positions, agent_velocities):
        """Get goal forces for all agents"""
        n_agents = len(agent_positions)
        forces = np.zeros((n_agents, 2))
        
        for i, (pos, vel) in enumerate(zip(agent_positions, agent_velocities)):
            total_force = np.zeros(2)
            
            for beacon in self.beacons:
                if beacon.is_active:
                    beacon_force = beacon.get_goal_force(pos, vel)
                    total_force += beacon_force
            
            forces[i] = total_force
        
        return forces
    
    def get_goal_force_for_agent(self, pos, vel, agent_flock_id=None):
        """Get goal force for single agent"""
        total = np.zeros(2)
        for beacon in self.beacons:
            if beacon.is_active:
                total += beacon.get_goal_force(pos, vel, agent_flock_id=agent_flock_id)
        return total
    
    def update(self):
        """Update beacons and spawn new ones"""
        active_beacons = []
        freed_agents = set()
        
        for beacon in self.beacons:
            if beacon.update():
                active_beacons.append(beacon)
            else:
                freed_agents.update(beacon.get_trapped_agents())
        
        for agent_idx in freed_agents:
            if agent_idx in self.agent_beacon_map:
                del self.agent_beacon_map[agent_idx]
        
        beacons_disappeared = len(self.beacons) > len(active_beacons)
        self.beacons = active_beacons
        
        if beacons_disappeared and len(self.beacons) == 0:
            self.frames_since_last_spawn = 0
        
        self.frames_since_last_spawn += 1
        
        # Spawn new beacons
        if len(self.beacons) < self.max_concurrent_beacons and self.frames_since_last_spawn >= self.spawn_interval:
            success = self._spawn_next_beacon()
            if success:
                self.frames_since_last_spawn = 0
        
        # Increment frame counter
        self.frame += 1
    
    def get_active_beacons(self):
        """Get active beacons for visualization"""
        return [(b.position, b.radius, len(b.trapped_agents), b.owner_flock_id)
                for b in self.beacons if b.is_active]

def create_comparison_graphs(results: ValidationResults):
    """Create comparison graphs"""
    iterations = range(1, len(results.lj_total_times) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graph 1: Total completion time over iterations
    ax1.plot(iterations, results.lj_total_times, 'b-o', label='LJ-Swarm', linewidth=2, markersize=6)
    ax1.plot(iterations, results.os_total_times, 'r-s', label='Olfati-Saber-Flock', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Time to Complete 10 Zones (s)')
    ax1.set_title('Total Completion Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graph 2: Average time per zone over iterations
    ax2.plot(iterations, results.lj_avg_times, 'b-o', label='LJ-Swarm', linewidth=2, markersize=6)
    ax2.plot(iterations, results.os_avg_times, 'r-s', label='Olfati-Saber-Flock', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Average Time per Zone (s)')
    ax2.set_title('Average Time per Zone Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "validation_comparison.png"), dpi=300, bbox_inches="tight")
    
    # Print summary statistics
    lj_total_mean = np.mean(results.lj_total_times)
    lj_total_std = np.std(results.lj_total_times)
    lj_avg_mean = np.mean(results.lj_avg_times)
    lj_avg_std = np.std(results.lj_avg_times)
    
    os_total_mean = np.mean(results.os_total_times)
    os_total_std = np.std(results.os_total_times)
    os_avg_mean = np.mean(results.os_avg_times)
    os_avg_std = np.std(results.os_avg_times)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"LJ-Swarm      - Total time: {lj_total_mean:.2f}±{lj_total_std:.2f}s")
    print(f"LJ-Swarm      - Avg per zone: {lj_avg_mean:.2f}±{lj_avg_std:.2f}s")
    print(f"Olfati-Saber  - Total time: {os_total_mean:.2f}±{os_total_std:.2f}s")
    print(f"Olfati-Saber  - Avg per zone: {os_avg_mean:.2f}±{os_avg_std:.2f}s")
    print(f"{'='*60}")
    
    # Calculate improvement percentages
    total_improvement = (lj_total_mean - os_total_mean) / lj_total_mean * 100
    avg_improvement = (lj_avg_mean - os_avg_mean) / lj_avg_mean * 100
    
    if total_improvement > 0:
        print(f"🏆 Olfati-Saber is {total_improvement:.1f}% FASTER for total completion time")
    else:
        print(f"🏆 LJ-Swarm is {-total_improvement:.1f}% FASTER for total completion time")
    
    if avg_improvement > 0:
        print(f"🏆 Olfati-Saber is {avg_improvement:.1f}% FASTER for average time per zone")
    else:
        print(f"🏆 LJ-Swarm is {-avg_improvement:.1f}% FASTER for average time per zone")

def main():
    """Run the simple validation study"""
    print("Simple Validation: LJ-Swarm vs Olfati-Saber-Flock")
    print("Running 10 iterations with identical zone sequences")
    print("="*60)
    
    
    results = ValidationResults()
    
    # Run 10 iterations
    for iteration in range(1, 11):
        print(f"\nIteration {iteration}/10")
        
        # Generate random zones for this iteration
        zones = generate_random_zones(num_zones=10, seed=12345 + iteration)
        print(f"  Generated 10 zones for iteration {iteration}")
        
        # Run LJ-Swarm simulation
        lj_total, lj_avg, lj_success = run_lj_simulation(zones, iteration)
        
        # Run Olfati-Saber simulation
        os_total, os_avg, os_success = run_olfati_simulation(zones, iteration)
        
        # Store results if both succeeded
        if lj_success and os_success:
            results.lj_total_times.append(lj_total)
            results.lj_avg_times.append(lj_avg)
            results.os_total_times.append(os_total)
            results.os_avg_times.append(os_avg)
            
            results.iteration_data.append({
                'iteration': iteration,
                'lj_total': lj_total,
                'lj_avg': lj_avg,
                'os_total': os_total,
                'os_avg': os_avg,
                'zones': [(z.position, z.radius) for z in zones]
            })
            
            print(f"  Iteration {iteration} completed successfully")
        else:
            print(f"  Iteration {iteration} failed (LJ: {lj_success}, OS: {os_success})")
    
    # Save results
    with open(OUTPUT_DIR / "validation_results.json", "w") as f:
        json.dump(results.iteration_data, f, indent=2)
    
    # Create comparison graphs
    if results.lj_total_times and results.os_total_times:
        print(f"\nCreating comparison graphs...")
        create_comparison_graphs(results)
        print(f"\nValidation completed! Results saved to: {OUTPUT_DIR}")
    else:
        print("\nNo successful iterations to analyze")

if __name__ == "__main__":
    main()