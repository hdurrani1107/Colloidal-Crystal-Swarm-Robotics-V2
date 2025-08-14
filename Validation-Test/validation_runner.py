#######################################################################
# validation_runner.py
#
# Validation Testing Framework for LJ-Swarm vs Olfati-Saber-Flock
# Compares time to complete 10 goals/cooling-zones and average completion time
#
# Author: Humzah Durrani
#######################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import threading
import signal
import traceback
import gc

# Add parent directories to path to import simulation modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LJ-Swarm'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Olfati-Saber-Flock'))
sys.path.append(os.path.dirname(__file__))  # Add current directory for custom systems

from custom_systems import CustomCoolingZoneSystem, CustomGoalBeaconSystem

@dataclass
class GoalData:
    """Data structure for pre-determined goal spawn data"""
    position: Tuple[float, float]
    radius: float
    spawn_order: int

@dataclass
class TrialResult:
    """Results from a single trial"""
    algorithm: str
    trial_number: int
    total_time: float
    average_time_per_goal: float
    completion_times: List[float]
    simulation_frames: int
    success: bool
    error_message: str = ""

class GoalGenerator:
    """Generates consistent goal sets for both algorithms"""
    
    def __init__(self, bounds=[0, 200], seed=None):
        self.bounds = bounds
        if seed is not None:
            np.random.seed(seed)
    
    def generate_goal_set(self, num_goals=10, mean_radius=12.0, radius_std=2.0):
        """Generate a set of goals with positions and radii"""
        goals = []
        
        for i in range(num_goals):
            # Sample position ensuring it's not too close to boundaries
            margin = 25
            x = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
            y = np.random.uniform(self.bounds[0] + margin, self.bounds[1] - margin)
            
            # Sample radius from normal distribution (matching both algorithms)
            radius = max(5.0, np.random.normal(mean_radius, radius_std))
            
            goals.append(GoalData(
                position=(x, y),
                radius=radius,
                spawn_order=i
            ))
        
        return goals

def run_lj_swarm_trial(trial_num: int, goal_set: List[GoalData], output_dir: str) -> TrialResult:
    """Run a single LJ-Swarm trial with predetermined goals"""
    try:
        # Clear any existing imports and force garbage collection
        gc.collect()
        
        # Import LJ-Swarm modules
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LJ-Swarm'))
        
        from engine import multi_agent, init_melt
        from cooling_zone import CoolingZoneSystem
        
        # Simulation parameters (matching LJ-Swarm main.py)
        agents = 150
        sigma = 3.0
        epsilon = 3.0
        sample_time = 0.005
        bounds = [0, 200]
        obstacles = []
        
        # Initialize simulation
        sim = multi_agent(agents, sigma, sample_time, bounds, obstacles)
        
        # Setup cooling zones with custom goal injection
        cooling_zone_config = {
            'zone_radius': 10.0,
            'radius_std': 1.0,
            'max_concurrent_zones': 3,
            'spawn_interval': 400,
            'base_lifetime': 1000,
            'zone_temperature': 0.0,
            'logger': None
        }
        
        # Initialize with melted condition
        initial_temperature = 150
        init_melt(sim.agents, init_temp=initial_temperature)
        sim.initialize_agent_temperatures(initial_temperature)
        
        # Create custom cooling zone system with predetermined goals
        sim.cooling_zones = CustomCoolingZoneSystem(
            bounds=bounds,
            goal_set=goal_set,
            **cooling_zone_config
        )
        
        # Run simulation
        frame_count = 0
        completed_zones = 0
        target_zones = 10
        completion_times = []
        start_time = time.time()
        max_frames = 2000000  # Safety limit
        
        while completed_zones < target_zones and frame_count < max_frames:
            # Track zones before update
            zones_before = len(sim.cooling_zones.zones)
            
            # Run simulation step
            forces = sim.compute_forces(sigma, epsilon, (2**(1/6)) * sigma, 150, 10, 10, 0.5)
            sim.update(forces, 150)
            
            # Check for completed zones
            zones_after = len(sim.cooling_zones.zones)
            if zones_before > zones_after:
                zones_completed_this_step = zones_before - zones_after
                completed_zones += zones_completed_this_step
                current_time = frame_count * sample_time
                completion_times.extend([current_time] * zones_completed_this_step)
            
            frame_count += 1
        
        total_time = frame_count * sample_time
        avg_time = total_time / target_zones if target_zones > 0 else 0
        
        # Save trial data
        trial_data = {
            'algorithm': 'LJ-Swarm',
            'trial': trial_num,
            'total_time': total_time,
            'avg_time_per_goal': avg_time,
            'completion_times': completion_times,
            'frames': frame_count,
            'completed_goals': completed_zones
        }
        
        with open(os.path.join(output_dir, f'lj_swarm_trial_{trial_num}.json'), 'w') as f:
            json.dump(trial_data, f, indent=2)
        
        # Cleanup
        del sim
        gc.collect()
        
        return TrialResult(
            algorithm='LJ-Swarm',
            trial_number=trial_num,
            total_time=total_time,
            average_time_per_goal=avg_time,
            completion_times=completion_times,
            simulation_frames=frame_count,
            success=completed_zones >= target_zones
        )
        
    except Exception as e:
        return TrialResult(
            algorithm='LJ-Swarm',
            trial_number=trial_num,
            total_time=0,
            average_time_per_goal=0,
            completion_times=[],
            simulation_frames=0,
            success=False,
            error_message=str(e)
        )

def run_olfati_saber_trial(trial_num: int, goal_set: List[GoalData], output_dir: str) -> TrialResult:
    """Run a single Olfati-Saber trial with predetermined goals"""
    try:
        # Clear any existing imports and force garbage collection
        gc.collect()
        
        # Import Olfati-Saber modules
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Olfati-Saber-Flock'))
        
        from engine import multi_agent
        from goal_beacon import GoalBeaconSystem
        
        # Simulation parameters (matching Olfati-Saber main.py)
        agents = 150
        sample_time = 0.005
        bounds = [0, 200]
        obstacles = []
        
        # Initialize simulation
        sim = multi_agent(agents, sample_time, bounds, obstacles)
        
        # Setup goal beacons with custom goal injection
        goal_beacon_config = {
            'beacon_radius': 12.0,
            'radius_std': 2.0,
            'max_concurrent_beacons': 3,
            'spawn_interval': 400,
            'base_lifetime': 1000,
            'velocity_damping': 0.985,
            'logger': None
        }
        
        # Create custom goal beacon system with predetermined goals
        sim.goal_beacons = CustomGoalBeaconSystem(
            bounds=bounds,
            goal_set=goal_set,
            **goal_beacon_config
        )
        sim.goal_beacons._owner = sim
        
        # Run simulation
        frame_count = 0
        completed_beacons = 0
        target_beacons = 10
        completion_times = []
        start_time = time.time()
        max_frames = 2000000  # Safety limit
        
        while completed_beacons < target_beacons and frame_count < max_frames:
            # Track beacons before update
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
                completion_times.extend([current_time] * beacons_completed_this_step)
            
            frame_count += 1
        
        total_time = frame_count * sample_time
        avg_time = total_time / target_beacons if target_beacons > 0 else 0
        
        # Save trial data
        trial_data = {
            'algorithm': 'Olfati-Saber',
            'trial': trial_num,
            'total_time': total_time,
            'avg_time_per_goal': avg_time,
            'completion_times': completion_times,
            'frames': frame_count,
            'completed_goals': completed_beacons
        }
        
        with open(os.path.join(output_dir, f'olfati_saber_trial_{trial_num}.json'), 'w') as f:
            json.dump(trial_data, f, indent=2)
        
        # Cleanup
        del sim
        gc.collect()
        
        return TrialResult(
            algorithm='Olfati-Saber',
            trial_number=trial_num,
            total_time=total_time,
            average_time_per_goal=avg_time,
            completion_times=completion_times,
            simulation_frames=frame_count,
            success=completed_beacons >= target_beacons
        )
        
    except Exception as e:
        return TrialResult(
            algorithm='Olfati-Saber',
            trial_number=trial_num,
            total_time=0,
            average_time_per_goal=0,
            completion_times=[],
            simulation_frames=0,
            success=False,
            error_message=str(e)
        )

def run_validation_study(num_trials=10, output_base_dir="../output/validation-test"):
    """Run the complete validation study"""
    print("🚀 Starting LJ-Swarm vs Olfati-Saber Validation Study")
    print(f"📊 Running {num_trials} trials for each algorithm")
    
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    goal_generator = GoalGenerator(bounds=[0, 200])
    
    for trial in range(1, num_trials + 1):
        print(f"\n🔄 Trial {trial}/{num_trials}")
        
        # Generate consistent goal set for this trial
        trial_seed = 12345 + trial  # Ensure reproducibility
        goal_generator = GoalGenerator(bounds=[0, 200], seed=trial_seed)
        goal_set = goal_generator.generate_goal_set(num_goals=10)
        
        print(f"   Generated {len(goal_set)} goals for trial {trial}")
        
        # Run both algorithms concurrently with timeout
        print("   🔄 Running both algorithms concurrently...")
        
        try:
            # Use ThreadPoolExecutor for true concurrency with timeout
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both simulations
                lj_future = executor.submit(run_lj_swarm_trial, trial, goal_set, output_base_dir)
                os_future = executor.submit(run_olfati_saber_trial, trial, goal_set, output_base_dir)
                
                # Wait for both to complete with timeout (10 minutes per simulation)
                timeout = 600  # 10 minutes
                
                try:
                    lj_result = lj_future.result(timeout=timeout)
                    print("   ✅ LJ-Swarm completed")
                except Exception as e:
                    print(f"   ❌ LJ-Swarm failed or timed out: {e}")
                    lj_result = TrialResult(
                        algorithm='LJ-Swarm', trial_number=trial, total_time=0, 
                        average_time_per_goal=0, completion_times=[], 
                        simulation_frames=0, success=False, error_message=str(e)
                    )
                
                try:
                    os_result = os_future.result(timeout=timeout)
                    print("   ✅ Olfati-Saber completed")
                except Exception as e:
                    print(f"   ❌ Olfati-Saber failed or timed out: {e}")
                    os_result = TrialResult(
                        algorithm='Olfati-Saber', trial_number=trial, total_time=0, 
                        average_time_per_goal=0, completion_times=[], 
                        simulation_frames=0, success=False, error_message=str(e)
                    )
                
        except Exception as e:
            print(f"   ❌ Concurrent execution failed: {e}")
            # Fallback to sequential execution
            print("   🔄 Falling back to sequential execution...")
            lj_result = run_lj_swarm_trial(trial, goal_set, output_base_dir)
            os_result = run_olfati_saber_trial(trial, goal_set, output_base_dir)
        
        # Report trial results
        if lj_result.success:
            print(f"   ✅ LJ-Swarm: {lj_result.total_time:.2f}s total, {lj_result.average_time_per_goal:.2f}s avg")
        else:
            print(f"   ❌ LJ-Swarm failed: {lj_result.error_message}")
            
        if os_result.success:
            print(f"   ✅ Olfati-Saber: {os_result.total_time:.2f}s total, {os_result.average_time_per_goal:.2f}s avg")
        else:
            print(f"   ❌ Olfati-Saber failed: {os_result.error_message}")
        
        all_results.extend([lj_result, os_result])
        
        # Force garbage collection between trials
        gc.collect()
        print(f"   🧹 Trial {trial} cleanup completed")
    
    # Generate final comparison analysis
    print("\n📈 Generating validation analysis...")
    generate_validation_analysis(all_results, output_base_dir)
    
    print(f"\n🎯 Validation study completed!")
    print(f"📁 Results saved to: {output_base_dir}")

def generate_validation_analysis(results: List[TrialResult], output_dir: str):
    """Generate comprehensive analysis of validation results"""
    
    # Separate results by algorithm
    lj_results = [r for r in results if r.algorithm == 'LJ-Swarm' and r.success]
    os_results = [r for r in results if r.algorithm == 'Olfati-Saber' and r.success]
    
    if len(lj_results) == 0 or len(os_results) == 0:
        print("❌ Insufficient successful trials for analysis")
        return
    
    # Extract metrics
    lj_total_times = [r.total_time for r in lj_results]
    lj_avg_times = [r.average_time_per_goal for r in lj_results]
    os_total_times = [r.total_time for r in os_results]
    os_avg_times = [r.average_time_per_goal for r in os_results]
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total completion times
    trials = range(1, len(lj_total_times) + 1)
    ax1.plot(trials, lj_total_times, 'b-o', label='LJ-Swarm', linewidth=2, markersize=6)
    ax1.plot(trials, os_total_times, 'r-s', label='Olfati-Saber', linewidth=2, markersize=6)
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Total Time to Complete 10 Goals (s)')
    ax1.set_title('Total Completion Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average time per goal
    ax2.plot(trials, lj_avg_times, 'b-o', label='LJ-Swarm', linewidth=2, markersize=6)
    ax2.plot(trials, os_avg_times, 'r-s', label='Olfati-Saber', linewidth=2, markersize=6)
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Average Time per Goal (s)')
    ax2.set_title('Average Time per Goal Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plots for total times
    ax3.boxplot([lj_total_times, os_total_times], labels=['LJ-Swarm', 'Olfati-Saber'])
    ax3.set_ylabel('Total Time to Complete 10 Goals (s)')
    ax3.set_title('Total Completion Time Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plots for average times
    ax4.boxplot([lj_avg_times, os_avg_times], labels=['LJ-Swarm', 'Olfati-Saber'])
    ax4.set_ylabel('Average Time per Goal (s)')
    ax4.set_title('Average Time per Goal Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary statistics
    summary_stats = {
        'LJ-Swarm': {
            'total_time_mean': np.mean(lj_total_times),
            'total_time_std': np.std(lj_total_times),
            'avg_time_mean': np.mean(lj_avg_times),
            'avg_time_std': np.std(lj_avg_times),
            'successful_trials': len(lj_results)
        },
        'Olfati-Saber': {
            'total_time_mean': np.mean(os_total_times),
            'total_time_std': np.std(os_total_times),
            'avg_time_mean': np.mean(os_avg_times),
            'avg_time_std': np.std(os_avg_times),
            'successful_trials': len(os_results)
        }
    }
    
    # Calculate performance comparison
    total_time_improvement = (summary_stats['LJ-Swarm']['total_time_mean'] - 
                            summary_stats['Olfati-Saber']['total_time_mean']) / summary_stats['LJ-Swarm']['total_time_mean'] * 100
    avg_time_improvement = (summary_stats['LJ-Swarm']['avg_time_mean'] - 
                          summary_stats['Olfati-Saber']['avg_time_mean']) / summary_stats['LJ-Swarm']['avg_time_mean'] * 100
    
    summary_stats['comparison'] = {
        'total_time_improvement_percent': total_time_improvement,
        'avg_time_improvement_percent': avg_time_improvement
    }
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'validation_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Print summary
    print(f"\n📊 VALIDATION RESULTS SUMMARY:")
    print(f"{'='*50}")
    print(f"LJ-Swarm - Total time: {summary_stats['LJ-Swarm']['total_time_mean']:.2f}±{summary_stats['LJ-Swarm']['total_time_std']:.2f}s")
    print(f"LJ-Swarm - Avg per goal: {summary_stats['LJ-Swarm']['avg_time_mean']:.2f}±{summary_stats['LJ-Swarm']['avg_time_std']:.2f}s")
    print(f"Olfati-Saber - Total time: {summary_stats['Olfati-Saber']['total_time_mean']:.2f}±{summary_stats['Olfati-Saber']['total_time_std']:.2f}s")
    print(f"Olfati-Saber - Avg per goal: {summary_stats['Olfati-Saber']['avg_time_mean']:.2f}±{summary_stats['Olfati-Saber']['avg_time_std']:.2f}s")
    print(f"{'='*50}")
    if total_time_improvement > 0:
        print(f"🏆 Olfati-Saber is {total_time_improvement:.1f}% FASTER for total completion time")
    else:
        print(f"🏆 LJ-Swarm is {-total_time_improvement:.1f}% FASTER for total completion time")
    if avg_time_improvement > 0:
        print(f"🏆 Olfati-Saber is {avg_time_improvement:.1f}% FASTER for average time per goal")
    else:
        print(f"🏆 LJ-Swarm is {-avg_time_improvement:.1f}% FASTER for average time per goal")

if __name__ == "__main__":
    # Check if number of trials specified
    num_trials = 10
    if len(sys.argv) > 1:
        try:
            num_trials = int(sys.argv[1])
        except ValueError:
            print("Invalid number of trials. Using default: 10")
    
    run_validation_study(num_trials=num_trials)