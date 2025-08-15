#######################################################################
# validation_runner_fixed.py
#
# Fixed Validation Testing Framework for LJ-Swarm vs Olfati-Saber-Flock
# Fixes concurrency and trial transition issues
#
# Author: Humzah Durrani
#######################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import gc
import signal
from dataclasses import dataclass
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback

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

def run_lj_swarm_trial_isolated(trial_num: int, goal_set: List[GoalData], output_dir: str) -> TrialResult:
    """Run LJ-Swarm trial in isolated environment"""
    try:
        print(f"      [LJ-Swarm Trial {trial_num}] Starting...")
        
        # Set up isolated import paths
        script_dir = os.path.dirname(__file__)
        lj_path = os.path.join(script_dir, '..', 'LJ-Swarm')
        val_path = script_dir
        
        # Clear module cache for clean imports
        modules_to_clear = [mod for mod in sys.modules.keys() if 'engine' in mod or 'cooling_zone' in mod]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Import with absolute paths
        sys.path.insert(0, lj_path)
        sys.path.insert(0, val_path)
        
        import engine
        import cooling_zone
        from custom_systems import create_custom_cooling_zone_system
        
        # Simulation parameters
        agents = 150
        sigma = 3.0
        epsilon = 3.0
        sample_time = 0.005
        bounds = [0, 200]
        obstacles = []
        
        print(f"      [LJ-Swarm Trial {trial_num}] Initializing simulation...")
        
        # Initialize simulation
        sim = engine.multi_agent(agents, sigma, sample_time, bounds, obstacles)
        
        # Setup cooling zones configuration
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
        engine.init_melt(sim.agents, init_temp=initial_temperature)
        sim.initialize_agent_temperatures(initial_temperature)
        
        # Create custom cooling zone system
        sim.cooling_zones = create_custom_cooling_zone_system(
            bounds=bounds,
            goal_set=goal_set,
            **cooling_zone_config
        )
        
        print(f"      [LJ-Swarm Trial {trial_num}] Running simulation...")
        
        # Run simulation with progress tracking
        frame_count = 0
        completed_zones = 0
        target_zones = 10
        completion_times = []
        max_frames = 2000000  # Safety limit
        last_progress_frame = 0
        
        while completed_zones < target_zones and frame_count < max_frames:
            # Progress reporting every 10000 frames
            if frame_count - last_progress_frame >= 10000:
                print(f"      [LJ-Swarm Trial {trial_num}] Frame {frame_count}, completed: {completed_zones}/{target_zones}")
                last_progress_frame = frame_count
            
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
                print(f"      [LJ-Swarm Trial {trial_num}] Zone completed! Total: {completed_zones}/{target_zones}")
            
            frame_count += 1
        
        total_time = frame_count * sample_time
        avg_time = total_time / target_zones if target_zones > 0 else 0
        
        print(f"      [LJ-Swarm Trial {trial_num}] Completed in {total_time:.2f}s")
        
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
        del engine
        del cooling_zone
        sys.path.remove(lj_path)
        if val_path in sys.path:
            sys.path.remove(val_path)
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
        print(f"      [LJ-Swarm Trial {trial_num}] ERROR: {e}")
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

def run_olfati_saber_trial_isolated(trial_num: int, goal_set: List[GoalData], output_dir: str) -> TrialResult:
    """Run Olfati-Saber trial in isolated environment"""
    try:
        print(f"      [Olfati-Saber Trial {trial_num}] Starting...")
        
        # Set up isolated import paths
        script_dir = os.path.dirname(__file__)
        os_path = os.path.join(script_dir, '..', 'Olfati-Saber-Flock')
        val_path = script_dir
        
        # Clear module cache for clean imports
        modules_to_clear = [mod for mod in sys.modules.keys() if 'engine' in mod or 'goal_beacon' in mod]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Import with absolute paths
        sys.path.insert(0, os_path)
        sys.path.insert(0, val_path)
        
        import engine
        import goal_beacon
        from custom_systems import create_custom_goal_beacon_system
        
        # Simulation parameters
        agents = 150
        sample_time = 0.005
        bounds = [0, 200]
        obstacles = []
        
        print(f"      [Olfati-Saber Trial {trial_num}] Initializing simulation...")
        
        # Initialize simulation
        sim = engine.multi_agent(agents, sample_time, bounds, obstacles)
        
        # Setup goal beacons configuration
        goal_beacon_config = {
            'beacon_radius': 12.0,
            'radius_std': 2.0,
            'max_concurrent_beacons': 3,
            'spawn_interval': 400,
            'base_lifetime': 1000,
            'velocity_damping': 0.985,
            'logger': None
        }
        
        # Create custom goal beacon system
        sim.goal_beacons = create_custom_goal_beacon_system(
            bounds=bounds,
            goal_set=goal_set,
            **goal_beacon_config
        )
        sim.goal_beacons._owner = sim
        
        print(f"      [Olfati-Saber Trial {trial_num}] Running simulation...")
        
        # Run simulation with progress tracking
        frame_count = 0
        completed_beacons = 0
        target_beacons = 10
        completion_times = []
        max_frames = 2000000  # Safety limit
        last_progress_frame = 0
        
        while completed_beacons < target_beacons and frame_count < max_frames:
            # Progress reporting every 10000 frames
            if frame_count - last_progress_frame >= 10000:
                print(f"      [Olfati-Saber Trial {trial_num}] Frame {frame_count}, completed: {completed_beacons}/{target_beacons}")
                last_progress_frame = frame_count
            
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
                print(f"      [Olfati-Saber Trial {trial_num}] Beacon completed! Total: {completed_beacons}/{target_beacons}")
            
            frame_count += 1
        
        total_time = frame_count * sample_time
        avg_time = total_time / target_beacons if target_beacons > 0 else 0
        
        print(f"      [Olfati-Saber Trial {trial_num}] Completed in {total_time:.2f}s")
        
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
        del engine
        del goal_beacon
        sys.path.remove(os_path)
        if val_path in sys.path:
            sys.path.remove(val_path)
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
        print(f"      [Olfati-Saber Trial {trial_num}] ERROR: {e}")
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
    """Run the complete validation study with proper concurrency and cleanup"""
    print("Starting LJ-Swarm vs Olfati-Saber Validation Study")
    print(f"Running {num_trials} trials for each algorithm")
    
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    for trial in range(1, num_trials + 1):
        print(f"\n=== Trial {trial}/{num_trials} ===")
        
        # Generate consistent goal set for this trial
        trial_seed = 12345 + trial
        goal_generator = GoalGenerator(bounds=[0, 200], seed=trial_seed)
        goal_set = goal_generator.generate_goal_set(num_goals=10)
        
        print(f"Generated {len(goal_set)} goals for trial {trial}")
        
        # Run both algorithms concurrently with timeout
        print("Running both algorithms concurrently...")
        
        lj_result = None
        os_result = None
        
        try:
            # Use ThreadPoolExecutor for true concurrency
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both simulations
                lj_future = executor.submit(run_lj_swarm_trial_isolated, trial, goal_set, output_base_dir)
                os_future = executor.submit(run_olfati_saber_trial_isolated, trial, goal_set, output_base_dir)
                
                # Wait for both with timeout (15 minutes each)
                timeout = 900  # 15 minutes
                
                try:
                    lj_result = lj_future.result(timeout=timeout)
                    print(f"   LJ-Swarm completed successfully")
                except TimeoutError:
                    print(f"   LJ-Swarm timed out after {timeout}s")
                    lj_result = TrialResult(
                        algorithm='LJ-Swarm', trial_number=trial, total_time=0,
                        average_time_per_goal=0, completion_times=[],
                        simulation_frames=0, success=False, error_message="Timeout"
                    )
                except Exception as e:
                    print(f"   LJ-Swarm failed: {e}")
                    lj_result = TrialResult(
                        algorithm='LJ-Swarm', trial_number=trial, total_time=0,
                        average_time_per_goal=0, completion_times=[],
                        simulation_frames=0, success=False, error_message=str(e)
                    )
                
                try:
                    os_result = os_future.result(timeout=timeout)
                    print(f"   Olfati-Saber completed successfully")
                except TimeoutError:
                    print(f"   Olfati-Saber timed out after {timeout}s")
                    os_result = TrialResult(
                        algorithm='Olfati-Saber', trial_number=trial, total_time=0,
                        average_time_per_goal=0, completion_times=[],
                        simulation_frames=0, success=False, error_message="Timeout"
                    )
                except Exception as e:
                    print(f"   Olfati-Saber failed: {e}")
                    os_result = TrialResult(
                        algorithm='Olfati-Saber', trial_number=trial, total_time=0,
                        average_time_per_goal=0, completion_times=[],
                        simulation_frames=0, success=False, error_message=str(e)
                    )
                
        except Exception as e:
            print(f"   Concurrent execution failed: {e}")
            print("   Falling back to sequential execution...")
            
            # Sequential fallback
            lj_result = run_lj_swarm_trial_isolated(trial, goal_set, output_base_dir)
            os_result = run_olfati_saber_trial_isolated(trial, goal_set, output_base_dir)
        
        # Report trial results
        if lj_result and lj_result.success:
            print(f"   LJ-Swarm: {lj_result.total_time:.2f}s total, {lj_result.average_time_per_goal:.2f}s avg")
        elif lj_result:
            print(f"   LJ-Swarm failed: {lj_result.error_message}")
            
        if os_result and os_result.success:
            print(f"   Olfati-Saber: {os_result.total_time:.2f}s total, {os_result.average_time_per_goal:.2f}s avg")
        elif os_result:
            print(f"   Olfati-Saber failed: {os_result.error_message}")
        
        if lj_result:
            all_results.append(lj_result)
        if os_result:
            all_results.append(os_result)
        
        # Force cleanup between trials
        gc.collect()
        print(f"Trial {trial} cleanup completed\\n")
    
    # Generate final comparison analysis
    print("\\nGenerating validation analysis...")
    generate_validation_analysis(all_results, output_base_dir)
    
    print(f"\\nValidation study completed!")
    print(f"Results saved to: {output_base_dir}")

def generate_validation_analysis(results: List[TrialResult], output_dir: str):
    """Generate comprehensive analysis of validation results"""
    
    # Separate results by algorithm
    lj_results = [r for r in results if r.algorithm == 'LJ-Swarm' and r.success]
    os_results = [r for r in results if r.algorithm == 'Olfati-Saber' and r.success]
    
    if len(lj_results) == 0 or len(os_results) == 0:
        print("Insufficient successful trials for analysis")
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
    print(f"\nVALIDATION RESULTS SUMMARY:")
    print(f"=" * 50)
    print(f"LJ-Swarm - Total time: {summary_stats['LJ-Swarm']['total_time_mean']:.2f}±{summary_stats['LJ-Swarm']['total_time_std']:.2f}s")
    print(f"LJ-Swarm - Avg per goal: {summary_stats['LJ-Swarm']['avg_time_mean']:.2f}±{summary_stats['LJ-Swarm']['avg_time_std']:.2f}s")
    print(f"Olfati-Saber - Total time: {summary_stats['Olfati-Saber']['total_time_mean']:.2f}±{summary_stats['Olfati-Saber']['total_time_std']:.2f}s")
    print(f"Olfati-Saber - Avg per goal: {summary_stats['Olfati-Saber']['avg_time_mean']:.2f}±{summary_stats['Olfati-Saber']['avg_time_std']:.2f}s")
    print(f"=" * 50)
    if total_time_improvement > 0:
        print(f"Olfati-Saber is {total_time_improvement:.1f}% FASTER for total completion time")
    else:
        print(f"LJ-Swarm is {-total_time_improvement:.1f}% FASTER for total completion time")
    if avg_time_improvement > 0:
        print(f"Olfati-Saber is {avg_time_improvement:.1f}% FASTER for average time per goal")
    else:
        print(f"LJ-Swarm is {-avg_time_improvement:.1f}% FASTER for average time per goal")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Validation Testing Framework')
    parser.add_argument('--trials', '-t', type=int, default=10, help='Number of trials (default: 10)')
    parser.add_argument('--output', '-o', type=str, default='../output/validation-test', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        run_validation_study(num_trials=args.trials, output_base_dir=args.output)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
    except Exception as e:
        print(f"Validation failed: {e}")
        traceback.print_exc()