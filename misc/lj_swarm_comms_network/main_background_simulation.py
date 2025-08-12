#######################################################################
# main_background_simulation.py
#
# Main entry point for background mesh network simulation with MP4 recording
#
# Author: Humzah Durrani
#######################################################################

import numpy as np
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\hummy\anaconda3\envs\my-env\Library\bin\ffmpeg.exe'
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from mesh_network_engine import MeshNetworkSimulation
from background_visualization import BackgroundMeshVisualizer, SimulationRecorder

def create_test_scenario(sim: MeshNetworkSimulation):
    """Create a test scenario with infrastructure, obstacles, and coverage gaps"""
    
    # Add infrastructure nodes (base stations)
    sim.add_infrastructure_node(np.array([10, 10]), signal_strength=15.0, coverage_radius=12.0)
    sim.add_infrastructure_node(np.array([40, 40]), signal_strength=15.0, coverage_radius=12.0)
    sim.add_infrastructure_node(np.array([10, 40]), signal_strength=12.0, coverage_radius=10.0)
    
    # Add obstacles (buildings/terrain)
    sim.add_obstacle(np.array([25, 25]), 6.0)  # Large central obstacle
    sim.add_obstacle(np.array([15, 30]), 3.0)  # Medium obstacle
    sim.add_obstacle(np.array([35, 15]), 2.5)  # Small obstacle
    
    # Add communication deadzones
    sim.add_deadzone(np.array([30, 10]), 4.0)  # Dead zone in bottom right
    sim.add_deadzone(np.array([20, 35]), 3.0)  # Dead zone in top middle
    
    # Add noise interference zones
    sim.add_noise_zone(np.array([5, 25]), 4.0, 15.0)   # High noise zone
    sim.add_noise_zone(np.array([45, 20]), 3.0, 10.0)  # Medium noise zone
    
    # Add mobile agents at random positions
    np.random.seed(42)  # For reproducible results
    n_agents = 20
    
    for i in range(n_agents):
        # Ensure agents don't spawn inside obstacles
        while True:
            pos = np.random.uniform(sim.bounds[0] + 2, sim.bounds[1] - 2, 2)
            valid_position = True
            
            # Check obstacles
            for obs_pos, obs_radius in sim.obstacles:
                if np.linalg.norm(pos - obs_pos) < obs_radius + 2.0:
                    valid_position = False
                    break
            
            if valid_position:
                break
        
        # Vary agent properties
        sigma = np.random.uniform(2.5, 3.5)
        epsilon = np.random.uniform(2.5, 3.5)
        rssi_strength = np.random.uniform(8.0, 12.0)
        initial_temp = np.random.uniform(3.0, 8.0)  # Start with high temperature for exploration
        
        sim.add_agent(pos, sigma=sigma, epsilon=epsilon, 
                     rssi_strength=rssi_strength, temp=initial_temp)

def run_background_simulation():
    """Run the simulation in background mode with MP4 recording"""
    
    # Simulation parameters
    bounds = [0, 50]
    sample_time = 0.01
    n_steps = 2000  # Reduced for reasonable video length
    fps = 20       # Frames per second for video
    
    print("Swarm Mesh Network - Background Simulation")
    print("=" * 50)
    
    # Initialize simulation
    sim = MeshNetworkSimulation(bounds, sample_time)
    create_test_scenario(sim)
    
    print(f"Simulation Setup:")
    print(f"  Agents: {len(sim.agents)}")
    print(f"  Infrastructure Nodes: {len(sim.infrastructure_nodes)}")
    print(f"  Obstacles: {len(sim.obstacles)}")
    print(f"  Deadzones: {len(sim.deadzones)}")
    print(f"  Noise Zones: {len(sim.noise_zones)}")
    print(f"  Total Steps: {n_steps}")
    print(f"  Video FPS: {fps}")
    
    # Initialize background visualizer
    visualizer = BackgroundMeshVisualizer(bounds)
    recorder = SimulationRecorder(sim, visualizer)
    
    print("\nStarting background simulation with video recording...")
    print("This will create two MP4 files:")
    print("  - agent_simulation.mp4 (agent movement and connections)")
    print("  - coverage_simulation.mp4 (coverage quality heatmap)")
    print("-" * 50)
    
    start_time = time.time()
    
    # Run simulation with recording
    try:
        metrics_data = recorder.run_simulation_with_recording(n_steps, fps=fps)
        
        end_time = time.time()
        simulation_time = end_time - start_time
        
        print(f"\nSimulation completed in {simulation_time:.2f} seconds")
        
        # Final metrics
        if metrics_data['coverage_percentage']:
            final_coverage = metrics_data['coverage_percentage'][-1]
            final_connectivity = metrics_data['connectivity_ratio'][-1]
            final_deployed = metrics_data['deployed_agents'][-1]
            final_efficiency = metrics_data['network_efficiency'][-1]
            
            print("\nFinal Results:")
            print(f"  Coverage: {final_coverage:.1f}%")
            print(f"  Connectivity: {final_connectivity:.3f}")
            print(f"  Deployed Agents: {final_deployed}/{len(sim.agents)}")
            print(f"  Network Efficiency: {final_efficiency:.3f}")
            print(f"  Data Packets Processed: {sim.packet_counter}")
        
        return sim, metrics_data
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return sim, recorder.metrics_data
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_analysis_plots(sim: MeshNetworkSimulation, metrics_data: dict):
    """Create final analysis plots after simulation"""
    
    if not metrics_data['time_steps']:
        print("No metrics data available for analysis")
        return
    
    print("\nCreating analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Swarm Mesh Network - Final Analysis', fontsize=16, fontweight='bold')
    
    time_steps = metrics_data['time_steps']
    
    # Coverage over time
    axes[0, 0].plot(time_steps, metrics_data['coverage_percentage'], 'g-', linewidth=3)
    axes[0, 0].set_title('Coverage Percentage Over Time', fontsize=14)
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Coverage (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 100)
    
    # Add coverage target line
    axes[0, 0].axhline(y=80, color='g', linestyle='--', alpha=0.7, label='Target (80%)')
    axes[0, 0].legend()
    
    # Connectivity over time
    axes[0, 1].plot(time_steps, metrics_data['connectivity_ratio'], 'b-', linewidth=3)
    axes[0, 1].set_title('Network Connectivity Over Time', fontsize=14)
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Connectivity Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Agent deployment
    axes[1, 0].plot(time_steps, metrics_data['deployed_agents'], 'r-', linewidth=3)
    axes[1, 0].axhline(y=len(sim.agents), color='r', linestyle='--', alpha=0.7, 
                       label=f'Total Agents ({len(sim.agents)})')
    axes[1, 0].set_title('Agent Deployment Over Time', fontsize=14)
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Deployed Agents')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, len(sim.agents) + 1)
    
    # Network efficiency and temperature
    ax_temp = axes[1, 1].twinx()
    
    # Network efficiency
    line1 = axes[1, 1].plot(time_steps, metrics_data['network_efficiency'], 
                           'm-', linewidth=3, label='Network Efficiency')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Network Efficiency', color='m')
    axes[1, 1].tick_params(axis='y', labelcolor='m')
    axes[1, 1].set_ylim(0, 1)
    
    # Average temperature
    line2 = ax_temp.plot(time_steps, metrics_data['avg_temperature'], 
                        'orange', linewidth=2, alpha=0.7, label='Avg Temperature')
    ax_temp.set_ylabel('Average Temperature', color='orange')
    ax_temp.tick_params(axis='y', labelcolor='orange')
    
    # Combined legend
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax_temp.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    axes[1, 1].set_title('Network Efficiency & Temperature', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add summary statistics text box
    if metrics_data['coverage_percentage']:
        max_coverage = max(metrics_data['coverage_percentage'])
        final_coverage = metrics_data['coverage_percentage'][-1]
        max_connectivity = max(metrics_data['connectivity_ratio'])
        final_deployed = metrics_data['deployed_agents'][-1]
        
        summary_text = f"""Summary Statistics:
Max Coverage: {max_coverage:.1f}%
Final Coverage: {final_coverage:.1f}%
Max Connectivity: {max_connectivity:.3f}
Final Deployed: {final_deployed}/{len(sim.agents)}
Deployment Rate: {(final_deployed/len(sim.agents)*100):.1f}%"""
        
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for summary text
    
    # Save analysis
    analysis_filename = "mesh_network_analysis.png"
    plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
    print(f"Analysis saved as '{analysis_filename}'")
    
    plt.show()

def main():
    """Main function"""
    print("Starting Swarm Mesh Network Background Simulation")
    
    # Change to the script's directory if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("mesh_network_engine.py"):
        print(f"Changing to script directory: {script_dir}")
        os.chdir(script_dir)
    
    # Check if required files exist
    if not os.path.exists("mesh_network_engine.py"):
        print("Error: mesh_network_engine.py not found")
        print(f"Current directory: {os.getcwd()}")
        return
    
    # Run background simulation
    sim, metrics_data = run_background_simulation()
    
    if sim is None:
        print("Simulation failed")
        return
    
    # Create analysis plots
    create_analysis_plots(sim, metrics_data)
    
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE")
    print("=" * 50)
    print("Generated files:")
    print("  - agent_simulation.mp4 (agent movement and network topology)")
    print("  - coverage_simulation.mp4 (coverage quality heatmap)")
    print("  - mesh_network_analysis.png (final analysis plots)")
    print("\nThe simulation ran in the background while generating videos.")
    print("No real-time plots were displayed during simulation.")

if __name__ == "__main__":
    main()