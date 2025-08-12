#######################################################################
# main_simulation.py
#
# Main entry point for swarm mesh network simulation
#
# Author: Humzah Durrani
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from mesh_network_engine import MeshNetworkSimulation
from mesh_visualization import MeshNetworkVisualizer

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

def run_simulation():
    """Run the main simulation"""
    
    # Simulation parameters
    bounds = [0, 50]
    sample_time = 0.01
    n_steps = 3000
    
    # Initialize simulation
    sim = MeshNetworkSimulation(bounds, sample_time)
    create_test_scenario(sim)
    
    # Initialize visualization
    visualizer = MeshNetworkVisualizer(bounds, figsize=(14, 10))
    visualizer.draw_static_elements(sim)
    visualizer.add_legend()
    
    # Metrics tracking
    step_data = {
        'time_steps': [],
        'coverage_percentage': [],
        'connectivity_ratio': [],
        'avg_temperature': [],
        'deployed_agents': [],
        'network_efficiency': []
    }
    
    # Enable interactive mode
    plt.ion()
    
    print("Starting Swarm Mesh Network Simulation...")
    print(f"Agents: {len(sim.agents)}")
    print(f"Infrastructure Nodes: {len(sim.infrastructure_nodes)}")
    print(f"Obstacles: {len(sim.obstacles)}")
    print(f"Deadzones: {len(sim.deadzones)}")
    print(f"Noise Zones: {len(sim.noise_zones)}")
    print("-" * 50)
    
    # Simulation loop
    try:
        for step in tqdm(range(n_steps), desc="Simulation Progress"):
            # Run simulation step
            sim.step()
            
            # Calculate metrics every 10 steps
            if step % 10 == 0:
                # Coverage metrics
                coverage_map = sim.compute_coverage_gaps()
                coverage_percentage = np.mean(coverage_map > 0.5) * 100
                
                # Connectivity metrics
                total_possible_connections = len(sim.agents) * (len(sim.infrastructure_nodes) + len(sim.agents) - 1)
                actual_connections = sum(len(agent.connected_nodes) + len(agent.connected_agents) 
                                       for agent in sim.agents)
                connectivity_ratio = actual_connections / max(1, total_possible_connections)
                
                # Temperature metrics
                avg_temperature = np.mean([agent.temperature for agent in sim.agents])
                
                # Deployment metrics
                deployed_count = sum(1 for agent in sim.agents if agent.is_deployed)
                
                # Network efficiency (simplified)
                network_efficiency = (coverage_percentage / 100) * connectivity_ratio
                
                # Store metrics
                step_data['time_steps'].append(step)
                step_data['coverage_percentage'].append(coverage_percentage)
                step_data['connectivity_ratio'].append(connectivity_ratio)
                step_data['avg_temperature'].append(avg_temperature)
                step_data['deployed_agents'].append(deployed_count)
                step_data['network_efficiency'].append(network_efficiency)
                
                # Print progress
                if step % 100 == 0:
                    print(f"Step {step}: Coverage={coverage_percentage:.1f}%, "
                          f"Connectivity={connectivity_ratio:.3f}, "
                          f"Deployed={deployed_count}/{len(sim.agents)}, "
                          f"Avg Temp={avg_temperature:.2f}")
            
            # Update visualization every 20 steps
            if step % 20 == 0:
                visualizer.update_display(sim, step_data)
                plt.pause(0.01)  # Small pause for visualization
            
            # Generate some data packets for testing
            if step % 50 == 0 and len(sim.agents) > 1:
                # Create random data transmission
                source_id = np.random.randint(0, len(sim.infrastructure_nodes))
                dest_agent_id = len(sim.infrastructure_nodes) + np.random.randint(0, len(sim.agents))
                sim.create_data_packet(source_id, dest_agent_id, f"Data packet {step}")
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    # Final metrics
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print("="*50)
    
    if step_data['coverage_percentage']:
        final_coverage = step_data['coverage_percentage'][-1]
        final_connectivity = step_data['connectivity_ratio'][-1]
        final_deployed = step_data['deployed_agents'][-1]
        final_efficiency = step_data['network_efficiency'][-1]
        
        print(f"Final Coverage: {final_coverage:.1f}%")
        print(f"Final Connectivity: {final_connectivity:.3f}")
        print(f"Deployed Agents: {final_deployed}/{len(sim.agents)}")
        print(f"Network Efficiency: {final_efficiency:.3f}")
        print(f"Data Packets Processed: {sim.packet_counter}")
    
    # Save final visualization
    from output_utils import get_output_path
    final_state_path = get_output_path('analysis', 'mesh_network_final_state.png', 'comms_network')
    visualizer.save_frame(final_state_path, dpi=300)
    print(f"Final state saved as '{final_state_path}'")
    
    # Keep visualization open
    plt.ioff()
    plt.show()
    
    return sim, visualizer, step_data

def analyze_results(sim: MeshNetworkSimulation, step_data: dict):
    """Analyze and plot simulation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Swarm Mesh Network Analysis', fontsize=16)
    
    time_steps = step_data['time_steps']
    
    # Coverage over time
    axes[0, 0].plot(time_steps, step_data['coverage_percentage'], 'g-', linewidth=2)
    axes[0, 0].set_title('Coverage Percentage Over Time')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Coverage (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Connectivity over time
    axes[0, 1].plot(time_steps, step_data['connectivity_ratio'], 'b-', linewidth=2)
    axes[0, 1].set_title('Network Connectivity Over Time')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Connectivity Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Agent deployment
    axes[1, 0].plot(time_steps, step_data['deployed_agents'], 'r-', linewidth=2)
    axes[1, 0].axhline(y=len(sim.agents), color='r', linestyle='--', alpha=0.7, label='Total Agents')
    axes[1, 0].set_title('Agent Deployment Over Time')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Deployed Agents')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Network efficiency
    axes[1, 1].plot(time_steps, step_data['network_efficiency'], 'm-', linewidth=2)
    axes[1, 1].set_title('Network Efficiency Over Time')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Efficiency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    analysis_path = get_output_path('analysis', 'mesh_network_analysis.png', 'comms_network')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    print(f"Analysis saved as '{analysis_path}'")
    plt.show()

if __name__ == "__main__":
    print("Swarm Mesh Network Simulation")
    print("============================")
    
    # Run the simulation
    sim, visualizer, step_data = run_simulation()
    
    # Analyze results
    analyze_results(sim, step_data)
    
    print("\nSimulation complete! Check the generated images for results.")