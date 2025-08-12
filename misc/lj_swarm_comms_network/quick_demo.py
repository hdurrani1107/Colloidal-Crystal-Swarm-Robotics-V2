#######################################################################
# quick_demo.py
#
# Quick demonstration of the background simulation system
#
# Author: Humzah Durrani
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from mesh_network_engine import MeshNetworkSimulation
from background_visualization import BackgroundMeshVisualizer

def create_demo_scenario(sim: MeshNetworkSimulation):
    """Create a small demo scenario"""
    
    # Add infrastructure nodes
    sim.add_infrastructure_node(np.array([8, 8]), signal_strength=10.0, coverage_radius=8.0)
    sim.add_infrastructure_node(np.array([22, 22]), signal_strength=10.0, coverage_radius=8.0)
    
    # Add one obstacle and deadzone
    sim.add_obstacle(np.array([15, 15]), 3.0)
    sim.add_deadzone(np.array([18, 12]), 2.5)
    sim.add_noise_zone(np.array([12, 18]), 2.5, 5.0)
    
    # Add just 6 agents
    positions = [
        [5, 5], [10, 5], [5, 10],
        [20, 25], [25, 20], [25, 25]
    ]
    
    for i, pos in enumerate(positions):
        sim.add_agent(np.array(pos), sigma=3.0, epsilon=3.0, 
                     rssi_strength=8.0, temp=np.random.uniform(3.0, 6.0))

def run_quick_demo():
    """Run a quick demo with limited steps"""
    
    bounds = [0, 30]
    sample_time = 0.01
    n_steps = 200  # Very short simulation
    
    print("Quick Mesh Network Demo")
    print("=" * 30)
    
    # Create simulation
    sim = MeshNetworkSimulation(bounds, sample_time)
    create_demo_scenario(sim)
    
    print(f"Setup: {len(sim.agents)} agents, {len(sim.infrastructure_nodes)} nodes")
    print(f"Running {n_steps} simulation steps...")
    
    # Initialize visualizer
    visualizer = BackgroundMeshVisualizer(bounds)
    
    # Run simulation and save a few key frames
    save_frames = [0, 50, 100, 150, 199]  # Save 5 frames total
    
    print("\nSimulation progress:")
    for step in range(n_steps):
        sim.step()
        
        if step in save_frames:
            print(f"  Step {step}: Saving frame...")
            
            # Create and save agent frame
            from output_utils import get_output_path
            agent_fig = visualizer.create_agent_frame(sim, step, n_steps)
            agent_path = get_output_path('temp_frames', f"demo_agents_step_{step:03d}.png", 'comms_network/agent_steps')
            agent_fig.savefig(agent_path, dpi=120, bbox_inches='tight')
            plt.close(agent_fig)
            
            # Create and save coverage frame
            coverage_fig = visualizer.create_coverage_frame(sim, step, n_steps)
            coverage_path = get_output_path('temp_frames', f"demo_coverage_step_{step:03d}.png", 'comms_network/coverage_steps')
            coverage_fig.savefig(coverage_path, dpi=120, bbox_inches='tight')
            plt.close(coverage_fig)
        
        elif step % 25 == 0:
            print(f"  Step {step}")
    
    # Final metrics
    coverage_map = sim.compute_coverage_gaps()
    coverage_pct = np.mean(coverage_map > 0.5) * 100 if coverage_map is not None else 0
    deployed = sum(1 for agent in sim.agents if agent.is_deployed)
    
    print(f"\nDemo Complete!")
    print(f"Final Coverage: {coverage_pct:.1f}%")
    print(f"Deployed Agents: {deployed}/{len(sim.agents)}")
    
    print(f"\nGenerated frames:")
    for step in save_frames:
        print(f"  demo_agents_step_{step:03d}.png")
        print(f"  demo_coverage_step_{step:03d}.png")
    
    return sim

if __name__ == "__main__":
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Running Quick Demo of Background Simulation System")
    sim = run_quick_demo()
    
    print("\nDemo shows the split visualization system working!")
    print("- Agent simulation frames show network topology and connections")
    print("- Coverage frames show signal quality heatmap") 
    print("- No real-time plots during simulation")
    print("- All visualization happens in background")