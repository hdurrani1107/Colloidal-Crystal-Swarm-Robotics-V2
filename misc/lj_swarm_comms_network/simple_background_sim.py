#######################################################################
# simple_background_sim.py
#
# Simplified background simulation that saves PNG sequence instead of MP4
# Works without ffmpeg dependency
#
# Author: Humzah Durrani
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from mesh_network_engine import MeshNetworkSimulation
from background_visualization import BackgroundMeshVisualizer, SimulationRecorder

def create_simple_scenario(sim: MeshNetworkSimulation):
    """Create a simple test scenario"""
    
    # Add infrastructure nodes
    sim.add_infrastructure_node(np.array([10, 10]), signal_strength=12.0, coverage_radius=10.0)
    sim.add_infrastructure_node(np.array([30, 30]), signal_strength=12.0, coverage_radius=10.0)
    
    # Add obstacles and deadzones
    sim.add_obstacle(np.array([20, 20]), 4.0)
    sim.add_deadzone(np.array([25, 15]), 3.0)
    sim.add_noise_zone(np.array([15, 25]), 3.0, 8.0)
    
    # Add agents
    np.random.seed(42)
    n_agents = 8
    
    for i in range(n_agents):
        # Random positions avoiding obstacles
        while True:
            pos = np.random.uniform(sim.bounds[0] + 2, sim.bounds[1] - 2, 2)
            valid = True
            for obs_pos, obs_radius in sim.obstacles:
                if np.linalg.norm(pos - obs_pos) < obs_radius + 2.0:
                    valid = False
                    break
            if valid:
                break
        
        sim.add_agent(pos, sigma=3.0, epsilon=3.0, rssi_strength=10.0, temp=np.random.uniform(2.0, 6.0))

def run_simple_simulation():
    """Run simulation with PNG frame saving"""
    
    bounds = [0, 40]
    sample_time = 0.01
    n_steps = 1000
    save_interval = 20  # Save every 20th frame
    
    print("Simple Background Simulation (PNG Sequence)")
    print("=" * 50)
    
    # Create simulation
    sim = MeshNetworkSimulation(bounds, sample_time)
    create_simple_scenario(sim)
    
    print(f"Setup: {len(sim.agents)} agents, {len(sim.infrastructure_nodes)} nodes")
    print(f"Steps: {n_steps}, saving every {save_interval} frames")
    
    # Create output directories
    os.makedirs("agent_frames", exist_ok=True)
    os.makedirs("coverage_frames", exist_ok=True)
    
    # Initialize visualizer
    visualizer = BackgroundMeshVisualizer(bounds)
    
    # Metrics collection
    metrics = {
        'coverage': [],
        'connectivity': [],
        'deployed': [],
        'steps': []
    }
    
    print("\nRunning simulation...")
    frame_count = 0
    
    for step in range(n_steps):
        # Run simulation step
        sim.step()
        
        # Save frames and collect metrics
        if step % save_interval == 0:
            # Create and save agent frame
            agent_fig = visualizer.create_agent_frame(sim, step, n_steps)
            agent_fig.savefig(f"agent_frames/frame_{frame_count:04d}.png", dpi=150, bbox_inches='tight')
            plt.close(agent_fig)
            
            # Create and save coverage frame
            coverage_fig = visualizer.create_coverage_frame(sim, step, n_steps)
            coverage_fig.savefig(f"coverage_frames/frame_{frame_count:04d}.png", dpi=150, bbox_inches='tight')
            plt.close(coverage_fig)
            
            # Collect metrics
            coverage_map = sim.compute_coverage_gaps()
            coverage_pct = np.mean(coverage_map > 0.5) * 100 if coverage_map is not None else 0
            
            total_connections = sum(len(agent.connected_nodes) + len(agent.connected_agents) for agent in sim.agents)
            max_connections = len(sim.agents) * (len(sim.infrastructure_nodes) + len(sim.agents) - 1)
            connectivity = total_connections / max(1, max_connections)
            
            deployed = sum(1 for agent in sim.agents if agent.is_deployed)
            
            metrics['coverage'].append(coverage_pct)
            metrics['connectivity'].append(connectivity)
            metrics['deployed'].append(deployed)
            metrics['steps'].append(step)
            
            frame_count += 1
            
            if step % 100 == 0:
                print(f"Step {step}: Coverage={coverage_pct:.1f}%, Deployed={deployed}/{len(sim.agents)}")
    
    print(f"\nSimulation complete! Saved {frame_count} frames")
    print(f"Agent frames: agent_frames/")
    print(f"Coverage frames: coverage_frames/")
    
    # Create final analysis
    create_final_analysis(sim, metrics)
    
    return sim, metrics

def create_final_analysis(sim, metrics):
    """Create final analysis plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Simple Mesh Network Simulation - Results', fontsize=14)
    
    steps = metrics['steps']
    
    # Coverage
    axes[0, 0].plot(steps, metrics['coverage'], 'g-', linewidth=2)
    axes[0, 0].set_title('Coverage Over Time')
    axes[0, 0].set_ylabel('Coverage (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Connectivity
    axes[0, 1].plot(steps, metrics['connectivity'], 'b-', linewidth=2)
    axes[0, 1].set_title('Connectivity Over Time')
    axes[0, 1].set_ylabel('Connectivity Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Deployment
    axes[1, 0].plot(steps, metrics['deployed'], 'r-', linewidth=2)
    axes[1, 0].axhline(y=len(sim.agents), color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Agent Deployment')
    axes[1, 0].set_ylabel('Deployed Agents')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary
    axes[1, 1].axis('off')
    
    if metrics['coverage']:
        final_coverage = metrics['coverage'][-1]
        final_connectivity = metrics['connectivity'][-1]
        final_deployed = metrics['deployed'][-1]
        
        summary_text = f"""Final Results:
        
Coverage: {final_coverage:.1f}%
Connectivity: {final_connectivity:.3f}
Deployed: {final_deployed}/{len(sim.agents)}
Deployment Rate: {(final_deployed/len(sim.agents)*100):.1f}%

Total Agents: {len(sim.agents)}
Infrastructure: {len(sim.infrastructure_nodes)}
Obstacles: {len(sim.obstacles)}
Deadzones: {len(sim.deadzones)}"""
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                       verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig("simple_simulation_results.png", dpi=200, bbox_inches='tight')
    print("Analysis saved as 'simple_simulation_results.png'")
    plt.show()

if __name__ == "__main__":
    print("Running Simple Background Simulation...")
    sim, metrics = run_simple_simulation()
    
    print("\nGenerated files:")
    print("- agent_frames/frame_XXXX.png - Agent simulation frames")
    print("- coverage_frames/frame_XXXX.png - Coverage heatmap frames") 
    print("- simple_simulation_results.png - Final analysis")
    print("\nNote: This creates PNG sequences instead of MP4 videos")
    print("You can create videos from PNG sequences using external tools if needed")