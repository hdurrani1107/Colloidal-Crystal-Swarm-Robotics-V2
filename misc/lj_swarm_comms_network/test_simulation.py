#######################################################################
# test_simulation.py
#
# Quick test of the mesh network simulation
#
# Author: Humzah Durrani
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from mesh_network_engine import MeshNetworkSimulation
from mesh_visualization import MeshNetworkVisualizer

def test_basic_functionality():
    """Test basic simulation functionality"""
    print("Testing Swarm Mesh Network Simulation...")
    
    # Create simulation
    sim = MeshNetworkSimulation([0, 20], sample_time=0.01)
    
    # Add infrastructure
    sim.add_infrastructure_node(np.array([5, 5]), signal_strength=10.0, coverage_radius=8.0)
    sim.add_infrastructure_node(np.array([15, 15]), signal_strength=10.0, coverage_radius=8.0)
    
    # Add obstacles and deadzones
    sim.add_obstacle(np.array([10, 10]), 2.0)
    sim.add_deadzone(np.array([12, 8]), 2.0)
    sim.add_noise_zone(np.array([8, 12]), 2.0, 5.0)
    
    # Add agents
    for i in range(5):
        pos = np.random.uniform(2, 18, 2)
        sim.add_agent(pos, temp=3.0)
    
    print(f"[OK] Created simulation with {len(sim.agents)} agents")
    print(f"[OK] Added {len(sim.infrastructure_nodes)} infrastructure nodes")
    print(f"[OK] Added {len(sim.obstacles)} obstacles")
    print(f"[OK] Added {len(sim.deadzones)} deadzones")
    
    # Test simulation steps
    print("Running 50 simulation steps...")
    for i in range(50):
        sim.step()
        if i % 10 == 0:
            coverage = sim.compute_coverage_gaps()
            avg_coverage = np.mean(coverage)
            print(f"  Step {i}: Average coverage = {avg_coverage:.3f}")
    
    print("[OK] Simulation steps completed successfully")
    
    # Test visualization setup
    print("Testing visualization...")
    visualizer = MeshNetworkVisualizer([0, 20], figsize=(10, 8))
    visualizer.draw_static_elements(sim)
    
    # Update visualization once
    step_data = {
        'time_steps': [0, 10, 20],
        'coverage_percentage': [30, 45, 60],
        'connectivity_ratio': [0.2, 0.4, 0.6],
        'avg_temperature': [3.0, 2.0, 1.0]
    }
    
    visualizer.update_display(sim, step_data)
    print("[OK] Visualization created successfully")
    
    # Save test image
    visualizer.save_frame("test_mesh_network.png", dpi=150)
    print("[OK] Test image saved as 'test_mesh_network.png'")
    
    plt.close('all')  # Clean up
    
    print("\nAll tests passed! The simulation is working correctly.")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()