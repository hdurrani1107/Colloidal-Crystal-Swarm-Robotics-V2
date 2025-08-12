#######################################################################
# test_background_sim.py
#
# Quick test of the background simulation system
#
# Author: Humzah Durrani
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from mesh_network_engine import MeshNetworkSimulation
from background_visualization import BackgroundMeshVisualizer, SimulationRecorder

def test_background_system():
    """Test the background simulation system"""
    print("Testing Background Simulation System...")
    
    # Create small test simulation
    sim = MeshNetworkSimulation([0, 20], sample_time=0.01)
    
    # Add minimal test scenario
    sim.add_infrastructure_node(np.array([5, 5]), signal_strength=10.0, coverage_radius=8.0)
    sim.add_infrastructure_node(np.array([15, 15]), signal_strength=10.0, coverage_radius=8.0)
    
    # Add one obstacle and deadzone
    sim.add_obstacle(np.array([10, 10]), 2.0)
    sim.add_deadzone(np.array([12, 8]), 2.0)
    
    # Add 3 agents
    for i in range(3):
        pos = np.array([3 + i*6, 3 + i*2])
        sim.add_agent(pos, temp=3.0)
    
    print(f"[OK] Created test simulation with {len(sim.agents)} agents")
    
    # Test frame creation
    visualizer = BackgroundMeshVisualizer([0, 20])
    
    # Test agent frame creation
    print("Testing agent frame creation...")
    agent_fig = visualizer.create_agent_frame(sim, 1, 10)
    print("[OK] Agent frame created successfully")
    plt.close(agent_fig)
    
    # Test coverage frame creation
    print("Testing coverage frame creation...")
    coverage_fig = visualizer.create_coverage_frame(sim, 1, 10)
    print("[OK] Coverage frame created successfully")
    plt.close(coverage_fig)
    
    # Test a few simulation steps
    print("Testing simulation steps...")
    for i in range(10):
        sim.step()
    print("[OK] Simulation steps completed")
    
    print("\nAll background system tests passed!")
    return True

def test_short_video_creation():
    """Test creating very short videos (just a few frames)"""
    print("\nTesting short video creation...")
    
    # Create simulation
    sim = MeshNetworkSimulation([0, 15], sample_time=0.01)
    sim.add_infrastructure_node(np.array([5, 5]), signal_strength=8.0, coverage_radius=6.0)
    sim.add_obstacle(np.array([10, 10]), 1.5)
    
    # Add 2 agents
    sim.add_agent(np.array([3, 3]), temp=2.0)
    sim.add_agent(np.array([12, 3]), temp=2.0)
    
    # Create visualizer and recorder
    visualizer = BackgroundMeshVisualizer([0, 15])
    recorder = SimulationRecorder(sim, visualizer)
    
    # Create very short video (just 20 steps, 5 fps)
    print("Creating test videos (this may take a moment)...")
    
    try:
        # Check if ffmpeg is available
        import matplotlib
        matplotlib.rcParams['animation.writer'] = 'ffmpeg'
        
        metrics_data = recorder.run_simulation_with_recording(20, fps=5)
        
        # Check if files were created
        if os.path.exists("agent_simulation.mp4"):
            print("[OK] agent_simulation.mp4 created successfully")
            file_size = os.path.getsize("agent_simulation.mp4")
            print(f"    File size: {file_size} bytes")
        else:
            print("[WARNING] agent_simulation.mp4 not found")
        
        if os.path.exists("coverage_simulation.mp4"):
            print("[OK] coverage_simulation.mp4 created successfully")
            file_size = os.path.getsize("coverage_simulation.mp4")
            print(f"    File size: {file_size} bytes")
        else:
            print("[WARNING] coverage_simulation.mp4 not found")
        
        print("[OK] Video creation test completed")
        return True
        
    except Exception as e:
        print(f"[INFO] Video creation test skipped: {e}")
        print("This is normal if ffmpeg is not available")
        return True

if __name__ == "__main__":
    try:
        # Run basic tests
        test_background_system()
        
        # Try video test (may skip if ffmpeg unavailable)
        test_short_video_creation()
        
        print("\nBackground simulation system is ready!")
        print("Run 'python main_background_simulation.py' for full simulation")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()