#######################################################################
# test_mp4.py
#
# Test MP4 creation with the configured ffmpeg path
#
# Author: Humzah Durrani  
#######################################################################

import numpy as np
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\hummy\anaconda3\envs\my-env\Library\bin\ffmpeg.exe'
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os
from mesh_network_engine import MeshNetworkSimulation
from background_visualization import BackgroundMeshVisualizer

def test_mp4_creation():
    """Test creating a very short MP4"""
    
    print("Testing MP4 creation with configured ffmpeg...")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create minimal simulation
    sim = MeshNetworkSimulation([0, 20], sample_time=0.01)
    sim.add_infrastructure_node(np.array([10, 10]), signal_strength=8.0, coverage_radius=6.0)
    sim.add_agent(np.array([5, 5]), temp=2.0)
    sim.add_agent(np.array([15, 15]), temp=2.0)
    
    # Create visualizer
    visualizer = BackgroundMeshVisualizer([0, 20])
    
    # Test very short video (just 10 steps, 5 fps)
    print("Creating test MP4 (10 frames)...")
    
    try:
        writer = FFMpegWriter(fps=5, metadata=dict(artist='TestMP4'))
        
        fig, ax = visualizer.setup_agent_figure()
        visualizer.draw_static_elements_agent(ax, sim)
        
        with writer.saving(fig, "test_short.mp4", 100):
            for step in range(10):
                sim.step()
                
                # Update the plot
                ax.clear()
                ax.set_xlim([0, 20])
                ax.set_ylim([0, 20])
                ax.set_aspect('equal')
                ax.set_title(f'Test MP4 - Step {step}')
                
                visualizer.draw_static_elements_agent(ax, sim)
                
                # Draw agents
                if sim.agents:
                    positions = np.array([agent.position for agent in sim.agents])
                    ax.scatter(positions[:, 0], positions[:, 1], 
                              c='blue', s=100, alpha=0.8, edgecolors='black')
                
                writer.grab_frame()
                print(f"  Frame {step}")
        
        plt.close(fig)
        
        if os.path.exists("test_short.mp4"):
            file_size = os.path.getsize("test_short.mp4")
            print(f"SUCCESS: test_short.mp4 created ({file_size} bytes)")
            return True
        else:
            print("ERROR: MP4 file was not created")
            return False
            
    except Exception as e:
        print(f"ERROR creating MP4: {e}")
        return False

if __name__ == "__main__":
    if test_mp4_creation():
        print("\nMP4 creation works! You can now use main_background_simulation.py")
    else:
        print("\nMP4 creation failed. Use simple_background_sim.py instead for PNG sequences")