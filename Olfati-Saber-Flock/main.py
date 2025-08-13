#######################################################################
# main.py
#
# Olfati-Saber flocking simulation with goal beacons
# Adapted from LJ-Swarm main.py
#
# Author: Humzah Durrani
#######################################################################

##########################
# Importing Libraries
##########################
 
import numpy as np
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\hummy\anaconda3\envs\my-env\Library\bin\ffmpeg.exe'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from tqdm import tqdm
from engine import multi_agent
# import schedule  # Not needed for Olfati-Saber algorithm
from ui import setup_visualization, create_update_function
from metrics import MetricsLogger

##########################
# Simulation Params
##########################

agents = 153
sample_time = 0.005  # Smaller timestep for flocking stability
bounds = [0, 200]

# Goal Beacon Configuration (equivalent to cooling zones)
goal_beacon_config = {
    'beacon_radius': 10.0,        # radius of goal beacons
    'spawn_interval': 400,        # frames between beacon spawns when no beacons exist
    'base_lifetime': 1000,        # base lifetime of beacons in frames
    'velocity_damping': 0.95,     # velocity damping factor inside beacon
    'logger': None                # will be set below
}

# Continuous simulation until target beacons completed
target_beacons = 10  # User-configurable number of beacons to complete
render_interval = 10  # Render every 10th frame for performance
frame_count = 0
counter = 0 
dummy_temp = 0  # Dummy temperature value - not used by Olfati-Saber algorithm
obstacles = [
#    (np.array([75, 75]), 10),  # center sphere
#    (np.array([30, 100]), 5),
#    (np.array([100, 30]), 5)
]

##########################
# Setup Output Directory and Metrics
##########################
import os
output_dir = "../output"
os.makedirs(f"{output_dir}/videos", exist_ok=True)
os.makedirs(f"{output_dir}/graphs", exist_ok=True)
os.makedirs(f"{output_dir}/metrics/OS_metric", exist_ok=True)
#metrics = MetricsLogger(out_dir=f"{output_dir}/metrics/OS_metric")
metrics = None

##########################
# Init Sim
##########################

sim = multi_agent(agents, sample_time, bounds, obstacles)
goal_beacon_config['logger'] = metrics
sim.setup_goal_beacons(goal_beacon_config)

##########################
# Visualization
###########################
fig, ax, scat, quiver, title, beacon_circles = setup_visualization(bounds, obstacles)
norm = Normalize(vmin=0, vmax=50)
scat.set_norm(norm)

##########################
# Logging
##########################
beacon_completion_times = []  # Track when each beacon was completed
beacon_completion_frames = []  # Track frame numbers when beacons completed
completed_beacons = 0
frame_data = []  # Store frame data for video creation

##########################
# Simulation Loop Functions
##########################

def capture_frame_data(frame):
    """Capture frame data for video creation"""
    positions = sim.agents[:, :2].copy()
    colors = get_agent_colors(sim).copy()
    
    # Get goal beacon data
    active_beacons = sim.goal_beacons.get_active_beacons() if sim.goal_beacons else []
    
    # Store frame data
    frame_info = {
        'frame': frame,
        'positions': positions,
        'colors': colors,
        'active_beacons': active_beacons,
        'completed_beacons': completed_beacons
    }
    frame_data.append(frame_info)
    return frame_info

def update_animation(animation_frame):
    """Animation function for FuncAnimation"""
    if animation_frame >= len(frame_data):
        return [scat, title] + beacon_circles
        
    frame_info = frame_data[animation_frame]
    
    # Clear previous beacon circles
    for circle in beacon_circles:
        circle.remove()
    beacon_circles.clear()
    
    # Update particle positions and colors
    scat.set_offsets(frame_info['positions'])
    scat.set_facecolor(frame_info['colors'])
    
    # Add goal beacons
    for beacon_pos, beacon_radius, num_agents in frame_info['active_beacons']:
        if num_agents == 0:
            color = 'cyan'
            beacon_alpha = 0.3
        elif num_agents <= 3:
            color = 'blue'
            beacon_alpha = 0.5
        else:
            color = 'darkblue'
            beacon_alpha = 0.7
        
        circle = plt.Circle(beacon_pos, beacon_radius, fill=False, color=color, 
                          linestyle='-', alpha=beacon_alpha, linewidth=3)
        ax.add_patch(circle)
        beacon_circles.append(circle)
        
        # Add goal point marker at center
        goal_point = plt.Circle(beacon_pos, 2, fill=True, color=color, alpha=0.8)
        ax.add_patch(goal_point)
        beacon_circles.append(goal_point)
    
    # Leaders now target beacons directly - no separate goal markers needed
    
    # Update title
    time_value = frame_info['frame'] * sample_time
    title.set_text(f"Time {time_value:.3f}s | 3 Flocks: Red(1), Green(2), Blue(3) | Leaders: Yellow | Completed: {frame_info['completed_beacons']}/{target_beacons}")
    
    return [scat, title] + beacon_circles

##########################
# Run Continuous Simulation
##########################

from ui import get_agent_colors

pbar = tqdm(desc=f"Completing beacons (0/{target_beacons})", unit="beacons")

try:
    while completed_beacons < target_beacons:
        # Track beacons before update
        beacons_before = len(sim.goal_beacons.beacons) if sim.goal_beacons else 0
        
        # Run one simulation step
        external_forces = np.zeros((len(sim.agents), 2))
        sim.update(external_forces, dummy_temp, frame_count)
        
        # Check for completed beacons
        beacons_after = len(sim.goal_beacons.beacons) if sim.goal_beacons else 0
        if beacons_before > beacons_after:
            beacons_completed_this_step = beacons_before - beacons_after
            completed_beacons += beacons_completed_this_step
            beacon_completion_times.append(frame_count * sample_time)
            beacon_completion_frames.append(frame_count)
            pbar.set_description(f"Completing beacons ({completed_beacons}/{target_beacons})")
            pbar.update(beacons_completed_this_step)
        
        # Capture frame data every 10th frame
        if frame_count % render_interval == 0:
            capture_frame_data(frame_count)
        
        frame_count += 1
        
        # Safety check to prevent infinite loop
        if frame_count > 1000000:  # 1 million frame limit
            print(f"Simulation reached maximum frame limit. Completed {completed_beacons}/{target_beacons} beacons.")
            break

except KeyboardInterrupt:
    print(f"\nSimulation interrupted by user at frame {frame_count}")
    print(f"Completed {completed_beacons}/{target_beacons} beacons")
    print(f"Captured {len(frame_data)} frames for video")

pbar.close()
print(f"Simulation completed! {completed_beacons} beacons were completed.")

##########################
# Create and Save Video
##########################

if len(frame_data) > 0:
    print(f"Creating video from {len(frame_data)} captured frames...")
    
    # Create animation from captured frame data
    anim = FuncAnimation(
        fig, update_animation, frames=len(frame_data), 
        interval=50, blit=True, repeat=False
    )
    
    # Save animation as MP4
    anim.save(f"{output_dir}/videos/olfati_saber_flocking_simulation.mp4", fps=60, dpi=150)
    print(f"Video saved to {output_dir}/videos/olfati_saber_flocking_simulation.mp4")
else:
    print("No frames captured - cannot create video")

# Also save final frame as image
if frame_data:
    final_frame = frame_data[-1]
    scat.set_offsets(final_frame['positions'])
    scat.set_facecolor(final_frame['colors'])
    
    # Clear and add final beacons
    for circle in beacon_circles:
        circle.remove()
    beacon_circles.clear()
    
    for beacon_pos, beacon_radius, num_agents in final_frame['active_beacons']:
        if num_agents == 0:
            color = 'cyan'
            beacon_alpha = 0.3
        elif num_agents <= 3:
            color = 'blue'
            beacon_alpha = 0.5
        else:
            color = 'darkblue'
            beacon_alpha = 0.7
        
        circle = plt.Circle(beacon_pos, beacon_radius, fill=False, color=color, 
                          linestyle='-', alpha=beacon_alpha, linewidth=3)
        ax.add_patch(circle)
        beacon_circles.append(circle)
        
        goal_point = plt.Circle(beacon_pos, 2, fill=True, color=color, alpha=0.8)
        ax.add_patch(goal_point)
        beacon_circles.append(goal_point)
    
    # Leaders now target beacons directly - no separate goal markers needed
    
    final_time = final_frame['frame'] * sample_time
    title.set_text(f"Final State - Time {final_time:.3f}s | 3 Flocks with Leaders | Completed: {completed_beacons}/{target_beacons}")
    plt.savefig(f"{output_dir}/videos/final_olfati_saber_state.png", dpi=150)
    print(f"Final state image saved to {output_dir}/videos/final_olfati_saber_state.png")

# Create beacon completion graph
plt.figure(figsize=(10, 6))
if beacon_completion_times:
    plt.scatter(beacon_completion_times, range(1, len(beacon_completion_times) + 1), 
                marker='o', s=50, alpha=0.7, label="Beacon Completions", color='blue')
    plt.plot(beacon_completion_times, range(1, len(beacon_completion_times) + 1), 
             linestyle='--', alpha=0.5, color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Beacons Completed")
    plt.title(f"Beacon Completion Timeline - Olfati-Saber Flocking (Total: {completed_beacons})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add completion rate info
    if len(beacon_completion_times) > 1:
        total_time = beacon_completion_times[-1] - beacon_completion_times[0]
        avg_rate = len(beacon_completion_times) / total_time if total_time > 0 else 0
        avg_time_per_beacon = total_time / len(beacon_completion_times) if len(beacon_completion_times) > 0 else 0
        plt.text(0.02, 0.98, f"Avg completion rate: {avg_rate:.3f} beacons/s\nAvg time per beacon: {avg_time_per_beacon:.3f}s", 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
else:
    plt.text(0.5, 0.5, "No beacons completed", 
             transform=plt.gca().transAxes, ha='center', va='center', fontsize=16)
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Beacons Completed")
    plt.title("Beacon Completion Timeline - Olfati-Saber Flocking")

plt.tight_layout()
plt.savefig(f"{output_dir}/graphs/olfati_saber_beacon_completion_analysis.png")
plt.show()

#metrics.save()
#print(f"Saved metrics to {output_dir}/metrics/OS_metric")
#print("Olfati-Saber flocking simulation completed!")