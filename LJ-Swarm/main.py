#######################################################################
# main.py
#
# Run simulation from here
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
from engine import multi_agent, init_melt
import schedule
from ui import setup_visualization, create_update_function, classify_state, state_colors
# from metrics import MetricsLogger  # Disabled metrics

##########################
# Simulation Params
##########################

agents = 150  
sigma = 3.0
epsilon = 3.0 
sample_time = 0.005
bounds = [0, 200]
distance = (2 ** (1 / 6)) * sigma
alpha = 0.5
c1_gamma = 10 
c2_gamma = 10
# Cooling Zone Configuration
cooling_zone_config = {
    'zone_radius': 10.0,        # mean radius of cooling zones
    'radius_std': 1.0,          # standard deviation for radius sampling
    'max_concurrent_zones': 3,  # maximum number of active zones at once
    'spawn_interval': 400,      # frames between zone spawns when no zones exist
    'base_lifetime': 1000,       # base lifetime of zones in frames
    'zone_temperature': 0.0,    # temperature of cooling zones
    'logger': None              # will be set below
}
# Continuous simulation until target cooling zones completed
target_cooling_zones = 10  # User-configurable number of zones to complete
render_interval = 10  # Render every 10th frame for performance
max_temp = 150
counter = 0
frame_count = 0
constant_temperature = 150  # Constant temperature throughout simulation
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
# os.makedirs(f"{output_dir}/metrics/LJ_metric", exist_ok=True)
# metrics = MetricsLogger(out_dir=f"{output_dir}/metrics/LJ_metric")  # Disabled metrics
metrics = None

##########################
# Init Sim
##########################

sim = multi_agent(agents, sigma, sample_time, bounds, obstacles)
cooling_zone_config['logger'] = None  # Disabled metrics
sim.setup_cooling_zones(cooling_zone_config)

#Melted Initial Condition
initial_temperature = 150
init_melt(sim.agents, init_temp=initial_temperature)
sim.initialize_agent_temperatures(initial_temperature)

##########################
# Visualization
###########################
fig, ax, scat, title, cooling_zone_circles = setup_visualization(bounds, obstacles)
norm = Normalize(vmin=0, vmax=50)
scat.set_norm(norm)

##########################
# Logging
##########################
zone_completion_times = []  # Track when each cooling zone was completed
zone_completion_frames = []  # Track frame numbers when zones completed
completed_zones = 0
rendered_frames = []  # Track which frames we rendered
frame_data = []  # Store frame data for video creation

##########################
# Simulation Loop Functions
##########################

def capture_frame_data(frame):
    """Capture frame data for video creation"""
    positions = sim.agents[:, :2].copy()
    velocities = sim.agents[:, 2:].copy()
    KE = 0.5 * np.sum(velocities**2, axis=1)
    states = np.array([classify_state(k) for k in KE])
    colors = state_colors[states].copy()
    
    # Get cooling zone data
    active_zones = sim.cooling_zones.get_active_zones() if sim.cooling_zones else []
    
    # Store frame data
    frame_info = {
        'frame': frame,
        'positions': positions,
        'colors': colors,
        'active_zones': active_zones,
        'completed_zones': completed_zones,
        'num_zones': len(sim.cooling_zones.zones) if sim.cooling_zones else 0,
        'total_trapped': sum(len(zone.trapped_agents) for zone in sim.cooling_zones.zones) if sim.cooling_zones else 0
    }
    frame_data.append(frame_info)
    return frame_info

def update_animation(animation_frame):
    """Animation function for FuncAnimation"""
    if animation_frame >= len(frame_data):
        return [scat, title] + cooling_zone_circles
        
    frame_info = frame_data[animation_frame]
    
    # Clear previous cooling zone circles
    for circle in cooling_zone_circles:
        circle.remove()
    cooling_zone_circles.clear()
    
    # Update particle positions and colors
    scat.set_offsets(frame_info['positions'])
    scat.set_facecolor(frame_info['colors'])
    
    # Add cooling zones
    for zone_pos, zone_radius, num_agents in frame_info['active_zones']:
        if num_agents == 0:
            color = 'cyan'
            zone_alpha = 0.3
        elif num_agents <= 3:
            color = 'blue'
            zone_alpha = 0.5
        else:
            color = 'darkblue'
            zone_alpha = 0.7
        
        circle = plt.Circle(zone_pos, zone_radius, fill=False, color=color, 
                          linestyle='-', alpha=zone_alpha, linewidth=3)
        ax.add_patch(circle)
        cooling_zone_circles.append(circle)
    
    # Update title
    time_value = frame_info['frame'] * sample_time
    title.set_text(f"Time {time_value:.3f}s | Temp = {constant_temperature:.1f} | Zones: {frame_info['num_zones']} | Trapped: {frame_info['total_trapped']} | Completed: {frame_info['completed_zones']}/{target_cooling_zones}")
    
    return [scat, title] + cooling_zone_circles

##########################
# Run Continuous Simulation
##########################

pbar = tqdm(desc=f"Completing cooling zones (0/{target_cooling_zones})", unit="zones")

try:
    while completed_zones < target_cooling_zones:
        # Track zones before update
        zones_before = len(sim.cooling_zones.zones) if sim.cooling_zones else 0
        
        # Run one simulation step
        forces = sim.compute_forces(sigma, epsilon, distance, constant_temperature, c1_gamma, c2_gamma, alpha)
        sim.update(forces, constant_temperature)
        
        # Check for completed zones
        zones_after = len(sim.cooling_zones.zones) if sim.cooling_zones else 0
        if zones_before > zones_after:
            zones_completed_this_step = zones_before - zones_after
            completed_zones += zones_completed_this_step
            zone_completion_times.append(frame_count * sample_time)
            zone_completion_frames.append(frame_count)
            pbar.set_description(f"Completing cooling zones ({completed_zones}/{target_cooling_zones})")
            pbar.update(zones_completed_this_step)
        
        # Capture frame data every 10th frame
        if frame_count % render_interval == 0:
            capture_frame_data(frame_count)
            rendered_frames.append(frame_count)
        
        frame_count += 1
        
        # Safety check to prevent infinite loop
        if frame_count > 1000000:  # 1 million frame limit
            print(f"Simulation reached maximum frame limit. Completed {completed_zones}/{target_cooling_zones} zones.")
            break

except KeyboardInterrupt:
    print(f"\nSimulation interrupted by user at frame {frame_count}")
    print(f"Completed {completed_zones}/{target_cooling_zones} cooling zones")
    print(f"Captured {len(frame_data)} frames for video")

pbar.close()
print(f"Simulation finished! {completed_zones} cooling zones were completed.")

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
    anim.save(f"{output_dir}/videos/lj_crystallization_simulation.mp4", fps=60, dpi=150)
    print(f"Video saved to {output_dir}/videos/lj_crystallization_simulation.mp4")
else:
    print("No frames captured - cannot create video")

# Also save final frame as image
if frame_data:
    final_frame = frame_data[-1]
    scat.set_offsets(final_frame['positions'])
    scat.set_facecolor(final_frame['colors'])
    
    # Clear and add final cooling zones
    for circle in cooling_zone_circles:
        circle.remove()
    cooling_zone_circles.clear()
    
    for zone_pos, zone_radius, num_agents in final_frame['active_zones']:
        if num_agents == 0:
            color = 'cyan'
            zone_alpha = 0.3
        elif num_agents <= 3:
            color = 'blue'
            zone_alpha = 0.5
        else:
            color = 'darkblue'
            zone_alpha = 0.7
        
        circle = plt.Circle(zone_pos, zone_radius, fill=False, color=color, 
                          linestyle='-', alpha=zone_alpha, linewidth=3)
        ax.add_patch(circle)
        cooling_zone_circles.append(circle)
    
    final_time = final_frame['frame'] * sample_time
    title.set_text(f"Final State - Time {final_time:.3f}s | Completed: {completed_zones}/{target_cooling_zones} zones")
    plt.savefig(f"{output_dir}/videos/final_simulation_state.png", dpi=150)
    print(f"Final state image saved to {output_dir}/videos/final_simulation_state.png")

# Create cooling zone completion graph
plt.figure(figsize=(10, 6))
if zone_completion_times:
    plt.scatter(zone_completion_times, range(1, len(zone_completion_times) + 1), 
                marker='o', s=50, alpha=0.7, label="Zone Completions")
    plt.plot(zone_completion_times, range(1, len(zone_completion_times) + 1), 
             linestyle='--', alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Cooling Zones Completed")
    plt.title(f"Cooling Zone Completion Timeline (Total: {completed_zones})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add completion rate info
    if len(zone_completion_times) > 1:
        total_time = zone_completion_times[-1] - zone_completion_times[0]
        avg_rate = len(zone_completion_times) / total_time if total_time > 0 else 0
        plt.text(0.02, 0.98, f"Avg completion rate: {avg_rate:.3f} zones/s", 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
else:
    plt.text(0.5, 0.5, "No cooling zones completed", 
             transform=plt.gca().transAxes, ha='center', va='center', fontsize=16)
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Cooling Zones Completed")
    plt.title("Cooling Zone Completion Timeline")

plt.tight_layout()
plt.savefig(f"{output_dir}/graphs/cooling_zone_completion_analysis.png")
plt.show()

# metrics.save()  # Disabled metrics
# print(f"Saved metrics to {output_dir}/metrics/LJ_metric")