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
import schedule
from ui import setup_visualization, create_update_function
from metrics import MetricsLogger

##########################
# Simulation Params
##########################

agents = 150
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

n_frames = 30000
render_interval = 10  # Render every 10th frame for performance
render_frames = n_frames // render_interval  # Number of frames to actually render
max_temp = 150
counter = 0 
temperature_schedule = schedule.temp_schedule_5(max_temp, n_frames)
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
metrics = MetricsLogger(out_dir=f"{output_dir}/metrics/OS_metric")

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
kinetic_temperature = []
time_log = []

##########################
# Update Sim
##########################
update = create_update_function(
    sim, temperature_schedule, sample_time, sigma=3.0, epsilon=3.0,
    distance=10.0, c1_gamma=5.0, c2_gamma=2.0, alpha=0.5, scat=scat, 
    quiver=quiver, title=title, kinetic_temperatures=kinetic_temperature, 
    time_log=time_log, beacon_circles=beacon_circles, ax=ax,
    metrics=metrics, render_interval=render_interval
)

############################
# Progress Bar
############################

def frame_generator():
    for i in tqdm(range(render_frames), desc="Simulating Olfati-Saber Flocking"):
        yield i

##########################
# Run Sim
##########################
anim = FuncAnimation(
    fig, update, frames=frame_generator(), interval=50, blit=True, save_count=render_frames
)

##########################
# Save Logs 
##########################

anim.save(f"{output_dir}/videos/olfati_saber_flocking_simulation.mp4", fps=60, dpi=150)

plt.figure(figsize=(6, 3))
plt.plot(time_log, kinetic_temperature, label="Kinetic Energy", color='green')
plt.xlabel("Time")
plt.ylabel("Average Kinetic Energy") 
plt.title("Average Kinetic Energy Over Time - Olfati-Saber Flocking")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/graphs/olfati_saber_kinetic_temperature_analysis.png")
plt.show()

metrics.save()
print(f"Saved metrics to {output_dir}/metrics/OS_metric")
print("Olfati-Saber flocking simulation completed!")