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
from ui import setup_visualization, create_update_function

##########################
# Simulation Params
##########################

agents = 50
sigma = 3.0
epsilon = 3.0 
sample_time = 0.005
bounds = [0, 45]
distance = (2 ** (1 / 6)) * sigma
alpha = 0.5
c1_gamma = 10 
c2_gamma = 10
gamma_pos = np.array([30, 30])

# Discovery-Based Goal Configuration
goal_config = {
    'detection_radius': 8.0,    # radius for discovery (stumbling upon goal)
    'trap_strength': 25.0,      # multiplier for gamma force when trapped
    'trap_radius': 5.0,         # radius for trap (smaller than detection)
    'max_capacity': 10          # max agents before broadcasting stops
}
n_frames = 30000
max_temp = 150
counter = 0
temperature_schedule = schedule.temp_schedule_1(max_temp, n_frames)
obstacles = [
    (np.array([22.5, 22.5]), 5),  # center sphere
    (np.array([10, 35]), 3),
    (np.array([35, 10]), 3)
]

##########################
# Init Sim
##########################

sim = multi_agent(agents, sigma, sample_time, bounds, obstacles)
sim.setup_goal(gamma_pos, goal_config)

#Melted Initial Condition
#initial_temperature = 50
#init_melt(sim.agents, init_temp=initial_temperature)

##########################
# Visualization
###########################
fig, ax, scat, title, goal_marker, trap_circle = setup_visualization(bounds, obstacles, gamma_pos)
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
    sim, temperature_schedule, sample_time, sigma, epsilon,
    distance, c1_gamma, c2_gamma, gamma_pos, alpha, scat, 
    title, kinetic_temperature, time_log, goal_marker, trap_circle
)

############################
# Progress Bar
############################

def frame_generator():
    for i in tqdm(range(n_frames), desc="Simulating"):
        yield i

##########################
# Run Sim
##########################
anim = FuncAnimation(
    fig, update, frames=frame_generator(), interval=50, blit=True, save_count=n_frames
)

##########################
# Save Logs
##########################
anim.save("lj_crystallization_simulation.mp4", fps=60, dpi=150)

plt.figure(figsize=(6, 3))
plt.plot(time_log, kinetic_temperature, label="Kinetic Energy")
plt.xlabel("Time")
plt.ylabel("Average Kinetic Energy")
plt.title("Average Kinetic Energy Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("kinetic_temperature_log.png")
plt.show()