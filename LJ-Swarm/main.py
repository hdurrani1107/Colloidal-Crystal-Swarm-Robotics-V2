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
from schedule import temp_schedule_1
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
gamma_pos = np.array([45/2, 45/2])
n_frames = 20000
max_temp = 0
counter = 0
temperature_schedule = temp_schedule_1(max_temp, n_frames)

##########################
# Init Sim
##########################

sim = multi_agent(agents, sigma, sample_time, bounds)

#Melted Initial Condition
#initial_temperature = 50
#init_melt(sim.agents, init_temp=initial_temperature)

##########################
# Visualization
##########################
fig, ax, scat, title = setup_visualization(bounds)
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
    title, kinetic_temperature, time_log
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