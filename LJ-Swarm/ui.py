#######################################################################
# ui.py
#
# Visualization
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

############################
# Phase Change Color Update
############################
state_colors = np.array([
    [0, 0.2, 1.0],   # solid: blue
    [0.0, 0.8, 0.2], # liquid: green
    [1.0, 0.1, 0.1]  # gas: red
])

############################
# Classify States based on Kinetic Energy
############################

def classify_state(KE):
    if KE < 10:
        return 0  # solid
    elif KE < 50:
        return 1  # liquid
    else:
        return 2  # gas

############################
# Init Scatter Plot
############################

def setup_visualization(bounds, obstacles, gamma_pos):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    #Equal Aspect Ratio
    ax.set_aspect('equal', adjustable='box')
    scat = ax.scatter([], [], s=10, animated=True, edgecolors='none')
    title = ax.text(0.02, 1.02, '', transform=ax.transAxes)

    #Obstacles Input
    if obstacles:
        for pos, radius in obstacles:
            circle = plt.Circle(pos, radius, color='gray', alpha=0.4)
            ax.add_patch(circle)

    #Gamma_Pos Input
    if gamma_pos is not None:
        ax.plot(gamma_pos[0], gamma_pos[1], marker='*', color='purple', markersize=5, label='Goal')
        #ax.legend(loc="upper right", fontsize="small", frameon=False)
    
    return fig, ax, scat, title

############################
# Updates entire frame
############################

def create_update_function(sim, temperature_schedule, sample_time, sigma, epsilon, distance, c1_gamma, c2_gamma, gamma_pos, alpha,
                           scat, title, kinetic_temperatures, time_log):
    counter = [0]

    def compute_kinetic_temperature(agents, mass=1, kB=1):
        velocities = agents[:, 2:]
        KE_total = 0.5 * mass * np.sum(velocities**2)
        N = len(agents)
        return KE_total / (N * kB)

    def update(frame):
        temp = temperature_schedule[frame]
        forces = sim.compute_forces(sigma, epsilon, distance, temp, c1_gamma, c2_gamma, gamma_pos, alpha)
        sim.update(forces, temp)
        positions = sim.agents[:, :2]
        velocities = sim.agents[:, 2:]
        KE = 0.5 * np.sum(velocities**2, axis = 1)
        states = np.array([classify_state(k) for k in KE])
        colors = state_colors[states]

        T_kin = compute_kinetic_temperature(sim.agents)
        if T_kin >= 1e18:
            print("Simulation Destabilized")

        kinetic_temperatures.append(T_kin)
        time_log.append(frame * sample_time)

        scat.set_offsets(positions)
        scat.set_facecolor(colors)
        title.set_text(f"Frame {frame} | Temp = {temp:.1f}")
        counter[0] += 1
        return scat, title
    
    return update

