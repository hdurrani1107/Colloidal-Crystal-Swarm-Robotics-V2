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

def setup_visualization(bounds, obstacles):
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

    # Cooling zone visualization elements
    cooling_zone_circles = []

    return fig, ax, scat, title, cooling_zone_circles

############################
# Updates entire frame
############################

def create_update_function(sim, temperature_schedule, sample_time, sigma, epsilon, distance, c1_gamma, c2_gamma, alpha,
                           scat, title, kinetic_temperatures, time_log, cooling_zone_circles, ax):
    counter = [0]

    def compute_kinetic_temperature(agents, mass=1, kB=1):
        velocities = agents[:, 2:]
        KE_total = 0.5 * mass * np.sum(velocities**2)
        N = len(agents)
        return KE_total / (N * kB)

    def update(frame):
        temp = temperature_schedule[frame]
        forces = sim.compute_forces(sigma, epsilon, distance, temp, c1_gamma, c2_gamma, alpha)
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

        # Update cooling zone visualization
        if sim.cooling_zones:
            # Remove old cooling zone circles
            for circle in cooling_zone_circles:
                circle.remove()
            cooling_zone_circles.clear()
            
            # Add current cooling zones
            active_zones = sim.cooling_zones.get_active_zones()
            if frame < 10:  # Debug print for first few frames
                print(f"Frame {frame}: {len(active_zones)} active zones")
            for zone_pos, zone_radius, num_agents in active_zones:
                # Color based on number of trapped agents
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

        scat.set_offsets(positions)
        scat.set_facecolor(colors)
        
        # Add cooling zone status to title if cooling zones exist
        if sim.cooling_zones:
            num_zones = len(sim.cooling_zones.zones)
            total_trapped = sum(len(zone.trapped_agents) for zone in sim.cooling_zones.zones)
            title.set_text(f"Frame {frame} | Temp = {temp:.1f} | Zones: {num_zones} | Trapped: {total_trapped}")
        else:
            title.set_text(f"Frame {frame} | Temp = {temp:.1f}")
            
        counter[0] += 1
        
        # Return all animated elements
        return_elements = [scat, title] + cooling_zone_circles
        
        return return_elements
    
    return update

