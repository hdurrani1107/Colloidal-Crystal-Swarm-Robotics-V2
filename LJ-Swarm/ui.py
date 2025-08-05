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

    # Goal visualization elements (initially hidden)
    goal_marker = None
    trap_circle = None
    if gamma_pos is not None:
        # Goal marker (initially hidden)
        goal_marker = ax.plot([], [], marker='*', color='purple', markersize=8, label='Goal')[0]
        # Trap radius circle (initially hidden, dotted line)
        trap_circle = plt.Circle(gamma_pos, 0, fill=False, color='purple', 
                               linestyle='--', alpha=0.6, linewidth=2)
        ax.add_patch(trap_circle)
    
    return fig, ax, scat, title, goal_marker, trap_circle

############################
# Updates entire frame
############################

def create_update_function(sim, temperature_schedule, sample_time, sigma, epsilon, distance, c1_gamma, c2_gamma, gamma_pos, alpha,
                           scat, title, kinetic_temperatures, time_log, goal_marker=None, trap_circle=None):
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

        # Update goal visibility based on discovery state
        if sim.goal and goal_marker and trap_circle:
            # Reset per-frame data
            sim.goal.reset_frame_data()
            
            # Process all agents for discovery/trapping
            for i, pos in enumerate(positions):
                sim.goal.process_agent_discovery(i, pos, velocities[i], c1_gamma, c2_gamma)
            
            # Show/hide goal based on discovery state
            if sim.goal.is_goal_visible():
                goal_marker.set_data([gamma_pos[0]], [gamma_pos[1]])
                trap_circle.set_radius(sim.goal.trap_radius)
                trap_circle.set_visible(True)
                
                # Change colors based on broadcasting state
                if sim.goal.is_broadcasting:
                    goal_marker.set_color('red')  # Broadcasting = red
                    trap_circle.set_color('red')
                elif sim.goal.is_full:
                    goal_marker.set_color('orange')  # Full = orange
                    trap_circle.set_color('orange')
                else:
                    goal_marker.set_color('purple')  # Default = purple
                    trap_circle.set_color('purple')
            else:
                goal_marker.set_data([], [])
                trap_circle.set_visible(False)

        scat.set_offsets(positions)
        scat.set_facecolor(colors)
        
        # Add goal status to title if goal exists
        if sim.goal:
            status = sim.goal.get_status_string()
            title.set_text(f"Frame {frame} | Temp = {temp:.1f} | Goal: {status}")
        else:
            title.set_text(f"Frame {frame} | Temp = {temp:.1f}")
            
        counter[0] += 1
        
        # Return all animated elements
        return_elements = [scat, title]
        if goal_marker:
            return_elements.append(goal_marker)
        if trap_circle:
            return_elements.append(trap_circle)
        
        return return_elements
    
    return update

