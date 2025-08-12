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

def setup_visualization(bounds, obstacles, infrastructure_manager):
    fig, ax = plt.subplots(figsize=(8, 8))
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

    # Infrastructure visualization elements
    infrastructure_elements = {'nodes': [], 'comm_circles': [], 'cryst_circles': []}
    
    if infrastructure_manager and infrastructure_manager.nodes:
        for i, node in enumerate(infrastructure_manager.nodes):
            # Infrastructure node marker
            node_marker = ax.plot(node.position[0], node.position[1], 
                                marker='*', color='green', markersize=15, 
                                label='Infrastructure Node' if i == 0 else '')[0]
            infrastructure_elements['nodes'].append(node_marker)
            
            # Communication radius circle (solid line)
            comm_circle = plt.Circle(node.position, node.comm_radius, 
                                   fill=False, color='green', alpha=0.3, 
                                   linewidth=2, linestyle='-')
            ax.add_patch(comm_circle)
            infrastructure_elements['comm_circles'].append(comm_circle)
            
            # Crystallization radius circle (dashed line)
            cryst_circle = plt.Circle(node.position, node.crystallization_radius, 
                                    fill=False, color='blue', alpha=0.4, 
                                    linewidth=2, linestyle='--')
            ax.add_patch(cryst_circle)
            infrastructure_elements['cryst_circles'].append(cryst_circle)

    return fig, ax, scat, title, infrastructure_elements

############################
# Updates entire frame
############################

def create_update_function(sim, temperature_schedule, sample_time, sigma, epsilon, distance, c1_gamma, c2_gamma, alpha,
                           scat, title, kinetic_temperatures, time_log, infrastructure_elements):
    counter = [0]

    def compute_kinetic_temperature(agents, mass=1, kB=1):
        velocities = agents[:, 2:]
        KE_total = 0.5 * mass * np.sum(velocities**2)
        N = len(agents)
        return KE_total / (N * kB)

    def update(frame):
        temp = temperature_schedule[frame]
        forces = sim.compute_forces(sigma, epsilon, distance, temp, c1_gamma, c2_gamma, None, alpha)
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

        # Update infrastructure node visualization and statistics
        if sim.infrastructure.nodes:
            # Reset per-frame data
            sim.infrastructure.reset_frame_data()
            
            # Get coverage statistics
            stats = sim.infrastructure.get_coverage_stats()
            
            # Update node colors based on status
            for i, node in enumerate(sim.infrastructure.nodes):
                node_marker = infrastructure_elements['nodes'][i]
                
                if node.is_saturated:
                    node_marker.set_color('red')  # Saturated = red
                elif node.get_connected_count() > 0:
                    node_marker.set_color('orange')  # Active = orange  
                else:
                    node_marker.set_color('green')  # Idle = green

        scat.set_offsets(positions)
        scat.set_facecolor(colors)
        
        # Add infrastructure statistics to title
        if sim.infrastructure.nodes:
            stats = sim.infrastructure.get_coverage_stats()
            title.set_text(f"Frame {frame} | Temp = {temp:.1f} | Connected: {stats['total_connected']} | Active Nodes: {stats['active_nodes']}/{stats['total_nodes']}")
        else:
            title.set_text(f"Frame {frame} | Temp = {temp:.1f}")
            
        counter[0] += 1
        
        # Return all animated elements
        return_elements = [scat, title]
        # Add infrastructure node markers to return elements for animation
        if infrastructure_elements and infrastructure_elements['nodes']:
            return_elements.extend(infrastructure_elements['nodes'])
        
        return return_elements
    
    return update

