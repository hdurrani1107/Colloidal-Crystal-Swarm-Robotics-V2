#######################################################################
# ui.py
#
# Visualization for Olfati-Saber flocking simulation
# Adapted from LJ-Swarm ui.py
# 
# Author: Humzah Durrani
# AI Disclosure: Debugging and Handling
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
# Flock Color Update
############################
flock_colors = np.array([
    [1.0, 0.2, 0.2],   # Flock 1: red
    [0.2, 0.8, 0.2],   # Flock 2: green
    [0.2, 0.2, 1.0]    # Flock 3: blue
])

#OLD CODE
#leader_color = np.array([1.0, 1.0, 0.0])  # Leaders: yellow

############################
# Get Agent Colors
############################

def get_agent_colors(sim):
    n = len(sim.agents)
    colors = np.zeros((n, 3))
    
    for i in range(n):
        flock_id = sim.get_agent_flock(i)
        colors[i] = flock_colors[flock_id]
    
    return colors

############################
# Init Scatter Plot
############################

def setup_visualization(bounds, obstacles):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    # Equal Aspect Ratio 
    ax.set_aspect('equal', adjustable='box')
    scat = ax.scatter([], [], s=10, animated=True, edgecolors='none')
    
    # Initialize quiver with dummy data that will be updated (disabled)
    quiver = None  # Force arrows disabled
    
    title = ax.text(0.02, 1.02, '', transform=ax.transAxes)

    # Obstacles Input
    if obstacles:
        for pos, radius in obstacles:
            circle = plt.Circle(pos, radius, color='gray', alpha=0.4)
            ax.add_patch(circle)

    # Goal beacon visualization elements
    beacon_circles = []

    return fig, ax, scat, quiver, title, beacon_circles

############################
# Updates frame
############################

def create_update_function(sim, temperature_schedule, sample_time, sigma, epsilon, distance, c1_gamma, c2_gamma, alpha,
                           scat, quiver, title, kinetic_temperatures, time_log, beacon_circles, ax, metrics=None, 
                           render_interval=10):
    counter = [0]
    # current_quiver disabled - no force arrows


    #Pulled from LJ-SWARM. OS Does not use Temperature
    def compute_kinetic_temperature(agents, mass=1, kB=1):
        velocities = agents[:, 2:]
        KE_total = 0.5 * mass * np.sum(velocities**2)
        N = len(agents)
        return KE_total / (N * kB)
    

    def compute_neighbor_counts(agents, R=12):
        positions = agents[:, :2]
        n = len(agents)
        neighbor_counts = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= R:
                        neighbor_counts[i] += 1
        
        return neighbor_counts

    def update(frame):
        # Calculate the starting frame for this render step
        start_frame = frame * render_interval
        
        # For frame 0, render first then simulate
        # For other frames, simulate then render
        if frame == 0:
            # Render the initial state (frame 0) before any simulation
            temp = temperature_schedule[0]
            last_temp = temp
            positions = sim.agents[:, :2]
            velocities = sim.agents[:, 2:]
            colors = get_agent_colors(sim)
            
            # Log initial state (OBSOLETE)
            T_kin = compute_kinetic_temperature(sim.agents)
            kinetic_temperatures.append(T_kin)
            time_log.append(0)
            
            if sim.goal_beacons and metrics:
                active_beacons = sim.goal_beacons.get_active_beacons()
                num_beacons = len(active_beacons)
                trapped_counts = [n for _, _, n in active_beacons]
                total_trapped = int(np.sum(trapped_counts)) if trapped_counts else 0
                mean_trapped = float(np.mean(trapped_counts)) if trapped_counts else 0.0

                #Redundant
                #Clean Up Metrics
                metrics.log_frame(
                    frame=0,
                    t=0.0,
                    system_temp=float(temp),
                    num_beacons=num_beacons,
                    total_trapped=total_trapped,
                    mean_trapped=mean_trapped,
                    arrivals_this_frame=0
                )
                
            # Now run simulation for render_interval steps
            for step in range(render_interval):
                actual_frame = step + 1  # Start from frame 1
                if actual_frame >= len(temperature_schedule):
                    break
                temp = temperature_schedule[actual_frame]
                last_temp = temp
                
                # No external forces needed for flocking - all handled internally
                external_forces = np.zeros((len(sim.agents), 2))
                sim.update(external_forces, temp)
                
                # OBSOLETE (Bad Coding Practice)
                T_kin = compute_kinetic_temperature(sim.agents)
                if T_kin >= 1e18:
                    print("Simulation Destabilized")
                kinetic_temperatures.append(T_kin)
                time_log.append(actual_frame * sample_time)
                
                # Log metrics for every simulation frame
                if sim.goal_beacons and metrics:
                    active_beacons = sim.goal_beacons.get_active_beacons()
                    num_beacons = len(active_beacons)
                    trapped_counts = [n for _, _, n in active_beacons]
                    total_trapped = int(np.sum(trapped_counts)) if trapped_counts else 0
                    mean_trapped = float(np.mean(trapped_counts)) if trapped_counts else 0.0

                    prev_total = getattr(sim.goal_beacons, f"_prev_total_trapped_{actual_frame-1}", 0)
                    arrivals_this_frame = max(0, total_trapped - prev_total)
                    setattr(sim.goal_beacons, f"_prev_total_trapped_{actual_frame}", total_trapped)
                
                    #Redundant
                    #Clean Up Metrics
                    metrics.log_frame(
                        frame=int(actual_frame),
                        t=float(actual_frame * sample_time),
                        system_temp=float(temp),
                        num_beacons=num_beacons,
                        total_trapped=total_trapped,
                        mean_trapped=mean_trapped,
                        arrivals_this_frame=arrivals_this_frame
                    )
        else:
            # For subsequent frames: simulate then render
            last_temp = 0
            for step in range(render_interval):
                actual_frame = start_frame + step
                if actual_frame >= len(temperature_schedule):
                    break
                temp = temperature_schedule[actual_frame]
                last_temp = temp
                
                # No external forces needed for flocking
                external_forces = np.zeros((len(sim.agents), 2))
                sim.update(external_forces, temp, actual_frame)
                
                # OBSOLETE (Bad Coding Practice)
                T_kin = compute_kinetic_temperature(sim.agents)
                if T_kin >= 1e18:
                    print("Simulation Destabilized")
                kinetic_temperatures.append(T_kin)
                time_log.append(actual_frame * sample_time)
                
                # Log metrics for every simulation frame
                if sim.goal_beacons and metrics:
                    active_beacons = sim.goal_beacons.get_active_beacons()
                    num_beacons = len(active_beacons)
                    trapped_counts = [n for _, _, n in active_beacons]
                    total_trapped = int(np.sum(trapped_counts)) if trapped_counts else 0
                    mean_trapped = float(np.mean(trapped_counts)) if trapped_counts else 0.0

                    prev_total = getattr(sim.goal_beacons, f"_prev_total_trapped_{actual_frame-1}", 0)
                    arrivals_this_frame = max(0, total_trapped - prev_total)
                    setattr(sim.goal_beacons, f"_prev_total_trapped_{actual_frame}", total_trapped)

                    #Redundant
                    #Clean Up Metrics
                    metrics.log_frame(
                        frame=int(actual_frame),
                        t=float(actual_frame * sample_time),
                        system_temp=float(temp),
                        num_beacons=num_beacons,
                        total_trapped=total_trapped,
                        mean_trapped=mean_trapped,
                        arrivals_this_frame=arrivals_this_frame
                    )
            
            # Compute visualization data after simulation steps
            positions = sim.agents[:, :2]
            velocities = sim.agents[:, 2:]
            colors = get_agent_colors(sim)

        # Update goal beacon visualization
        if sim.goal_beacons:
            # Remove old beacon circles
            for circle in beacon_circles:
                circle.remove()
            beacon_circles.clear()
            
            # Add current goal beacons
            active_beacons = sim.goal_beacons.get_active_beacons()
            if frame < 10:  # Debug print for first few frames
                print(f"Frame {frame}: {len(active_beacons)} active beacons")
            for beacon_pos, beacon_radius, num_agents in active_beacons:
                # Color based on number of trapped agents
                if num_agents == 0:
                    color = 'cyan'
                    beacon_alpha = 0.3
                elif num_agents <= 3:
                    color = 'blue'
                    beacon_alpha = 0.5
                else:
                    color = 'darkblue'
                    beacon_alpha = 0.7
                
                # Draw beacon as circle with goal point at center
                circle = plt.Circle(beacon_pos, beacon_radius, fill=False, color=color, 
                                  linestyle='-', alpha=beacon_alpha, linewidth=3)
                ax.add_patch(circle)
                beacon_circles.append(circle)
                
                # Add goal point marker at center
                goal_point = plt.Circle(beacon_pos, 2, fill=True, color=color, alpha=0.8)
                ax.add_patch(goal_point)
                beacon_circles.append(goal_point)

        # Update scatter plot
        scat.set_offsets(positions)
        scat.set_facecolor(colors)
        
        # Force arrows disabled - no quiver updates needed
        
        # Show the rendered frame number (0, 10, 20, 30, ...)
        display_frame = frame * render_interval
        if sim.goal_beacons:
            num_beacons = len(sim.goal_beacons.beacons)
            total_trapped = sum(len(beacon.trapped_agents) for beacon in sim.goal_beacons.beacons)
            title.set_text(f"Time {display_frame * 0.005} | Beacons: {num_beacons} | Trapped: {total_trapped}")
        else:
            title.set_text(f"Time {display_frame * 0.005} | Flocking Simulation")
            
        counter[0] += 1
        
        # Return all animated elements (note: quiver is recreated each frame)
        return_elements = [scat, title] + beacon_circles
        
        return return_elements
    
    return update