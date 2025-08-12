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
import os
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

agents = 150
sigma = 4.0
epsilon = 4.0 
sample_time = 0.005
bounds = [0, 100]
distance = (2 ** (1 / 6)) * sigma
alpha = 0.5
c1_gamma = 10 
c2_gamma = 1/2 * np.sqrt(c1_gamma)
n_frames = 30000
max_temp = 150
counter = 0
temperature_schedule = schedule.temp_schedule_3(max_temp,n_frames)
obstacles = [
    #(np.array([75, 75]), 10),  # center sphere
    #(np.array([30, 100]), 5),
    #(np.array([100, 30]), 5),
    #(np.array([100,75]), 10),
    #(np.array([75,100]), 10)
]
infrastructure = [np.array([25,25]), np.array([75,25]), np.array([50,75])]
infrastructure_config = {
    'broadcast radius': 25,       # communication radius
    'attraction_strength': 12.0,  # strength of attraction force
    'max_agents': 25              # max agents per node
}

##########################
# Init Sim
##########################

sim = multi_agent(agents, sigma, sample_time, bounds, obstacles)
sim.setup_infrastructure(infrastructure, infrastructure_config)

#Melted Initial Condition
initial_temperature = 150
init_melt(sim.agents, init_temp=initial_temperature)

##########################
# Visualization
###########################
fig, ax, scat, title, infrastructure_elements = setup_visualization(bounds, obstacles, sim.infrastructure)
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
    distance, c1_gamma, c2_gamma, alpha, scat, 
    title, kinetic_temperature, time_log, infrastructure_elements
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
output_dir = "../output"
os.makedirs(f"{output_dir}/videos", exist_ok=True)
os.makedirs(f"{output_dir}/graphs", exist_ok=True)

anim.save(f"{output_dir}/videos/comm_network_simulation.mp4", fps=60, dpi=150)

# Save infrastructure coverage statistics
if sim.infrastructure.nodes:
    stats = sim.infrastructure.get_coverage_stats()
    print(f"\n=== Final Infrastructure Statistics ===")
    print(f"Total Infrastructure Nodes: {stats['total_nodes']}")
    print(f"Active Nodes: {stats['active_nodes']}")
    print(f"Saturated Nodes: {stats['saturated_nodes']}")
    print(f"Total Connected Agents: {stats['total_connected']}")
    print(f"Average Connections per Node: {stats['avg_connections']:.2f}")

plt.figure(figsize=(10, 6))

# Kinetic energy plot
plt.subplot(1, 2, 1)
plt.plot(time_log, kinetic_temperature, label="Kinetic Energy", color='blue')
plt.xlabel("Time")
plt.ylabel("Average Kinetic Energy") 
plt.title("Average Kinetic Energy Over Time")
plt.grid(True)
plt.legend()

# Infrastructure connections plot (if data available)
plt.subplot(1, 2, 2)
if sim.infrastructure.nodes:
    node_connections = [node.get_connected_count() for node in sim.infrastructure.nodes]
    node_labels = [f"Node {i+1}" for i in range(len(sim.infrastructure.nodes))]
    plt.bar(node_labels, node_connections, color=['green' if c < 15 else 'orange' if c < 25 else 'red' for c in node_connections])
    plt.xlabel("Infrastructure Nodes")
    plt.ylabel("Connected Agents")
    plt.title("Final Agent Distribution Across Nodes")
    plt.xticks(rotation=45)
    
plt.tight_layout()
plt.savefig(f"{output_dir}/graphs/comm_network_analysis.png")
plt.show()