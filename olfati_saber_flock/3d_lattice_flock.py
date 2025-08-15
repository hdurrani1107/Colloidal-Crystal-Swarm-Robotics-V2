#######################################################################
# This is an updated flocking simulation based on:
# Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory
# by Reza Olfati-Saber
#
# Author: Humzah Durrani
# STATUS: Complete
# To do:
# 
# Sources/References:
#  1. R. Olfati-Saber. Flocking for multi-agent dynamic systems: 
#     algorithms and theory. IEEE Transactions on Automatic Control, 
#     51(3):401–420, 2006.
#  2. https://github.com/arbit3rr/Flocking-Multi-Agent/tree/main
#  3. https://github.com/tjards/flocking_network?tab=readme-ov-file
#  4. ChatGPT
# 
#######################################################################

##########################
# Importing Libraries
##########################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##########################
# Parameters
##########################

#Number of Agents
agents = 50

#Max Steps (Animation run-time)
max_steps = 1000

#Smoothing factor for sigma_norm
eps = 0.1

#Shapes phi function
A = 5
B = 5

#Normalization Constant
C = np.abs(A - B) / np.sqrt(4 * A * B)

#Bump Function Threshold
H = 0.2

#Range and Distance
R = 12 
D = 10

#Repulsion Agents
obstacles = np.array([[25,25,25], [75,50,50], [10,90,90]])
R_obs = 10


#Control Gain Parameters from the paper
#Inter-agent interactions
c1_alpha = 3
c2_alpha = 2 * np.sqrt(c1_alpha)
# Global Control
c1_gamma = 5
c2_gamma = 0.2 * np.sqrt(c1_gamma)

#Counter for gamme change
counter = 0

##########################
# Core Math Functions
##########################

#Smooths transition between 1 and 0 when z crosses threshold
def bump_funct(z):
    Ph = np.zeros_like(z)
    Ph[z <= 1] = (1+ np.cos(np.pi * (z[z <= 1] - H) / (1 - H))) / 2
    Ph[z < H] = 1
    Ph[z < 0] = 0
    return Ph

#Smooths version of identity function that avoids singularities at 0
def sigma_1(z):
    return z / np.sqrt(1 + z ** 2)

#Used to compute distances with smooth behavior
def sigma_norm(z):
    return (np.sqrt(1 + eps * np.linalg.norm(z, axis =-1, keepdims=True) ** 2) - 1) / eps

#Gradient of sigma norm, to compute normalized direction vectors 
#for force field
def sigma_grad(z):
    return z / np.sqrt(1 + eps * np.linalg.norm(z, axis =-1, keepdims=True) ** 2)

#Base Potential Function
def phi(z):
    return ((A + B) * sigma_1(z + C) + (A - B)) / 2

#Function that uses bump function to restrict the range of interaction
def phi_alpha(z):
    r_alpha = sigma_norm([R])
    d_alpha = sigma_norm([D])
    return bump_funct(z / r_alpha) * phi(z - d_alpha)

#Function for repulsion-only agents
def phi_obs(z):
    if z < 1e-3:
        return 1e6
    elif z < R_obs:
        return 100/ (z**2 + 1e-3)
    else:
        return 0.0


##########################
# Multi-Agent Class
##########################
class multi_agent:
    
    #Initialize agents position and velocity
    def __init__(self, number, sampletime=0.1):
        self.dt = sampletime

        #Calculates if agent is outside obstacle at start
        def is_outside_obstacles(pos, obstacles, R_obs):
            return all(np.linalg.norm(pos-obs) > R_obs for obs in obstacles)
        
        #Ensures agent is in a valid position outside obstacles at start
        def generate_valid_positions(n_agents, obstacles, R_obs, bounds=(0,100)):
            valid_positions = []
            while len(valid_positions) < n_agents:
                candidate = np.random.uniform(bounds[0], bounds[1], 3)
                if is_outside_obstacles(candidate, obstacles, R_obs):
                    valid_positions.append(candidate)
            return np.array(valid_positions)
        
        #Generate valid positions
        positions = generate_valid_positions(number, obstacles, R_obs)
        self.agents = np.hstack([positions, np.zeros((number,3))])

    #Update agents position and velocity
    def update(self,u=3):
        q_dot = u
        self.agents[:,3:] += q_dot * self.dt
        p_dot = self.agents[:,3:]
        self.agents[:,:3] += p_dot * self.dt

##########################
# Helper Functions
##########################

#Function that indicates whether two agents are within interaction range
def get_adj_mat(nodes, r):
    n = len(nodes)
    adj = np.zeros((n,n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(nodes[i,:3] - nodes[j, :3])
                adj[i,j] = dist <= r
    return adj

#Function for influence coeffs based on distance (velocity matching)
def influence(q_i, q_js):
    r_alpha = sigma_norm([R])
    return bump_funct(sigma_norm(q_js - q_i) / r_alpha)

#Function to calculate direction vectors (agents to neighbors)
def local_dir(q_i, q_js):
    return sigma_grad(q_js - q_i)

#Function to calculate repulsive forces from agents to obstacles
def obs_rep(obstacles, agent_p):
    u_obs = np.zeros(3)
    for obs in obstacles:
        d_vec = agent_p - obs
        d_norm = sigma_norm(d_vec)[0]
        if d_norm < R_obs:
            u_obs += phi_obs(d_norm) * (d_vec / (d_norm + 1e-3))
    return u_obs

##########################
# Main Loop
##########################

#Initialize Agents
multi_agent_sys = multi_agent(number = agents)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

for i in range(max_steps):
    ax.cla()
    plt.title(f't = {i}')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    #Compute Adjacency
    adj_mat = get_adj_mat(multi_agent_sys.agents, R)
    u = np.zeros((agents, 3))

    #Loop through agents
    for j in range(agents):
        #Get positions and velocity
        agent_p = multi_agent_sys.agents[j, :3]
        agent_q = multi_agent_sys.agents[j, 3:]

        #Init control input
        u_alpha = np.zeros(3)

        #Identify and process neighbors
        neighbor_idx = adj_mat[j]
        if np.sum(neighbor_idx) > 1:
            neighbor_p = multi_agent_sys.agents[neighbor_idx, :3]
            neighbor_q = multi_agent_sys.agents[neighbor_idx, 3:]
            direction = local_dir(agent_p, neighbor_p)

            #Interaction with neighbors
            u1 = c2_alpha * np.sum(phi_alpha(sigma_norm(neighbor_p - agent_p)) * direction, axis=0)

            #Velocity alignment with neighbors
            n_influence = influence(agent_p, neighbor_p)
            u2 = c2_alpha * np.sum(n_influence * (neighbor_q - agent_q), axis=0)
        
            #Total Influence
            u_alpha = u1 + u2 

        #Feedback from gamma agent
        gamma_pos = [75,75,75]
        if counter >= 150:
            gamma_pos = [25,25,25]
        
        u_gamma = -c1_gamma * sigma_1(agent_p - gamma_pos) - c2_gamma * agent_q

        #Feedback from obstacle
        u_obs = 20 * obs_rep(obstacles, agent_p)

        #Total control input
        u[j] = u_alpha + u_gamma + u_obs

    #Update agent states
    multi_agent_sys.update(u)

    #Plot Agents and their connections

    def plot_sphere(ax, center, radius, color='red', alpha=0.2):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = radius * np.cos(u) * np.sin(v) + center[0]
        y = radius * np.sin(u) * np.sin(v) + center[1]
        z = radius * np.cos(v) + center[2]
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    for obs in obstacles:
        plot_sphere(ax, obs, R_obs)

    for k in range(agents):
        for l in range(agents):
            if k != l and adj_mat[k,l] == 1:
                x_vals = [multi_agent_sys.agents[k, 0], multi_agent_sys.agents[l, 0]]
                y_vals = [multi_agent_sys.agents[k, 1], multi_agent_sys.agents[l, 1]]
                z_vals = [multi_agent_sys.agents[k, 2], multi_agent_sys.agents[l, 2]]
                ax.plot(x_vals, y_vals, z_vals, linewidth=0.5)

    for m, agent in enumerate(multi_agent_sys.agents):
        x,y,z = agent[:3]
        ax.scatter(x,y,z, s=5, c='green', marker = 's')

    xg, yg, zg = gamma_pos
    ax.scatter(xg, yg, zg, s=5, c='blue', marker = '*')
    
    counter += 1

    plt.pause(0.01)

plt.show()