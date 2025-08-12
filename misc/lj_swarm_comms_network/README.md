# Swarm Mesh Network Simulation

A 2D simulation of rapidly deployable swarm-based mesh networking using Lennard-Jones physics and RSSI-based communication.

## Overview

This simulation combines the collective behavior of LJ-Swarm agents with communication network principles to create a rapidly deployable mesh network that can provide robust coverage in environments with obstacles, deadzones, and interference.

## Features

### Agent Properties
- **Temperature-based behavior**: Agents can dynamically adjust their temperature to move quickly to deployment positions and then cool down for stable operation
- **LJ Physics**: Agents use Lennard-Jones potential for natural spacing and collision avoidance
- **RSSI Communication**: Each agent has configurable signal strength affecting communication range
- **Adaptive Deployment**: Agents autonomously identify and fill coverage gaps

### Network Elements
- **Infrastructure Nodes**: Fixed base stations with predefined signal strength and coverage
- **Mobile Agents**: Swarm agents that form the adaptive mesh network
- **Data Transmission**: Simulated packet routing through the network
- **Coverage Analysis**: Real-time coverage quality assessment

### Environmental Challenges
- **Physical Obstacles**: Buildings, terrain features that block movement and signals
- **Communication Deadzones**: Areas with poor signal propagation
- **Noise Interference**: Zones that degrade signal quality
- **Dynamic Coverage Gaps**: Areas requiring additional network coverage

## Architecture

### Core Components

1. **mesh_network_engine.py**: Main simulation engine
   - `MeshNetworkAgent`: Mobile agents with LJ physics and RSSI capabilities
   - `NetworkNode`: Fixed infrastructure communication nodes
   - `MeshNetworkSimulation`: Main simulation controller
   - `DataPacket`: Network data transmission simulation

2. **mesh_visualization.py**: Real-time visualization system
   - Network topology display
   - Coverage quality heatmap
   - Agent temperature monitoring
   - Network metrics tracking

3. **main_simulation.py**: Simulation runner and analysis
   - Test scenario creation
   - Metrics collection
   - Results analysis

### Key Algorithms

#### Coverage Optimization
Agents use a multi-stage approach:
1. **Exploration Phase**: High temperature for rapid movement
2. **Gap Detection**: Identify areas with poor coverage quality
3. **Deployment**: Navigate to optimal positions using LJ forces
4. **Stabilization**: Cool down temperature for stable operation

#### RSSI-based Communication
- Signal strength calculation using path loss model
- Connection quality assessment
- Interference from noise zones
- Dynamic connectivity updates

#### Network Routing
- Simple hop-based routing algorithm
- Connection quality consideration
- Packet success/failure tracking

## Usage

### Running the Simulation

```bash
cd lj_swarm_comms_network
python main_simulation.py
```

### Customizing Parameters

Edit the simulation parameters in `main_simulation.py`:

```python
# Simulation parameters
bounds = [0, 50]          # Environment size
sample_time = 0.01        # Physics timestep
n_steps = 3000           # Simulation duration
n_agents = 20            # Number of mobile agents

# Agent properties
sigma = 3.0              # LJ interaction range
epsilon = 3.0            # LJ interaction strength
rssi_strength = 10.0     # Communication power
initial_temp = 5.0       # Starting temperature
```

### Adding Network Elements

```python
# Infrastructure nodes
sim.add_infrastructure_node(position, signal_strength, coverage_radius)

# Obstacles
sim.add_obstacle(position, radius)

# Communication deadzones
sim.add_deadzone(position, radius)

# Noise interference zones
sim.add_noise_zone(position, radius, noise_level)

# Mobile agents
sim.add_agent(position, sigma=3.0, epsilon=3.0, rssi_strength=10.0, temp=5.0)
```

## Visualization

The simulation provides real-time visualization with four panels:

1. **Network Topology** (top-left): Shows agents, infrastructure, connections, and environmental features
2. **Coverage Quality** (top-right): Heatmap of communication coverage quality
3. **Network Metrics** (bottom-left): Time-series plots of coverage, connectivity, and temperature
4. **Agent Temperatures** (bottom-right): Individual agent temperature states

### Visual Legend
- **Green squares**: Active infrastructure nodes
- **Colored circles**: Mobile agents (color = temperature, size = RSSI strength)
- **Gray circles**: Physical obstacles
- **Red dashed circles**: Communication deadzones
- **Orange circles**: Noise interference zones
- **Green lines**: Agent-infrastructure connections
- **Blue lines**: Agent-agent connections

## Metrics

### Coverage Quality
- Percentage of environment with adequate signal strength
- Calculated using RSSI propagation model
- Accounts for obstacles and interference

### Network Connectivity
- Ratio of active connections to possible connections
- Measures network robustness
- Includes agent-infrastructure and agent-agent links

### Deployment Efficiency
- Number of agents in deployed (stable) state
- Agent temperature distribution
- Network formation time

### Data Transmission
- Packet routing success rate
- Average hop count
- Network throughput simulation

## Applications

This simulation is suitable for modeling:

- **Emergency Response Networks**: Rapid deployment in disaster areas
- **Military Communications**: Adaptive mesh networks in contested environments
- **IoT Sensor Networks**: Self-organizing sensor coverage
- **Drone Swarm Communications**: UAV mesh networking
- **Smart City Infrastructure**: Adaptive urban communication networks

## Dependencies

- numpy
- matplotlib
- tqdm
- dataclasses (Python 3.7+)
- typing (Python 3.5+)

## Future Enhancements

- Advanced routing algorithms (AODV, OLSR)
- Multi-frequency interference modeling
- 3D environment support
- Mobile infrastructure nodes
- Energy consumption modeling
- Machine learning-based optimization