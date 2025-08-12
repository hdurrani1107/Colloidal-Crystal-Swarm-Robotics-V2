# LJ Swarm Communication Network V2

A 2D simulation where swarm agents assemble and crystallize around infrastructure nodes to create robust and scalable coverage networks for disaster-resilient communication.

## Features

### Infrastructure Nodes
- **Active Nodes**: Static infrastructure points that attract nearby agents
- **Communication Radius**: Large radius for agent attraction and network coverage
- **Crystallization Radius**: Smaller radius where agents form stable connections
- **Capacity Management**: Each node has maximum agent capacity to prevent oversaturation

### Agent Behavior
- **Lennard-Jones Forces**: Agents maintain optimal spacing through repulsion/attraction
- **Infrastructure Attraction**: Agents are pulled toward infrastructure nodes via goal forces
- **Crystallization**: Agents form stable clusters around nodes once connected
- **Dynamic Assignment**: Agents can switch between nodes based on proximity and capacity

### Visualization
- **Green Stars**: Infrastructure nodes (idle)
- **Orange Stars**: Active nodes with connected agents
- **Red Stars**: Saturated nodes at maximum capacity
- **Green Circles**: Communication radius (solid line)
- **Blue Circles**: Crystallization radius (dashed line)
- **Colored Agents**: Blue (solid), Green (liquid), Red (gas) based on kinetic energy

## Configuration

### Key Parameters
```python
agents = 100                    # Number of swarm agents
infrastructure_config = {
    'broadcast radius': 35,     # Communication/attraction radius
    'attraction_strength': 12.0, # Force strength for attracting agents
    'max_agents': 25            # Maximum agents per node
}
```

### Infrastructure Placement
```python
infrastructure = [
    np.array([20,130]),   # Node 1: Top-left
    np.array([75,75]),    # Node 2: Center  
    np.array([130,20])    # Node 3: Bottom-right
]
```

## Output
- **Video**: `output/videos/comm_network_simulation.mp4`
- **Analysis**: `output/graphs/comm_network_analysis.png`
- **Statistics**: Printed coverage metrics at simulation end

## Usage
```bash
cd lj_swarm_comm_network_2
python main.py
```

## System Architecture
- `engine.py`: Core simulation engine with LJ physics and infrastructure forces
- `infrastructure.py`: Infrastructure node management and force calculations
- `ui.py`: Visualization and real-time statistics display
- `main.py`: Simulation setup and execution
- `schedule.py`: Temperature scheduling for crystallization control