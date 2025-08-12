#######################################################################
# mesh_visualization.py
#
# Visualization system for swarm mesh network simulation
#
# Author: Humzah Durrani
#######################################################################
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from typing import List, Tuple, Optional

class MeshNetworkVisualizer:
    """Visualization system for mesh network simulation"""
    
    def __init__(self, bounds: List[float], figsize: Tuple[float, float] = (12, 8)):
        self.bounds = bounds
        self.fig = None
        self.axes = None
        self.setup_figure(figsize)
        
        # Visual elements
        self.agent_scatter = None
        self.node_scatter = None
        self.coverage_im = None
        self.connection_lines = None
        self.obstacle_patches = []
        self.deadzone_patches = []
        self.noise_patches = []
        
        # Color schemes
        self.agent_colormap = cm.viridis
        self.coverage_colormap = cm.RdYlGn
        
    def setup_figure(self, figsize: Tuple[float, float]):
        """Initialize the figure and subplots"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('Swarm Mesh Network Simulation', fontsize=16)
        
        # Main simulation view (top-left)
        self.axes[0, 0].set_xlim(self.bounds)
        self.axes[0, 0].set_ylim(self.bounds)
        self.axes[0, 0].set_aspect('equal')
        self.axes[0, 0].set_title('Network Topology')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Coverage heatmap (top-right)
        self.axes[0, 1].set_xlim(self.bounds)
        self.axes[0, 1].set_ylim(self.bounds)
        self.axes[0, 1].set_aspect('equal')
        self.axes[0, 1].set_title('Coverage Quality')
        
        # Network metrics (bottom-left)
        self.axes[1, 0].set_title('Network Metrics')
        self.axes[1, 0].set_xlabel('Time Steps')
        
        # Agent temperatures (bottom-right)
        self.axes[1, 1].set_title('Agent Temperatures')
        self.axes[1, 1].set_xlabel('Agent ID')
        self.axes[1, 1].set_ylabel('Temperature')
        
        plt.tight_layout()
    
    def draw_static_elements(self, sim):
        """Draw static elements like obstacles, deadzones, and infrastructure"""
        # Clear previous static elements
        for patch in self.obstacle_patches + self.deadzone_patches + self.noise_patches:
            patch.remove()
        self.obstacle_patches.clear()
        self.deadzone_patches.clear()
        self.noise_patches.clear()
        
        # Draw obstacles
        for obs_pos, obs_radius in sim.obstacles:
            circle = Circle(obs_pos, obs_radius, color='gray', alpha=0.7, label='Obstacle')
            self.axes[0, 0].add_patch(circle)
            self.obstacle_patches.append(circle)
            
            # Also add to coverage view
            circle_cov = Circle(obs_pos, obs_radius, color='gray', alpha=0.7)
            self.axes[0, 1].add_patch(circle_cov)
            self.obstacle_patches.append(circle_cov)
        
        # Draw deadzones
        for dead_pos, dead_radius in sim.deadzones:
            circle = Circle(dead_pos, dead_radius, color='red', alpha=0.3, 
                          linestyle='--', fill=False, linewidth=2, label='Deadzone')
            self.axes[0, 0].add_patch(circle)
            self.deadzone_patches.append(circle)
            
            circle_cov = Circle(dead_pos, dead_radius, color='red', alpha=0.3, 
                              linestyle='--', fill=False, linewidth=2)
            self.axes[0, 1].add_patch(circle_cov)
            self.deadzone_patches.append(circle_cov)
        
        # Draw noise zones
        for noise_pos, noise_radius, noise_level in sim.noise_zones:
            alpha = min(0.5, noise_level / 20.0)  # Scale alpha based on noise level
            circle = Circle(noise_pos, noise_radius, color='orange', alpha=alpha, 
                          linestyle=':', fill=True, label='Noise Zone')
            self.axes[0, 0].add_patch(circle)
            self.noise_patches.append(circle)
            
            circle_cov = Circle(noise_pos, noise_radius, color='orange', alpha=alpha, 
                              linestyle=':', fill=True)
            self.axes[0, 1].add_patch(circle_cov)
            self.noise_patches.append(circle_cov)
        
        # Draw infrastructure nodes
        if sim.infrastructure_nodes:
            node_positions = np.array([node.position for node in sim.infrastructure_nodes])
            node_sizes = np.array([node.signal_strength * 5 for node in sim.infrastructure_nodes])
            node_colors = ['green' if node.is_active else 'red' for node in sim.infrastructure_nodes]
            
            self.node_scatter = self.axes[0, 0].scatter(
                node_positions[:, 0], node_positions[:, 1],
                s=node_sizes, c=node_colors, marker='s', alpha=0.8,
                edgecolors='black', linewidth=2, label='Infrastructure'
            )
            
            # Draw coverage circles for active nodes
            for node in sim.infrastructure_nodes:
                if node.is_active:
                    coverage_circle = Circle(node.position, node.coverage_radius, 
                                           color='green', alpha=0.1, fill=True)
                    self.axes[0, 0].add_patch(coverage_circle)
    
    def update_agents(self, sim):
        """Update agent visualization"""
        if not sim.agents:
            return
        
        # Get agent data
        positions = np.array([agent.position for agent in sim.agents])
        temperatures = np.array([agent.temperature for agent in sim.agents])
        rssi_strengths = np.array([agent.rssi_strength for agent in sim.agents])
        
        # Normalize temperatures for color mapping
        temp_norm = Normalize(vmin=0, vmax=max(5.0, np.max(temperatures)))
        
        # Update or create agent scatter plot
        if self.agent_scatter is None:
            self.agent_scatter = self.axes[0, 0].scatter(
                positions[:, 0], positions[:, 1],
                c=temperatures, s=rssi_strengths * 2, 
                cmap=self.agent_colormap, norm=temp_norm,
                alpha=0.8, edgecolors='black', linewidth=1,
                label='Mobile Agents'
            )
        else:
            self.agent_scatter.set_offsets(positions)
            self.agent_scatter.set_array(temperatures)
            self.agent_scatter.set_sizes(rssi_strengths * 2)
    
    def update_connections(self, sim):
        """Update network connection visualization"""
        # Clear previous connection lines
        if self.connection_lines is not None:
            self.connection_lines.remove()
        
        lines = []
        line_colors = []
        
        # Agent to infrastructure connections
        for agent in sim.agents:
            for node_id in agent.connected_nodes:
                if node_id < len(sim.infrastructure_nodes):
                    node = sim.infrastructure_nodes[node_id]
                    lines.append([agent.position, node.position])
                    
                    # Color based on connection quality
                    dist = np.linalg.norm(agent.position - node.position)
                    quality = max(0, 1 - dist / 20.0)  # Simplified quality metric
                    line_colors.append((0, 1, 0, quality))  # Green with variable alpha
        
        # Agent to agent connections
        for i, agent1 in enumerate(sim.agents):
            for agent2_id in agent1.connected_agents:
                if agent2_id > i:  # Avoid duplicate lines
                    agent2 = sim.agents[agent2_id]
                    lines.append([agent1.position, agent2.position])
                    
                    # Color based on connection quality
                    dist = np.linalg.norm(agent1.position - agent2.position)
                    quality = max(0, 1 - dist / 15.0)
                    line_colors.append((0, 0, 1, quality))  # Blue with variable alpha
        
        # Draw connection lines
        if lines:
            self.connection_lines = LineCollection(lines, colors=line_colors, linewidths=1)
            self.axes[0, 0].add_collection(self.connection_lines)
    
    def update_coverage_heatmap(self, sim):
        """Update coverage quality heatmap"""
        if sim.coverage_map is not None:
            # Clear previous heatmap
            if self.coverage_im is not None:
                self.coverage_im.remove()
            
            # Create new heatmap
            extent = [self.bounds[0], self.bounds[1], self.bounds[0], self.bounds[1]]
            self.coverage_im = self.axes[0, 1].imshow(
                sim.coverage_map, extent=extent, origin='lower',
                cmap=self.coverage_colormap, alpha=0.7, vmin=0, vmax=1
            )
            
            # Add colorbar if not present
            if not hasattr(self, 'coverage_colorbar'):
                self.coverage_colorbar = plt.colorbar(self.coverage_im, ax=self.axes[0, 1])
                self.coverage_colorbar.set_label('Coverage Quality')
    
    def update_metrics(self, step_data: dict):
        """Update network metrics plots"""
        # Network metrics (bottom-left)
        ax_metrics = self.axes[1, 0]
        ax_metrics.clear()
        ax_metrics.set_title('Network Metrics')
        ax_metrics.set_xlabel('Time Steps')
        
        if 'time_steps' in step_data and len(step_data['time_steps']) > 1:
            time_steps = step_data['time_steps']
            
            if 'coverage_percentage' in step_data:
                ax_metrics.plot(time_steps, step_data['coverage_percentage'], 
                              'g-', label='Coverage %', linewidth=2)
            
            if 'connectivity_ratio' in step_data:
                ax_metrics.plot(time_steps, step_data['connectivity_ratio'], 
                              'b-', label='Connectivity', linewidth=2)
            
            if 'avg_temperature' in step_data:
                ax_temp_norm = np.array(step_data['avg_temperature']) / max(step_data['avg_temperature'])
                ax_metrics.plot(time_steps, ax_temp_norm, 
                              'r--', label='Avg Temp (norm)', alpha=0.7)
            
            ax_metrics.legend()
            ax_metrics.grid(True, alpha=0.3)
            ax_metrics.set_ylim(0, 1.1)
    
    def update_agent_temperatures(self, sim):
        """Update agent temperature bar chart"""
        ax_temp = self.axes[1, 1]
        ax_temp.clear()
        ax_temp.set_title('Agent Temperatures')
        ax_temp.set_xlabel('Agent ID')
        ax_temp.set_ylabel('Temperature')
        
        if sim.agents:
            agent_ids = range(len(sim.agents))
            temperatures = [agent.temperature for agent in sim.agents]
            deployment_status = ['Deployed' if agent.is_deployed else 'Moving' for agent in sim.agents]
            
            # Color bars based on deployment status
            colors = ['green' if status == 'Deployed' else 'red' for status in deployment_status]
            
            bars = ax_temp.bar(agent_ids, temperatures, color=colors, alpha=0.7)
            ax_temp.grid(True, alpha=0.3, axis='y')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', alpha=0.7, label='Deployed'),
                             Patch(facecolor='red', alpha=0.7, label='Moving')]
            ax_temp.legend(handles=legend_elements)
    
    def update_display(self, sim, step_data: dict = None):
        """Update the entire visualization"""
        # Update dynamic elements
        self.update_agents(sim)
        self.update_connections(sim)
        self.update_coverage_heatmap(sim)
        self.update_agent_temperatures(sim)
        
        if step_data:
            self.update_metrics(step_data)
        
        # Refresh the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def add_legend(self):
        """Add legend to the main plot"""
        self.axes[0, 0].legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    def save_frame(self, filename: str, dpi: int = 150):
        """Save current frame as image"""
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    def close(self):
        """Close the visualization"""
        plt.close(self.fig)