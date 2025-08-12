#######################################################################
# background_visualization.py
#
# Background visualization system for mesh network simulation
# Saves agent simulation and coverage as separate MP4 files
#
# Author: Humzah Durrani
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as patches
from typing import List, Tuple, Optional
import os
from output_utils import get_output_path

class BackgroundMeshVisualizer:
    """Background visualization system that saves MP4 files"""
    
    def __init__(self, bounds: List[float], figsize: Tuple[float, float] = (10, 5)):
        self.bounds = bounds
        self.figsize = figsize
        
        # Storage for frame data
        self.agent_frames = []
        self.coverage_frames = []
        self.sim_data = None
        
        # Visual elements storage
        self.obstacle_patches = []
        self.deadzone_patches = []
        self.noise_patches = []
        
    def setup_agent_figure(self):
        """Setup figure for agent simulation"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.bounds)
        ax.set_ylim(self.bounds)
        ax.set_aspect('equal')
        ax.set_title('Swarm Mesh Network - Agent Simulation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        return fig, ax
    
    def setup_coverage_figure(self):
        """Setup figure for coverage heatmap"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.bounds)
        ax.set_ylim(self.bounds)
        ax.set_aspect('equal')
        ax.set_title('Swarm Mesh Network - Coverage Quality', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        return fig, ax
    
    def draw_static_elements_agent(self, ax, sim):
        """Draw static elements for agent visualization"""
        # Clear previous static elements
        for patch in self.obstacle_patches:
            patch.remove()
        self.obstacle_patches.clear()
        
        # Draw obstacles
        for obs_pos, obs_radius in sim.obstacles:
            circle = Circle(obs_pos, obs_radius, color='gray', alpha=0.7, 
                          edgecolor='black', linewidth=2, label='Obstacles')
            ax.add_patch(circle)
            self.obstacle_patches.append(circle)
        
        # Draw deadzones
        for dead_pos, dead_radius in sim.deadzones:
            circle = Circle(dead_pos, dead_radius, color='red', alpha=0.3, 
                          linestyle='--', fill=False, linewidth=2, label='Deadzones')
            ax.add_patch(circle)
            self.obstacle_patches.append(circle)
        
        # Draw noise zones
        for noise_pos, noise_radius, noise_level in sim.noise_zones:
            alpha = min(0.5, noise_level / 20.0)
            circle = Circle(noise_pos, noise_radius, color='orange', alpha=alpha, 
                          linestyle=':', fill=True, label='Noise Zones')
            ax.add_patch(circle)
            self.obstacle_patches.append(circle)
        
        # Draw infrastructure nodes
        if sim.infrastructure_nodes:
            for node in sim.infrastructure_nodes:
                # Node itself
                node_circle = Circle(node.position, 0.8, color='green' if node.is_active else 'red', 
                                   alpha=0.9, edgecolor='black', linewidth=2)
                ax.add_patch(node_circle)
                self.obstacle_patches.append(node_circle)
                
                # Coverage area
                if node.is_active:
                    coverage_circle = Circle(node.position, node.coverage_radius, 
                                           color='green', alpha=0.1, fill=True, 
                                           linestyle='-', linewidth=1)
                    ax.add_patch(coverage_circle)
                    self.obstacle_patches.append(coverage_circle)
        
        # Add legend only once
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.7, label='Obstacles'),
            Patch(facecolor='red', alpha=0.3, label='Deadzones'),
            Patch(facecolor='orange', alpha=0.3, label='Noise Zones'),
            Patch(facecolor='green', alpha=0.9, label='Infrastructure'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=8, alpha=0.8, label='Mobile Agents'),
            plt.Line2D([0], [0], color='green', alpha=0.7, label='Infra Connection'),
            plt.Line2D([0], [0], color='blue', alpha=0.7, label='Agent Connection')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    def draw_static_elements_coverage(self, ax, sim):
        """Draw static elements for coverage visualization"""
        # Draw obstacles
        for obs_pos, obs_radius in sim.obstacles:
            circle = Circle(obs_pos, obs_radius, color='black', alpha=0.8, 
                          edgecolor='white', linewidth=2)
            ax.add_patch(circle)
        
        # Draw deadzones
        for dead_pos, dead_radius in sim.deadzones:
            circle = Circle(dead_pos, dead_radius, color='darkred', alpha=0.6, 
                          linestyle='--', fill=False, linewidth=3)
            ax.add_patch(circle)
        
        # Draw noise zones
        for noise_pos, noise_radius, noise_level in sim.noise_zones:
            circle = Circle(noise_pos, noise_radius, color='purple', alpha=0.4, 
                          linestyle=':', fill=False, linewidth=2)
            ax.add_patch(circle)
    
    def create_agent_frame(self, sim, step_num, total_steps):
        """Create a single frame for agent visualization"""
        fig, ax = self.setup_agent_figure()
        self.draw_static_elements_agent(ax, sim)
        
        # Update network connectivity
        sim.update_network_connectivity()
        
        # Draw agents
        if sim.agents:
            positions = np.array([agent.position for agent in sim.agents])
            temperatures = np.array([agent.temperature for agent in sim.agents])
            rssi_strengths = np.array([agent.rssi_strength for agent in sim.agents])
            
            # Color by temperature
            temp_norm = Normalize(vmin=0, vmax=8.0)
            agent_scatter = ax.scatter(
                positions[:, 0], positions[:, 1],
                c=temperatures, s=rssi_strengths * 8, 
                cmap='viridis', norm=temp_norm, alpha=0.8, 
                edgecolors='black', linewidth=1
            )
        
        # Draw connections
        lines = []
        line_colors = []
        
        # Agent to infrastructure connections
        for agent in sim.agents:
            for node_id in agent.connected_nodes:
                if node_id < len(sim.infrastructure_nodes):
                    node = sim.infrastructure_nodes[node_id]
                    lines.append([agent.position, node.position])
                    dist = np.linalg.norm(agent.position - node.position)
                    quality = max(0.3, 1 - dist / 20.0)
                    line_colors.append((0, 1, 0, quality))
        
        # Agent to agent connections
        for i, agent1 in enumerate(sim.agents):
            for agent2_id in agent1.connected_agents:
                if agent2_id > i:
                    agent2 = sim.agents[agent2_id]
                    lines.append([agent1.position, agent2.position])
                    dist = np.linalg.norm(agent1.position - agent2.position)
                    quality = max(0.3, 1 - dist / 15.0)
                    line_colors.append((0, 0, 1, quality))
        
        if lines:
            connection_lines = LineCollection(lines, colors=line_colors, linewidths=2)
            ax.add_collection(connection_lines)
        
        # Add step information
        deployed_count = sum(1 for agent in sim.agents if agent.is_deployed)
        avg_temp = np.mean([agent.temperature for agent in sim.agents]) if sim.agents else 0
        ax.text(0.02, 0.98, f'Step: {step_num}/{total_steps}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.93, f'Deployed: {deployed_count}/{len(sim.agents)}', transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.88, f'Avg Temp: {avg_temp:.2f}', transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return fig
    
    def create_coverage_frame(self, sim, step_num, total_steps):
        """Create a single frame for coverage visualization"""
        fig, ax = self.setup_coverage_figure()
        self.draw_static_elements_coverage(ax, sim)
        
        # Compute and display coverage
        coverage_map = sim.compute_coverage_gaps()
        if coverage_map is not None:
            extent = [self.bounds[0], self.bounds[1], self.bounds[0], self.bounds[1]]
            im = ax.imshow(coverage_map, extent=extent, origin='lower',
                          cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Coverage Quality', fontsize=12)
        
        # Mark agent positions
        if sim.agents:
            positions = np.array([agent.position for agent in sim.agents])
            deployment_status = [agent.is_deployed for agent in sim.agents]
            
            # Different markers for deployed vs moving agents
            deployed_pos = positions[[i for i, deployed in enumerate(deployment_status) if deployed]]
            moving_pos = positions[[i for i, deployed in enumerate(deployment_status) if not deployed]]
            
            if len(deployed_pos) > 0:
                ax.scatter(deployed_pos[:, 0], deployed_pos[:, 1], 
                          c='darkgreen', s=100, marker='s', alpha=0.9, 
                          edgecolors='white', linewidth=2, label='Deployed Agents')
            
            if len(moving_pos) > 0:
                ax.scatter(moving_pos[:, 0], moving_pos[:, 1], 
                          c='red', s=80, marker='o', alpha=0.9, 
                          edgecolors='white', linewidth=2, label='Moving Agents')
        
        # Mark infrastructure nodes
        if sim.infrastructure_nodes:
            infra_positions = np.array([node.position for node in sim.infrastructure_nodes])
            ax.scatter(infra_positions[:, 0], infra_positions[:, 1],
                      c='blue', s=150, marker='*', alpha=1.0,
                      edgecolors='white', linewidth=2, label='Infrastructure')
        
        # Add coverage statistics
        if coverage_map is not None:
            coverage_pct = np.mean(coverage_map > 0.5) * 100
            avg_coverage = np.mean(coverage_map)
            
            ax.text(0.02, 0.98, f'Step: {step_num}/{total_steps}', transform=ax.transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0.02, 0.93, f'Coverage: {coverage_pct:.1f}%', transform=ax.transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0.02, 0.88, f'Avg Quality: {avg_coverage:.3f}', transform=ax.transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        return fig
    
    def save_simulation_videos(self, sim, n_steps, fps=30, progress_callback=None):
        """Run simulation and save both agent and coverage videos"""
        print("Starting background simulation with video recording...")
        print(f"Steps: {n_steps}, FPS: {fps}")
        agent_video_path = get_output_path('videos', 'agent_simulation.mp4', 'comms_network')
        coverage_video_path = get_output_path('videos', 'coverage_simulation.mp4', 'comms_network')
        print(f"This will create two MP4 files: {agent_video_path} and {coverage_video_path}")
        
        # Initialize video writers
        agent_writer = FFMpegWriter(fps=fps, metadata=dict(artist='SwarmMeshNetwork'))
        coverage_writer = FFMpegWriter(fps=fps, metadata=dict(artist='SwarmMeshNetwork'))
        
        # Run simulation and collect frames
        with agent_writer.saving(self.setup_agent_figure()[0], agent_video_path, 150), \
             coverage_writer.saving(self.setup_coverage_figure()[0], coverage_video_path, 150):
            
            for step in range(n_steps):
                # Run simulation step
                sim.step()
                
                # Save every nth frame to control video length
                save_interval = max(1, n_steps // (fps * 30))  # Target ~30 second video max
                
                if step % save_interval == 0:
                    # Create agent frame
                    agent_fig = self.create_agent_frame(sim, step, n_steps)
                    agent_writer.grab_frame()
                    plt.close(agent_fig)
                    
                    # Create coverage frame  
                    coverage_fig = self.create_coverage_frame(sim, step, n_steps)
                    coverage_writer.grab_frame()
                    plt.close(coverage_fig)
                
                # Progress callback
                if progress_callback and step % 50 == 0:
                    progress_callback(step, n_steps)
        
        print(f"✓ Videos saved: {agent_video_path} and {coverage_video_path}")

class SimulationRecorder:
    """Manages the entire simulation recording process"""
    
    def __init__(self, sim, visualizer):
        self.sim = sim
        self.visualizer = visualizer
        self.metrics_data = {
            'time_steps': [],
            'coverage_percentage': [],
            'connectivity_ratio': [],
            'avg_temperature': [],
            'deployed_agents': [],
            'network_efficiency': []
        }
    
    def progress_callback(self, current_step, total_steps):
        """Callback for simulation progress"""
        percentage = (current_step / total_steps) * 100
        print(f"Progress: {current_step}/{total_steps} ({percentage:.1f}%)")
        
        # Collect metrics
        if current_step % 10 == 0:
            coverage_map = self.sim.compute_coverage_gaps()
            coverage_percentage = np.mean(coverage_map > 0.5) * 100 if coverage_map is not None else 0
            
            total_possible = len(self.sim.agents) * (len(self.sim.infrastructure_nodes) + len(self.sim.agents) - 1)
            actual_connections = sum(len(agent.connected_nodes) + len(agent.connected_agents) 
                                   for agent in self.sim.agents)
            connectivity_ratio = actual_connections / max(1, total_possible)
            
            avg_temperature = np.mean([agent.temperature for agent in self.sim.agents]) if self.sim.agents else 0
            deployed_count = sum(1 for agent in self.sim.agents if agent.is_deployed)
            network_efficiency = (coverage_percentage / 100) * connectivity_ratio
            
            self.metrics_data['time_steps'].append(current_step)
            self.metrics_data['coverage_percentage'].append(coverage_percentage)
            self.metrics_data['connectivity_ratio'].append(connectivity_ratio)
            self.metrics_data['avg_temperature'].append(avg_temperature)
            self.metrics_data['deployed_agents'].append(deployed_count)
            self.metrics_data['network_efficiency'].append(network_efficiency)
    
    def run_simulation_with_recording(self, n_steps, fps=20):
        """Run the complete simulation with background recording"""
        self.visualizer.save_simulation_videos(
            self.sim, n_steps, fps=fps, 
            progress_callback=self.progress_callback
        )
        return self.metrics_data