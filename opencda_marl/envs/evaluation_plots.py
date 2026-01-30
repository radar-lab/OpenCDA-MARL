'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 12:00:00
FilePath     : /OpenCDA-MARL/opencda_marl/envs/evaluation_plots.py
Description  : Reusable plotting functions for evaluation system
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


class EvaluationPlotter:
    """
    Reusable plotting class for evaluation system.
    
    Provides standardized plotting functions for:
    - Step analysis (within episode)
    - Episode analysis (single episode summary)  
    - Episode comparison (across episodes)
    """
    
    def __init__(self, output_dir: Path, scenario_name: str, agent_name: str):
        """
        Initialize plotter.
        
        Args:
            output_dir: Directory to save plots
            scenario_name: Name of scenario
            agent_name: Name of agent
        """
        self.output_dir = Path(output_dir)
        self.scenario_name = scenario_name
        self.agent_name = agent_name
        
        # Use non-interactive backend for server environments
        matplotlib.use('Agg')
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_step_analysis(self, history: List[Dict], episode_num: int = 0) -> Path:
        """
        Generate detailed step-by-step analysis plot for single episode.
        
        Args:
            history: List of step statistics dictionaries
            episode_num: Episode number for filename
            
        Returns:
            Path to saved plot file
        """
        if not history:
            logger.warning("No step history data provided for plotting")
            return None
        
        steps = [h['step'] for h in history]
        success = [h['success'] for h in history]
        collision = [h['collision'] for h in history]
        active_agents = [h['active_agents'] for h in history]
        success_rate = [h['success_rate'] for h in history]
        collision_rate = [h['collision_rate'] for h in history]
        throughput = [h['throughput'] for h in history]
        total_vehicles = [h['total_vehicles'] for h in history]
        
        # Extract reward data
        step_rewards = [h.get('step_reward', 0.0) for h in history]
        cumulative_rewards = [h.get('cumulative_reward', 0.0) for h in history]
        avg_rewards_per_agent = [h.get('avg_reward_per_agent', 0.0) for h in history]
        
        # Create comprehensive 3x2 subplot for traffic and reward analysis
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle(f'Traffic Evaluation - {self.scenario_name} - {self.agent_name}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Throughput Analysis
        ax1 = axes[0, 0]
        if throughput:
            ax1.plot(steps, throughput, color='blue', linewidth=2, label='Throughput (vph)')
            ax1.fill_between(steps, throughput, alpha=0.3, color='blue')
                
            # Mark peak throughput
            if len(throughput) > 0:
                peak_idx = np.argmax(throughput)
                peak_value = throughput[peak_idx]
                peak_step = steps[peak_idx]
                ax1.scatter(peak_step, peak_value, color='red', s=100, zorder=5)
                ax1.annotate(f'Peak: {peak_value:.1f} vph\n@ step {peak_step}',
                            xy=(peak_step, peak_value), xytext=(10, 10),
                            textcoords='offset points', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
            
        ax1.set_xlabel('Simulation Steps')
        ax1.set_ylabel('Throughput (vehicles/hour)')
        ax1.set_title('Traffic Throughput Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success vs Collision Rates
        ax2 = axes[0, 1]
        if success_rate and collision_rate:
            ax2.fill_between(steps, success_rate, alpha=0.5, color='green', label='Success Rate')
            ax2.fill_between(steps, collision_rate, alpha=0.5, color='red', label='Collision Rate')
            ax2.plot(steps, success_rate, color='darkgreen', linewidth=1.5)
            ax2.plot(steps, collision_rate, color='darkred', linewidth=1.5)
                
        ax2.set_xlabel('Simulation Steps')
        ax2.set_ylabel('Rate (%)')
        ax2.set_title('Success vs Collision Rates')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
         # Plot 3: Vehicle Counts
        ax3 = axes[1, 0]
        if active_agents and total_vehicles:
            ax3.fill_between(steps, total_vehicles, alpha=0.4, color='lightblue', label='Total Vehicles')
            ax3.fill_between(steps, active_agents, alpha=0.6, color='purple', label='Active Vehicles')
            ax3.plot(steps, total_vehicles, color='blue', linewidth=1.5)
            ax3.plot(steps, active_agents, color='darkviolet', linewidth=1.5)
                
            # Add average lines
            if len(active_agents) > 0:
                avg_active = np.mean(active_agents)
                ax3.axhline(y=avg_active, color='orange', linestyle='--', 
                               label=f'Avg Active: {avg_active:.1f}')
                       
        ax3.set_xlabel('Simulation Steps')
        ax3.set_ylabel('Number of Vehicles')
        ax3.set_title('Vehicle Count Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
            
        # Plot 4: Cumulative Outcomes
        ax4 = axes[1, 1]
        if success and collision:
            ax4.plot(steps, success, color='green', linewidth=2, label='Cumulative Success')
            ax4.plot(steps, collision, color='red', linewidth=2, label='Cumulative Collision')

            # Add final statistics text box - use total_vehicles to include timeouts
            final_success = success[-1] if success else 0
            final_collision = collision[-1] if collision else 0
            final_total = total_vehicles[-1] if total_vehicles else (final_success + final_collision)
            final_timeout = final_total - final_success - final_collision
            final_success_pct = (final_success / final_total * 100) if final_total > 0 else 0
            final_collision_pct = (final_collision / final_total * 100) if final_total > 0 else 0

            stats_text = f'Final Results:\n{final_success} successes ({final_success_pct:.1f}%)\n{final_collision} collisions ({final_collision_pct:.1f}%)\n{final_timeout} timeouts'
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
        ax4.set_xlabel('Simulation Steps')
        ax4.set_ylabel('Cumulative Count')
        ax4.set_title('Cumulative Outcomes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Reward Evolution
        ax5 = axes[2, 0]
        if any(r != 0 for r in step_rewards + cumulative_rewards):  # Only plot if we have reward data
            # Use twin axes for different scales
            ax5_twin = ax5.twinx()
            
            # Step rewards on left axis
            line1 = ax5.plot(steps, step_rewards, 'b-', linewidth=2, label='Step Reward')
            ax5.fill_between(steps, step_rewards, alpha=0.3, color='blue')
            ax5.set_ylabel('Step Reward', color='b')
            ax5.tick_params(axis='y', labelcolor='b')
            
            # Cumulative rewards on right axis
            line2 = ax5_twin.plot(steps, cumulative_rewards, 'r-', linewidth=2, label='Cumulative Reward')
            ax5_twin.set_ylabel('Cumulative Reward', color='r')
            ax5_twin.tick_params(axis='y', labelcolor='r')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax5.legend(lines, labels, loc='upper left')
            
        ax5.set_xlabel('Simulation Steps')
        ax5.set_title('Reward Evolution Over Time')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Reward Efficiency Analysis
        ax6 = axes[2, 1]
        if any(r != 0 for r in avg_rewards_per_agent):  # Only plot if we have reward data
            ax6.plot(steps, avg_rewards_per_agent, 'g-', linewidth=2, label='Avg Reward per Agent')
            ax6.fill_between(steps, avg_rewards_per_agent, alpha=0.3, color='green')
            
            # Add correlation with collision rate if available
            if collision_rate and any(c > 0 for c in collision_rate):
                ax6_twin = ax6.twinx()
                ax6_twin.plot(steps, collision_rate, 'r--', alpha=0.7, label='Collision Rate %')
                ax6_twin.set_ylabel('Collision Rate (%)', color='r')
                ax6_twin.tick_params(axis='y', labelcolor='r')
                
                # Combined legend
                lines1, labels1 = ax6.get_legend_handles_labels()
                lines2, labels2 = ax6_twin.get_legend_handles_labels()
                ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax6.legend()
                
        ax6.set_xlabel('Simulation Steps')
        ax6.set_ylabel('Average Reward per Agent', color='g')
        ax6.tick_params(axis='y', labelcolor='g')
        ax6.set_title('Reward Efficiency vs Safety')
        ax6.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"episode_{episode_num}_step_analysis.png"
        plot_path = self.output_dir / plot_filename
            
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
            
        logger.info(f"Step analysis plot saved: {plot_path}")
        return plot_path
    
    def plot_episode_analysis(self, episode_data: Dict, episode_num: int = 0, fixed_dt: float = None) -> Path:
        """
        Generate single episode summary visualization.
        
        Args:
            episode_data: Episode statistics dictionary
            episode_num: Episode number for filename
            fixed_dt: Fixed time step for simulation time calculation
            
        Returns:
            Path to saved plot file
        """
        if not episode_data:
            logger.warning("No episode data provided for plotting")
            return None
            
        # Create episode summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Episode {episode_num} Summary - {self.scenario_name} - {self.agent_name}', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Final Performance Metrics
        ax1 = axes[0, 0]
        metrics = ['Success', 'Collision', 'Active']
        values = [episode_data['success'], episode_data['collision'], episode_data['active_agents']]
        colors = ['green', 'red', 'blue']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Vehicle Count')
        ax1.set_title('Final Vehicle Counts')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(value)}', ha='center', va='bottom')
        
        # Plot 2: Success vs Collision vs Timeout Rates
        ax2 = axes[0, 1]
        rates = ['Success', 'Collision', 'Timeout']
        rate_values = [
            episode_data.get('success_rate', 0),
            episode_data.get('collision_rate', 0),
            episode_data.get('timeout_rate', 0)
        ]
        rate_colors = ['green', 'red', 'orange']

        # Only include non-zero rates in pie chart
        filtered_rates = []
        filtered_values = []
        filtered_colors = []
        for rate, value, color in zip(rates, rate_values, rate_colors):
            if value > 0:
                filtered_rates.append(rate)
                filtered_values.append(value)
                filtered_colors.append(color)

        if filtered_values:
            wedges, texts, autotexts = ax2.pie(filtered_values, labels=filtered_rates, colors=filtered_colors,
                                              autopct='%1.1f%%', startangle=90)
        ax2.set_title('Vehicle Outcome Distribution')
        
        # Plot 3: Episode Throughput
        ax3 = axes[1, 0]
        ax3.bar(['Throughput'], [episode_data['throughput']], color='orange', alpha=0.7)
        ax3.set_ylabel('Vehicles per Hour')
        ax3.set_title('Episode Throughput')
        ax3.text(0, episode_data['throughput'] + 1, f"{episode_data['throughput']:.1f} vph", 
                ha='center', va='bottom')
        
        # Plot 4: Episode Statistics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')  # Turn off axis
        
        # Create summary text
        total_vehicles = episode_data['total_vehicles']
        steps = episode_data['step']
        sim_time = steps * fixed_dt / 60 if fixed_dt else 0  # minutes
        
        summary_text = f"""Episode {episode_num} Statistics:
        
Total Steps: {steps}
Simulation Time: {sim_time:.1f} min
Total Vehicles: {total_vehicles}
Success: {episode_data['success']} ({episode_data['success_rate']:.1f}%)
Collision: {episode_data['collision']} ({episode_data['collision_rate']:.1f}%)
Active: {episode_data['active_agents']}
Throughput: {episode_data['throughput']:.1f} vph"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"episode_{episode_num}_analysis.png"
        plot_path = self.output_dir / plot_filename
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        logger.info(f"Episode analysis plot saved: {plot_path}")
        return plot_path
    
    def plot_episode_comparison(self, episode_history: List[Dict]) -> Path:
        """
        Generate multi-episode comparison plots for MARL training analysis.
        
        Args:
            episode_history: List of episode statistics dictionaries
            
        Returns:
            Path to saved plot file
        """
        if not episode_history or len(episode_history) < 2:
            logger.warning("Episode comparison requires at least 2 episodes of data")
            return None
            
        # Extract data for all episodes
        episodes = [i for i in range(len(episode_history))]
        success_rates = [ep['success_rate'] for ep in episode_history]
        collision_rates = [ep['collision_rate'] for ep in episode_history]
        throughputs = [ep['throughput'] for ep in episode_history]
        total_successes = [ep['success'] for ep in episode_history]
        total_collisions = [ep['collision'] for ep in episode_history]
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Multi-Episode Comparison - {self.scenario_name} - {self.agent_name}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Success Rate Trend
        ax1 = axes[0, 0]
        ax1.plot(episodes, success_rates, 'o-', color='green', linewidth=2, markersize=6, 
                label='Success Rate')
        
        # Add trend line if enough data points
        if len(episodes) > 2:
            z = np.polyfit(episodes, success_rates, 1)
            p = np.poly1d(z)
            ax1.plot(episodes, p(episodes), '--', color='lightgreen', alpha=0.8, 
                    label=f'Trend: {z[0]:.2f}%/episode')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate Improvement Over Episodes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Collision Rate Trend
        ax2 = axes[0, 1]
        ax2.plot(episodes, collision_rates, 'o-', color='red', linewidth=2, markersize=6, 
                label='Collision Rate')
        
        # Add trend line
        if len(episodes) > 2:
            z = np.polyfit(episodes, collision_rates, 1)
            p = np.poly1d(z)
            ax2.plot(episodes, p(episodes), '--', color='lightcoral', alpha=0.8, 
                    label=f'Trend: {z[0]:.2f}%/episode')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Collision Rate (%)')
        ax2.set_title('Collision Rate Change Over Episodes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Throughput Performance
        ax3 = axes[1, 0]
        ax3.plot(episodes, throughputs, 'o-', color='blue', linewidth=2, markersize=6, 
                label='Throughput')
        ax3.fill_between(episodes, throughputs, alpha=0.3, color='blue')
        
        # Mark best performance
        best_idx = np.argmax(throughputs)
        best_throughput = throughputs[best_idx]
        ax3.scatter(episodes[best_idx], best_throughput, color='gold', s=150, zorder=5, 
                   label=f'Best: {best_throughput:.1f} vph')
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Throughput (vph)')
        ax3.set_title('Throughput Performance Across Episodes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative Performance Comparison
        ax4 = axes[1, 1]
        
        # Stacked bar chart of success vs collisions
        width = 0.6
        ax4.bar(episodes, total_successes, width, label='Success', color='green', alpha=0.7)
        ax4.bar(episodes, total_collisions, width, bottom=total_successes, 
               label='Collision', color='red', alpha=0.7)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Vehicle Count')
        ax4.set_title('Success vs Collision Count per Episode')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot with episode range to prevent overwriting
        episode_range = f"0-{len(episode_history)-1}"
        plot_filename = f"episodes_{self.scenario_name}_{self.agent_name}_comparison_{episode_range}.png"
        plot_path = self.output_dir / plot_filename
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        logger.info(f"Episode comparison plot saved: {plot_path} (episodes {episode_range})")
        return plot_path
    
    @staticmethod
    def create_summary_plot(data: Dict[str, List], title: str, output_path: Path) -> Path:
        """
        Create a summary comparison plot for multiple agents or scenarios.
        
        Args:
            data: Dictionary with agent names as keys and metric lists as values
            title: Plot title
            output_path: Path to save the plot
            
        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Plot success rates
        ax1 = axes[0, 0]
        for i, (agent_name, metrics) in enumerate(data.items()):
            if 'success_rates' in metrics:
                episodes = list(range(len(metrics['success_rates'])))
                ax1.plot(episodes, metrics['success_rates'], 'o-', 
                        color=colors[i % len(colors)], label=agent_name, linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot collision rates
        ax2 = axes[0, 1]
        for i, (agent_name, metrics) in enumerate(data.items()):
            if 'collision_rates' in metrics:
                episodes = list(range(len(metrics['collision_rates'])))
                ax2.plot(episodes, metrics['collision_rates'], 'o-', 
                        color=colors[i % len(colors)], label=agent_name, linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Collision Rate (%)')
        ax2.set_title('Collision Rate Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot throughput
        ax3 = axes[1, 0]
        for i, (agent_name, metrics) in enumerate(data.items()):
            if 'throughputs' in metrics:
                episodes = list(range(len(metrics['throughputs'])))
                ax3.plot(episodes, metrics['throughputs'], 'o-', 
                        color=colors[i % len(colors)], label=agent_name, linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Throughput (vph)')
        ax3.set_title('Throughput Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot average performance summary
        ax4 = axes[1, 1]
        agent_names = list(data.keys())
        avg_success_rates = []
        avg_throughputs = []
        
        for agent_name, metrics in data.items():
            if 'success_rates' in metrics and metrics['success_rates']:
                avg_success_rates.append(np.mean(metrics['success_rates']))
            else:
                avg_success_rates.append(0)
                
            if 'throughputs' in metrics and metrics['throughputs']:
                avg_throughputs.append(np.mean(metrics['throughputs']))
            else:
                avg_throughputs.append(0)
        
        x_pos = np.arange(len(agent_names))
        width = 0.35
        
        ax4_twin = ax4.twinx()
        bars1 = ax4.bar(x_pos - width/2, avg_success_rates, width, 
                       label='Avg Success Rate (%)', color='green', alpha=0.7)
        bars2 = ax4_twin.bar(x_pos + width/2, avg_throughputs, width, 
                           label='Avg Throughput (vph)', color='blue', alpha=0.7)
        
        ax4.set_xlabel('Agent')
        ax4.set_ylabel('Success Rate (%)', color='green')
        ax4_twin.set_ylabel('Throughput (vph)', color='blue')
        ax4.set_title('Average Performance Summary')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(agent_names, rotation=45, ha='right')
        
        # Add legends
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Summary comparison plot saved: {output_path}")
        return output_path

    def plot_learning_progress(self, episode_history: List[Dict],
                                window_size: int = 100) -> Path:
        """
        Generate comprehensive learning progress visualization with improvement metrics.

        This is designed for paper-ready figures showing:
        - Smoothed learning curves (moving averages)
        - Performance improvement table
        - Time-to-target statistics
        - Relative improvement percentages

        Args:
            episode_history: List of episode statistics dictionaries
            window_size: Window size for moving averages (default 100)

        Returns:
            Path to saved plot file
        """
        if not episode_history or len(episode_history) < window_size:
            logger.warning(f"Learning progress plot requires at least {window_size} episodes")
            return None

        # Extract data
        num_episodes = len(episode_history)
        episodes = list(range(num_episodes))
        success_rates = [ep['success_rate'] for ep in episode_history]
        collision_rates = [ep['collision_rate'] for ep in episode_history]
        throughputs = [ep['throughput'] for ep in episode_history]
        cumulative_rewards = [ep.get('cumulative_reward', 0) for ep in episode_history]

        # Calculate moving averages
        def moving_average(data, window):
            return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]

        success_ma = moving_average(success_rates, window_size)
        collision_ma = moving_average(collision_rates, window_size)
        throughput_ma = moving_average(throughputs, window_size)
        reward_ma = moving_average(cumulative_rewards, window_size)

        # Calculate improvement statistics
        baseline_window = min(window_size, num_episodes // 4)  # First quarter or window_size
        final_window = min(window_size, num_episodes // 4)  # Last quarter or window_size

        baseline_success = np.mean(success_rates[:baseline_window])
        final_success = np.mean(success_rates[-final_window:])
        baseline_collision = np.mean(collision_rates[:baseline_window])
        final_collision = np.mean(collision_rates[-final_window:])
        baseline_throughput = np.mean(throughputs[:baseline_window])
        final_throughput = np.mean(throughputs[-final_window:])
        baseline_reward = np.mean(cumulative_rewards[:baseline_window])
        final_reward = np.mean(cumulative_rewards[-final_window:])

        # Calculate percentage improvements
        def calc_improvement_pct(baseline, final):
            if abs(baseline) < 0.01:
                return 0.0
            return (final - baseline) / abs(baseline) * 100

        success_improvement = calc_improvement_pct(baseline_success, final_success)
        collision_reduction = calc_improvement_pct(baseline_collision, final_collision) * -1  # Negative is good
        throughput_improvement = calc_improvement_pct(baseline_throughput, final_throughput)
        reward_improvement = calc_improvement_pct(baseline_reward, final_reward)

        # Calculate time-to-target (episodes to reach X% success rate)
        milestones = [50, 60, 70, 80, 90, 95]
        time_to_target = {}
        for target in milestones:
            for i, rate in enumerate(success_rates):
                if rate >= target:
                    time_to_target[target] = i
                    break
            if target not in time_to_target:
                time_to_target[target] = None  # Never reached

        # Create figure with subplots and table
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.3)

        fig.suptitle(f'Learning Progress Analysis - {self.scenario_name} - {self.agent_name}\n'
                    f'({num_episodes} Episodes, {window_size}-Episode Moving Average)',
                    fontsize=14, fontweight='bold')

        # ============ Plot 1: Success Rate with Moving Average ============
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(episodes, success_rates, alpha=0.3, color='green', linewidth=0.5, label='Raw')
        ax1.plot(episodes, success_ma, color='green', linewidth=2, label=f'MA-{window_size}')
        ax1.axhline(y=baseline_success, color='gray', linestyle='--', alpha=0.5, label=f'Baseline: {baseline_success:.1f}%')
        ax1.axhline(y=final_success, color='darkgreen', linestyle='--', alpha=0.7, label=f'Final: {final_success:.1f}%')
        ax1.fill_between(episodes, success_ma, alpha=0.2, color='green')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title(f'Success Rate (Δ{success_improvement:+.1f}%)')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        # ============ Plot 2: Collision Rate with Moving Average ============
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(episodes, collision_rates, alpha=0.3, color='red', linewidth=0.5, label='Raw')
        ax2.plot(episodes, collision_ma, color='red', linewidth=2, label=f'MA-{window_size}')
        ax2.axhline(y=baseline_collision, color='gray', linestyle='--', alpha=0.5, label=f'Baseline: {baseline_collision:.1f}%')
        ax2.axhline(y=final_collision, color='darkred', linestyle='--', alpha=0.7, label=f'Final: {final_collision:.1f}%')
        ax2.fill_between(episodes, collision_ma, alpha=0.2, color='red')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Collision Rate (%)')
        ax2.set_title(f'Collision Rate (Δ{-collision_reduction:+.1f}%)')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(collision_rates) * 1.1 + 5)

        # ============ Plot 3: Throughput with Moving Average ============
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(episodes, throughputs, alpha=0.3, color='blue', linewidth=0.5, label='Raw')
        ax3.plot(episodes, throughput_ma, color='blue', linewidth=2, label=f'MA-{window_size}')
        ax3.axhline(y=baseline_throughput, color='gray', linestyle='--', alpha=0.5, label=f'Baseline: {baseline_throughput:.0f}')
        ax3.axhline(y=final_throughput, color='darkblue', linestyle='--', alpha=0.7, label=f'Final: {final_throughput:.0f}')
        ax3.fill_between(episodes, throughput_ma, alpha=0.2, color='blue')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Throughput (vph)')
        ax3.set_title(f'Throughput (Δ{throughput_improvement:+.1f}%)')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # ============ Plot 4: Cumulative Reward with Moving Average ============
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(episodes, cumulative_rewards, alpha=0.3, color='purple', linewidth=0.5, label='Raw')
        ax4.plot(episodes, reward_ma, color='purple', linewidth=2, label=f'MA-{window_size}')
        ax4.axhline(y=baseline_reward, color='gray', linestyle='--', alpha=0.5)
        ax4.axhline(y=final_reward, color='darkviolet', linestyle='--', alpha=0.7)
        ax4.fill_between(episodes, reward_ma, alpha=0.2, color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Cumulative Reward')
        ax4.set_title(f'Episode Reward (Δ{reward_improvement:+.1f}%)')
        ax4.legend(loc='lower right', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # ============ Plot 5: Success vs Collision Tradeoff ============
        ax5 = fig.add_subplot(gs[1, 1])
        # Color by episode (gradient from blue to red)
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_episodes))
        scatter = ax5.scatter(collision_rates, success_rates, c=episodes, cmap='coolwarm',
                             alpha=0.6, s=20, label='Episodes')
        ax5.scatter([baseline_collision], [baseline_success], color='blue', s=200,
                   marker='s', edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax5.scatter([final_collision], [final_success], color='red', s=200,
                   marker='*', edgecolors='black', linewidths=2, label='End', zorder=5)
        # Draw arrow from start to end
        ax5.annotate('', xy=(final_collision, final_success),
                    xytext=(baseline_collision, baseline_success),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax5.set_xlabel('Collision Rate (%)')
        ax5.set_ylabel('Success Rate (%)')
        ax5.set_title('Learning Trajectory (Success vs Collision)')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Episode')

        # ============ Plot 6: Time-to-Target Bar Chart ============
        ax6 = fig.add_subplot(gs[1, 2])
        reached_targets = [(t, e) for t, e in time_to_target.items() if e is not None]
        if reached_targets:
            targets, episodes_to_reach = zip(*reached_targets)
            bars = ax6.barh([f'{t}%' for t in targets], episodes_to_reach, color='teal', alpha=0.7)
            ax6.set_xlabel('Episodes to Reach Target')
            ax6.set_ylabel('Success Rate Target')
            ax6.set_title('Time-to-Target Analysis')
            # Add value labels
            for bar, val in zip(bars, episodes_to_reach):
                ax6.text(val + max(episodes_to_reach) * 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val}', va='center', fontsize=9)
            ax6.grid(True, alpha=0.3, axis='x')
        else:
            ax6.text(0.5, 0.5, 'No targets reached yet', transform=ax6.transAxes,
                    ha='center', va='center', fontsize=12)
            ax6.set_title('Time-to-Target Analysis')

        # ============ Performance Improvement Table ============
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')

        # Create table data
        table_data = [
            ['Metric', 'Baseline\n(First 100 ep)', 'Final\n(Last 100 ep)', 'Absolute Δ', 'Relative Δ', 'Status'],
            ['Success Rate (%)', f'{baseline_success:.1f}', f'{final_success:.1f}',
             f'{final_success - baseline_success:+.1f}', f'{success_improvement:+.1f}%',
             '✓ Improved' if success_improvement > 0 else '✗ Declined'],
            ['Collision Rate (%)', f'{baseline_collision:.1f}', f'{final_collision:.1f}',
             f'{final_collision - baseline_collision:+.1f}', f'{-collision_reduction:+.1f}%',
             '✓ Reduced' if collision_reduction > 0 else '✗ Increased'],
            ['Throughput (vph)', f'{baseline_throughput:.0f}', f'{final_throughput:.0f}',
             f'{final_throughput - baseline_throughput:+.0f}', f'{throughput_improvement:+.1f}%',
             '✓ Improved' if throughput_improvement > 0 else '✗ Declined'],
            ['Episode Reward', f'{baseline_reward:.0f}', f'{final_reward:.0f}',
             f'{final_reward - baseline_reward:+.0f}', f'{reward_improvement:+.1f}%',
             '✓ Improved' if reward_improvement > 0 else '✗ Declined'],
        ]

        # Add time-to-target summary row
        reached_str = ', '.join([f'{t}%@ep{e}' for t, e in time_to_target.items() if e is not None])
        if not reached_str:
            reached_str = 'None reached'
        table_data.append(['Time-to-Target', '-', '-', '-', '-', reached_str])

        # Create and style table
        table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                               loc='center', cellLoc='center',
                               colColours=['lightgray']*6)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Color code the status column
        for i in range(1, len(table_data)):
            cell = table[(i-1, 5)]
            if '✓' in table_data[i][5]:
                cell.set_facecolor('#90EE90')  # Light green
            elif '✗' in table_data[i][5]:
                cell.set_facecolor('#FFB6C1')  # Light red
            else:
                cell.set_facecolor('#FFFACD')  # Light yellow

        ax_table.set_title('Performance Improvement Summary', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save plot
        plot_filename = f"learning_progress_{self.scenario_name}_{self.agent_name}.png"
        plot_path = self.output_dir / plot_filename

        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Learning progress plot saved: {plot_path}")
        return plot_path

    def generate_improvement_report(self, episode_history: List[Dict],
                                     window_size: int = 100) -> Dict[str, Any]:
        """
        Generate a comprehensive improvement statistics report.

        Returns a dictionary with all improvement metrics for paper/report use.

        Args:
            episode_history: List of episode statistics dictionaries
            window_size: Window size for baseline/final calculations

        Returns:
            Dictionary with improvement statistics
        """
        if not episode_history or len(episode_history) < 2:
            return {'error': 'Insufficient data for improvement analysis'}

        num_episodes = len(episode_history)

        # Extract data
        success_rates = [ep['success_rate'] for ep in episode_history]
        collision_rates = [ep['collision_rate'] for ep in episode_history]
        throughputs = [ep['throughput'] for ep in episode_history]
        cumulative_rewards = [ep.get('cumulative_reward', 0) for ep in episode_history]

        # Calculate baseline and final windows
        baseline_window = min(window_size, num_episodes // 4)
        final_window = min(window_size, num_episodes // 4)

        def calc_stats(data, baseline_w, final_w):
            baseline = data[:baseline_w]
            final = data[-final_w:]

            baseline_mean = np.mean(baseline)
            final_mean = np.mean(final)

            improvement_abs = final_mean - baseline_mean
            improvement_pct = (improvement_abs / abs(baseline_mean) * 100) if abs(baseline_mean) > 0.01 else 0

            return {
                'baseline_mean': baseline_mean,
                'baseline_std': np.std(baseline),
                'final_mean': final_mean,
                'final_std': np.std(final),
                'improvement_absolute': improvement_abs,
                'improvement_percent': improvement_pct,
                'min': min(data),
                'max': max(data),
                'overall_mean': np.mean(data),
                'overall_std': np.std(data),
            }

        # Time-to-target analysis
        milestones = [50, 60, 70, 80, 90, 95]
        time_to_target = {}
        for target in milestones:
            for i, rate in enumerate(success_rates):
                if rate >= target:
                    time_to_target[target] = i
                    break
            if target not in time_to_target:
                time_to_target[target] = None

        report = {
            'num_episodes': num_episodes,
            'window_size': window_size,
            'success_rate': calc_stats(success_rates, baseline_window, final_window),
            'collision_rate': calc_stats(collision_rates, baseline_window, final_window),
            'throughput': calc_stats(throughputs, baseline_window, final_window),
            'cumulative_reward': calc_stats(cumulative_rewards, baseline_window, final_window),
            'time_to_target': time_to_target,
            'scenario_name': self.scenario_name,
            'agent_name': self.agent_name,
        }

        # Add summary statistics
        report['summary'] = {
            'success_improvement_pct': report['success_rate']['improvement_percent'],
            'collision_reduction_pct': -report['collision_rate']['improvement_percent'],
            'throughput_improvement_pct': report['throughput']['improvement_percent'],
            'reward_improvement_pct': report['cumulative_reward']['improvement_percent'],
            'best_success_rate': max(success_rates),
            'best_throughput': max(throughputs),
            'lowest_collision_rate': min(collision_rates),
            'episodes_to_80pct_success': time_to_target.get(80),
            'episodes_to_90pct_success': time_to_target.get(90),
        }

        return report