'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 12:00:00
FilePath     : /OpenCDA-MARL/opencda_marl/envs/cross_agent_evaluator.py
Description  : Cross-agent comparison tool for evaluation system
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from loguru import logger

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opencda_marl.envs.evaluation_plots import EvaluationPlotter


class CrossAgentEvaluator:
    """
    Cross-agent comparison tool for analyzing multiple agents' performance.

    Usage:
        evaluator = CrossAgentEvaluator("/evaluation_outputs/intersection/2025_09_05_12")
        evaluator.generate_comparison_report()
    """

    def __init__(self, evaluation_dir: str):
        """
        Initialize cross-agent evaluator.

        Args:
            evaluation_dir: Path to evaluation directory containing agent subdirectories
                          Example: "/evaluation_outputs/intersection/2025_09_05_12"
        """
        self.evaluation_dir = Path(evaluation_dir)
        self.scenario_name = self.evaluation_dir.parent.name  # Extract scenario from path
        self.timestamp = self.evaluation_dir.name

        if not self.evaluation_dir.exists():
            raise FileNotFoundError(
                f"Evaluation directory not found: {evaluation_dir}")

        # Discover agent directories
        self.agent_dirs = self._discover_agents()
        if not self.agent_dirs:
            raise ValueError(
                f"No agent data found in directory: {evaluation_dir}")

        logger.info(
            f"Discovered {len(self.agent_dirs)} agents: {list(self.agent_dirs.keys())}")

        # Storage for loaded data
        self.agent_data = {}
        self.comparison_summary = {}

        # Output directory for comparison results
        self.output_dir = self.evaluation_dir / "cross_agent_comparison"
        self.output_dir.mkdir(exist_ok=True)

    def _discover_agents(self) -> Dict[str, Path]:
        """
        Discover agent subdirectories in evaluation directory.

        Returns:
            Dictionary mapping agent names to their data directories
        """
        agent_dirs = {}

        for item in self.evaluation_dir.iterdir():
            if item.is_dir() and item.name != "cross_agent_comparison":
                data_dir = item / "data"
                if data_dir.exists():
                    agent_dirs[item.name] = data_dir
                    logger.debug(f"Found agent: {item.name} at {data_dir}")

        return agent_dirs

    def load_agent_data(self, agent_name: str) -> Dict[str, Any]:
        """
        Load evaluation data for a specific agent.

        Args:
            agent_name: Name of agent to load

        Returns:
            Dictionary containing agent's evaluation data
        """
        if agent_name not in self.agent_dirs:
            raise ValueError(f"Agent {agent_name} not found")

        data_dir = self.agent_dirs[agent_name]
        agent_data = {
            'name': agent_name,
            'episode_summaries': [],
            'step_histories': {},
            'final_summary': None
        }

        try:
            # Load episode summaries from CSV
            csv_path = data_dir / "episodes_summary.csv"
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert numeric strings to appropriate types
                        processed_row = {}
                        for key, value in row.items():
                            try:
                                if '.' in value:
                                    processed_row[key] = float(value)
                                else:
                                    processed_row[key] = int(value)
                            except (ValueError, TypeError):
                                processed_row[key] = value
                        agent_data['episode_summaries'].append(processed_row)

                logger.info(
                    f"Loaded {len(agent_data['episode_summaries'])} episode summaries for {agent_name}")

            # Load step histories (NPZ files)
            for npz_file in data_dir.glob("episode_*_steps.npz"):
                episode_num = int(npz_file.stem.split(
                    '_')[1])  # Extract episode number
                try:
                    with np.load(npz_file) as data:
                        agent_data['step_histories'][episode_num] = {
                            'episode': int(data['episode']) if 'episode' in data else episode_num,
                            'scenario_name': str(data['scenario_name']) if 'scenario_name' in data else self.scenario_name,
                            'agent_name': str(data['agent_name']) if 'agent_name' in data else agent_name,
                            'steps': data['steps'],
                            'success': data['success'],
                            'collision': data['collision'],
                            'active_agents': data['active_agents'],
                            'total_vehicles': data['total_vehicles'],
                            'success_rate': data['success_rate'],
                            'collision_rate': data['collision_rate'],
                            'throughput': data['throughput'],
                            # Add reward data if available (backward compatibility)
                            'step_reward': data.get('step_reward', np.zeros_like(data['steps'])),
                            'cumulative_reward': data.get('cumulative_reward', np.zeros_like(data['steps'])),
                            'avg_reward_per_agent': data.get('avg_reward_per_agent', np.zeros_like(data['steps']))
                        }
                except Exception as e:
                    logger.warning(
                        f"Failed to load step history for episode {episode_num}: {e}")

            logger.info(
                f"Loaded step histories for {len(agent_data['step_histories'])} episodes for {agent_name}")

            # Load final summary if available
            summary_path = data_dir / "final_summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    agent_data['final_summary'] = json.load(f)
                logger.debug(f"Loaded final summary for {agent_name}")

        except Exception as e:
            logger.error(f"Error loading data for agent {agent_name}: {e}")
            raise

        return agent_data

    def load_all_agents(self) -> Dict[str, Dict]:
        """
        Load evaluation data for all discovered agents.

        Returns:
            Dictionary mapping agent names to their loaded data
        """
        logger.info("Loading data for all agents...")

        for agent_name in self.agent_dirs.keys():
            try:
                self.agent_data[agent_name] = self.load_agent_data(agent_name)
            except Exception as e:
                logger.error(
                    f"Failed to load data for agent {agent_name}: {e}")
                continue

        logger.info(
            f"Successfully loaded data for {len(self.agent_data)} agents")
        return self.agent_data

    def compare_performance(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance comparison across all agents.

        Returns:
            Dictionary containing comparison statistics
        """
        if not self.agent_data:
            self.load_all_agents()

        comparison = {
            'scenario': self.scenario_name,
            'timestamp': self.timestamp,
            'agents': list(self.agent_data.keys()),
            'num_agents': len(self.agent_data),
            'agent_stats': {},
            'rankings': {},
            'best_performers': {},
            'summary_statistics': {}
        }

        # Collect metrics for each agent
        agent_metrics = {}

        for agent_name, data in self.agent_data.items():
            if not data['episode_summaries']:
                logger.warning(f"No episode data for agent {agent_name}")
                continue

            # Extract episode-level metrics
            episodes = data['episode_summaries']
            success_rates = [ep['success_rate'] for ep in episodes]
            collision_rates = [ep['collision_rate'] for ep in episodes]
            throughputs = [ep['throughput'] for ep in episodes]
            total_successes = [ep['success'] for ep in episodes]
            total_collisions = [ep['collision'] for ep in episodes]
            
            # Extract reward metrics if available
            cumulative_rewards = [ep.get('cumulative_reward', 0.0) for ep in episodes]
            step_rewards = [ep.get('step_reward', 0.0) for ep in episodes]
            avg_rewards_per_agent = [ep.get('avg_reward_per_agent', 0.0) for ep in episodes]

            # Calculate statistics
            agent_stats = {
                'num_episodes': len(episodes),
                'avg_success_rate': np.mean(success_rates),
                'std_success_rate': np.std(success_rates),
                'best_success_rate': np.max(success_rates),
                'worst_success_rate': np.min(success_rates),
                'final_success_rate': success_rates[-1] if success_rates else 0,
                'avg_collision_rate': np.mean(collision_rates),
                'std_collision_rate': np.std(collision_rates),
                # Lower is better
                'best_collision_rate': np.min(collision_rates),
                'worst_collision_rate': np.max(collision_rates),
                'final_collision_rate': collision_rates[-1] if collision_rates else 0,
                'avg_throughput': np.mean(throughputs),
                'std_throughput': np.std(throughputs),
                'best_throughput': np.max(throughputs),
                'worst_throughput': np.min(throughputs),
                'final_throughput': throughputs[-1] if throughputs else 0,
                'total_success': sum(total_successes),
                'total_collision': sum(total_collisions),
                'total_vehicles': sum(total_successes) + sum(total_collisions),
                # Reward statistics
                'avg_cumulative_reward': np.mean(cumulative_rewards) if cumulative_rewards else 0,
                'std_cumulative_reward': np.std(cumulative_rewards) if cumulative_rewards else 0,
                'best_cumulative_reward': np.max(cumulative_rewards) if cumulative_rewards else 0,
                'worst_cumulative_reward': np.min(cumulative_rewards) if cumulative_rewards else 0,
                'final_cumulative_reward': cumulative_rewards[-1] if cumulative_rewards else 0,
                'avg_reward_per_agent': np.mean(avg_rewards_per_agent) if avg_rewards_per_agent else 0,
                'total_reward': sum(cumulative_rewards) if cumulative_rewards else 0
            }

            # Calculate improvement trends if multiple episodes
            if len(episodes) > 1:
                success_trend = np.polyfit(
                    range(len(success_rates)), success_rates, 1)[0]
                collision_trend = np.polyfit(
                    range(len(collision_rates)), collision_rates, 1)[0]
                throughput_trend = np.polyfit(
                    range(len(throughputs)), throughputs, 1)[0]
                
                # Calculate reward trends if available
                reward_trend = 0.0
                if cumulative_rewards and len(cumulative_rewards) > 1:
                    reward_trend = np.polyfit(
                        range(len(cumulative_rewards)), cumulative_rewards, 1)[0]

                agent_stats.update({
                    'success_rate_trend': success_trend,  # % per episode
                    'collision_rate_trend': collision_trend,  # % per episode
                    'throughput_trend': throughput_trend,  # vph per episode
                    'reward_trend': reward_trend,  # reward per episode
                    'improvement_factor': success_rates[-1] / max(success_rates[0], 0.1),
                    'reward_efficiency': agent_stats['avg_cumulative_reward'] / max(agent_stats['avg_collision_rate'], 0.1)  # reward per collision rate
                })

            comparison['agent_stats'][agent_name] = agent_stats
            agent_metrics[agent_name] = {
                'success_rates': success_rates,
                'collision_rates': collision_rates,
                'throughputs': throughputs,
                'cumulative_rewards': cumulative_rewards,
                'avg_rewards_per_agent': avg_rewards_per_agent
            }

        # Generate rankings
        comparison['rankings'] = self._calculate_rankings(
            comparison['agent_stats'])

        # Identify best performers
        comparison['best_performers'] = self._identify_best_performers(
            comparison['agent_stats'])

        # Generate summary statistics
        comparison['summary_statistics'] = self._calculate_summary_statistics(
            comparison['agent_stats'])

        # Store metrics for plotting
        self.agent_metrics = agent_metrics
        self.comparison_summary = comparison

        return comparison

    def _calculate_rankings(self, agent_stats: Dict) -> Dict[str, List[Tuple[str, float]]]:
        """Calculate rankings for different performance metrics."""
        rankings = {}

        # Success rate ranking (higher is better)
        success_ranking = sorted(agent_stats.items(),
                                 key=lambda x: x[1]['avg_success_rate'], reverse=True)
        rankings['success_rate'] = [
            (name, stats['avg_success_rate']) for name, stats in success_ranking]

        # Collision rate ranking (lower is better)
        collision_ranking = sorted(agent_stats.items(),
                                   key=lambda x: x[1]['avg_collision_rate'])
        rankings['collision_rate'] = [
            (name, stats['avg_collision_rate']) for name, stats in collision_ranking]

        # Throughput ranking (higher is better)
        throughput_ranking = sorted(agent_stats.items(),
                                    key=lambda x: x[1]['avg_throughput'], reverse=True)
        rankings['throughput'] = [(name, stats['avg_throughput'])
                                  for name, stats in throughput_ranking]

        # Cumulative reward ranking (higher is better)
        reward_ranking = sorted(agent_stats.items(),
                               key=lambda x: x[1]['avg_cumulative_reward'], reverse=True)
        rankings['cumulative_reward'] = [(name, stats['avg_cumulative_reward'])
                                        for name, stats in reward_ranking]

        # Reward efficiency ranking (higher is better) - reward per collision rate
        reward_efficiency_ranking = sorted(agent_stats.items(),
                                         key=lambda x: x[1].get('reward_efficiency', 0), reverse=True)
        rankings['reward_efficiency'] = [(name, stats.get('reward_efficiency', 0))
                                        for name, stats in reward_efficiency_ranking]

        return rankings

    def _identify_best_performers(self, agent_stats: Dict) -> Dict[str, str]:
        """Identify best performing agents for each metric."""
        best_performers = {}

        # Best success rate
        best_success = max(agent_stats.items(),
                           key=lambda x: x[1]['avg_success_rate'])
        best_performers['success_rate'] = best_success[0]

        # Best collision rate (lowest)
        best_collision = min(agent_stats.items(),
                             key=lambda x: x[1]['avg_collision_rate'])
        best_performers['collision_rate'] = best_collision[0]

        # Best throughput
        best_throughput = max(agent_stats.items(),
                              key=lambda x: x[1]['avg_throughput'])
        best_performers['throughput'] = best_throughput[0]

        # Best cumulative reward
        best_reward = max(agent_stats.items(),
                         key=lambda x: x[1]['avg_cumulative_reward'])
        best_performers['cumulative_reward'] = best_reward[0]

        # Best reward efficiency
        best_efficiency = max(agent_stats.items(),
                             key=lambda x: x[1].get('reward_efficiency', 0))
        best_performers['reward_efficiency'] = best_efficiency[0]

        return best_performers

    def _calculate_summary_statistics(self, agent_stats: Dict) -> Dict[str, Any]:
        """Calculate cross-agent summary statistics."""
        all_success_rates = [stats['avg_success_rate']
                             for stats in agent_stats.values()]
        all_collision_rates = [stats['avg_collision_rate']
                               for stats in agent_stats.values()]
        all_throughputs = [stats['avg_throughput']
                           for stats in agent_stats.values()]
        all_cumulative_rewards = [stats['avg_cumulative_reward']
                                 for stats in agent_stats.values()]
        all_reward_efficiencies = [stats.get('reward_efficiency', 0)
                                  for stats in agent_stats.values()]

        return {
            'success_rate': {
                'mean': np.mean(all_success_rates),
                'std': np.std(all_success_rates),
                'min': np.min(all_success_rates),
                'max': np.max(all_success_rates),
                'range': np.max(all_success_rates) - np.min(all_success_rates)
            },
            'collision_rate': {
                'mean': np.mean(all_collision_rates),
                'std': np.std(all_collision_rates),
                'min': np.min(all_collision_rates),
                'max': np.max(all_collision_rates),
                'range': np.max(all_collision_rates) - np.min(all_collision_rates)
            },
            'throughput': {
                'mean': np.mean(all_throughputs),
                'std': np.std(all_throughputs),
                'min': np.min(all_throughputs),
                'max': np.max(all_throughputs),
                'range': np.max(all_throughputs) - np.min(all_throughputs)
            },
            'cumulative_reward': {
                'mean': np.mean(all_cumulative_rewards),
                'std': np.std(all_cumulative_rewards),
                'min': np.min(all_cumulative_rewards),
                'max': np.max(all_cumulative_rewards),
                'range': np.max(all_cumulative_rewards) - np.min(all_cumulative_rewards)
            },
            'reward_efficiency': {
                'mean': np.mean(all_reward_efficiencies),
                'std': np.std(all_reward_efficiencies),
                'min': np.min(all_reward_efficiencies),
                'max': np.max(all_reward_efficiencies),
                'range': np.max(all_reward_efficiencies) - np.min(all_reward_efficiencies)
            }
        }

    def generate_comparison_plots(self) -> List[Path]:
        """
        Generate comprehensive comparison plots.

        Returns:
            List of paths to generated plot files
        """
        if not hasattr(self, 'agent_metrics') or not self.agent_metrics:
            self.compare_performance()

        plot_paths = []

        # Create main comparison plot
        title = f"Cross-Agent Performance Comparison - {self.scenario_name} ({self.timestamp})"
        main_plot_path = self.output_dir / "agent_comparison.png"

        plot_path = EvaluationPlotter.create_summary_plot(
            self.agent_metrics, title, main_plot_path
        )
        plot_paths.append(plot_path)

        # Create unified behavior comparison plot if step histories available
        agents_with_histories = {name: data for name, data in self.agent_data.items() 
                               if data['step_histories']}
        
        if agents_with_histories:
            unified_plot_path = self.output_dir / "unified_behavior_comparison.png"
            behavior_plot = self._create_unified_behavior_plot(
                agents_with_histories, unified_plot_path)
            if behavior_plot:
                plot_paths.append(behavior_plot)

        logger.info(f"Generated {len(plot_paths)} comparison plots")
        return plot_paths

    def _create_unified_behavior_plot(self, agents_with_histories: Dict, output_path: Path) -> Path:
        """
        Create unified behavior comparison plot showing all agents' step-by-step performance.
        
        Args:
            agents_with_histories: Dictionary of agents with their step history data
            output_path: Path where to save the plot
            
        Returns:
            Path to the generated plot file
        """
        try:
            import matplotlib.pyplot as plt
            
            # Aggregate data from all agents and episodes
            unified_data = {}
            
            for agent_name, agent_data in agents_with_histories.items():
                agent_metrics = {
                    'steps': [],
                    'success_rates': [],
                    'collision_rates': [],
                    'throughputs': [],
                    'success_counts': [],
                    'collision_counts': [],
                    'step_rewards': [],
                    'cumulative_rewards': [],
                    'avg_rewards_per_agent': []
                }
                
                # Combine data from all episodes for this agent
                for episode_num, step_data in agent_data['step_histories'].items():
                    steps = step_data['steps']
                    success_rates = step_data['success_rate']
                    collision_rates = step_data['collision_rate']
                    throughputs = step_data['throughput']
                    success_counts = step_data['success']
                    collision_counts = step_data['collision']
                    
                    # Extract reward data if available
                    step_rewards = step_data.get('step_reward', np.zeros_like(steps))
                    cumulative_rewards = step_data.get('cumulative_reward', np.zeros_like(steps))
                    avg_rewards_per_agent = step_data.get('avg_reward_per_agent', np.zeros_like(steps))
                    
                    # Check if rates are already in percentage (0-100) or decimal (0-1) format
                    # If max value <= 1, assume decimal format and convert to percentage
                    if len(success_rates) > 0 and max(success_rates) <= 1:
                        success_rates = success_rates * 100
                        collision_rates = collision_rates * 100
                    
                    agent_metrics['steps'].extend(steps)
                    agent_metrics['success_rates'].extend(success_rates)
                    agent_metrics['collision_rates'].extend(collision_rates)
                    agent_metrics['throughputs'].extend(throughputs)
                    agent_metrics['success_counts'].extend(success_counts)
                    agent_metrics['collision_counts'].extend(collision_counts)
                    agent_metrics['step_rewards'].extend(step_rewards)
                    agent_metrics['cumulative_rewards'].extend(cumulative_rewards)
                    agent_metrics['avg_rewards_per_agent'].extend(avg_rewards_per_agent)
                
                unified_data[agent_name] = agent_metrics
            
            # Create the unified plot with 7 subplots (added reward plots)
            fig, axes = plt.subplots(7, 1, figsize=(12, 18))
            fig.suptitle(f"Unified Behavior Comparison - {self.scenario_name} ({self.timestamp})", 
                        fontsize=14, fontweight='bold')
            
            colors = plt.cm.Set1(range(len(unified_data)))
            
            for i, (agent_name, metrics) in enumerate(unified_data.items()):
                color = colors[i]
                
                # Success Rate plot
                axes[0].scatter(metrics['steps'], metrics['success_rates'], 
                              alpha=0.6, c=[color], label=agent_name, s=20)
                axes[0].set_ylabel('Success Rate (%)')
                axes[0].set_title('Success Rate vs Simulation Steps')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                
                # Success Count plot
                axes[1].scatter(metrics['steps'], metrics['success_counts'], 
                              alpha=0.6, c=[color], label=agent_name, s=20)
                axes[1].set_ylabel('Success Count')
                axes[1].set_title('Success Count vs Simulation Steps')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
                
                # Collision Rate plot
                axes[2].scatter(metrics['steps'], metrics['collision_rates'], 
                              alpha=0.6, c=[color], label=agent_name, s=20)
                axes[2].set_ylabel('Collision Rate (%)')
                axes[2].set_title('Collision Rate vs Simulation Steps')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
                
                # Collision Count plot
                axes[3].scatter(metrics['steps'], metrics['collision_counts'], 
                              alpha=0.6, c=[color], label=agent_name, s=20)
                axes[3].set_ylabel('Collision Count')
                axes[3].set_title('Collision Count vs Simulation Steps')
                axes[3].grid(True, alpha=0.3)
                axes[3].legend()
                
                # Throughput plot
                axes[4].scatter(metrics['steps'], metrics['throughputs'], 
                              alpha=0.6, c=[color], label=agent_name, s=20)
                axes[4].set_ylabel('Throughput (vph)')
                axes[4].set_title('Throughput vs Simulation Steps')
                axes[4].grid(True, alpha=0.3)
                axes[4].legend()
                
                # Step Reward plot
                axes[5].scatter(metrics['steps'], metrics['step_rewards'], 
                              alpha=0.6, c=[color], label=agent_name, s=20)
                axes[5].set_ylabel('Step Reward')
                axes[5].set_title('Step Reward vs Simulation Steps')
                axes[5].grid(True, alpha=0.3)
                axes[5].legend()
                
                # Cumulative Reward plot
                axes[6].scatter(metrics['steps'], metrics['cumulative_rewards'], 
                              alpha=0.6, c=[color], label=agent_name, s=20)
                axes[6].set_ylabel('Cumulative Reward')
                axes[6].set_xlabel('Simulation Steps')
                axes[6].set_title('Cumulative Reward vs Simulation Steps')
                axes[6].grid(True, alpha=0.3)
                axes[6].legend()
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated unified behavior comparison plot: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create unified behavior plot: {e}")
            return None

    def generate_comparison_report(self) -> Path:
        """
        Generate comprehensive comparison report with plots and statistics.

        Returns:
            Path to generated report directory
        """
        logger.info("Generating cross-agent comparison report...")

        # Perform comparison analysis
        comparison = self.compare_performance()

        # Generate plots
        plot_paths = self.generate_comparison_plots()

        # Save comparison data to JSON
        comparison_json_path = self.output_dir / "comparison_summary.json"
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

        # Generate text report
        report_path = self._generate_text_report(comparison)

        # Generate CSV summary
        csv_path = self._generate_csv_summary(comparison)

        logger.info(f"Comparison report generated in: {self.output_dir}")
        logger.info(f"- JSON summary: {comparison_json_path}")
        logger.info(f"- Text report: {report_path}")
        logger.info(f"- CSV summary: {csv_path}")
        logger.info(f"- {len(plot_paths)} plot files")

        return self.output_dir

    def _generate_text_report(self, comparison: Dict) -> Path:
        """Generate human-readable text report."""
        report_path = self.output_dir / "comparison_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Cross-Agent Performance Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Scenario: {comparison['scenario']}\n")
            f.write(f"Timestamp: {comparison['timestamp']}\n")
            f.write(f"Number of Agents: {comparison['num_agents']}\n")
            f.write(f"Agents: {', '.join(comparison['agents'])}\n\n")

            # Rankings section
            f.write("PERFORMANCE RANKINGS\n")
            f.write("-" * 30 + "\n\n")

            f.write("Success Rate Ranking (Higher is Better):\n")
            for i, (agent, score) in enumerate(comparison['rankings']['success_rate'], 1):
                f.write(f"  {i}. {agent}: {score:.1f}%\n")
            f.write("\n")

            f.write("Collision Rate Ranking (Lower is Better):\n")
            for i, (agent, score) in enumerate(comparison['rankings']['collision_rate'], 1):
                f.write(f"  {i}. {agent}: {score:.1f}%\n")
            f.write("\n")

            f.write("Throughput Ranking (Higher is Better):\n")
            for i, (agent, score) in enumerate(comparison['rankings']['throughput'], 1):
                f.write(f"  {i}. {agent}: {score:.1f} vph\n")
            f.write("\n")

            f.write("Cumulative Reward Ranking (Higher is Better):\n")
            for i, (agent, score) in enumerate(comparison['rankings']['cumulative_reward'], 1):
                f.write(f"  {i}. {agent}: {score:.2f}\n")
            f.write("\n")

            f.write("Reward Efficiency Ranking (Higher is Better):\n")
            for i, (agent, score) in enumerate(comparison['rankings']['reward_efficiency'], 1):
                f.write(f"  {i}. {agent}: {score:.2f}\n")
            f.write("\n")

            # Best performers section
            f.write("BEST PERFORMERS\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Best Success Rate: {comparison['best_performers']['success_rate']}\n")
            f.write(
                f"Best Collision Rate: {comparison['best_performers']['collision_rate']}\n")
            f.write(
                f"Best Throughput: {comparison['best_performers']['throughput']}\n")
            f.write(
                f"Best Cumulative Reward: {comparison['best_performers']['cumulative_reward']}\n")
            f.write(
                f"Best Reward Efficiency: {comparison['best_performers']['reward_efficiency']}\n\n")

            # Detailed agent statistics
            f.write("DETAILED AGENT STATISTICS\n")
            f.write("-" * 30 + "\n\n")

            for agent_name, stats in comparison['agent_stats'].items():
                f.write(f"{agent_name}:\n")
                f.write(f"  Episodes: {stats['num_episodes']}\n")
                f.write(
                    f"  Success Rate: {stats['avg_success_rate']:.1f}% +/- {stats['std_success_rate']:.1f}%\n")
                f.write(
                    f"  Collision Rate: {stats['avg_collision_rate']:.1f}% +/- {stats['std_collision_rate']:.1f}%\n")
                f.write(
                    f"  Throughput: {stats['avg_throughput']:.1f} +/- {stats['std_throughput']:.1f} vph\n")
                f.write(
                    f"  Cumulative Reward: {stats['avg_cumulative_reward']:.2f} +/- {stats['std_cumulative_reward']:.2f}\n")
                f.write(
                    f"  Reward Efficiency: {stats.get('reward_efficiency', 0):.2f}\n")
                f.write(
                    f"  Total Vehicles: {stats['total_vehicles']} ({stats['total_success']} success, {stats['total_collision']} collision)\n")

                if 'success_rate_trend' in stats:
                    f.write(
                        f"  Trends: Success {stats['success_rate_trend']:+.2f}%/ep, ")
                    f.write(
                        f"Collision {stats['collision_rate_trend']:+.2f}%/ep, ")
                    f.write(
                        f"Throughput {stats['throughput_trend']:+.1f} vph/ep, ")
                    f.write(
                        f"Reward {stats.get('reward_trend', 0):+.2f}/ep\n")
                    f.write(
                        f"  Improvement Factor: {stats['improvement_factor']:.2f}x\n")

                f.write("\n")

        return report_path

    def _generate_csv_summary(self, comparison: Dict) -> Path:
        """Generate CSV summary for easy analysis."""
        csv_path = self.output_dir / "agent_comparison_summary.csv"

        # Prepare data for CSV
        rows = []
        for agent_name, stats in comparison['agent_stats'].items():
            row = {
                'agent_name': agent_name,
                'num_episodes': stats['num_episodes'],
                'avg_success_rate': stats['avg_success_rate'],
                'std_success_rate': stats['std_success_rate'],
                'best_success_rate': stats['best_success_rate'],
                'final_success_rate': stats['final_success_rate'],
                'avg_collision_rate': stats['avg_collision_rate'],
                'std_collision_rate': stats['std_collision_rate'],
                'best_collision_rate': stats['best_collision_rate'],
                'final_collision_rate': stats['final_collision_rate'],
                'avg_throughput': stats['avg_throughput'],
                'std_throughput': stats['std_throughput'],
                'best_throughput': stats['best_throughput'],
                'final_throughput': stats['final_throughput'],
                'total_success': stats['total_success'],
                'total_collision': stats['total_collision'],
                'total_vehicles': stats['total_vehicles'],
                # Reward statistics
                'avg_cumulative_reward': stats['avg_cumulative_reward'],
                'std_cumulative_reward': stats['std_cumulative_reward'],
                'best_cumulative_reward': stats['best_cumulative_reward'],
                'final_cumulative_reward': stats['final_cumulative_reward'],
                'avg_reward_per_agent': stats['avg_reward_per_agent'],
                'total_reward': stats['total_reward'],
                'reward_efficiency': stats.get('reward_efficiency', 0)
            }

            # Add trend data if available
            if 'success_rate_trend' in stats:
                row.update({
                    'success_rate_trend': stats['success_rate_trend'],
                    'collision_rate_trend': stats['collision_rate_trend'],
                    'throughput_trend': stats['throughput_trend'],
                    'reward_trend': stats.get('reward_trend', 0),
                    'improvement_factor': stats['improvement_factor']
                })

            rows.append(row)

        # Write to CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return csv_path


# Convenience function for direct usage
def compare_agents(evaluation_dir: str) -> Path:
    """
    Convenience function to quickly compare agents from an evaluation directory.

    Args:
        evaluation_dir: Path to evaluation directory containing agent subdirectories

    Returns:
        Path to generated comparison report directory
    """
    evaluator = CrossAgentEvaluator(evaluation_dir)
    return evaluator.generate_comparison_report()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python cross_agent_evaluator.py <evaluation_directory>")
        print("Example: python cross_agent_evaluator.py evaluation_outputs/intersection/2025_09_05_12")
        sys.exit(1)

    evaluation_dir = sys.argv[1]
    try:
        report_dir = compare_agents(evaluation_dir)
        print(f"Comparison report generated: {report_dir}")
    except Exception as e:
        logger.error(f"Failed to generate comparison: {e}")
        sys.exit(1)
