'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-04 12:00:00
FilePath     : /OpenCDA-MARL/opencda_marl/envs/evaluation.py
Description  : Modular evaluation system for MARL training with memory efficiency
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .evaluation_plots import EvaluationPlotter


class EvaluationManager:
    """
    Memory-efficient evaluation manager for MARL training.

    Provides comprehensive traffic analysis and metrics tracking with three modes:
    - 'full': Detailed time-series tracking with plotting
    - 'lightweight': Summary-only tracking for training efficiency
    - 'disabled': No evaluation tracking
    """

    def __init__(self, config: Optional[Dict] = None, 
                 scenario_name: Optional[str] = None,
                 agent_name: Optional[str] = None):
        """
        Initialize evaluation manager.

        Args:
            config: Configuration dictionary containing evaluation settings
        """
        self.config = config or {}
        self.scenario_name = scenario_name
        self.agent_name = agent_name
        self.fixed_dt = None

        # Core configuration
        # 'full', 'lightweight', 'disabled'
        self.mode = self.config.get('mode', 'full')

        # Data management
        self.root_dir = Path(self.config.get('root_dir', 'evaluation_outputs'))
        self.output_dir = self.root_dir / self.scenario_name / \
            datetime.now().strftime("%Y_%m_%d_%H")
        self.data_dir = self.output_dir / self.agent_name / 'data'
        self.plot_dir = self.output_dir / self.agent_name / 'plots'

         # Data storage configuration
        self.save_format = self.config.get(
            'save_format', 'npz')  # 'npz' or 'hdf5'
        self.save_interval = self.config.get(
            'save_interval', 1)  # Save every N episodes
        self.compression = self.config.get('compression', True)
        self.keep_last_n_files = self.config.get(
            'keep_last_n_files', 0)  # 0 = keep all
        self.keep_summaries = self.config.get(
            'keep_summaries', 10)  # Lightweight mode
        
        # Track last saved episode count to prevent double saves
        self.last_saved_episode_count = 0

        # Plotting configuration
        # Use non-interactive backend for server environments
        matplotlib.use('Agg')
        self.plot_interval = self.config.get('plot_interval', 10)

        # Initialize plotter
        self.plotter = None

        # Initialize tracking based on mode
        self._init_tracking()

    def _init_tracking(self):
        """Initialize evaluation tracking based on mode"""
        self.history = []
        self.episode_history = []
        # Initialize reward tracking
        self.cumulative_reward = 0.0
        self.agent_cumulative_rewards = {}
        # Track when all vehicles complete for accurate throughput calculation
        self.last_completion_step = None
        
        if not self.is_enabled():    
            logger.info("Evaluation disabled - no history tracking")
        else:  # 'full' mode
            # Track detailed time-series data
            # history = [{
            # 'total_vehicles': int, 'success': int, 'collision': int, 'active_agents': int,
            # 'success_rate': float, 'collision_rate': float, 'throughput': float,
            # 'step_reward': float, 'cumulative_reward': float, 'avg_reward_per_agent': float
            # }]
            logger.info(
                "Full evaluation mode enabled - tracking detailed metrics and rewards")

        # Ensure directories exist
        if self.is_enabled():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize plotter
            self.plotter = EvaluationPlotter(self.plot_dir, self.scenario_name, self.agent_name)
    # --------------------------------------------------------------------- #
    # Core Update Methods
    # --------------------------------------------------------------------- #

    def update_step(self, metrics: Dict[str, Any], rewards: Dict[int, float] = None):
        """
        Update evaluation with current step metrics and rewards.

        Args:
            metrics: Current step metrics dictionary
            rewards: Dictionary mapping agent_id to reward value
        """
        if not self.is_enabled():
            return

        if self.fixed_dt is None:
            self.fixed_dt = metrics.get('fixed_dt')

        assert self.fixed_dt is not None, "fixed_dt is not set"

        current_step = metrics.get('step', 0)
        max_steps = metrics.get('max_steps', 0)
        current_episode = metrics.get('episode', 0)
        max_episodes = metrics.get('max_episodes', 0)

        # Full mode updates detailed history at each step
        if self.get_mode() == 'full':
            self.history.append(self._step_statistics(metrics, rewards))

        # Handle end of episode (when current step reaches max_steps - 1)
        if current_step == max_steps - 1:
            # Save step history FIRST (before clearing) for full mode
            if self.get_mode() == 'full' and len(self.history) > 0:
                self._save_step_history(current_episode)
            
            # Always add episode summary to episode_history
            self.episode_history.append(self._step_statistics(metrics, rewards))
            
            # Generate plots for this episode
            if self.get_mode() == 'full':
                self.plot_step_analysis(current_episode)
            
            self.plot_episode_analysis(current_episode)
            
            # Generate comparison plot if we have enough episodes and it's interval time
            if (len(self.episode_history) >= 2 and 
                len(self.episode_history) % self.plot_interval == 0):
                self.plot_episode_comparison()
            
            # Clear history for next episode (full mode only)
            if self.get_mode() == 'full':
                self.history.clear()
                logger.debug(f"Cleared step history after episode {current_episode}")
            
            # Reset reward tracking for new episode
            self.cumulative_reward = 0.0
            self.agent_cumulative_rewards.clear()
            self.last_completion_step = None  # Reset for next episode

        self._mem_management()
        self._data_management()

        # End of simulation - generate final summary
        if current_episode == max_episodes - 1 and current_step == max_steps - 1:
            self.summary_evaluation()

    # --------------------------------------------------------------------- #
    # Private Methods
    # --------------------------------------------------------------------- #

    def _step_statistics(self, metrics: Dict[str, Any], rewards: Dict[int, float] = None):
        # get current step metrics
        step = metrics.get('step', 0)
        success = metrics.get('success', 0)
        collision = metrics.get('collision', 0)
        active_agents = metrics.get('active_agents', 0)
        pending_spawns = metrics.get('pending_spawns', 0)
        total_vehicles = success + collision + active_agents

        # Track the step when ALL vehicles complete:
        # - No pending spawns (all vehicles have been spawned)
        # - No active agents (all spawned vehicles have finished)
        if pending_spawns == 0 and active_agents == 0 and (success > 0 or collision > 0):
            if self.last_completion_step is None:
                self.last_completion_step = step

        # calculate current rates
        # Calculate rates including timeout vehicles for complete accountability
        success_rate = (success / total_vehicles *
                        100) if total_vehicles > 0 else 0
        collision_rate = (collision / total_vehicles *
                          100) if total_vehicles > 0 else 0
        timeout_rate = (active_agents / total_vehicles *
                       100) if total_vehicles > 0 else 0

        # throughput: how many vehicles have completed per hour
        # Use completion step if all vehicles finished, otherwise current step
        effective_step = self.last_completion_step if self.last_completion_step else step
        if effective_step > 0:
            vps = success / effective_step / self.fixed_dt
            throughput = vps * 3600
        else:
            throughput = 0.0

        # Calculate reward metrics
        step_reward = 0.0
        avg_reward_per_agent = 0.0
        rewards_by_agent = {}
        
        if rewards:
            step_reward = sum(rewards.values())
            self.cumulative_reward += step_reward
            
            # Update agent cumulative rewards
            for agent_id, reward in rewards.items():
                if agent_id not in self.agent_cumulative_rewards:
                    self.agent_cumulative_rewards[agent_id] = 0.0
                self.agent_cumulative_rewards[agent_id] += reward
                rewards_by_agent[agent_id] = reward
            
            # Calculate average reward per agent
            if len(rewards) > 0:
                avg_reward_per_agent = step_reward / len(rewards)

        stat = {
            'step': step,
            'success': success,
            'collision': collision,
            'active_agents': active_agents,
            'total_vehicles': total_vehicles,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'throughput': throughput,
            # Episode length: step when ALL vehicles completed (for consistent reporting)
            'episode_length': effective_step,  # Same as used in throughput calc
            'total_simulation_steps': step,    # Total steps for reference
            'step_reward': step_reward,
            'cumulative_reward': self.cumulative_reward,
            'avg_reward_per_agent': avg_reward_per_agent,
            'rewards_by_agent': rewards_by_agent,
            'agent_cumulative_rewards': dict(self.agent_cumulative_rewards)  # Copy for safety
        }

        return stat

    # --------------------------------------------------------------------- #
    # Data Storage Methods
    # --------------------------------------------------------------------- #
    def _mem_management(self):
        """Manage memory usage by limiting history sizes"""
        if not self.is_enabled():
            return
            
        # Memory management for step history (full mode)
        if self.get_mode() == 'full' and len(self.history) > 0:
            # Keep sliding window of recent history to prevent memory overflow
            max_history_steps = self.config.get('max_history_steps', 10000)
            if len(self.history) > max_history_steps:
                # Keep only the most recent steps
                excess = len(self.history) - max_history_steps
                self.history = self.history[excess:]
                logger.debug(f"Trimmed step history: removed {excess} old steps")
        
        # Memory management for episode history
        if len(self.episode_history) > self.keep_summaries:
            excess = len(self.episode_history) - self.keep_summaries
            self.episode_history = self.episode_history[excess:]
            logger.debug(f"Trimmed episode history: removed {excess} old episodes")

    def _data_management(self):
        """Manage data saving to disk"""
        if not self.is_enabled():
            return
        
        # Save data based on save_interval
        current_episode_count = len(self.episode_history)
        
        # Only save if we have new episodes AND haven't saved them yet
        if (current_episode_count > self.last_saved_episode_count and 
            current_episode_count % self.save_interval == 0):
            self._save_data_to_disk(current_episode_count - 1)
            self.last_saved_episode_count = current_episode_count  # Mark as saved
            
        # Clean up old files if configured
        if self.keep_last_n_files > 0:
            self._cleanup_old_files()
    
    def _save_data_to_disk(self, episode_num: int):
        """Save current data to disk"""
        try:
            # Only save episode summary (step history is saved at episode end)
            self._save_episode_summary()
            
            logger.info(f"Data saved for episode {episode_num}")
            
        except Exception as e:
            logger.warning(f"Failed to save data for episode {episode_num}: {e}")
    
    def _save_step_history(self, episode_num: int):
        """Save step-by-step history for current episode"""
        if not self.history:
            return
            
        filepath = self.data_dir / f"episode_{episode_num}_steps.npz"
        
        # Convert history to arrays
        save_data = {
            'episode': episode_num,
            'scenario_name': self.scenario_name,
            'agent_name': self.agent_name,
            'steps': np.array([h['step'] for h in self.history]),
            'success': np.array([h['success'] for h in self.history]),
            'collision': np.array([h['collision'] for h in self.history]),
            'active_agents': np.array([h['active_agents'] for h in self.history]),
            'total_vehicles': np.array([h['total_vehicles'] for h in self.history]),
            'success_rate': np.array([h['success_rate'] for h in self.history]),
            'collision_rate': np.array([h['collision_rate'] for h in self.history]),
            'throughput': np.array([h['throughput'] for h in self.history]),
            # Add reward data
            'step_reward': np.array([h.get('step_reward', 0.0) for h in self.history]),
            'cumulative_reward': np.array([h.get('cumulative_reward', 0.0) for h in self.history]),
            'avg_reward_per_agent': np.array([h.get('avg_reward_per_agent', 0.0) for h in self.history])
        }
        
        if self.compression:
            np.savez_compressed(str(filepath), **save_data)
        else:
            np.savez(str(filepath), **save_data)
        
        logger.debug(f"Step history saved: {filepath}")
    
    def _save_episode_summary(self):
        """Save latest episode summary to consolidated file"""
        if not self.episode_history:
            return
        
        csv_path = self.data_dir / "episodes_summary.csv"
        file_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            # Get only the latest episode data
            latest_episode = self.episode_history[-1]
            writer = csv.DictWriter(f, fieldnames=latest_episode.keys())
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write only the latest episode
            writer.writerow(latest_episode)
        
        logger.debug(f"Latest episode summary saved: {csv_path}")
    
    def _cleanup_old_files(self):
        """Remove old step history files to save disk space"""
        try:
            import glob
            
            pattern = str(self.data_dir / "episode_*_steps.npz")
            files = sorted(glob.glob(pattern), 
                          key=lambda x: int(x.split('_')[-2]))  # Sort by episode number
            
            if len(files) > self.keep_last_n_files:
                for filepath in files[:-self.keep_last_n_files]:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up old file: {filepath}")
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup old files: {e}")
    # --------------------------------------------------------------------- #
    # Plotting and Visualization
    # --------------------------------------------------------------------- #

    def plot_step_analysis(self, episode_num: int = 0):
        """Generate step analysis plot using the plotter module"""
        if not self.plotter:
            logger.warning("Plotter not initialized")
            return
        
        if not self.history:
            logger.warning("No step history data available for plotting")
            return
        
        self.plotter.plot_step_analysis(self.history, episode_num)
    
    def plot_episode_analysis(self, episode_num: int = 0):
        """Generate single episode summary visualization using the plotter module"""
        if not self.is_enabled() or not self.episode_history:
            logger.info("Episode analysis plotting requires episode data")
            return
        
        if not self.plotter:
            logger.warning("Plotter not initialized")
            return
            
        # Get the latest episode data
        episode_data = self.episode_history[-1]
        
        self.plotter.plot_episode_analysis(episode_data, episode_num, self.fixed_dt)
    
    def plot_episode_comparison(self):
        """Generate multi-episode comparison plots using the plotter module"""
        if not self.is_enabled() or len(self.episode_history) < 2:
            logger.info("Episode comparison requires at least 2 episodes of data")
            return
        
        if not self.plotter:
            logger.warning("Plotter not initialized")
            return
            
        self.plotter.plot_episode_comparison(self.episode_history)
    
        
    # --------------------------------------------------------------------- #
    # Analysis and Summary Methods
    # --------------------------------------------------------------------- #

    def summary_evaluation(self):
        """Generate comprehensive final evaluation summary"""
        if not self.is_enabled():
            logger.info("Evaluation disabled - no summary generated")
            return
            
        if not self.episode_history:
            logger.warning("No episode data available for summary evaluation")
            return
            
        try:
            logger.info("=== FINAL EVALUATION SUMMARY ===")
            
            # Extract metrics from all episodes
            num_episodes = len(self.episode_history)
            success_rates = [ep['success_rate'] for ep in self.episode_history]
            collision_rates = [ep['collision_rate'] for ep in self.episode_history]
            timeout_rates = [ep.get('timeout_rate', 0.0) for ep in self.episode_history]
            throughputs = [ep['throughput'] for ep in self.episode_history]
            total_successes = [ep['success'] for ep in self.episode_history]
            total_collisions = [ep['collision'] for ep in self.episode_history]
            total_timeouts = [ep.get('active_agents', 0) for ep in self.episode_history]
            
            # Calculate comprehensive statistics
            final_stats = {
                'scenario_name': self.scenario_name,
                'agent_name': self.agent_name,
                'total_episodes': num_episodes,
                'avg_success_rate': np.mean(success_rates),
                'std_success_rate': np.std(success_rates),
                'best_success_rate': np.max(success_rates),
                'worst_success_rate': np.min(success_rates),
                'avg_collision_rate': np.mean(collision_rates),
                'std_collision_rate': np.std(collision_rates),
                'best_collision_rate': np.min(collision_rates),  # Lower is better
                'worst_collision_rate': np.max(collision_rates),
                'avg_timeout_rate': np.mean(timeout_rates),
                'std_timeout_rate': np.std(timeout_rates),
                'best_timeout_rate': np.min(timeout_rates),  # Lower is better
                'worst_timeout_rate': np.max(timeout_rates),
                'avg_throughput': np.mean(throughputs),
                'std_throughput': np.std(throughputs),
                'best_throughput': np.max(throughputs),
                'worst_throughput': np.min(throughputs),
                'total_success': sum(total_successes),
                'total_collision': sum(total_collisions),
                'total_timeout': sum(total_timeouts),
                'total_vehicles': sum(total_successes) + sum(total_collisions) + sum(total_timeouts),
                'final_episode_success_rate': success_rates[-1],
                'final_episode_collision_rate': collision_rates[-1],
                'final_episode_timeout_rate': timeout_rates[-1],
                'final_episode_throughput': throughputs[-1]
            }
            
            # Calculate improvement trends
            if num_episodes > 1:
                # Success rate trend
                success_trend = np.polyfit(range(num_episodes), success_rates, 1)[0]
                collision_trend = np.polyfit(range(num_episodes), collision_rates, 1)[0]
                throughput_trend = np.polyfit(range(num_episodes), throughputs, 1)[0]
                
                final_stats.update({
                    'success_rate_trend': success_trend,  # % per episode
                    'collision_rate_trend': collision_trend,  # % per episode  
                    'throughput_trend': throughput_trend,  # vph per episode
                    'improvement_factor': success_rates[-1] / max(success_rates[0], 0.1)  # Avoid div by 0
                })
            
            # Log comprehensive summary
            logger.info(f"Scenario: {self.scenario_name} | Agent: {self.agent_name}")
            logger.info(f"Episodes Completed: {num_episodes}")
            logger.info(f"Success Rate: {final_stats['avg_success_rate']:.1f}% ± {final_stats['std_success_rate']:.1f}% "
                       f"(Best: {final_stats['best_success_rate']:.1f}%)")
            logger.info(f"Collision Rate: {final_stats['avg_collision_rate']:.1f}% ± {final_stats['std_collision_rate']:.1f}% "
                       f"(Best: {final_stats['best_collision_rate']:.1f}%)")
            logger.info(f"Timeout Rate: {final_stats['avg_timeout_rate']:.1f}% ± {final_stats['std_timeout_rate']:.1f}% "
                       f"(Best: {final_stats['best_timeout_rate']:.1f}%)")
            logger.info(f"Throughput: {final_stats['avg_throughput']:.1f} ± {final_stats['std_throughput']:.1f} vph "
                       f"(Best: {final_stats['best_throughput']:.1f} vph)")
            logger.info(f"Total Vehicles: {final_stats['total_vehicles']} "
                       f"({final_stats['total_success']} success, {final_stats['total_collision']} collision, "
                       f"{final_stats['total_timeout']} timeout)")
            
            if num_episodes > 1:
                logger.info(f"Trends: Success {final_stats['success_rate_trend']:+.2f}%/ep, "
                           f"Collision {final_stats['collision_rate_trend']:+.2f}%/ep, "
                           f"Throughput {final_stats['throughput_trend']:+.1f} vph/ep")
                logger.info(f"Overall Improvement: {final_stats['improvement_factor']:.2f}x")
            
            # Save summary to JSON file
            self._save_final_summary(final_stats)

            # Generate final comparison plot if we have multiple episodes
            if num_episodes > 1:
                self.plot_episode_comparison()

            # Generate learning progress visualization with improvement metrics
            # This is the paper-ready plot with tables and statistics
            if num_episodes >= 100 and self.plotter:
                logger.info("Generating learning progress analysis...")
                self.plotter.plot_learning_progress(self.episode_history, window_size=100)

                # Generate and save improvement report
                improvement_report = self.plotter.generate_improvement_report(
                    self.episode_history, window_size=100)
                self._save_improvement_report(improvement_report)

                # Log key improvement metrics
                if 'summary' in improvement_report:
                    summary = improvement_report['summary']
                    logger.info(f"=== IMPROVEMENT SUMMARY ===")
                    logger.info(f"Success Rate: {summary['success_improvement_pct']:+.1f}% improvement")
                    logger.info(f"Collision Rate: {summary['collision_reduction_pct']:+.1f}% reduction")
                    logger.info(f"Throughput: {summary['throughput_improvement_pct']:+.1f}% improvement")
                    if summary.get('episodes_to_80pct_success'):
                        logger.info(f"Reached 80% success at episode {summary['episodes_to_80pct_success']}")
                    if summary.get('episodes_to_90pct_success'):
                        logger.info(f"Reached 90% success at episode {summary['episodes_to_90pct_success']}")

            logger.info("=== EVALUATION COMPLETE ===")
            
        except Exception as e:
            logger.error(f"Failed to generate summary evaluation: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _save_improvement_report(self, report: dict):
        """Save improvement report to JSON file"""
        try:
            # Add timestamp
            report['generated_at'] = datetime.now().isoformat()

            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj

            report = convert_numpy(report)

            # Save to JSON
            json_path = self.data_dir / "improvement_report.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Improvement report saved: {json_path}")

        except Exception as e:
            logger.warning(f"Failed to save improvement report: {e}")

    def _save_final_summary(self, summary_stats: dict):
        """Save final summary statistics to file"""
        try:


            # Add timestamp
            summary_stats['generated_at'] = datetime.now().isoformat()
            
            # Save to JSON
            json_path = self.data_dir / "final_summary.json"
            with open(json_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            logger.info(f"Final summary saved: {json_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save final summary: {e}")
    # --------------------------------------------------------------------- #
    # Utility Methods
    # --------------------------------------------------------------------- #

    def is_enabled(self) -> bool:
        """Check if evaluation is enabled"""
        return self.get_mode() != 'disabled'

    def get_mode(self) -> str:
        """Get current evaluation mode"""
        return self.mode

    def reset_config(self, new_config: Dict):
        """Update configuration and reinitialize if needed"""
        self.config = new_config
        old_mode = self.mode
        self.mode = self.config.get('mode', 'full')

        # Reinitialize if mode changed
        if old_mode != self.mode:
            logger.info(
                f"Evaluation mode changed from {old_mode} to {self.mode}")
            self._init_tracking()
