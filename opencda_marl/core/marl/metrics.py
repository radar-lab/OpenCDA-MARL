from typing import Dict, Any, Optional, List
from collections import deque
import numpy as np
import pickle
import os
from loguru import logger


class TrainingMetrics:
    """Track and compute training metrics including traffic performance.

    Uses rolling windows for real-time stats and periodic file export for full history.
    """

    def __init__(self, export_interval: int = 100, export_dir: str = "metrics_history"):
        """
        Initialize metrics tracker.

        Args:
            export_interval: Export history to file every N episodes (0 = disabled)
            export_dir: Directory to save history files
        """
        # Rolling windows for real-time stats (bounded memory)
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_states: deque = deque(maxlen=100)

        # Traffic performance history (bounded)
        self.episode_avg_speeds: deque = deque(maxlen=100)
        self.episode_speed_variances: deque = deque(maxlen=100)

        # Full history for file export (cleared after export)
        self._full_history = {
            'rewards': [],
            'avg_speeds': [],
            'speed_variances': [],
            'episode_lengths': [],
            'success_counts': [],
            'collision_counts': [],
            'near_miss_counts': []  # Track near-miss events for learning analysis
        }

        # Export configuration
        self.export_interval = export_interval
        self.export_dir = export_dir
        self._episode_counter = 0

        # Create export directory if needed
        if export_interval > 0:
            os.makedirs(export_dir, exist_ok=True)

        self.reset()

    def reset(self):
        """Reset current episode metrics."""
        self.current_reward = 0.0
        self.current_length = 0
        self.current_total_reward = 0.0
        self.agent_rewards = {}
        self.collisions = 0
        self.successes = 0

        # Track step when LAST vehicle completes (for accurate episode length)
        self.last_completion_step = 0

        # Traffic performance tracking (per episode)
        self.step_speeds: List[float] = []  # All speeds at each step
        self.step_avg_speeds: List[float] = []  # Average speed per step
        self.agent_speeds: Dict[int, List[float]] = {}  # Speed history per agent
        self.step_target_speeds: List[float] = []  # Target (commanded) speeds for comparison
    # --------------------------------------------------------------------- #
    # Main steps for updating metrics
    # --------------------------------------------------------------------- #

    def update_step(self, rewards: Dict[int, float], observations: Optional[Dict] = None,
                    target_speeds: Optional[Dict[int, float]] = None,
                    step_successes: int = 0):
        """
        Update metrics for current step.

        Args:
            rewards: Dictionary of agent_id -> reward
            observations: Optional dictionary of agent observations (containing speed data)
            target_speeds: Optional dictionary of agent_id -> RL-commanded target speed (km/h)
            step_successes: Number of vehicles that succeeded this step (for tracking last completion)
        """
        self.current_reward = sum(rewards.values()) if rewards else 0.0
        self.current_total_reward += self.current_reward
        self.current_length += 1
        for agent_id, reward in rewards.items():
            if agent_id not in self.agent_rewards:
                self.agent_rewards[agent_id] = 0.0
            self.agent_rewards[agent_id] += reward

        # Track step when vehicles complete (for accurate episode length)
        if step_successes > 0:
            self.last_completion_step = self.current_length

        # Track traffic performance metrics if observations provided
        if observations:
            self._update_traffic_metrics(observations)

        # Track RL-commanded target speeds directly (more reliable than observations)
        # This captures what the RL algorithm actually commanded, not adapter cached values
        if target_speeds:
            for agent_id, ts in target_speeds.items():
                if ts is not None and ts > 0:
                    self.step_target_speeds.append(ts)

    def _update_traffic_metrics(self, observations: Dict):
        """
        Update traffic performance metrics from observations.

        Args:
            observations: Dictionary of agent observations
        """
        step_speeds_list = []
        step_target_speeds_list = []

        for agent_id, obs in observations.items():
            # Extract speed (handle different observation formats)
            speed = None
            target_speed = None
            if isinstance(obs, dict):
                speed = obs.get('speed')
                target_speed = obs.get('target_speed')  # Track commanded speed
            elif hasattr(obs, '__getitem__'):
                # Numpy array or list - speed is typically index 2 in 7D features
                try:
                    speed = float(obs[2]) if len(obs) > 2 else None
                except (IndexError, TypeError):
                    pass

            if speed is not None:
                step_speeds_list.append(speed)

                # Track per-agent speed history
                if agent_id not in self.agent_speeds:
                    self.agent_speeds[agent_id] = []
                self.agent_speeds[agent_id].append(speed)

            # Filter out 0.0 values (uninitialized or stopped vehicles)
            if target_speed is not None and target_speed > 0:
                step_target_speeds_list.append(target_speed)

        # Store step-level speed data
        if step_speeds_list:
            self.step_speeds.extend(step_speeds_list)
            self.step_avg_speeds.append(np.mean(step_speeds_list))

        # Store target speeds
        if step_target_speeds_list:
            self.step_target_speeds.extend(step_target_speeds_list)

    # --------------------------------------------------------------------- #
    # Public Methods
    # --------------------------------------------------------------------- #

    def get_agent_reward(self, agent_id: int) -> float:
        return self.agent_rewards.get(agent_id, 0.0)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get metrics for current episode including traffic performance."""
        if len(self.agent_rewards) == 0:
            avg_reward = 0.0
        else:
            avg_reward = self.current_total_reward / \
                len(self.agent_rewards)

        # Compute traffic performance metrics for current episode
        traffic_metrics = self._compute_traffic_metrics()

        # Compute success/collision rates from tracked counts
        # Note: Don't add active_agents because successes+collisions already accounts
        # for all completed agents. agent_rewards contains ALL agents ever rewarded
        # (including those that succeeded/collided), causing double-counting.
        total_vehicles = self.successes + self.collisions
        success_rate = (self.successes / total_vehicles * 100) if total_vehicles > 0 else 0.0
        collision_rate = (self.collisions / total_vehicles * 100) if total_vehicles > 0 else 0.0

        if not self.episode_rewards:
            metrics = {
                'step_reward': self.current_reward,
                'step_length': self.current_length,
                'total_reward': self.current_total_reward,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'successes': self.successes,
                'collisions': self.collisions,
            }
            metrics.update(traffic_metrics)
            return metrics

        # Compute running averages (convert deque to list for slicing)
        rewards_list = list(self.episode_rewards)
        window_size = min(10, len(rewards_list))
        recent_rewards = rewards_list[-window_size:]

        metrics = {
            'step_reward': self.current_reward,
            'step_length': self.current_length,
            'total_reward': self.current_total_reward,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'successes': self.successes,
            'collisions': self.collisions,
            f'mean_reward_episode_{window_size}': float(np.mean(recent_rewards)),
            f'std_reward_episode_{window_size}': float(np.std(recent_rewards)),
            'max_reward_episode': max(rewards_list),
            'total_episodes': len(rewards_list),
        }
        metrics.update(traffic_metrics)

        # Add multi-episode traffic trends if available
        if len(self.episode_avg_speeds) > 0:
            metrics['mean_episode_speed'] = float(np.mean(self.episode_avg_speeds))
            metrics['mean_episode_speed_var'] = float(np.mean(self.episode_speed_variances))

        return metrics

    def _compute_traffic_metrics(self) -> Dict[str, float]:
        """
        Compute traffic performance metrics for current episode.

        Returns:
            Dictionary with traffic performance metrics
        """
        metrics = {}

        # Average speed across all agents and steps
        if self.step_speeds:
            metrics['avg_speed'] = float(np.mean(self.step_speeds))
            metrics['speed_std'] = float(np.std(self.step_speeds))
            metrics['speed_variance'] = float(np.var(self.step_speeds))
            metrics['min_speed'] = float(np.min(self.step_speeds))
            metrics['max_speed'] = float(np.max(self.step_speeds))
        else:
            metrics['avg_speed'] = 0.0
            metrics['speed_std'] = 0.0
            metrics['speed_variance'] = 0.0
            metrics['min_speed'] = 0.0
            metrics['max_speed'] = 0.0

        # Target (commanded) speed metrics - for comparing RL output vs actual vehicle speed
        if self.step_target_speeds:
            metrics['target_speed_mean'] = float(np.mean(self.step_target_speeds))
            metrics['target_speed_max'] = float(np.max(self.step_target_speeds))
            metrics['target_speed_min'] = float(np.min(self.step_target_speeds))
        else:
            metrics['target_speed_mean'] = 0.0
            metrics['target_speed_max'] = 0.0
            metrics['target_speed_min'] = 0.0

        # Speed smoothness (variance of step-average speeds = how stable traffic flow is)
        if self.step_avg_speeds:
            metrics['speed_smoothness'] = float(np.var(self.step_avg_speeds))
            # Lower variance = smoother flow
            metrics['avg_step_speed'] = float(np.mean(self.step_avg_speeds))
        else:
            metrics['speed_smoothness'] = 0.0
            metrics['avg_step_speed'] = 0.0

        # Per-agent speed consistency (how consistent each agent maintains speed)
        if self.agent_speeds:
            agent_speed_vars = []
            for agent_id, speeds in self.agent_speeds.items():
                if len(speeds) > 1:
                    agent_speed_vars.append(np.var(speeds))
            if agent_speed_vars:
                metrics['avg_agent_speed_var'] = float(np.mean(agent_speed_vars))
            else:
                metrics['avg_agent_speed_var'] = 0.0
        else:
            metrics['avg_agent_speed_var'] = 0.0

        return metrics

    def finish_episode(self, states: Dict[str, Any]):
        """Finish current episode and compute metrics."""
        self._episode_counter += 1

        # Update success/collision counters from episode states
        # BUG FIX: These were never being updated from episode states
        episode_successes = states.get('success', 0)
        episode_collisions = states.get('collision', 0)
        episode_near_misses = states.get('near_miss_count', 0)
        self.successes += episode_successes
        self.collisions += episode_collisions

        # Compute traffic metrics first
        traffic_metrics = self._compute_traffic_metrics()
        avg_speed = traffic_metrics.get('avg_speed', 0.0)
        speed_variance = traffic_metrics.get('speed_variance', 0.0)

        # Calculate actual episode length: step when LAST vehicle reaches destination
        # If no vehicles completed, use total simulation steps as fallback
        effective_episode_length = self.last_completion_step if self.last_completion_step > 0 else self.current_length

        # Calculate throughput: successful vehicles per hour
        # Formula: (successes / effective_steps / fixed_dt) * 3600
        fixed_dt = states.get('fixed_dt', 0.05)  # 20 FPS = 0.05s per step
        if effective_episode_length > 0 and fixed_dt > 0:
            vehicles_per_second = episode_successes / effective_episode_length / fixed_dt
            throughput = vehicles_per_second * 3600  # Convert to vehicles per hour
        else:
            throughput = 0.0

        # Add to rolling windows (bounded memory)
        self.episode_rewards.append(self.current_total_reward)
        self.episode_states.append(states)
        self.episode_avg_speeds.append(avg_speed)
        self.episode_speed_variances.append(speed_variance)

        # Add to full history for export (use effective length, not total steps)
        self._full_history['rewards'].append(self.current_total_reward)
        self._full_history['avg_speeds'].append(avg_speed)
        self._full_history['speed_variances'].append(speed_variance)
        self._full_history['episode_lengths'].append(effective_episode_length)
        self._full_history['success_counts'].append(episode_successes)
        self._full_history['collision_counts'].append(episode_collisions)
        self._full_history['near_miss_counts'].append(episode_near_misses)

        # Check if it's time to export
        if self.export_interval > 0 and self._episode_counter % self.export_interval == 0:
            self.export_history()

        metrics = self.get_current_metrics()
        metrics.update({
            'episode_states': states,
            'near_miss_count': episode_near_misses,  # Add for TensorBoard logging
            'ttc_violation_rate': states.get('ttc_violation_rate', 0.0),  # % of TTC checks with violations
            'throughput': throughput,  # Vehicles per hour
            'episode_length': effective_episode_length,  # Step when LAST vehicle completed
            'total_simulation_steps': self.current_length  # Total steps for reference
        })
        self.reset()
        return metrics

    def export_history(self, filepath: str = None):
        """
        Export full history to file and clear memory.

        Args:
            filepath: Optional custom filepath. If None, auto-generates based on episode count.
        """
        if not self._full_history['rewards']:
            logger.debug("No history to export")
            return

        if filepath is None:
            filepath = os.path.join(
                self.export_dir,
                f"metrics_history_ep{self._episode_counter}.pkl"
            )

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self._full_history, f)

            exported_count = len(self._full_history['rewards'])
            logger.info(f"Exported {exported_count} episodes to {filepath}")

            # Clear full history to free memory
            self._full_history = {
                'rewards': [],
                'avg_speeds': [],
                'speed_variances': [],
                'episode_lengths': [],
                'success_counts': [],
                'collision_counts': [],
                'near_miss_counts': []
            }
        except Exception as e:
            logger.error(f"Failed to export metrics history: {e}")

    def get_traffic_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive traffic performance statistics for paper reporting.

        Returns:
            Dictionary with traffic performance metrics across all episodes
        """
        stats = {
            'total_episodes': len(self.episode_avg_speeds),
        }

        if len(self.episode_avg_speeds) > 0:
            stats.update({
                'mean_avg_speed': float(np.mean(self.episode_avg_speeds)),
                'std_avg_speed': float(np.std(self.episode_avg_speeds)),
                'mean_speed_variance': float(np.mean(self.episode_speed_variances)),
                'std_speed_variance': float(np.std(self.episode_speed_variances)),
                'episode_avg_speeds': list(self.episode_avg_speeds),
                'episode_speed_variances': list(self.episode_speed_variances),
            })

            # Trend analysis
            if len(self.episode_avg_speeds) > 1:
                speed_trend = np.polyfit(
                    range(len(self.episode_avg_speeds)),
                    self.episode_avg_speeds, 1
                )[0]
                var_trend = np.polyfit(
                    range(len(self.episode_speed_variances)),
                    self.episode_speed_variances, 1
                )[0]
                stats['speed_trend'] = float(speed_trend)  # km/h per episode
                stats['variance_trend'] = float(var_trend)  # Decreasing = improving smoothness

        return stats
