from typing import Dict, Any, Optional, List
from collections import deque
import numpy as np


class TrainingMetrics:
    """Track and compute training metrics including traffic performance."""

    def __init__(self):
        self.episode_rewards = []
        self.episode_states = []

        # Traffic performance history (for multi-episode analysis)
        self.episode_avg_speeds = []
        self.episode_speed_variances = []

        self.reset()

    def reset(self):
        """Reset current episode metrics."""
        self.current_reward = 0.0
        self.current_length = 0
        self.current_total_reward = 0.0
        self.agent_rewards = {}
        self.collisions = 0
        self.successes = 0

        # Traffic performance tracking (per episode)
        self.step_speeds: List[float] = []  # All speeds at each step
        self.step_avg_speeds: List[float] = []  # Average speed per step
        self.agent_speeds: Dict[int, List[float]] = {}  # Speed history per agent
    # --------------------------------------------------------------------- #
    # Main steps for updating metrics
    # --------------------------------------------------------------------- #

    def update_step(self, rewards: Dict[int, float], observations: Optional[Dict] = None):
        """
        Update metrics for current step.

        Args:
            rewards: Dictionary of agent_id -> reward
            observations: Optional dictionary of agent observations (containing speed data)
        """
        self.current_reward = sum(rewards.values()) if rewards else 0.0
        self.current_total_reward += self.current_reward
        self.current_length += 1
        for agent_id, reward in rewards.items():
            if agent_id not in self.agent_rewards:
                self.agent_rewards[agent_id] = 0.0
            self.agent_rewards[agent_id] += reward

        # Track traffic performance metrics if observations provided
        if observations:
            self._update_traffic_metrics(observations)

    def _update_traffic_metrics(self, observations: Dict):
        """
        Update traffic performance metrics from observations.

        Args:
            observations: Dictionary of agent observations
        """
        step_speeds_list = []

        for agent_id, obs in observations.items():
            # Extract speed (handle different observation formats)
            speed = None
            if isinstance(obs, dict):
                speed = obs.get('speed')
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

        # Store step-level speed data
        if step_speeds_list:
            self.step_speeds.extend(step_speeds_list)
            self.step_avg_speeds.append(np.mean(step_speeds_list))

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

        if not self.episode_rewards:
            metrics = {
                'step_reward': self.current_reward,
                'step_length': self.current_length,
                'total_reward': self.current_total_reward,
                'avg_reward': avg_reward
            }
            metrics.update(traffic_metrics)
            return metrics

        # Compute running averages
        window_size = min(10, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-window_size:]

        metrics = {
            'step_reward': self.current_reward,
            'step_length': self.current_length,
            'total_reward': self.current_total_reward,
            'avg_reward': avg_reward,
            f'mean_reward_episode_{window_size}': float(np.mean(recent_rewards)),
            f'std_reward_episode_{window_size}': float(np.std(recent_rewards)),
            'max_reward_episode': max(self.episode_rewards),
            'total_episodes': len(self.episode_rewards),
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
        self.episode_rewards.append(self.current_total_reward)
        self.episode_states.append(states)

        # Save traffic performance for multi-episode analysis
        traffic_metrics = self._compute_traffic_metrics()
        self.episode_avg_speeds.append(traffic_metrics.get('avg_speed', 0.0))
        self.episode_speed_variances.append(traffic_metrics.get('speed_variance', 0.0))

        metrics = self.get_current_metrics()
        metrics.update({
            'episode_states': states
        })
        self.reset()
        return metrics

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
