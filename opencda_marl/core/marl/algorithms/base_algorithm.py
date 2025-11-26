'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 18:29:51
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/base_algorithm.py
Description  : Base class for all reinforcement learning algorithms
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from collections import deque
import os
from datetime import datetime
import numpy as np
from loguru import logger

# TensorBoard for training visualization
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")


class BaseAlgorithm(ABC):
    """
    Abstract base class for all reinforcement learning algorithms.

    This class defines the interface that all RL algorithms must implement
    for use with the MARLAgent.
    """

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        """
        Initialize the algorithm.

        Args:
            config: Algorithm-specific configuration
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Common parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.discount_factor = config.get('discount_factor', 0.95)

        # Training state
        self.training_step = 0
        self.episode_count = 0

        # Reward tracking for convergence analysis
        self.reward_window_size = config.get('reward_window_size', 10)
        self.reward_history: deque = deque(maxlen=100)  # Full history (last 100 episodes)
        self.episode_length_history: deque = deque(maxlen=100)

        # Convergence detection parameters
        self.convergence_threshold = config.get('convergence_threshold', 0.05)  # 5% variance threshold
        self.convergence_window = config.get('convergence_window', 10)  # Episodes to check
        self.is_converged = False
        self.convergence_episode = None  # Episode when convergence was detected

        # Initialize TensorBoard logging
        self._init_tensorboard(config)

        logger.success(
            f"Initialized {self.__class__.__name__} with state_dim={state_dim}, action_dim={action_dim}")

    def _init_tensorboard(self, config: Dict[str, Any]):
        """Initialize TensorBoard logging based on configuration."""
        # Get tensorboard config (can be nested under algorithm or at root level)
        tb_config = config.get('tensorboard', {})
        if isinstance(tb_config, bool):
            # Handle simple boolean config
            tb_config = {'enabled': tb_config}

        self.tb_enabled = tb_config.get('enabled', True) and TENSORBOARD_AVAILABLE
        self.writer: Optional[SummaryWriter] = None

        if self.tb_enabled:
            # Configurable log directory
            base_dir = tb_config.get('log_dir', 'runs')
            algo_name = self.__class__.__name__.lower().replace('algorithm', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Allow custom run name
            run_name = tb_config.get('run_name', timestamp)
            self.tensorboard_dir = os.path.join(base_dir, algo_name, run_name)

            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
            logger.info(f"TensorBoard logging enabled: {self.tensorboard_dir}")
            logger.info(f"View with: tensorboard --logdir={base_dir}")

            # Configurable metrics to log
            self.tb_log_frequency = tb_config.get('log_frequency', 1)  # Log every N steps
            self.tb_metrics = tb_config.get('metrics', {
                'losses': True,
                'q_values': True,
                'buffer': True,
                'episode': True,
                'rewards': True
            })

    def log_scalar(self, tag: str, value: float, step: int = None, category: str = None):
        """
        Log a scalar value to TensorBoard.

        Args:
            tag: Metric name
            value: Scalar value
            step: Global step (defaults to training_step)
            category: Optional category to check if logging is enabled
        """
        if self.writer is None:
            return

        # Check if this category of metrics should be logged
        if category and not self.tb_metrics.get(category, True):
            return

        # Use training_step if step not provided
        if step is None:
            step = self.training_step

        # Respect log frequency
        if step % self.tb_log_frequency != 0:
            return

        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float],
                    step: int = None, category: str = None):
        """
        Log multiple scalars under a main tag.

        Args:
            main_tag: Main category tag
            tag_scalar_dict: Dictionary of {tag: value}
            step: Global step
            category: Optional category to check if logging is enabled
        """
        if self.writer is None:
            return

        if category and not self.tb_metrics.get(category, True):
            return

        if step is None:
            step = self.training_step

        if step % self.tb_log_frequency != 0:
            return

        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int = None):
        """Log a histogram of values."""
        if self.writer is None:
            return

        if step is None:
            step = self.training_step

        self.writer.add_histogram(tag, values, step)

    def log_episode_metrics(self, episode_reward: float, episode_length: int,
                           success_rate: float = 0.0, collision_rate: float = 0.0,
                           additional_metrics: Dict[str, float] = None,
                           traffic_metrics: Dict[str, float] = None):
        """
        Log episode-level metrics to TensorBoard with learning quality analysis.

        Args:
            episode_reward: Total episode reward
            episode_length: Number of steps in episode
            success_rate: Success rate (0-1)
            collision_rate: Collision rate (0-1)
            additional_metrics: Additional custom metrics to log
            traffic_metrics: Traffic performance metrics (avg_speed, speed_variance, etc.)
        """
        # Track reward and episode length history (always track, even without TensorBoard)
        self.reward_history.append(episode_reward)
        self.episode_length_history.append(episode_length)

        # Compute learning quality metrics
        reward_ma, reward_var, reward_std = self._compute_reward_statistics()
        length_ma = self._compute_episode_length_ma()

        # Check for convergence
        self._check_convergence()

        if self.writer is None or not self.tb_metrics.get('episode', True):
            return

        # Core episode metrics
        self.writer.add_scalar('Episode/reward', episode_reward, self.episode_count)
        self.writer.add_scalar('Episode/length', episode_length, self.episode_count)
        self.writer.add_scalar('Episode/success_rate', success_rate, self.episode_count)
        self.writer.add_scalar('Episode/collision_rate', collision_rate, self.episode_count)

        # Learning quality metrics (MARL paper-ready)
        self.writer.add_scalar('Learning/reward_moving_avg', reward_ma, self.episode_count)
        self.writer.add_scalar('Learning/reward_variance', reward_var, self.episode_count)
        self.writer.add_scalar('Learning/reward_std', reward_std, self.episode_count)
        self.writer.add_scalar('Learning/episode_length_ma', length_ma, self.episode_count)
        self.writer.add_scalar('Learning/converged', 1.0 if self.is_converged else 0.0, self.episode_count)

        # Normalized coefficient of variation (for convergence visualization)
        if reward_ma != 0:
            cv = reward_std / abs(reward_ma)  # Coefficient of variation
            self.writer.add_scalar('Learning/reward_cv', cv, self.episode_count)

        # Traffic performance metrics (RA-L paper-ready)
        if traffic_metrics:
            # Core traffic metrics
            if 'avg_speed' in traffic_metrics:
                self.writer.add_scalar('Traffic/avg_speed', traffic_metrics['avg_speed'], self.episode_count)
            if 'speed_std' in traffic_metrics:
                self.writer.add_scalar('Traffic/speed_std', traffic_metrics['speed_std'], self.episode_count)
            if 'speed_variance' in traffic_metrics:
                self.writer.add_scalar('Traffic/speed_variance', traffic_metrics['speed_variance'], self.episode_count)
            if 'min_speed' in traffic_metrics:
                self.writer.add_scalar('Traffic/min_speed', traffic_metrics['min_speed'], self.episode_count)
            if 'max_speed' in traffic_metrics:
                self.writer.add_scalar('Traffic/max_speed', traffic_metrics['max_speed'], self.episode_count)

            # Traffic flow quality metrics
            if 'speed_smoothness' in traffic_metrics:
                self.writer.add_scalar('Traffic/speed_smoothness', traffic_metrics['speed_smoothness'], self.episode_count)
            if 'avg_step_speed' in traffic_metrics:
                self.writer.add_scalar('Traffic/avg_step_speed', traffic_metrics['avg_step_speed'], self.episode_count)
            if 'avg_agent_speed_var' in traffic_metrics:
                self.writer.add_scalar('Traffic/avg_agent_speed_var', traffic_metrics['avg_agent_speed_var'], self.episode_count)

        # Log additional custom metrics
        if additional_metrics:
            for name, value in additional_metrics.items():
                self.writer.add_scalar(f'Episode/{name}', value, self.episode_count)

        # Flush periodically (every 10 episodes) instead of every episode
        # Reduces I/O blocking during training
        if self.episode_count % 10 == 0:
            self.writer.flush()

    def flush_tensorboard(self):
        """Flush TensorBoard writer to ensure all metrics are written."""
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        """Close TensorBoard writer and cleanup resources."""
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")

    # ------------------------------------------------------------------ #
    # Learning Quality Analysis Methods (for RA-L paper metrics)
    # ------------------------------------------------------------------ #

    def _compute_reward_statistics(self) -> tuple:
        """
        Compute reward moving average, variance, and standard deviation.

        Returns:
            (moving_average, variance, std_deviation)
        """
        if len(self.reward_history) == 0:
            return 0.0, 0.0, 0.0

        # Use last N episodes for moving average
        window = list(self.reward_history)[-self.reward_window_size:]
        reward_ma = float(np.mean(window))
        reward_var = float(np.var(window))
        reward_std = float(np.std(window))

        return reward_ma, reward_var, reward_std

    def _compute_episode_length_ma(self) -> float:
        """Compute moving average of episode lengths."""
        if len(self.episode_length_history) == 0:
            return 0.0

        window = list(self.episode_length_history)[-self.reward_window_size:]
        return float(np.mean(window))

    def _check_convergence(self):
        """
        Check if training has converged based on reward stability.

        Convergence is detected when:
        1. We have enough episodes (at least convergence_window)
        2. Coefficient of variation (std/mean) is below threshold
        3. Convergence persists for convergence_window episodes
        """
        if self.is_converged:
            return  # Already converged

        if len(self.reward_history) < self.convergence_window:
            return  # Not enough data

        # Get recent rewards
        recent_rewards = list(self.reward_history)[-self.convergence_window:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        # Avoid division by zero
        if abs(mean_reward) < 1e-8:
            return

        # Coefficient of variation
        cv = std_reward / abs(mean_reward)

        # Check convergence condition
        if cv < self.convergence_threshold:
            self.is_converged = True
            self.convergence_episode = self.episode_count
            logger.success(f"🎯 Convergence detected at episode {self.episode_count}! "
                         f"CV={cv:.4f} < threshold={self.convergence_threshold}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics for paper reporting.

        Returns:
            Dictionary with learning quality metrics
        """
        reward_ma, reward_var, reward_std = self._compute_reward_statistics()
        length_ma = self._compute_episode_length_ma()

        stats = {
            'episode_count': self.episode_count,
            'reward_moving_avg': reward_ma,
            'reward_variance': reward_var,
            'reward_std': reward_std,
            'episode_length_ma': length_ma,
            'is_converged': self.is_converged,
            'convergence_episode': self.convergence_episode,
            'reward_history_size': len(self.reward_history),
        }

        # Add coefficient of variation if mean is non-zero
        if reward_ma != 0:
            stats['coefficient_of_variation'] = reward_std / abs(reward_ma)

        # Add full reward history for plotting
        if len(self.reward_history) > 0:
            stats['reward_history'] = list(self.reward_history)
            stats['episode_length_history'] = list(self.episode_length_history)

        return stats

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> Any:
        """
        Select an action given the current state.

        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)

        Returns:
            Action to take (format depends on algorithm)
        """
        pass

    @abstractmethod
    def store_transition(self, state: np.ndarray, action: Any, reward: float,
                         next_state: np.ndarray, done: bool):
        """
        Store a transition for learning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        pass

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """
        Update the algorithm (learning step).

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def reset_episode(self):
        """Reset algorithm state for new episode."""
        pass

    @abstractmethod
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get information about training progress.
        Must be implemented by all algorithms.

        Returns:
            Dictionary with training statistics
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save algorithm state."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load algorithm state."""
        pass
