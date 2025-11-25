'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 18:29:51
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/base_algorithm.py
Description  : Base class for all reinforcement learning algorithms
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
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
                           additional_metrics: Dict[str, float] = None):
        """
        Log episode-level metrics to TensorBoard.

        Args:
            episode_reward: Total episode reward
            episode_length: Number of steps in episode
            success_rate: Success rate (0-1)
            collision_rate: Collision rate (0-1)
            additional_metrics: Additional custom metrics to log
        """
        if self.writer is None or not self.tb_metrics.get('episode', True):
            return

        self.writer.add_scalar('Episode/reward', episode_reward, self.episode_count)
        self.writer.add_scalar('Episode/length', episode_length, self.episode_count)
        self.writer.add_scalar('Episode/success_rate', success_rate, self.episode_count)
        self.writer.add_scalar('Episode/collision_rate', collision_rate, self.episode_count)

        # Log additional custom metrics
        if additional_metrics:
            for name, value in additional_metrics.items():
                self.writer.add_scalar(f'Episode/{name}', value, self.episode_count)

        # Flush to ensure metrics are written
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
