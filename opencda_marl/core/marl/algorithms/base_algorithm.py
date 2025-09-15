'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 18:29:51
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/base_algorithm.py
Description  : Base class for all reinforcement learning algorithms
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
from loguru import logger


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

        logger.success(
            f"Initialized {self.__class__.__name__} with state_dim={state_dim}, action_dim={action_dim}")

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
