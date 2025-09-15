'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 18:33:45
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/q_learning.py
Description  : Q-learning algorithm implementation for tabular RL
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import numpy as np
import pickle
import random
from typing import Dict, Any
from collections import deque
from loguru import logger

from .base_algorithm import BaseAlgorithm


class QLearningAlgorithm(BaseAlgorithm):
    """
    Tabular Q-learning algorithm implementation.

    Uses epsilon-greedy exploration and tabular Q-values for discrete state/action spaces.
    """

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        """
        Initialize Q-learning algorithm.

        Args:
            config: Q-learning specific configuration
            state_dim: Total number of discrete states
            action_dim: Number of discrete actions
        """
        super().__init__(config, state_dim=state_dim, action_dim=action_dim)

        # Q-learning specific parameters
        self.epsilon = config.get('epsilon', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)

        # Store speed mapping from config
        self.speed_actions = config.get('speed_actions', [0, 15, 30, 45, 60])

        # Verify action_dim matches speed_actions
        if len(self.speed_actions) != action_dim:
            logger.warning(
                f"Mismatch: speed_actions length ({len(self.speed_actions)}) != action_dim ({action_dim})")
            logger.warning(f"Speed actions: {self.speed_actions}")

        # Initialize Q-table
        self.q_table = np.zeros((state_dim, action_dim))

        # Experience replay for batch updates (optional)
        self.use_experience_replay = config.get('use_experience_replay', False)
        self.memory_size = config.get('memory_size', 1000)
        self.batch_size = config.get('batch_size', 32)

        if self.use_experience_replay:
            self.memory = deque(maxlen=self.memory_size)

        # Training metrics
        self.training_metrics = {
            'q_values_mean': 0.0,
            'q_values_std': 0.0,
            'epsilon': self.epsilon,
            'exploration_actions': 0,
            'exploitation_actions': 0
        }

        self.training = True

        logger.info(
            f"Q-learning initialized with {state_dim} states, {action_dim} actions")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (should be discrete state index)
            training: Whether in training mode

        Returns:
            Action index
        """
        try:
            # Convert state to integer if needed
            if isinstance(state, np.ndarray):
                state_idx = int(
                    state.item()) if state.size == 1 else int(state[0])
            else:
                state_idx = int(state)

            # Ensure valid state index
            state_idx = max(0, min(state_idx, self.state_dim - 1))

            if training and random.random() < self.epsilon:
                # Exploration
                action = random.randint(0, self.action_dim - 1)
                self.training_metrics['exploration_actions'] += 1
            else:
                # Exploitation
                action = np.argmax(self.q_table[state_idx])
                if training:
                    self.training_metrics['exploitation_actions'] += 1

            return action

        except Exception as e:
            logger.error(f"Error in action selection: {e}")
            return random.randint(0, self.action_dim - 1)

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """
        Store transition for learning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        try:
            # Convert states to integers
            state_idx = int(state.item()) if isinstance(
                state, np.ndarray) and state.size == 1 else int(state)
            next_state_idx = int(next_state.item()) if isinstance(
                next_state, np.ndarray) and next_state.size == 1 else int(next_state)

            # Ensure valid indices
            state_idx = max(0, min(state_idx, self.state_dim - 1))
            next_state_idx = max(0, min(next_state_idx, self.state_dim - 1))

            if self.use_experience_replay:
                # Store in replay buffer
                self.memory.append(
                    (state_idx, action, reward, next_state_idx, done))
            else:
                # Direct Q-learning update
                self._update_q_value(
                    state_idx, action, reward, next_state_idx, done)

        except Exception as e:
            logger.error(f"Error storing transition: {e}")

    def update(self) -> Dict[str, float]:
        """
        Update Q-values (batch update if using experience replay).

        Returns:
            Training metrics
        """
        try:
            if self.use_experience_replay and len(self.memory) >= self.batch_size:
                # Sample random batch
                batch = random.sample(self.memory, self.batch_size)

                for state_idx, action, reward, next_state_idx, done in batch:
                    self._update_q_value(
                        state_idx, action, reward, next_state_idx, done)

            # Update epsilon
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Update metrics
            self.training_metrics.update({
                'q_values_mean': float(np.mean(self.q_table)),
                'q_values_std': float(np.std(self.q_table)),
                'epsilon': self.epsilon,
                'nonzero_q_values': int(np.count_nonzero(self.q_table))
            })

            self.training_step += 1

            return self.training_metrics.copy()

        except Exception as e:
            logger.error(f"Error in Q-learning update: {e}")
            return self.training_metrics.copy()

    def _update_q_value(self, state_idx: int, action: int, reward: float,
                        next_state_idx: int, done: bool):
        """
        Perform single Q-value update.

        Args:
            state_idx: Current state index
            action: Action taken
            reward: Reward received
            next_state_idx: Next state index
            done: Whether episode is done
        """
        try:
            current_q = self.q_table[state_idx, action]

            if done:
                target_q = reward
            else:
                max_next_q = np.max(self.q_table[next_state_idx])
                target_q = reward + self.discount_factor * max_next_q

            # Q-learning update
            self.q_table[state_idx, action] = current_q + \
                self.learning_rate * (target_q - current_q)

        except Exception as e:
            logger.error(f"Error in Q-value update: {e}")

    def reset_episode(self):
        """Reset for new episode."""
        self.episode_count += 1

        # Reset action counters
        self.training_metrics['exploration_actions'] = 0
        self.training_metrics['exploitation_actions'] = 0

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information."""
        return {
            'algorithm': 'Q-Learning',
            'algorithm_type': 'tabular_q_learning',
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'epsilon': self.epsilon,
            'q_table_shape': self.q_table.shape,
            'nonzero_q_values': int(np.count_nonzero(self.q_table)),
            'memory_size': len(self.memory) if self.use_experience_replay else 0,
            'use_experience_replay': self.use_experience_replay,
            'speed_actions': self.speed_actions
        }

    def save(self, path: str):
        """Save Q-table and algorithm state."""
        try:
            save_data = {
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                'config': self.config
            }

            with open(path, 'wb') as f:
                pickle.dump(save_data, f)

            logger.success(f"Q-learning model saved to {path}")

        except Exception as e:
            logger.error(f"Error saving Q-learning model: {e}")

    def load(self, path: str):
        """Load Q-table and algorithm state."""
        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)

            self.q_table = save_data['q_table']
            self.epsilon = save_data['epsilon']
            self.training_step = save_data['training_step']
            self.episode_count = save_data['episode_count']

            logger.success(f"Q-learning model loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading Q-learning model: {e}")

    def get_q_table(self) -> np.ndarray:
        """Get current Q-table (for debugging/visualization)."""
        return self.q_table.copy()
