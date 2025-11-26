'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-06
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/dqn.py
Description  : Deep Q-Network (DQN) algorithm implementation using PyTorch
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gc
from collections import deque
from typing import Dict, Any
from loguru import logger

from .base_algorithm import BaseAlgorithm


class QNetwork(nn.Module):
    """Deep Q-Network for value function approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [64, 32]):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Input state dimension
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """Forward pass through network"""
        return self.network(state)


class ExperienceReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Store transition"""
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        """Sample random batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNAlgorithm(BaseAlgorithm):
    """
    Deep Q-Network algorithm implementation with PyTorch.
    
    Uses neural networks for Q-value approximation, experience replay,
    and target networks for stable learning.
    """
    
    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        """
        Initialize DQN algorithm.
        
        Args:
            config: DQN specific configuration
            state_dim: Dimension of continuous state space
            action_dim: Number of discrete actions
        """
        super().__init__(config, state_dim, action_dim)
        
        # DQN specific parameters
        self.epsilon = config.get('epsilon', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.target_update_freq = config.get('target_update_freq', 100)
        
        # Network architecture
        hidden_dims = config.get('hidden_dims', [64, 32])
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory_size = config.get('memory_size', 10000)
        self.batch_size = config.get('batch_size', 64)
        self.memory = ExperienceReplayBuffer(self.memory_size)
        
        # Define discrete action space (speeds in m/s)
        self.speed_actions = config.get('speed_actions', [0.0, 5.0, 8.0, 12.0, 15.0])
        if len(self.speed_actions) != action_dim:
            logger.warning(f"Speed actions count ({len(self.speed_actions)}) != action_dim ({action_dim})")
            # Create evenly spaced speeds
            max_speed = config.get('max_speed', 15.0)
            self.speed_actions = [i * max_speed / (action_dim - 1) for i in range(action_dim)]
        
        # Training metrics
        self.training_metrics = {
            'loss': 0.0,
            'q_values_mean': 0.0,
            'epsilon': self.epsilon,
            'target_updates': 0,
            'memory_size': 0
        }
        
        self.training = True
        
        logger.info(f"DQN initialized with {state_dim}D state, {action_dim} actions, device: {self.device}")
        logger.info(f"Speed actions: {self.speed_actions}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode
            
        Returns:
            Target speed (float)
        """
        try:
            # Convert to tensor
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.FloatTensor([state]).to(self.device)
            
            if training and random.random() < self.epsilon:
                # Random action (exploration)
                action_idx = random.randint(0, self.action_dim - 1)
            else:
                # Greedy action (exploitation)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    action_idx = q_values.argmax().item()
            
            # Convert action index to target speed
            target_speed = self.speed_actions[action_idx]
            return float(target_speed)
            
        except Exception as e:
            logger.error(f"Error in DQN action selection: {e}")
            # Return safe default speed
            return 8.0
    
    def store_transition(self, state: np.ndarray, action: float, reward: float, 
                        next_state: np.ndarray, done: bool):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Target speed that was used
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        try:
            # Convert speed action to action index
            action_idx = self._speed_to_action_idx(action)
            
            # Convert to tensors
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            
            self.memory.push(
                state_tensor, 
                action_idx, 
                reward, 
                next_state_tensor, 
                done
            )
            
        except Exception as e:
            logger.error(f"Error storing DQN transition: {e}")
    
    def update(self) -> Dict[str, float]:
        """
        Update Q-network using experience replay.

        Returns:
            Training metrics
        """
        try:
            # Only update if we have enough samples
            if len(self.memory) < self.batch_size:
                return self.training_metrics

            # Sample batch
            transitions = self.memory.sample(self.batch_size)

            # Unpack batch
            states = torch.stack([t[0] for t in transitions]).to(self.device)
            actions = torch.LongTensor([t[1] for t in transitions]).to(self.device)
            rewards = torch.FloatTensor([t[2] for t in transitions]).to(self.device)
            next_states = torch.stack([t[3] for t in transitions]).to(self.device)
            dones = torch.BoolTensor([t[4] for t in transitions]).to(self.device)

            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

            # Next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)

            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

            # Optimize with gradient clipping for stability
            self.optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm BEFORE clipping (for monitoring)
            grad_norm_pre = self._compute_grad_norm(self.q_network.parameters())

            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

            # Compute gradient norm AFTER clipping
            grad_norm_post = self._compute_grad_norm(self.q_network.parameters())

            self.optimizer.step()

            # Update target network periodically
            if self.training_step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.training_metrics['target_updates'] += 1

            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Update metrics (compute before deleting tensors)
            with torch.no_grad():
                q_mean = self.q_network(states).mean().item()
            loss_value = loss.item()

            self.training_metrics.update({
                'loss': loss_value,
                'q_values_mean': q_mean,
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'grad_norm_pre': grad_norm_pre,
                'grad_norm_post': grad_norm_post
            })

            # TensorBoard logging (using base class methods)
            self.log_scalar('Loss/dqn', loss_value, category='losses')
            self.log_scalar('Q_values/mean', q_mean, category='q_values')
            self.log_scalar('Exploration/epsilon', self.epsilon, category='episode')
            self.log_scalar('Buffer/size', len(self.memory), category='buffer')
            self.log_scalar('Gradients/q_network_pre_clip', grad_norm_pre, category='losses')
            self.log_scalar('Gradients/q_network_post_clip', grad_norm_post, category='losses')

            # Explicit cleanup to prevent memory leaks (do this AFTER using tensors for metrics)
            del states, actions, rewards, next_states, dones
            del current_q_values, next_q_values, target_q_values
            del transitions, loss

            self.training_step += 1

            # Periodic deep cleanup to prevent CUDA memory fragmentation
            if self.training_step % 500 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return self.training_metrics.copy()

        except Exception as e:
            logger.error(f"Error in DQN update: {e}")
            # Cleanup even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return self.training_metrics.copy()
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_count += 1

        # GPU memory cleanup to prevent slowdown over episodes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # Log GPU memory usage periodically for debugging
            if self.episode_count % 5 == 0:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
                cached = torch.cuda.memory_reserved(self.device) / 1024**2  # MB
                logger.debug(f"DQN Episode {self.episode_count} GPU memory: "
                           f"allocated={allocated:.1f}MB, cached={cached:.1f}MB")

        # Force garbage collection every few episodes to prevent memory leaks
        if self.episode_count % 3 == 0:
            gc.collect()

    def _compute_grad_norm(self, parameters) -> float:
        """
        Compute the L2 norm of gradients for monitoring training stability.

        Args:
            parameters: Iterator of model parameters

        Returns:
            Total gradient norm (float)
        """
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information"""
        return {
            'algorithm': 'DQN',
            'algorithm_type': 'deep_q_network',
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'memory_capacity': self.memory_size,
            'target_updates': self.training_metrics.get('target_updates', 0),
            'device': str(self.device),
            'speed_actions': self.speed_actions
        }
    
    def save(self, path: str):
        """Save DQN model"""
        try:
            save_data = {
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                'config': self.config,
                'speed_actions': self.speed_actions
            }
            
            torch.save(save_data, path)
            logger.info(f"DQN model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving DQN model: {e}")
    
    def load(self, path: str):
        """Load DQN model"""
        try:
            save_data = torch.load(path, map_location=self.device, weights_only=False)
            
            self.q_network.load_state_dict(save_data['q_network_state_dict'])
            self.target_network.load_state_dict(save_data['target_network_state_dict'])
            self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
            self.epsilon = save_data['epsilon']
            self.training_step = save_data['training_step']
            self.episode_count = save_data['episode_count']
            self.speed_actions = save_data.get('speed_actions', self.speed_actions)
            
            logger.info(f"DQN model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading DQN model: {e}")
    
    def _speed_to_action_idx(self, speed: float) -> int:
        """Convert target speed to closest action index"""
        distances = [abs(speed - action_speed) for action_speed in self.speed_actions]
        return distances.index(min(distances))