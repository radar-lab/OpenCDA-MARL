'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-09
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/td3.py
Description  : Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm for multi-agent intersection control
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Any, List, Tuple
from loguru import logger

from .base_algorithm import BaseAlgorithm
from .smart_replay_buffer import SmartReplayBuffer, PrioritizedReplayBuffer


class LSTMEncoder(nn.Module):
    """LSTM encoder for processing multi-agent observations"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)

        # Initialize LSTM weights properly
        self._initialize_weights()

    def forward(self, agent_sequences):
        """
        Process sequences of agent observations

        Args:
            agent_sequences: Tensor of shape [batch_size, seq_len, input_size]

        Returns:
            Hidden state from final timestep [batch_size, hidden_size]
        """
        # Initialize hidden states
        batch_size = agent_sequences.size(0)
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size, device=agent_sequences.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size, device=agent_sequences.device)

        # Forward through LSTM
        lstm_out, (hn, cn) = self.lstm(agent_sequences, (h0, c0))

        # Return final hidden state
        return hn[-1]  # Take the last layer's hidden state

    def _initialize_weights(self):
        """Initialize LSTM weights properly for stable training"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights: use Xavier initialization
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights: use orthogonal initialization
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Initialize biases to zero, except forget gate bias to 1
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                with torch.no_grad():
                    param.data[n//4:n//2] = 1.0


class Actor(nn.Module):
    """Actor network for TD3 based on AdvRAIM architecture"""

    def __init__(self, state_dim: int, lstm_hidden_dim: int, action_dim: int,
                 max_action: float = 1.0, min_action: float = 0.0, motion_planner_config: dict = None, 
                 is_target: bool = False):
        super(Actor, self).__init__()

        self.max_action = max_action
        self.min_action = min_action
        self.is_target = is_target
        self.forward_count = 0  # Counter for logging frequency

        # Use configurable motion planner architecture
        if motion_planner_config is None:
            # Default ITS_Sim architecture
            input_dim = state_dim + lstm_hidden_dim
            layer_dims = [input_dim, 1024, 1024, 512, 256]
            output_dim = 256
            num_layers = len(layer_dims) - 1
        else:
            layer_dims = motion_planner_config['input_dim']
            output_dims = motion_planner_config['output_dim']
            num_layers = motion_planner_config['num_layers']
            output_dim = output_dims[-1]

        # Motion planner architecture from config
        motion_planner_layers = []
        for i in range(num_layers):
            if motion_planner_config is None:
                # Default case
                in_dim = layer_dims[i]
                out_dim = layer_dims[i+1]
            else:
                # Config case - properly pair input_dim[i] -> output_dim[i]
                in_dim = layer_dims[i]
                out_dim = output_dims[i]
            
            motion_planner_layers.append(nn.Linear(in_dim, out_dim))
            # Add ReLU for all layers except the last one
            if i < num_layers - 1:
                motion_planner_layers.append(nn.ReLU())
        
        self.motion_planner = nn.Sequential(*motion_planner_layers)
        
        # Final output layer (if the last layer doesn't output action_dim directly)
        if output_dim != action_dim:
            self.output_layer = nn.Linear(output_dim, action_dim)
        else:
            # Motion planner already outputs the right dimension
            self.output_layer = None

        # Initialize weights properly (only log for non-target networks)
        self._initialize_weights()

    def forward(self, ego_state, multi_agent_context):
        """
        Forward pass through actor network

        Args:
            ego_state: Ego agent features [batch_size, state_dim]
            multi_agent_context: LSTM encoding [batch_size, lstm_hidden_dim]

        Returns:
            Action scaled to [min_action, max_action] representing speed in km/h
        """
        # Concatenate ego state and multi-agent context
        combined_input = torch.cat([ego_state, multi_agent_context], dim=1)

        # Forward through motion planner
        x = self.motion_planner(combined_input)
        
        # Output layer with sigmoid activation (if separate output layer exists)
        if self.output_layer is not None:
            x = self.output_layer(x)
        
        # Apply tanh activation for better gradient flow and exploration
        # tanh outputs in [-1, 1], then scale to [min_action, max_action]
        tanh_out = torch.tanh(x)
        # Scale from [-1, 1] to [0, 1] then to [min_action, max_action]
        action = self.min_action + (self.max_action - self.min_action) * (tanh_out + 1) * 0.5
        self.forward_count = (self.forward_count + 1) % 100000
        
        # Debug logging for network outputs (periodic)
        if not self.is_target and self.forward_count % 5000 == 0:
            logger.info(f"Actor Debug (batch mean) - Pre-tanh: {x.mean().item():.3f}, Tanh: {tanh_out.mean().item():.3f}, Action: {action.mean().item():.1f} km/h")
        
        return action

    def _initialize_weights(self):
        """Initialize network weights properly for stable training"""
        # Find the final layer for special initialization
        final_layer_idx = -1
        for i, layer in enumerate(self.motion_planner):
            if isinstance(layer, nn.Linear):
                final_layer_idx = i
        
        # Initialize motion planner layers
        for i, layer in enumerate(self.motion_planner):
            if isinstance(layer, nn.Linear):
                if i == final_layer_idx and self.output_layer is None:
                    # Final layer before tanh - initialize for middle speed range
                    nn.init.normal_(layer.weight, mean=0, std=0.05)  # Small random weights
                    nn.init.constant_(layer.bias, 0.5)  # tanh(0.5) ≈ 0.46 → ~47 km/h start
                else:
                    # Hidden layers use Xavier
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize separate output layer (if it exists) 
        if self.output_layer is not None:
            # Output layer before tanh - initialize for middle speed range
            nn.init.normal_(self.output_layer.weight, mean=0, std=0.05)
            nn.init.constant_(self.output_layer.bias, 0.5)  # tanh(0.5) ≈ 0.46 → ~47 km/h start


class Critic(nn.Module):
    """Twin Critic networks for TD3"""

    def __init__(self, state_dim: int, lstm_hidden_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256], motion_planner_config: dict = None):
        super(Critic, self).__init__()

        # Combined input dimension (ego features + LSTM encoding + action)
        input_dim = state_dim + lstm_hidden_dim + action_dim

        # Use motion planner architecture for critic or fallback to hidden_dims
        if motion_planner_config is not None:
            # Use motion planner output dimensions for critic architecture
            critic_dims = motion_planner_config['output_dim'][:-1]  # Exclude final output (which is 1 for actor)
            critic_dims.append(1)  # Add final output dimension for Q-value
        else:
            critic_dims = hidden_dims + [1]

        # Q1 network
        q1_layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(critic_dims[:-1]):
            q1_layers.append(nn.Linear(prev_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q1_layers.append(nn.Linear(prev_dim, critic_dims[-1]))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2 network (twin)
        q2_layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(critic_dims[:-1]):
            q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q2_layers.append(nn.Linear(prev_dim, critic_dims[-1]))
        self.q2 = nn.Sequential(*q2_layers)

        # Initialize weights properly
        self._initialize_weights()

    def forward(self, ego_state, multi_agent_context, action):
        """Forward pass through both critic networks"""
        # Concatenate all inputs
        combined_input = torch.cat(
            [ego_state, multi_agent_context, action], dim=1)

        q1_value = self.q1(combined_input)
        q2_value = self.q2(combined_input)

        return q1_value, q2_value

    def Q1(self, ego_state, multi_agent_context, action):
        """Forward pass through first critic only"""
        combined_input = torch.cat(
            [ego_state, multi_agent_context, action], dim=1)
        return self.q1(combined_input)

    def _initialize_weights(self):
        """Initialize network weights properly for stable training"""
        for network in [self.q1, self.q2]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    # Use Xavier initialization for critic networks
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)


class ExperienceReplayBuffer:
    """Experience replay buffer for TD3"""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, ego_state, multi_agent_context, action, reward,
             next_ego_state, next_multi_agent_context, done):
        """Store transition"""
        self.memory.append((ego_state, multi_agent_context, action, reward,
                           next_ego_state, next_multi_agent_context, done))

    def sample(self, batch_size: int):
        """Sample random batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TD3Algorithm(BaseAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm
    for multi-agent intersection control.

    Features:
    - LSTM encoder for multi-agent context
    - Twin critics to reduce overestimation bias
    - Delayed policy updates
    - Target policy smoothing
    """

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        """
        Initialize TD3 algorithm

        Args:
            config: TD3 specific configuration
            state_dim: Individual agent state dimension (7D)
            action_dim: Action dimension (1 for continuous speed control)
        """
        super().__init__(config, state_dim, action_dim)

        # Read configuration for network architectures
        self.conflict_encoder_config = config.get('conflict_encoder', {
            'input_size': state_dim,
            'hidden_size': 256,
            'num_layers': 1
        })
        
        self.motion_planner_config = config.get('motion_planner', {
            'num_layers': 5,
            'input_dim': [state_dim + self.conflict_encoder_config['hidden_size'], 1024, 1024, 512, 256],
            'output_dim': [1024, 1024, 512, 256, 1]
        })

        # TD3 specific parameters
        self.lstm_hidden_size = self.conflict_encoder_config['hidden_size']
        self.lstm_num_layers = self.conflict_encoder_config['num_layers']
        self.tau = config.get('tau', 0.005)
        self.policy_noise = config.get('policy_noise', 0.2)
        self.noise_clip = config.get('noise_clip', 0.3)
        self.exploration_noise = config.get('exploration_noise', 0.1)
        self.policy_freq = config.get('policy_freq', 2)
        self.max_action = config.get('max_action', 60.0)  # Maximum speed in km/h
        self.min_action = config.get('min_action', 0.0)   # Minimum speed in km/h
        
        # Warmup configuration
        self.warmup_steps = config.get('warmup_steps', 100)  # Use vanilla agent during warmup

        # Legacy network architecture (for compatibility)
        critic_hidden_dims = config.get('critic_hidden_dims', [256, 256])

        # Learning rates
        self.actor_lr = config.get('learning_rate_actor', 1e-5)
        self.critic_lr = config.get('learning_rate_critic', 1e-4)

        # Device configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize LSTM encoder
        self.lstm_encoder = LSTMEncoder(
            state_dim, self.lstm_hidden_size, self.lstm_num_layers
        ).to(self.device)

        # Initialize Actor networks with motion planner config
        self.actor = Actor(
            state_dim, self.lstm_hidden_size, action_dim, self.max_action, self.min_action, self.motion_planner_config, is_target=False
        ).to(self.device)
        self.actor_target = Actor(
            state_dim, self.lstm_hidden_size, action_dim, self.max_action, self.min_action, self.motion_planner_config, is_target=True
        ).to(self.device)

        # Copy parameters to target
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Initialize Critic networks with motion planner config
        self.critic = Critic(
            state_dim, self.lstm_hidden_size, action_dim, critic_hidden_dims, self.motion_planner_config
        ).to(self.device)
        self.critic_target = Critic(
            state_dim, self.lstm_hidden_size, action_dim, critic_hidden_dims, self.motion_planner_config
        ).to(self.device)

        # Copy parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)

        # Experience replay buffer configuration
        self.memory_size = config.get('memory_size', 25000)  # Sized to prevent FIFO losses (~14k/episode)
        self.batch_size = config.get('batch_size', 100)
        
        # Prioritized Experience Replay (PER) configuration
        self.use_per = config.get('use_per', True)  # Enable PER by default
        self.per_alpha = config.get('per_alpha', 0.6)  # Priority exponent
        self.per_beta = config.get('per_beta', 0.4)   # Initial importance sampling
        self.per_beta_increment = config.get('per_beta_increment', 0.001)
        
        # Auto-clear configuration for preventing stale experiences and FIFO losses
        self.clear_episodes = config.get('clear_episodes', 1)  # Clear every episode to prevent overflow
        self.clear_keep_ratio = config.get('clear_keep_ratio', 0.6)  # Keep 60% newest
        self.clear_interval = config.get('clear_interval', None)  # Alternative: clear every N transitions
        
        # Choose replay buffer based on configuration
        if self.use_per:
            logger.info(f"Using Prioritized Experience Replay (alpha={self.per_alpha}, beta={self.per_beta})")
            self.memory = PrioritizedReplayBuffer(
                capacity=self.memory_size,
                alpha=self.per_alpha,
                beta=self.per_beta,
                beta_increment=self.per_beta_increment
            )
        else:
            logger.info("Using SmartReplayBuffer with recency bias")
            self.recency_ratio = config.get('recency_ratio', 0.5)  # 50% recent samples
            self.memory = SmartReplayBuffer(capacity=self.memory_size, 
                                           recency_ratio=self.recency_ratio)

        # Training metrics
        self.training_metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'q1_mean': 0.0,
            'q2_mean': 0.0,
            # Delayed policy updates (actor updates every policy_freq steps)
            'target_updates': 0,
            'memory_size': 0
        }

        self.training = True
        self.training_step = 0  # Required for delayed policy updates

        logger.info(
            f"TD3 initialized with {state_dim}D states, LSTM hidden: {self.lstm_hidden_size}, device: {self.device}")

    def select_action(self, multi_agent_obs: Dict[str, np.ndarray], ego_agent_id: str, training: bool = True) -> float:
        """
        Select action for ego agent using multi-agent context

        Args:
            multi_agent_obs: Dict of all agents' observations
            ego_agent_id: ID of the ego agent
            training: Whether in training mode

        Returns:
            Continuous action (speed in km/h)
        """
        try:
            # During warmup phase, return None to use vanilla agent
            if len(self.memory) < self.warmup_steps:
                # Log warmup progress only every 2000 samples to reduce verbosity
                if len(self.memory) % 2000 == 0:
                    logger.info(f"TD3: Warmup phase {len(self.memory)}/{self.warmup_steps}, using vanilla agent")
                return None
            
            # Convert to tensors and get multi-agent context
            ego_state, multi_agent_context = self._prepare_inputs(
                multi_agent_obs, ego_agent_id)

            with torch.no_grad():
                # Get action from actor
                raw_action = self.actor(ego_state, multi_agent_context)

                # Add exploration noise during training
                if training:
                    noise = torch.randn_like(
                        raw_action) * self.exploration_noise
                    action = torch.clamp(
                        raw_action + noise, self.min_action, self.max_action)
                else:
                    action = raw_action

                # Extract scalar value from action tensor
                final_speed = action.squeeze().item()

                if final_speed > self.max_action:
                    final_speed = self.max_action
                
                # Debug logging for action selection (periodic)
                if len(self.memory) % 2000 == 0:
                    raw_val = raw_action.squeeze().item()
                    logger.info(f"TD3 Action Debug - Raw: {raw_val:.1f}, +Noise: {final_speed:.1f}, Episode: {self.episode_count}")
                
                # Force exploration in early episodes to break out of 30 km/h trap
                if self.episode_count < 3 and training and not self._pretrained:
                    forced_speed = np.random.uniform(35, 60)  # Force higher speeds
                    logger.debug(f"Forced exploration Episode {self.episode_count}: {forced_speed:.1f} km/h (was {final_speed:.1f})")
                    return forced_speed

                return final_speed

        except Exception as e:
            logger.error(f"Error in TD3 action selection: {e}")
            return None

    def store_transition(self, multi_agent_obs: Dict[str, np.ndarray], ego_agent_id: str,
                         action: float, reward: float, next_multi_agent_obs: Dict[str, np.ndarray],
                         done: bool):
        """
        Store transition in replay buffer

        Args:
            multi_agent_obs: Current multi-agent observations
            ego_agent_id: Ego agent ID
            action: Action taken
            reward: Reward received
            next_multi_agent_obs: Next multi-agent observations
            done: Whether episode is done
        """
        try:
            # Prepare current state
            ego_state, multi_agent_context = self._prepare_inputs(
                multi_agent_obs, ego_agent_id)

            # Prepare next state
            next_ego_state, next_multi_agent_context = self._prepare_inputs(
                next_multi_agent_obs, ego_agent_id)

            # Store transition - SmartReplayBuffer handles the storage
            self.memory.push(
                ego_state.detach().cpu().numpy(),
                multi_agent_context.detach().cpu().numpy(),
                action,  # Store as scalar, not array
                reward,
                next_ego_state.detach().cpu().numpy(),
                next_multi_agent_context.detach().cpu().numpy(),
                done
            )
            
            # Pre-emptive clearing to prevent FIFO losses (before hitting capacity)
            buffer_usage = len(self.memory) / self.memory_size
            if buffer_usage >= 0.90 and hasattr(self.memory, 'clear_old'):  # At 90% capacity, pre-emptively clear
                old_size = len(self.memory)
                self.memory.clear_old(self.clear_keep_ratio)
                new_size = len(self.memory)
                logger.info(f"TD3: Pre-emptive clear at {buffer_usage:.1%} capacity: "
                           f"{old_size} → {new_size} (kept {self.clear_keep_ratio*100:.0f}% newest)")
            
            # Auto-clear based on transition count if configured
            elif self.clear_interval and hasattr(self.memory, 'total_stored') and hasattr(self.memory, 'clear_old'):
                if self.memory.total_stored > 0 and self.memory.total_stored % self.clear_interval == 0:
                    old_size = len(self.memory)
                    self.memory.clear_old(self.clear_keep_ratio)
                    new_size = len(self.memory)
                    logger.info(f"TD3: Interval-based clear after {self.memory.total_stored} transitions: "
                               f"{old_size} → {new_size} (kept {self.clear_keep_ratio*100:.0f}% newest)")
            
            # Log buffer stats occasionally with capacity usage
            if len(self.memory) % 5000 == 0:
                stats = self.memory.get_stats() if hasattr(self.memory, 'get_stats') else {}
                usage_pct = len(self.memory) / self.memory_size * 100
                
                if self.use_per:
                    # PER buffer stats
                    logger.debug(f"TD3 PER Buffer: {len(self.memory)}/{self.memory_size} ({usage_pct:.1f}%), "
                               f"beta={stats.get('beta', 0):.3f}, avg_priority={stats.get('avg_priority', 0):.4f}")
                else:
                    # SmartReplayBuffer stats
                    logger.debug(f"TD3 Buffer: {len(self.memory)}/{self.memory_size} ({usage_pct:.1f}%), "
                               f"total_stored={stats.get('total_stored', 0)}")
                
                # Warn if approaching capacity
                if usage_pct > 85:
                    logger.warning(f"Buffer at {usage_pct:.1f}% capacity - may trigger pre-emptive clearing soon")

        except Exception as e:
            logger.error(f"Error storing TD3 transition: {e}")
            import traceback
            traceback.print_exc()

    def update(self) -> Dict[str, float]:
        """
        Update TD3 networks using experience replay

        Returns:
            Training metrics
        """
        try:
            # Only update if we have enough samples
            if len(self.memory) < self.batch_size:
                return self.training_metrics

            # Sample batch (different handling for PER vs SmartReplay)
            if self.use_per:
                transitions, indices, weights = self.memory.sample(self.batch_size)
                # Convert weights to torch tensor WITHOUT gradients
                importance_weights = torch.FloatTensor(weights).to(self.device).detach()
            else:
                transitions = self.memory.sample(self.batch_size)
                indices = None
                importance_weights = None

            # Unpack batch - squeeze the stored [1, dim] arrays to [dim]
            # Create tensors with gradient tracking for critic learning
            ego_states = torch.FloatTensor(
                np.stack([t[0].squeeze(0) for t in transitions])).to(self.device).requires_grad_(True)
            multi_agent_contexts = torch.FloatTensor(
                np.stack([t[1].squeeze(0) for t in transitions])).to(self.device).requires_grad_(True)
            # Actions: collect scalar actions and reshape to [batch_size, 1] for critic
            actions = torch.FloatTensor(
                np.array([t[2] for t in transitions])).unsqueeze(1).to(self.device).requires_grad_(False)
            rewards = torch.FloatTensor(
                np.array([t[3] for t in transitions])).to(self.device)
            next_ego_states = torch.FloatTensor(
                np.stack([t[4].squeeze(0) for t in transitions])).to(self.device)
            next_multi_agent_contexts = torch.FloatTensor(
                np.stack([t[5].squeeze(0) for t in transitions])).to(self.device)
            dones = torch.BoolTensor(
                np.array([t[6] for t in transitions])).to(self.device)
            
            # Debug logging for tensor shapes (only every 1000 updates)
            if self.training_step % 1000 == 0:
                logger.info(f"TD3 Update batch shapes: ego_states={ego_states.shape}, actions={actions.shape}, rewards={rewards.shape}")
                logger.info(f"Sample action values: {actions[:5].flatten().tolist()}")

            # Update Critic (with TD-error calculation for PER)
            if self.use_per:
                critic_loss, td_errors = self._update_critic_with_per(
                    ego_states, multi_agent_contexts, actions, rewards,
                    next_ego_states, next_multi_agent_contexts, dones, importance_weights
                )
                # Update priorities based on TD-errors
                self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
            else:
                critic_loss = self._update_critic(ego_states, multi_agent_contexts, actions, rewards,
                                                  next_ego_states, next_multi_agent_contexts, dones)

            # Update Actor (delayed)
            if self.training_step % self.policy_freq == 0:
                actor_loss = self._update_actor(
                    ego_states, multi_agent_contexts)
                self._update_target_networks()
                # Count delayed policy updates
                self.training_metrics['target_updates'] += 1
                # Update actor_loss in metrics
                self.training_metrics['actor_loss'] = actor_loss
            # Note: actor_loss remains from previous update when no policy update occurs

            # Update metrics
            self.training_metrics.update({
                'critic_loss': critic_loss,
                'memory_size': len(self.memory)
            })

            # Log training progress periodically (every 200 steps)
            if self.training_step % 200 == 0:
                current_actor_loss = self.training_metrics.get('actor_loss', 0.0)
                target_updates = self.training_metrics.get('target_updates', 0)
                logger.info(
                    f"TD3 Step {self.training_step}: critic_loss={critic_loss:.4f}, actor_loss={current_actor_loss:.4f}, policy_updates={target_updates}, memory={len(self.memory)}")

            self.training_step += 1
            return self.training_metrics.copy()

        except Exception as e:
            logger.error(f"Error in TD3 update (step {self.training_step}): {e}")
            import traceback
            logger.error(f"TD3 update traceback:\n{traceback.format_exc()}")
            return self.training_metrics.copy()

    def _prepare_inputs(self, multi_agent_obs: Dict[str, np.ndarray], ego_agent_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for TD3 networks

        Returns:
            ego_state: Ego agent features [1, state_dim]
            multi_agent_context: LSTM encoding [1, lstm_hidden_size]
        """
        # Get ego agent observation
        if ego_agent_id not in multi_agent_obs:
            # Fallback to first available agent
            ego_agent_id = list(multi_agent_obs.keys())[0]

        ego_obs = multi_agent_obs[ego_agent_id]
        ego_state = torch.FloatTensor(ego_obs).unsqueeze(0).to(self.device)

        # Prepare sequence of all agents for LSTM
        agent_sequence = []

        # Add ego agent first
        agent_sequence.append(ego_obs)

        # Add other agents sorted by distance
        other_agents = []
        for agent_id, obs in multi_agent_obs.items():
            if agent_id != ego_agent_id:
                # Calculate rough distance using relative position
                # First two features are relative position
                rel_x, rel_y = obs[0], obs[1]
                distance = np.sqrt(rel_x*rel_x + rel_y*rel_y)
                other_agents.append((obs, distance))

        # Sort by distance and add to sequence
        other_agents.sort(key=lambda x: x[1])
        for obs, _ in other_agents:
            agent_sequence.append(obs)

        # Convert to tensor and encode with LSTM
        if len(agent_sequence) > 1:
            # Convert list of observations to tensor
            sequence_array = np.array(agent_sequence)
            sequence_tensor = torch.FloatTensor(
                sequence_array).unsqueeze(0).to(self.device)
            multi_agent_context = self.lstm_encoder(sequence_tensor)
        else:
            # Single agent case - use zero context
            multi_agent_context = torch.zeros(
                1, self.lstm_hidden_size).to(self.device)

        return ego_state, multi_agent_context

    def _update_critic(self, ego_states, multi_agent_contexts, actions, rewards,
                       next_ego_states, next_multi_agent_contexts, dones):
        """Update critic networks"""
        with torch.no_grad():
            # Target policy smoothing
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            # Get next actions from target actor
            next_actions = self.actor_target(
                next_ego_states, next_multi_agent_contexts)
            next_actions = torch.clamp(
                next_actions + noise, self.min_action, self.max_action)

            # Get target Q values (take minimum of twin critics)
            target_q1, target_q2 = self.critic_target(
                next_ego_states, next_multi_agent_contexts, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Debug target Q shapes (periodic logging)
            should_log_details = (self.training_step % 1000 == 0)
            if should_log_details:
                logger.info(f"TD3 Target Q shapes: target_q1={target_q1.shape}, target_q2={target_q2.shape}, target_q={target_q.shape}")

            # Compute target with proper broadcasting
            reward_term = rewards.unsqueeze(1)  # [batch_size, 1]
            discount_term = self.discount_factor * target_q * ~dones.unsqueeze(1)  # [batch_size, 1]
            target_q = reward_term + discount_term
            
            # Debug target computation (periodic logging)
            if should_log_details:
                logger.info(f"TD3 Target computation: rewards={reward_term.shape}, discount={discount_term.shape}, final_target={target_q.shape}")
            
            # Explicitly detach target to ensure no gradient computation
            target_q = target_q.detach()

        # Get current Q estimates
        current_q1, current_q2 = self.critic(
            ego_states, multi_agent_contexts, actions)
        
        # Debug logging for tensor shapes (periodic logging)
        should_log_details = (self.training_step % 1000 == 0)
        if should_log_details:
            logger.info(f"TD3 Critic shapes: current_q1={current_q1.shape}, current_q2={current_q2.shape}, target_q={target_q.shape}")
            logger.info(f"Sample Q values: q1={current_q1[:3].flatten().tolist()}, target={target_q[:3].flatten().tolist()}")

        # Compute critic loss - ensure both Q estimates are same shape as target
        critic_loss_1 = F.mse_loss(current_q1, target_q, reduction='mean')
        critic_loss_2 = F.mse_loss(current_q2, target_q, reduction='mean')
        critic_loss = critic_loss_1 + critic_loss_2

        # Update metrics before backward pass
        with torch.no_grad():
            self.training_metrics['q1_mean'] = current_q1.mean().item()
            self.training_metrics['q2_mean'] = current_q2.mean().item()

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Log loss values after backward pass (safe to call .item() now)
        with torch.no_grad():
            loss_value = critic_loss.item()
            loss1_value = critic_loss_1.item()
            loss2_value = critic_loss_2.item()
            
            # Log every 100 updates or when loss is significantly high/low
            should_log_loss = (self.training_step % 100 == 0) or (loss_value > 5.0) or (loss_value < 0.01)
            if should_log_loss:
                logger.info(f"TD3 Critic Step {self.training_step}: loss1={loss1_value:.4f}, loss2={loss2_value:.4f}, total={loss_value:.4f}")
            
        return loss_value
    
    def _update_critic_with_per(self, ego_states, multi_agent_contexts, actions, rewards,
                                next_ego_states, next_multi_agent_contexts, dones, importance_weights):
        """Update critic networks with Prioritized Experience Replay"""
        with torch.no_grad():
            # Target policy smoothing
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            # Get next actions from target actor
            next_actions = self.actor_target(
                next_ego_states, next_multi_agent_contexts)
            next_actions = torch.clamp(
                next_actions + noise, self.min_action, self.max_action)

            # Get target Q values (take minimum of twin critics)
            target_q1, target_q2 = self.critic_target(
                next_ego_states, next_multi_agent_contexts, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Compute target with proper broadcasting
            reward_term = rewards.unsqueeze(1)  # [batch_size, 1]
            discount_term = self.discount_factor * target_q * ~dones.unsqueeze(1)  # [batch_size, 1]
            target_q_values = reward_term + discount_term

        # Get current Q estimates (ensure critic networks can learn)
        # The critic needs gradients to update its parameters
        current_q1, current_q2 = self.critic(
            ego_states, multi_agent_contexts, actions)
        
        # Calculate TD-errors for priority updates (use Q1 for consistency)
        with torch.no_grad():
            td_errors = torch.abs(current_q1 - target_q_values).squeeze()
        
        # Compute losses with importance sampling weights
        # Detach target_q_values to prevent gradient tracking errors
        critic_loss1 = F.mse_loss(current_q1, target_q_values.detach(), reduction='none').squeeze()
        critic_loss2 = F.mse_loss(current_q2, target_q_values.detach(), reduction='none').squeeze()
        
        # Apply importance sampling weights (ensure no gradients)
        weighted_loss1 = (critic_loss1 * importance_weights.detach()).mean()
        weighted_loss2 = (critic_loss2 * importance_weights.detach()).mean()
        
        total_loss = weighted_loss1 + weighted_loss2
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Store loss values for metrics
        loss_value = total_loss.item()
        
        with torch.no_grad():
            loss1_value = weighted_loss1.item()
            loss2_value = weighted_loss2.item()
            
            # Update training metrics
            self.training_metrics.update({
                'critic_loss1': loss1_value,
                'critic_loss2': loss2_value,
                'q1_mean': current_q1.mean().item(),
                'q2_mean': current_q2.mean().item(),
                'target_q_mean': target_q_values.mean().item(),
                'td_error_mean': td_errors.mean().item(),
                'importance_weights_mean': importance_weights.mean().item()
            })
            
            # Log every 100 updates or when loss is significantly high/low
            should_log_loss = (self.training_step % 100 == 0) or (loss_value > 5.0) or (loss_value < 0.01)
            if should_log_loss:
                logger.info(f"TD3 PER Step {self.training_step}: loss1={loss1_value:.4f}, loss2={loss2_value:.4f}, "
                           f"td_error_avg={td_errors.mean().item():.4f}, weights_avg={importance_weights.mean().item():.3f}")
            
        return loss_value, td_errors

    def _update_actor(self, ego_states, multi_agent_contexts):
        """Update actor network"""
        # Freeze critic parameters
        for param in self.critic.parameters():
            param.requires_grad = False

        # Compute actor loss
        actions = self.actor(ego_states, multi_agent_contexts)
        actor_loss = -self.critic.Q1(ego_states,
                                     multi_agent_contexts, actions).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for param in self.critic.parameters():
            param.requires_grad = True

        return actor_loss.item()

    def _update_target_networks(self):
        """Soft update target networks"""
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update actor target
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def reset_episode(self):
        """Reset for new episode"""
        self.episode_count += 1
        
        # Auto-clear buffer every N episodes to prevent stale experiences
        if self.clear_episodes and self.episode_count > 0:
            if self.episode_count % self.clear_episodes == 0:
                old_size = len(self.memory)
                if old_size > 5000 and hasattr(self.memory, 'clear_old'):  # Only clear if buffer has meaningful data
                    self.memory.clear_old(self.clear_keep_ratio)
                    new_size = len(self.memory)
                    logger.info(f"Episode {self.episode_count}: Cleared old experiences: "
                               f"{old_size} → {new_size} (kept {self.clear_keep_ratio*100:.0f}% newest)")

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information"""
        return {
            'algorithm': 'TD3',
            'algorithm_type': 'twin_delayed_ddpg',
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'lstm_hidden_size': self.lstm_hidden_size,
            'memory_size': len(self.memory),
            'memory_capacity': self.memory_size,
            'buffer_stats': self.memory.get_stats() if hasattr(self.memory, 'get_stats') else {},
            'clear_episodes': self.clear_episodes,
            'clear_keep_ratio': self.clear_keep_ratio,
            'target_updates': self.training_metrics.get('target_updates', 0),
            # TD3 uses exploration noise instead of epsilon
            'epsilon': self.exploration_noise,
            'actor_loss': self.training_metrics.get('actor_loss', 0.0),
            'critic_loss': self.training_metrics.get('critic_loss', 0.0),
            'device': str(self.device),
            'max_action': self.max_action
        }

    def save(self, path: str):
        """Save TD3 model"""
        try:
            save_data = {
                'actor_state_dict': self.actor.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'lstm_encoder_state_dict': self.lstm_encoder.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'episode_count': self.episode_count,
                'training_metrics': self.training_metrics,
                'config': self.config
            }

            torch.save(save_data, path)
            logger.info(f"TD3 model saved to {path}")

        except Exception as e:
            logger.error(f"Error saving TD3 model: {e}")

    def load(self, path: str):
        """Load TD3 model"""
        try:
            save_data = torch.load(
                path, map_location=self.device, weights_only=False)

            self.actor.load_state_dict(save_data['actor_state_dict'])
            self.actor_target.load_state_dict(
                save_data['actor_target_state_dict'])
            self.critic.load_state_dict(save_data['critic_state_dict'])
            self.critic_target.load_state_dict(
                save_data['critic_target_state_dict'])
            self.lstm_encoder.load_state_dict(
                save_data['lstm_encoder_state_dict'])
            self.actor_optimizer.load_state_dict(
                save_data['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(
                save_data['critic_optimizer_state_dict'])
            self.episode_count = save_data['episode_count']
            if 'training_metrics' in save_data:
                self.training_metrics = save_data['training_metrics']

            logger.info(f"TD3 model loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading TD3 model: {e}")
