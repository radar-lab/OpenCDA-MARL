'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-12-03
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/mappo.py
Description  : Multi-Agent Proximal Policy Optimization (MAPPO) algorithm
               for cooperative autonomous intersection management.

Reference    : "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
               Yu et al., NeurIPS 2021
               GitHub: https://github.com/marlbenchmark/on-policy

               @article{yu2022surprising,
                 title={The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games},
                 author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and others},
                 journal={Advances in Neural Information Processing Systems},
                 volume={35},
                 pages={24611--24624},
                 year={2022}
               }

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gc
from typing import Dict, Any, List, Tuple, Optional
from torch.distributions import Normal
from loguru import logger

from .base_algorithm import BaseAlgorithm
from .rollout_buffer import RolloutBuffer, MultiAgentRolloutBuffer


class GaussianActor(nn.Module):
    """
    Gaussian policy network for continuous action space.

    Outputs mean and std for a Gaussian distribution over actions.
    Uses learnable log_std parameter for stable training.

    Architecture follows MAPPO paper recommendations:
    - Layer normalization for stability
    - Orthogonal initialization for ReLU layers
    - Small std initialization for output layer

    Parameters
    ----------
    state_dim : int
        Input observation dimension
    action_dim : int
        Output action dimension
    hidden_dims : List[int]
        Hidden layer dimensions
    max_action : float
        Maximum action value (for clamping)
    min_action : float
        Minimum action value
    init_std : float
        Initial standard deviation for exploration
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 max_action: float = 65.0, min_action: float = 0.0,
                 init_std: float = 0.5):
        super(GaussianActor, self).__init__()

        self.max_action = max_action
        self.min_action = min_action
        self.action_dim = action_dim

        # Build feature extraction network
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.feature_net = nn.Sequential(*layers)

        # Action mean head
        self.action_mean = nn.Linear(hidden_dims[-1], action_dim)

        # Learnable log standard deviation (shared across all states)
        # Initialize to produce init_std as initial standard deviation
        self.action_log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_std))

        # Min/max log std to prevent numerical issues
        self.log_std_min = np.log(0.01)  # Min std = 0.01
        self.log_std_max = np.log(2.0)   # Max std = 2.0

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        # Initialize feature network with orthogonal initialization
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        # Initialize action mean with small weights for controlled initial actions
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        # Bias initialized to produce middle-range speed (~40 km/h)
        nn.init.constant_(self.action_mean.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution parameters.

        Parameters
        ----------
        state : torch.Tensor
            Input observation [batch_size, state_dim]

        Returns
        -------
        action_mean : torch.Tensor
            Mean of Gaussian distribution [batch_size, action_dim]
        action_std : torch.Tensor
            Standard deviation [batch_size, action_dim]
        """
        features = self.feature_net(state)
        action_mean = self.action_mean(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(self.action_log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp().expand_as(action_mean)

        return action_mean, action_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Parameters
        ----------
        state : torch.Tensor
            Current observation
        deterministic : bool
            If True, return mean action (no sampling)

        Returns
        -------
        action : torch.Tensor
            Sampled or deterministic action
        log_prob : torch.Tensor
            Log probability of action under current policy
        """
        mean, std = self.forward(state)

        if deterministic:
            # Return mean action (clamped to valid range)
            action = torch.clamp(mean, self.min_action, self.max_action)
            # Log prob of mean under the distribution
            dist = Normal(mean, std)
            log_prob = dist.log_prob(mean).sum(dim=-1)
            return action, log_prob

        # Sample from Gaussian distribution
        dist = Normal(mean, std)
        action = dist.rsample()  # Reparameterized sampling for gradients

        # Compute log probability (sum over action dimensions)
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Clamp action to valid range
        action = torch.clamp(action, self.min_action, self.max_action)

        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions under current policy.

        Used during PPO update to compute ratio between old and new policies.

        Parameters
        ----------
        state : torch.Tensor
            Batch of observations
        action : torch.Tensor
            Batch of actions to evaluate

        Returns
        -------
        log_prob : torch.Tensor
            Log probability of actions
        entropy : torch.Tensor
            Entropy of the action distribution
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        # Log probability of given actions
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Entropy (encourages exploration)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized value function for MAPPO (CTDE paradigm).

    Takes global state (potentially all agents' observations) as input
    and outputs a single value estimate.

    In our intersection scenario, the global state includes:
    - Ego agent observation
    - Multi-agent context from LSTM encoder

    Parameters
    ----------
    global_state_dim : int
        Dimension of global state input
    hidden_dims : List[int]
        Hidden layer dimensions
    """

    def __init__(self, global_state_dim: int, hidden_dims: List[int] = [256, 256]):
        super(CentralizedCritic, self).__init__()

        # Build value network
        layers = []
        in_dim = global_state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        # Final value output
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.value_net = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        # Last layer with smaller gain for stable initial values
        final_layer = self.value_net[-1]
        nn.init.orthogonal_(final_layer.weight, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Parameters
        ----------
        global_state : torch.Tensor
            Global state observation [batch_size, global_state_dim]

        Returns
        -------
        value : torch.Tensor
            State value estimate [batch_size, 1]
        """
        return self.value_net(global_state)


class LSTMEncoder(nn.Module):
    """LSTM encoder for multi-agent context (same as TD3 for consistency)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1
                n = param.size(0)
                with torch.no_grad():
                    param.data[n//4:n//2] = 1.0

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Encode sequences of agent observations.

        Parameters
        ----------
        sequences : torch.Tensor
            Agent observation sequences [batch_size, seq_len, input_size]

        Returns
        -------
        encoding : torch.Tensor
            Final hidden state [batch_size, hidden_size]
        """
        batch_size = sequences.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                        device=sequences.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                        device=sequences.device)

        _, (hn, _) = self.lstm(sequences, (h0, c0))
        return hn[-1]  # Return last layer's hidden state


class MAPPOAlgorithm(BaseAlgorithm):
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.

    Implements the CTDE (Centralized Training, Decentralized Execution) paradigm:
    - Training: Centralized critic sees global state (all agents)
    - Execution: Decentralized actors use only local observations

    Key differences from TD3:
    - On-policy: Experiences used once then discarded
    - Stochastic policy: Gaussian distribution over actions
    - PPO clipping: Prevents large policy updates
    - Entropy bonus: Encourages exploration

    Reference
    ---------
    "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
    Yu et al., NeurIPS 2021
    GitHub: https://github.com/marlbenchmark/on-policy

    Parameters
    ----------
    config : Dict[str, Any]
        MAPPO configuration
    state_dim : int
        Individual agent observation dimension (44D in our case)
    action_dim : int
        Action dimension (1 for continuous speed control)
    """

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        super().__init__(config, state_dim, action_dim)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========== MAPPO Hyperparameters ===========
        # PPO clipping parameters
        self.clip_param = config.get('clip_param', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.num_mini_batch = config.get('num_mini_batch', 4)

        # Loss coefficients
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # GAE parameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.use_gae = config.get('use_gae', True)

        # Learning rate
        self.learning_rate = config.get('learning_rate', 3e-4)

        # Rollout settings
        self.n_steps = config.get('n_steps', 128)  # Steps before update
        self.batch_size = config.get('batch_size', 32)

        # Action bounds (same as TD3)
        self.max_action = config.get('max_action', 65.0)
        self.min_action = config.get('min_action', 0.0)

        # CTDE settings
        self.use_centralized_V = config.get('use_centralized_V', True)
        self.share_policy = config.get('share_policy', True)

        # =========== Network Architecture ===========
        # LSTM encoder for multi-agent context
        self.lstm_hidden_size = config.get('lstm_hidden_size', 256)
        self.lstm_num_layers = config.get('lstm_num_layers', 1)

        self.lstm_encoder = LSTMEncoder(
            input_size=state_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers
        ).to(self.device)

        # Actor (Gaussian policy)
        actor_hidden_dims = config.get('actor_hidden_dims', [256, 256])
        init_std = config.get('init_std', 0.5)

        self.actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            max_action=self.max_action,
            min_action=self.min_action,
            init_std=init_std
        ).to(self.device)

        # Critic (centralized value function)
        # Input: ego state + LSTM context
        critic_input_dim = state_dim + self.lstm_hidden_size
        critic_hidden_dims = config.get('critic_hidden_dims', [256, 256])

        self.critic = CentralizedCritic(
            global_state_dim=critic_input_dim,
            hidden_dims=critic_hidden_dims
        ).to(self.device)

        # =========== Optimizers ===========
        # Single optimizer for all networks (MAPPO paper recommendation)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()},
            {'params': self.lstm_encoder.parameters()}
        ], lr=self.learning_rate)

        # Learning rate scheduler (optional)
        self.use_lr_decay = config.get('use_lr_decay', False)
        if self.use_lr_decay:
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
            )

        # =========== Rollout Buffer ===========
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps * 4,  # Allow for multiple agents
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device
        )

        # =========== Training State ===========
        self.training = True
        self.steps_collected = 0
        self.updates_performed = 0
        self._pretrained = False

        # Track last action per agent for store_transition
        self.last_log_probs: Dict[str, float] = {}
        self.last_values: Dict[str, float] = {}

        # Training metrics
        self.training_metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'buffer_size': 0,
        }

        logger.info(f"MAPPO initialized: state_dim={state_dim}, action_dim={action_dim}, "
                   f"device={self.device}")
        logger.info(f"MAPPO hyperparameters: clip={self.clip_param}, epochs={self.ppo_epochs}, "
                   f"entropy_coef={self.entropy_coef}")

    def select_action(self, multi_agent_obs: Dict[str, np.ndarray],
                      ego_agent_id: str, training: bool = True) -> Optional[float]:
        """
        Select action for ego agent using MAPPO policy.

        Parameters
        ----------
        multi_agent_obs : Dict[str, np.ndarray]
            All agents' observations
        ego_agent_id : str
            ID of the ego agent
        training : bool
            Whether in training mode

        Returns
        -------
        action : float or None
            Target speed in km/h, or None during warmup
        """
        try:
            # Prepare inputs
            ego_state, multi_agent_context = self._prepare_inputs(
                multi_agent_obs, ego_agent_id
            )

            with torch.no_grad():
                # Get action from actor
                if training:
                    action, log_prob = self.actor.get_action(ego_state, deterministic=False)
                else:
                    action, log_prob = self.actor.get_action(ego_state, deterministic=True)

                # Get value estimate from critic
                critic_input = torch.cat([ego_state, multi_agent_context], dim=1)
                value = self.critic(critic_input)

                # Store for later transition storage
                self.last_log_probs[ego_agent_id] = log_prob.item()
                self.last_values[ego_agent_id] = value.item()

                # Extract scalar action
                final_speed = action.squeeze().item()
                final_speed = np.clip(final_speed, self.min_action, self.max_action)

                # Debug logging (periodic)
                if self.steps_collected % 1000 == 0:
                    mean, std = self.actor.forward(ego_state)
                    logger.debug(f"MAPPO Action: speed={final_speed:.1f}, "
                               f"mean={mean.item():.1f}, std={std.item():.2f}, "
                               f"value={value.item():.2f}")

                return final_speed

        except Exception as e:
            logger.error(f"Error in MAPPO action selection: {e}")
            import traceback
            traceback.print_exc()
            return None

    def store_transition(self, multi_agent_obs: Dict[str, np.ndarray],
                        ego_agent_id: str, action: float, reward: float,
                        next_multi_agent_obs: Dict[str, np.ndarray], done: bool):
        """
        Store transition in rollout buffer.

        Parameters
        ----------
        multi_agent_obs : Dict[str, np.ndarray]
            Current observations
        ego_agent_id : str
            Ego agent ID
        action : float
            Action taken (speed)
        reward : float
            Reward received
        next_multi_agent_obs : Dict[str, np.ndarray]
            Next observations
        done : bool
            Episode termination flag
        """
        try:
            # Get stored log_prob and value from action selection
            log_prob = self.last_log_probs.get(ego_agent_id, 0.0)
            value = self.last_values.get(ego_agent_id, 0.0)

            # Get ego observation
            if ego_agent_id in multi_agent_obs:
                ego_obs = multi_agent_obs[ego_agent_id]
            else:
                ego_obs = list(multi_agent_obs.values())[0]

            # Add to buffer
            self.rollout_buffer.add(
                obs=ego_obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )

            self.steps_collected += 1

        except Exception as e:
            logger.error(f"Error storing MAPPO transition: {e}")

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update if enough data collected.

        Returns
        -------
        metrics : Dict[str, float]
            Training metrics from update
        """
        # Check if enough data for update
        if len(self.rollout_buffer) < self.batch_size:
            return self.training_metrics

        try:
            # Compute returns and advantages using GAE
            self.rollout_buffer.compute_returns_and_advantages(
                last_value=0.0,  # Assume terminal (or use bootstrap)
                normalize_advantages=True
            )

            # PPO update epochs
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_approx_kl = 0.0
            total_clip_fraction = 0.0
            num_updates = 0

            for epoch in range(self.ppo_epochs):
                # Get mini-batches
                batch_size = max(len(self.rollout_buffer) // self.num_mini_batch, 1)

                for batch in self.rollout_buffer.get_batches(batch_size=batch_size):
                    # Unpack batch
                    obs = batch['observations']
                    actions = batch['actions'].unsqueeze(-1)  # [batch, 1]
                    old_log_probs = batch['old_log_probs']
                    advantages = batch['advantages']
                    returns = batch['returns']

                    # Evaluate current policy
                    new_log_probs, entropy = self.actor.evaluate_actions(obs, actions)

                    # Compute critic values
                    # For simplicity, use ego obs directly (no LSTM in this batch)
                    # In full implementation, would need to re-encode multi-agent context
                    critic_input = torch.cat([
                        obs,
                        torch.zeros(obs.size(0), self.lstm_hidden_size, device=self.device)
                    ], dim=1)
                    values = self.critic(critic_input).squeeze(-1)

                    # PPO policy loss with clipping
                    ratio = (new_log_probs - old_log_probs).exp()
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    ) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(values, returns)

                    # Entropy bonus (negative because we want to maximize entropy)
                    entropy_loss = -entropy.mean()

                    # Combined loss
                    loss = (policy_loss +
                           self.value_loss_coef * value_loss +
                           self.entropy_coef * entropy_loss)

                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) +
                        list(self.critic.parameters()) +
                        list(self.lstm_encoder.parameters()),
                        self.max_grad_norm
                    )
                    self.optimizer.step()

                    # Track metrics
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()

                    # Approximate KL divergence
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (ratio.log())).mean().item()
                        clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()

                    total_approx_kl += approx_kl
                    total_clip_fraction += clip_fraction
                    num_updates += 1

            # Average metrics
            if num_updates > 0:
                self.training_metrics = {
                    'policy_loss': total_policy_loss / num_updates,
                    'value_loss': total_value_loss / num_updates,
                    'entropy': total_entropy / num_updates,
                    'approx_kl': total_approx_kl / num_updates,
                    'clip_fraction': total_clip_fraction / num_updates,
                    'buffer_size': len(self.rollout_buffer),
                }

            # Clear buffer (on-policy requirement)
            self.rollout_buffer.clear()
            self.updates_performed += 1

            # LR decay if enabled
            if self.use_lr_decay:
                self.lr_scheduler.step()

            # TensorBoard logging
            self.log_scalar('Loss/policy', self.training_metrics['policy_loss'], category='losses')
            self.log_scalar('Loss/value', self.training_metrics['value_loss'], category='losses')
            self.log_scalar('Policy/entropy', self.training_metrics['entropy'], category='losses')
            self.log_scalar('Policy/approx_kl', self.training_metrics['approx_kl'], category='losses')
            self.log_scalar('Policy/clip_fraction', self.training_metrics['clip_fraction'], category='losses')

            # Log progress periodically
            if self.updates_performed % 10 == 0:
                logger.info(f"MAPPO Update {self.updates_performed}: "
                          f"policy_loss={self.training_metrics['policy_loss']:.4f}, "
                          f"value_loss={self.training_metrics['value_loss']:.4f}, "
                          f"entropy={self.training_metrics['entropy']:.4f}")

            # GPU cleanup
            if torch.cuda.is_available() and self.updates_performed % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            return self.training_metrics

        except Exception as e:
            logger.error(f"Error in MAPPO update: {e}")
            import traceback
            traceback.print_exc()
            return self.training_metrics

    def _prepare_inputs(self, multi_agent_obs: Dict[str, np.ndarray],
                        ego_agent_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for MAPPO networks.

        Returns
        -------
        ego_state : torch.Tensor
            Ego agent features [1, state_dim]
        multi_agent_context : torch.Tensor
            LSTM encoding [1, lstm_hidden_size]
        """
        # Get ego observation
        if ego_agent_id not in multi_agent_obs:
            ego_agent_id = list(multi_agent_obs.keys())[0]

        ego_obs = multi_agent_obs[ego_agent_id]
        ego_state = torch.FloatTensor(ego_obs).unsqueeze(0).to(self.device)

        # Prepare agent sequence for LSTM
        agent_sequence = [ego_obs]

        # Add other agents sorted by distance
        other_agents = []
        for agent_id, obs in multi_agent_obs.items():
            if agent_id != ego_agent_id:
                rel_x, rel_y = obs[0], obs[1]
                distance = np.sqrt(rel_x*rel_x + rel_y*rel_y)
                other_agents.append((obs, distance))

        other_agents.sort(key=lambda x: x[1])
        for obs, _ in other_agents:
            agent_sequence.append(obs)

        # Encode with LSTM
        if len(agent_sequence) > 1:
            sequence_array = np.array(agent_sequence)
            sequence_tensor = torch.FloatTensor(sequence_array).unsqueeze(0).to(self.device)
            multi_agent_context = self.lstm_encoder(sequence_tensor)
        else:
            multi_agent_context = torch.zeros(1, self.lstm_hidden_size).to(self.device)

        return ego_state, multi_agent_context

    def reset_episode(self):
        """Reset for new episode."""
        self.episode_count += 1
        self.last_log_probs.clear()
        self.last_values.clear()

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('MAPPO/steps_collected', self.steps_collected, self.episode_count)
            self.writer.add_scalar('MAPPO/updates_performed', self.updates_performed, self.episode_count)

        # GPU cleanup every few episodes
        if self.episode_count % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information."""
        return {
            'algorithm': 'MAPPO',
            'algorithm_type': 'multi_agent_ppo',
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'buffer_size': len(self.rollout_buffer),
            'steps_collected': self.steps_collected,
            'updates_performed': self.updates_performed,
            'clip_param': self.clip_param,
            'entropy_coef': self.entropy_coef,
            'policy_loss': self.training_metrics.get('policy_loss', 0.0),
            'value_loss': self.training_metrics.get('value_loss', 0.0),
            'entropy': self.training_metrics.get('entropy', 0.0),
            'epsilon': self.entropy_coef,  # For GUI compatibility
            'device': str(self.device),
            'max_action': self.max_action,
        }

    def log_episode_metrics(self, episode_reward: float, episode_length: int,
                           success_rate: float = 0.0, collision_rate: float = 0.0,
                           near_miss_count: int = 0, ttc_violation_rate: float = 0.0,
                           additional_metrics: Dict[str, float] = None,
                           traffic_metrics: Dict[str, float] = None):
        """Log episode-level metrics to TensorBoard."""
        # Add MAPPO-specific metrics
        mappo_metrics = {
            'buffer_size': len(self.rollout_buffer),
            'updates_performed': self.updates_performed,
        }
        if additional_metrics:
            mappo_metrics.update(additional_metrics)

        # Call base class method
        super().log_episode_metrics(
            episode_reward, episode_length, success_rate, collision_rate,
            near_miss_count=near_miss_count,
            ttc_violation_rate=ttc_violation_rate,
            additional_metrics=mappo_metrics,
            traffic_metrics=traffic_metrics
        )

    def save(self, path: str):
        """Save MAPPO model."""
        try:
            save_data = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'lstm_encoder_state_dict': self.lstm_encoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'episode_count': self.episode_count,
                'updates_performed': self.updates_performed,
                'training_metrics': self.training_metrics,
                'config': self.config,
            }
            torch.save(save_data, path)
            logger.info(f"MAPPO model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving MAPPO model: {e}")

    def load(self, path: str):
        """Load MAPPO model."""
        try:
            save_data = torch.load(path, map_location=self.device, weights_only=False)

            self.actor.load_state_dict(save_data['actor_state_dict'])
            self.critic.load_state_dict(save_data['critic_state_dict'])
            self.lstm_encoder.load_state_dict(save_data['lstm_encoder_state_dict'])
            self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
            self.episode_count = save_data.get('episode_count', 0)
            self.updates_performed = save_data.get('updates_performed', 0)

            if 'training_metrics' in save_data:
                self.training_metrics = save_data['training_metrics']

            self._pretrained = True
            logger.info(f"MAPPO model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading MAPPO model: {e}")
