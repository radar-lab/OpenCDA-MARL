'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-12-05
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/sac.py
Description  : Soft Actor-Critic (SAC) algorithm with auto-tuning entropy
               for multi-agent intersection control.

Reference    : "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
                with a Stochastic Actor" - Haarnoja et al., ICML 2018 (arXiv:1801.01290)
               "Soft Actor-Critic Algorithms and Applications" - Haarnoja et al., 2018
               (arXiv:1812.05905)

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
from .smart_replay_buffer import SmartReplayBuffer


class LSTMEncoder(nn.Module):
    """LSTM encoder for multi-agent context (shared with TD3/MAPPO)."""

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


class SquashedGaussianActor(nn.Module):
    """
    Stochastic actor network for SAC with squashed Gaussian policy.

    Outputs mean and log_std for a Gaussian distribution, then applies
    tanh squashing to bound actions. Uses reparameterization trick for
    gradient backpropagation.

    Reference: SAC paper Section 4.2
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 max_action: float = 65.0, min_action: float = 0.0,
                 log_std_min: float = -20.0, log_std_max: float = 2.0):
        super(SquashedGaussianActor, self).__init__()

        self.max_action = max_action
        self.min_action = min_action
        self.action_scale = (max_action - min_action) / 2.0
        self.action_bias = (max_action + min_action) / 2.0
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build feature network
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.feature_net = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        # Small initialization for output heads
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get mean and log_std.

        Parameters
        ----------
        state : torch.Tensor
            Input state [batch_size, state_dim]

        Returns
        -------
        mean : torch.Tensor
            Mean of Gaussian [batch_size, action_dim]
        log_std : torch.Tensor
            Log standard deviation [batch_size, action_dim]
        """
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action using reparameterization trick with tanh squashing.

        Parameters
        ----------
        state : torch.Tensor
            Input state

        Returns
        -------
        action : torch.Tensor
            Squashed action in [min_action, max_action]
        log_prob : torch.Tensor
            Log probability of action (corrected for tanh)
        mean : torch.Tensor
            Mean action (deterministic)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample from Gaussian using reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample

        # Apply tanh squashing
        y_t = torch.tanh(x_t)

        # Scale to action bounds: [-1, 1] -> [min_action, max_action]
        action = y_t * self.action_scale + self.action_bias

        # Compute log probability with tanh correction
        # log_prob = log_pi(a|s) = log_pi(u|s) - sum(log(1 - tanh(u)^2))
        # where u is pre-tanh action
        log_prob = normal.log_prob(x_t)
        # Tanh correction (numerically stable version)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Mean action (deterministic)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action for inference.

        Parameters
        ----------
        state : torch.Tensor
            Input state
        deterministic : bool
            If True, return mean action

        Returns
        -------
        action : torch.Tensor
            Action value
        log_prob : torch.Tensor
            Log probability
        """
        action, log_prob, mean_action = self.sample(state)
        if deterministic:
            return mean_action, log_prob
        return action, log_prob


class TwinQNetwork(nn.Module):
    """
    Twin Q-networks for SAC (reduces overestimation bias).

    Takes state and action as input, outputs Q-values from both networks.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super(TwinQNetwork, self).__init__()

        input_dim = state_dim + action_dim

        # Q1 network
        q1_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            q1_layers.append(nn.Linear(in_dim, h_dim))
            q1_layers.append(nn.ReLU())
            in_dim = h_dim
        q1_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2 network
        q2_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            q2_layers.append(nn.Linear(in_dim, h_dim))
            q2_layers.append(nn.ReLU())
            in_dim = h_dim
        q2_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q2 = nn.Sequential(*q2_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize Q-network weights."""
        for network in [self.q1, self.q2]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.

        Parameters
        ----------
        state : torch.Tensor
            State input [batch_size, state_dim]
        action : torch.Tensor
            Action input [batch_size, action_dim]

        Returns
        -------
        q1 : torch.Tensor
            Q-value from first network
        q2 : torch.Tensor
            Q-value from second network
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value from first network only."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


class SACAlgorithm(BaseAlgorithm):
    """
    Soft Actor-Critic (SAC) algorithm with auto-tuning entropy.

    SAC maximizes both expected return and entropy, encouraging exploration
    while learning optimal policies. Key features:
    - Maximum entropy framework
    - Twin Q-networks (reduces overestimation)
    - Auto-tuning temperature (alpha)
    - Off-policy learning with replay buffer

    The objective is: J(pi) = E[sum(r + alpha * H(pi(.|s)))]
    where H is the entropy and alpha is automatically tuned.

    Reference
    ---------
    Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor", ICML 2018

    Parameters
    ----------
    config : Dict[str, Any]
        SAC configuration
    state_dim : int
        Individual agent observation dimension
    action_dim : int
        Action dimension (1 for continuous speed control)
    """

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        super().__init__(config, state_dim, action_dim)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========== SAC Hyperparameters ===========
        self.tau = config.get('tau', 0.005)  # Soft update coefficient
        self.gamma = config.get('gamma', 0.99)  # Discount factor

        # Action bounds
        self.max_action = config.get('max_action', 65.0)
        self.min_action = config.get('min_action', 0.0)

        # Learning rates
        self.actor_lr = config.get('learning_rate_actor', 3e-4)
        self.critic_lr = config.get('learning_rate_critic', 3e-4)
        self.alpha_lr = config.get('learning_rate_alpha', 3e-4)

        # Entropy tuning
        self.auto_entropy_tuning = config.get('auto_entropy_tuning', True)
        self.target_entropy = config.get('target_entropy', -float(action_dim))
        init_alpha = config.get('init_alpha', 0.2)

        # Warmup and buffer
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.memory_size = config.get('memory_size', 25000)
        self.batch_size = config.get('batch_size', 256)

        # Network architecture
        self.lstm_hidden_size = config.get('lstm_hidden_size', 128)
        self.lstm_num_layers = config.get('lstm_num_layers', 1)
        actor_hidden_dims = config.get('actor_hidden_dims', [256, 256])
        critic_hidden_dims = config.get('critic_hidden_dims', [256, 256])

        # Log std bounds for actor
        log_std_min = config.get('log_std_min', -20)
        log_std_max = config.get('log_std_max', 2)

        # =========== Networks ===========
        # LSTM encoder for multi-agent context
        self.lstm_encoder = LSTMEncoder(
            input_size=state_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers
        ).to(self.device)

        # Combined state dim for actor/critic: ego_state + lstm_context
        combined_state_dim = state_dim + self.lstm_hidden_size

        # Actor (squashed Gaussian policy)
        self.actor = SquashedGaussianActor(
            state_dim=combined_state_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            max_action=self.max_action,
            min_action=self.min_action,
            log_std_min=log_std_min,
            log_std_max=log_std_max
        ).to(self.device)

        # Twin Q-networks
        self.critic = TwinQNetwork(
            state_dim=combined_state_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims
        ).to(self.device)

        # Target Q-networks
        self.critic_target = TwinQNetwork(
            state_dim=combined_state_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # =========== Optimizers ===========
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Auto-tuning alpha (temperature)
        if self.auto_entropy_tuning:
            self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True,
                                          device=self.device, dtype=torch.float32)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = init_alpha

        # =========== Replay Buffer ===========
        self.memory = SmartReplayBuffer(capacity=self.memory_size, recency_ratio=0.5)

        # =========== Training State ===========
        self.training = True
        self.training_step = 0
        self._pretrained = False

        # Training metrics
        self.training_metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'alpha_loss': 0.0,
            'alpha': self.alpha,
            'q1_mean': 0.0,
            'q2_mean': 0.0,
            'entropy': 0.0,
            'memory_size': 0,
        }

        logger.info(f"SAC initialized: state_dim={state_dim}, action_dim={action_dim}, "
                   f"device={self.device}")
        logger.info(f"SAC hyperparameters: tau={self.tau}, gamma={self.gamma}, "
                   f"auto_entropy={self.auto_entropy_tuning}, target_entropy={self.target_entropy}")

    def select_action(self, multi_agent_obs: Dict[str, np.ndarray],
                      ego_agent_id: str, training: bool = True) -> Optional[float]:
        """
        Select action for ego agent using SAC policy.

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
            # Warmup phase: random exploration
            if training and len(self.memory) < self.warmup_steps and not self._pretrained:
                if len(self.memory) % 500 == 0:
                    logger.debug(f"SAC Warmup: {len(self.memory)}/{self.warmup_steps}")
                return np.random.uniform(self.min_action, self.max_action)

            # Prepare inputs
            ego_state, multi_agent_context = self._prepare_inputs(
                multi_agent_obs, ego_agent_id
            )

            # Combine ego state and multi-agent context
            combined_state = torch.cat([ego_state, multi_agent_context], dim=1)

            with torch.no_grad():
                if training:
                    action, log_prob, _ = self.actor.sample(combined_state)
                else:
                    _, _, action = self.actor.sample(combined_state)  # Deterministic (mean)

                final_speed = action.squeeze().item()
                final_speed = np.clip(final_speed, self.min_action, self.max_action)

                # Debug logging (periodic)
                if len(self.memory) % 2000 == 0:
                    logger.debug(f"SAC Action: speed={final_speed:.1f}, alpha={self.alpha:.4f}")

                return final_speed

        except Exception as e:
            logger.error(f"Error in SAC action selection: {e}")
            import traceback
            traceback.print_exc()
            return None

    def store_transition(self, multi_agent_obs: Dict[str, np.ndarray],
                        ego_agent_id: str, action: float, reward: float,
                        next_multi_agent_obs: Dict[str, np.ndarray], done: bool):
        """
        Store transition in replay buffer.

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
            # Prepare current state
            ego_state, multi_agent_context = self._prepare_inputs(
                multi_agent_obs, ego_agent_id
            )

            # Prepare next state
            next_ego_state, next_multi_agent_context = self._prepare_inputs(
                next_multi_agent_obs, ego_agent_id
            )

            # Store transition
            self.memory.push(
                ego_state.detach().cpu().numpy(),
                multi_agent_context.detach().cpu().numpy(),
                action,
                reward,
                next_ego_state.detach().cpu().numpy(),
                next_multi_agent_context.detach().cpu().numpy(),
                done
            )

            # Log buffer stats periodically
            if len(self.memory) % 5000 == 0:
                logger.debug(f"SAC Buffer: {len(self.memory)}/{self.memory_size}")

        except Exception as e:
            logger.error(f"Error storing SAC transition: {e}")

    def update(self) -> Dict[str, float]:
        """
        Perform SAC update.

        Returns
        -------
        metrics : Dict[str, float]
            Training metrics from update
        """
        # Only update if we have enough samples
        if len(self.memory) < self.batch_size:
            return self.training_metrics

        try:
            # Sample batch
            transitions = self.memory.sample(self.batch_size)

            # Unpack batch
            ego_states_np = np.stack([t[0] for t in transitions]).squeeze(1).astype(np.float32)
            ego_states = torch.from_numpy(ego_states_np).to(self.device)

            multi_contexts_np = np.stack([t[1] for t in transitions]).squeeze(1).astype(np.float32)
            multi_agent_contexts = torch.from_numpy(multi_contexts_np).to(self.device)

            actions_np = np.array([t[2] for t in transitions], dtype=np.float32).reshape(-1, 1)
            actions = torch.from_numpy(actions_np).to(self.device)

            rewards_np = np.array([t[3] for t in transitions], dtype=np.float32).reshape(-1, 1)
            rewards = torch.from_numpy(rewards_np).to(self.device)

            next_ego_np = np.stack([t[4] for t in transitions]).squeeze(1).astype(np.float32)
            next_ego_states = torch.from_numpy(next_ego_np).to(self.device)

            next_contexts_np = np.stack([t[5] for t in transitions]).squeeze(1).astype(np.float32)
            next_multi_agent_contexts = torch.from_numpy(next_contexts_np).to(self.device)

            dones_np = np.array([t[6] for t in transitions], dtype=np.float32).reshape(-1, 1)
            dones = torch.from_numpy(dones_np).to(self.device)

            # Combine states for networks
            states = torch.cat([ego_states, multi_agent_contexts], dim=1)
            next_states = torch.cat([next_ego_states, next_multi_agent_contexts], dim=1)

            # =========== Update Critics ===========
            with torch.no_grad():
                # Sample next actions from current policy
                next_actions, next_log_probs, _ = self.actor.sample(next_states)

                # Target Q-values (minimum of two targets for conservatism)
                target_q1, target_q2 = self.critic_target(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)

                # Soft Q-target with entropy
                target_q = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_probs)

            # Current Q-values
            current_q1, current_q2 = self.critic(states, actions)

            # Critic loss (MSE)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # Update critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # =========== Update Actor ===========
            # Sample new actions for actor update
            new_actions, log_probs, _ = self.actor.sample(states)
            q1_new, q2_new = self.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)

            # Actor loss: maximize Q - alpha * log_prob
            actor_loss = (self.alpha * log_probs - q_new).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # =========== Update Alpha (Temperature) ===========
            if self.auto_entropy_tuning:
                # Alpha loss: minimize -alpha * (log_prob + target_entropy)
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp().item()
            else:
                alpha_loss = torch.tensor(0.0)

            # =========== Soft Update Target Networks ===========
            self._soft_update_target()

            # =========== Update Metrics ===========
            self.training_step += 1
            self.training_metrics.update({
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0.0,
                'alpha': self.alpha,
                'q1_mean': current_q1.mean().item(),
                'q2_mean': current_q2.mean().item(),
                'entropy': -log_probs.mean().item(),
                'memory_size': len(self.memory),
            })

            # TensorBoard logging
            self.log_scalar('Loss/actor', actor_loss.item(), category='losses')
            self.log_scalar('Loss/critic', critic_loss.item(), category='losses')
            self.log_scalar('SAC/alpha', self.alpha, category='losses')
            self.log_scalar('SAC/entropy', -log_probs.mean().item(), category='losses')
            self.log_scalar('Q_values/Q1_mean', current_q1.mean().item(), category='q_values')
            self.log_scalar('Q_values/Q2_mean', current_q2.mean().item(), category='q_values')
            self.log_scalar('Buffer/size', len(self.memory), category='buffer')

            # Log progress periodically
            if self.training_step % 200 == 0:
                logger.info(f"SAC Step {self.training_step}: "
                          f"critic_loss={critic_loss.item():.4f}, "
                          f"actor_loss={actor_loss.item():.4f}, "
                          f"alpha={self.alpha:.4f}, "
                          f"entropy={-log_probs.mean().item():.4f}")

            # GPU cleanup
            if torch.cuda.is_available() and self.training_step % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            return self.training_metrics

        except Exception as e:
            logger.error(f"Error in SAC update: {e}")
            import traceback
            traceback.print_exc()
            return self.training_metrics

    def _soft_update_target(self):
        """Soft update target networks."""
        for param, target_param in zip(self.critic.parameters(),
                                       self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _prepare_inputs(self, multi_agent_obs: Dict[str, np.ndarray],
                        ego_agent_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for SAC networks.

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

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('SAC/memory_size', len(self.memory), self.episode_count)
            self.writer.add_scalar('SAC/alpha', self.alpha, self.episode_count)

        # GPU cleanup every few episodes
        if self.episode_count % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information."""
        return {
            'algorithm': 'SAC',
            'algorithm_type': 'soft_actor_critic',
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'buffer_size': len(self.memory),
            'training_step': self.training_step,
            'tau': self.tau,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'target_entropy': self.target_entropy,
            'actor_loss': self.training_metrics.get('actor_loss', 0.0),
            'critic_loss': self.training_metrics.get('critic_loss', 0.0),
            'entropy': self.training_metrics.get('entropy', 0.0),
            'epsilon': self.alpha,  # For GUI compatibility
            'device': str(self.device),
            'max_action': self.max_action,
            'warmup_steps': self.warmup_steps,
            'in_warmup': len(self.memory) < self.warmup_steps and not self._pretrained,
        }

    def log_episode_metrics(self, episode_reward: float, episode_length: int,
                           success_rate: float = 0.0, collision_rate: float = 0.0,
                           near_miss_count: int = 0, ttc_violation_rate: float = 0.0,
                           additional_metrics: Dict[str, float] = None,
                           traffic_metrics: Dict[str, float] = None):
        """Log episode-level metrics to TensorBoard."""
        # Add SAC-specific metrics
        sac_metrics = {
            'alpha': self.alpha,
            'training_step': self.training_step,
        }
        if additional_metrics:
            sac_metrics.update(additional_metrics)

        # Call base class method
        super().log_episode_metrics(
            episode_reward, episode_length, success_rate, collision_rate,
            near_miss_count=near_miss_count,
            ttc_violation_rate=ttc_violation_rate,
            additional_metrics=sac_metrics,
            traffic_metrics=traffic_metrics
        )

    def save(self, path: str):
        """Save SAC model."""
        try:
            save_data = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'lstm_encoder_state_dict': self.lstm_encoder.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'episode_count': self.episode_count,
                'training_step': self.training_step,
                'training_metrics': self.training_metrics,
                'alpha': self.alpha,
                'config': self.config,
            }

            if self.auto_entropy_tuning:
                save_data['log_alpha'] = self.log_alpha.detach().cpu()
                save_data['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()

            torch.save(save_data, path)
            logger.info(f"SAC model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving SAC model: {e}")

    def load(self, path: str):
        """Load SAC model."""
        try:
            save_data = torch.load(path, map_location=self.device, weights_only=False)

            self.actor.load_state_dict(save_data['actor_state_dict'])
            self.critic.load_state_dict(save_data['critic_state_dict'])
            self.critic_target.load_state_dict(save_data['critic_target_state_dict'])
            self.lstm_encoder.load_state_dict(save_data['lstm_encoder_state_dict'])
            self.actor_optimizer.load_state_dict(save_data['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(save_data['critic_optimizer_state_dict'])
            self.episode_count = save_data.get('episode_count', 0)
            self.training_step = save_data.get('training_step', 0)
            self.alpha = save_data.get('alpha', 0.2)

            if 'training_metrics' in save_data:
                self.training_metrics = save_data['training_metrics']

            if self.auto_entropy_tuning and 'log_alpha' in save_data:
                self.log_alpha = save_data['log_alpha'].to(self.device).requires_grad_(True)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
                if 'alpha_optimizer_state_dict' in save_data:
                    self.alpha_optimizer.load_state_dict(save_data['alpha_optimizer_state_dict'])

            self._pretrained = True
            logger.info(f"SAC model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading SAC model: {e}")
