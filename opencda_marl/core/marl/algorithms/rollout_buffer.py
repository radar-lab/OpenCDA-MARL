'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-12-03
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/rollout_buffer.py
Description  : Rollout buffer for on-policy algorithms (MAPPO, PPO)
               Implements GAE (Generalized Advantage Estimation) for advantage computation.

Reference    : "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
               GitHub: https://github.com/marlbenchmark/on-policy

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Generator
from loguru import logger


class RolloutBuffer:
    """
    On-policy rollout buffer for MAPPO/PPO algorithms.

    Stores trajectories and computes GAE (Generalized Advantage Estimation)
    for policy gradient updates. Unlike off-policy buffers, this buffer is
    cleared after each policy update.

    Key differences from TD3's replay buffer:
    - On-policy: Data used only once per update, then cleared
    - Stores log_probs and values for PPO's clipped objective
    - Computes advantages using GAE for variance reduction

    Parameters
    ----------
    buffer_size : int
        Maximum number of transitions to store per rollout
    state_dim : int
        Dimension of the observation space
    action_dim : int
        Dimension of the action space
    gamma : float
        Discount factor for rewards
    gae_lambda : float
        GAE lambda for advantage estimation (tradeoff between bias and variance)
    device : torch.device
        Device to store tensors on
    """

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: torch.device = None):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Storage lists (reset after each update)
        self.observations: List[np.ndarray] = []
        self.actions: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

        # Multi-agent context storage (for LSTM encoder if used)
        self.multi_agent_contexts: List[np.ndarray] = []

        # Computed after rollout collection
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

        # Statistics tracking
        self.total_transitions_stored = 0
        self.num_rollouts = 0

    def add(self, obs: np.ndarray, action: float, reward: float,
            value: float, log_prob: float, done: bool,
            multi_agent_context: np.ndarray = None):
        """
        Add a single transition to the buffer.

        Parameters
        ----------
        obs : np.ndarray
            Current observation
        action : float
            Action taken
        reward : float
            Reward received
        value : float
            Value estimate V(s) from critic
        log_prob : float
            Log probability of action under current policy
        done : bool
            Whether episode terminated
        multi_agent_context : np.ndarray, optional
            LSTM encoding of multi-agent context (for centralized critic)
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

        if multi_agent_context is not None:
            self.multi_agent_contexts.append(multi_agent_context)

        self.total_transitions_stored += 1

    def compute_returns_and_advantages(self, last_value: float = 0.0,
                                        normalize_advantages: bool = True):
        """
        Compute GAE advantages and returns for collected trajectories.

        Uses Generalized Advantage Estimation (GAE):
        A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Parameters
        ----------
        last_value : float
            Value estimate for the last state (bootstrap value)
        normalize_advantages : bool
            Whether to normalize advantages to zero mean, unit variance
        """
        if len(self.rewards) == 0:
            logger.warning("RolloutBuffer: No transitions to compute returns")
            return

        # Convert to numpy for efficient computation
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute GAE advantages
        advantages = np.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            # GAE: A_t = δ_t + γλA_{t+1}
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        # Returns = advantages + values (used for value function fitting)
        returns = advantages + values

        # Convert to tensors
        self.advantages = torch.from_numpy(advantages).float().to(self.device)
        self.returns = torch.from_numpy(returns).float().to(self.device)

        # Normalize advantages for stable training
        if normalize_advantages and len(advantages) > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        logger.debug(f"GAE computed: advantages mean={self.advantages.mean():.4f}, "
                    f"std={self.advantages.std():.4f}, returns mean={self.returns.mean():.4f}")

    def get_batches(self, batch_size: int = None, shuffle: bool = True) -> Generator:
        """
        Generate mini-batches for PPO updates.

        Parameters
        ----------
        batch_size : int, optional
            Size of each mini-batch (default: use all data)
        shuffle : bool
            Whether to shuffle data before batching

        Yields
        ------
        Dict containing batch of:
            - observations, actions, old_log_probs, advantages, returns
            - multi_agent_contexts (if available)
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Must call compute_returns_and_advantages before get_batches")

        n_samples = len(self.observations)
        if batch_size is None or batch_size >= n_samples:
            batch_size = n_samples

        # Convert all data to tensors
        obs_tensor = torch.FloatTensor(np.stack(self.observations)).to(self.device)
        actions_tensor = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)

        # Multi-agent contexts if available
        contexts_tensor = None
        if len(self.multi_agent_contexts) > 0:
            contexts_tensor = torch.FloatTensor(np.stack(self.multi_agent_contexts)).to(self.device)

        # Generate batch indices
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        # Yield batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch = {
                'observations': obs_tensor[batch_indices],
                'actions': actions_tensor[batch_indices],
                'old_log_probs': old_log_probs_tensor[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices],
            }

            if contexts_tensor is not None:
                batch['multi_agent_contexts'] = contexts_tensor[batch_indices]

            yield batch

    def clear(self):
        """
        Clear the buffer after policy update (on-policy requirement).

        Unlike off-policy algorithms that reuse experiences, on-policy
        methods like MAPPO must clear old data after each update.
        """
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.multi_agent_contexts.clear()

        self.advantages = None
        self.returns = None

        self.num_rollouts += 1

    def __len__(self) -> int:
        """Return number of transitions in buffer."""
        return len(self.observations)

    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return len(self.observations) >= self.buffer_size

    def get_stats(self) -> Dict:
        """Get buffer statistics for debugging."""
        return {
            'current_size': len(self.observations),
            'buffer_capacity': self.buffer_size,
            'total_stored': self.total_transitions_stored,
            'num_rollouts': self.num_rollouts,
            'has_advantages': self.advantages is not None,
            'has_returns': self.returns is not None,
        }


class MultiAgentRolloutBuffer:
    """
    Rollout buffer for multi-agent scenarios.

    Maintains separate buffers for each agent while supporting
    centralized critic (which sees all agents' observations).

    This is specifically designed for MAPPO's CTDE paradigm:
    - Centralized Training: Critic sees global state
    - Decentralized Execution: Actor uses local observation

    Parameters
    ----------
    buffer_size : int
        Maximum transitions per agent per rollout
    state_dim : int
        Individual agent observation dimension
    action_dim : int
        Action dimension (shared across agents)
    max_agents : int
        Maximum number of concurrent agents
    gamma : float
        Discount factor
    gae_lambda : float
        GAE lambda
    device : torch.device
        Compute device
    """

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int,
                 max_agents: int = 4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, device: torch.device = None):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_agents = max_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Per-agent buffers
        self.agent_buffers: Dict[str, RolloutBuffer] = {}

        # Global state storage for centralized critic
        self.global_states: List[np.ndarray] = []

    def get_agent_buffer(self, agent_id: str) -> RolloutBuffer:
        """Get or create buffer for specific agent."""
        if agent_id not in self.agent_buffers:
            self.agent_buffers[agent_id] = RolloutBuffer(
                buffer_size=self.buffer_size,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                device=self.device
            )
        return self.agent_buffers[agent_id]

    def add(self, agent_id: str, obs: np.ndarray, action: float,
            reward: float, value: float, log_prob: float, done: bool,
            global_state: np.ndarray = None):
        """
        Add transition for specific agent.

        Parameters
        ----------
        agent_id : str
            Agent identifier
        obs : np.ndarray
            Agent's local observation
        action : float
            Action taken
        reward : float
            Reward received
        value : float
            Value estimate from centralized critic
        log_prob : float
            Log probability under current policy
        done : bool
            Whether agent's episode terminated
        global_state : np.ndarray, optional
            Global state for centralized critic (all agents' observations)
        """
        buffer = self.get_agent_buffer(agent_id)
        buffer.add(obs, action, reward, value, log_prob, done)

        if global_state is not None:
            self.global_states.append(global_state)

    def compute_all_returns_and_advantages(self, last_values: Dict[str, float] = None,
                                            normalize: bool = True):
        """
        Compute returns and advantages for all agents.

        Parameters
        ----------
        last_values : Dict[str, float], optional
            Bootstrap values for each agent
        normalize : bool
            Whether to normalize advantages
        """
        if last_values is None:
            last_values = {}

        for agent_id, buffer in self.agent_buffers.items():
            last_value = last_values.get(agent_id, 0.0)
            buffer.compute_returns_and_advantages(last_value, normalize)

    def get_combined_batches(self, batch_size: int = None) -> Generator:
        """
        Get batches combining all agents' experiences.

        Useful for shared policy training where all agents share parameters.

        Yields
        ------
        Dict with combined batch data from all agents
        """
        # Combine all agent data
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []

        for agent_id, buffer in self.agent_buffers.items():
            if buffer.advantages is None:
                continue

            all_obs.extend(buffer.observations)
            all_actions.extend(buffer.actions)
            all_log_probs.extend(buffer.log_probs)
            all_advantages.append(buffer.advantages)
            all_returns.append(buffer.returns)

        if len(all_advantages) == 0:
            return

        # Concatenate tensors
        obs_tensor = torch.FloatTensor(np.stack(all_obs)).to(self.device)
        actions_tensor = torch.FloatTensor(all_actions).to(self.device)
        log_probs_tensor = torch.FloatTensor(all_log_probs).to(self.device)
        advantages_tensor = torch.cat(all_advantages)
        returns_tensor = torch.cat(all_returns)

        n_samples = len(obs_tensor)
        if batch_size is None or batch_size >= n_samples:
            batch_size = n_samples

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            yield {
                'observations': obs_tensor[batch_indices],
                'actions': actions_tensor[batch_indices],
                'old_log_probs': log_probs_tensor[batch_indices],
                'advantages': advantages_tensor[batch_indices],
                'returns': returns_tensor[batch_indices],
            }

    def clear(self):
        """Clear all agent buffers."""
        for buffer in self.agent_buffers.values():
            buffer.clear()
        self.global_states.clear()

    def total_transitions(self) -> int:
        """Get total transitions across all agents."""
        return sum(len(b) for b in self.agent_buffers.values())

    def get_stats(self) -> Dict:
        """Get combined statistics."""
        return {
            'num_agents': len(self.agent_buffers),
            'total_transitions': self.total_transitions(),
            'per_agent': {aid: b.get_stats() for aid, b in self.agent_buffers.items()}
        }
