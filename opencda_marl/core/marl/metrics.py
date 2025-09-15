from typing import Dict, Any
import numpy as np


class TrainingMetrics:
    """Track and compute training metrics."""

    def __init__(self):
        self.episode_rewards = []
        self.episode_states = []
        self.reset()

    def reset(self):
        """Reset current episode metrics."""
        self.current_reward = 0.0
        self.current_length = 0
        self.current_total_reward = 0.0
        self.agent_rewards = {}
    # --------------------------------------------------------------------- #
    # Main steps for updating metrics
    # --------------------------------------------------------------------- #

    def update_step(self, rewards: Dict[int, float]):
        """Update metrics for current step."""
        self.current_reward = sum(rewards.values()) if rewards else 0.0
        self.current_total_reward += self.current_reward
        self.current_length += 1
        for agent_id, reward in rewards.items():
            if agent_id not in self.agent_rewards:
                self.agent_rewards[agent_id] = 0.0
            self.agent_rewards[agent_id] += reward

    # --------------------------------------------------------------------- #
    # Public Methods
    # --------------------------------------------------------------------- #

    def get_agent_reward(self, agent_id: int) -> float:
        return self.agent_rewards.get(agent_id, 0.0)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get metrics for current episode."""
        if len(self.agent_rewards) == 0:
            avg_reward = 0.0
        else:
            avg_reward = self.current_total_reward / \
                len(self.agent_rewards)

        if not self.episode_rewards:
            return {
                'step_reward': self.current_reward,
                'step_length': self.current_length,
                'total_reward': self.current_total_reward,
                'avg_reward': avg_reward
            }

        # Compute running averages
        window_size = min(10, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-window_size:]

        return {
            'step_reward': self.current_reward,
            'step_length': self.current_length,
            'total_reward': self.current_total_reward,
            'avg_reward': avg_reward,
            f'mean_reward_episode_{window_size}': float(np.mean(recent_rewards)),
            f'std_reward_episode_{window_size}': float(np.std(recent_rewards)),
            'max_reward_episode': max(self.episode_rewards),
            'total_episodes': len(self.episode_rewards),
        }

    def finish_episode(self, states: Dict[str, Any]):
        """Finish current episode and compute metrics."""
        self.episode_rewards.append(self.current_total_reward)
        self.episode_states.append(states)
        metrics = self.get_current_metrics()
        metrics.update({
            'episode_states': states
        })
        self.reset()
        return metrics
