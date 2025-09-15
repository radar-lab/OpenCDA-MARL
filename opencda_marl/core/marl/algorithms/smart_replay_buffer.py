"""
Simple Smart Replay Buffer for RL algorithms.
Optimized for small-scale scenarios (~15k transitions).
"""

import random
import numpy as np
from collections import deque
from typing import List, Tuple, Any
from loguru import logger


class SmartReplayBuffer:
    """
    A simple but smart replay buffer with recency bias.
    Designed for scenarios with ~15k transitions.
    """
    
    def __init__(self, capacity: int = 20000, recency_ratio: float = 0.5):
        """
        Initialize smart replay buffer.
        
        Args:
            capacity: Maximum buffer size (default 20k for your scenario)
            recency_ratio: Fraction of samples from recent experiences (0.5 = 50% recent)
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.recency_ratio = recency_ratio
        
        # Track statistics
        self.total_stored = 0
        self.sample_count = 0
        
        logger.info(f"SmartReplayBuffer initialized: capacity={capacity}, recency_ratio={recency_ratio}")
    
    def push(self, *args):
        """Store a transition (compatible with all algorithms)."""
        self.memory.append(args)
        self.total_stored += 1
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample batch with recency bias.
        
        50% from most recent 20% of buffer (recent experiences)
        50% from entire buffer (diverse experiences)
        """
        if len(self.memory) < batch_size:
            return list(self.memory)
        
        self.sample_count += 1
        
        # Calculate split
        recent_samples = int(batch_size * self.recency_ratio)
        random_samples = batch_size - recent_samples
        
        batch = []
        
        # Sample from recent experiences (last 20% of buffer)
        if recent_samples > 0:
            recent_size = max(1, int(len(self.memory) * 0.2))
            recent_portion = list(self.memory)[-recent_size:]
            batch.extend(random.sample(recent_portion, 
                                      min(recent_samples, len(recent_portion))))
        
        # Sample from entire buffer for diversity
        if random_samples > 0:
            remaining_batch = random.sample(self.memory, random_samples)
            batch.extend(remaining_batch)
        
        # Shuffle to mix recent and random samples
        random.shuffle(batch)
        
        # Log statistics occasionally
        if self.sample_count % 100 == 0:
            logger.debug(f"Buffer stats: size={len(self.memory)}/{self.capacity}, "
                        f"total_stored={self.total_stored}, samples={self.sample_count}")
        
        return batch
    
    def clear_old(self, keep_ratio: float = 0.7):
        """
        Clear oldest experiences, keeping only recent ones.
        Useful for non-stationary environments.
        
        Args:
            keep_ratio: Fraction of buffer to keep (0.7 = keep 70% newest)
        """
        if len(self.memory) < 1000:
            return  # Don't clear if buffer is small
        
        keep_size = int(len(self.memory) * keep_ratio)
        old_size = len(self.memory)
        
        # Convert to list, slice, convert back
        kept_memories = list(self.memory)[-keep_size:]
        self.memory = deque(kept_memories, maxlen=self.capacity)
        
        logger.info(f"Cleared old experiences: {old_size} -> {len(self.memory)}")
    
    def __len__(self):
        return len(self.memory)
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'current_size': len(self.memory),
            'capacity': self.capacity,
            'total_stored': self.total_stored,
            'sample_count': self.sample_count,
            'recency_ratio': self.recency_ratio
        }


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using TD-error as priority.
    Based on the PER paper: https://arxiv.org/abs/1511.05952
    """
    
    def __init__(self, capacity: int = 20000, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling weight (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sampling
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.epsilon = 1e-6  # Small constant to ensure non-zero priorities
        
        # Statistics
        self.total_stored = 0
        self.sample_count = 0
        
        logger.info(f"PrioritizedReplayBuffer initialized: capacity={capacity}, "
                   f"alpha={alpha}, beta={beta}")
    
    def push(self, *args, td_error: float = None):
        """Store transition with priority based on TD-error."""
        self.memory.append(args)
        
        # Use max priority for new transitions (optimistic initialization)
        # This ensures new experiences get sampled at least once
        if td_error is None:
            priority = self.max_priority
        else:
            # Convert TD-error to priority: |TD-error| + epsilon
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
        self.total_stored += 1
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample based on priorities with importance sampling weights."""
        if len(self.memory) < batch_size:
            batch = list(self.memory)
            indices = list(range(len(self.memory)))
            weights = np.ones(len(self.memory))
            return batch, indices, weights
        
        # Increment beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.sample_count += 1
        
        # Convert priorities to probabilities
        priorities = np.array(list(self.priorities))
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        # w_i = (1 / (N * P(i))) ^ beta
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Get transitions
        batch = [self.memory[idx] for idx in indices]
        
        # Log statistics occasionally
        if self.sample_count % 100 == 0:
            avg_priority = priorities.mean()
            logger.debug(f"PER stats: size={len(self.memory)}, avg_priority={avg_priority:.4f}, "
                        f"beta={self.beta:.3f}, samples={self.sample_count}")
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD-errors using vectorized operations."""
        # Filter valid indices
        valid_mask = (indices >= 0) & (indices < len(self.priorities))
        valid_indices = indices[valid_mask]
        valid_td_errors = td_errors[valid_mask]
        
        if len(valid_indices) == 0:
            return
            
        # Vectorized priority calculation
        priorities = (np.abs(valid_td_errors) + self.epsilon) ** self.alpha
        
        # Update priorities in batch
        for idx, priority in zip(valid_indices, priorities):
            self.priorities[idx] = priority
            
        # Update max priority
        self.max_priority = max(self.max_priority, np.max(priorities))
    
    def clear_old(self, keep_ratio: float = 0.5):
        """Clear oldest experiences while preserving priorities."""
        if len(self.memory) < 5000:
            return
        
        keep_size = int(len(self.memory) * keep_ratio)
        old_size = len(self.memory)
        
        # Keep newest experiences and their priorities
        kept_memories = list(self.memory)[-keep_size:]
        kept_priorities = list(self.priorities)[-keep_size:]
        
        self.memory = deque(kept_memories, maxlen=self.capacity)
        self.priorities = deque(kept_priorities, maxlen=self.capacity)
        
        logger.info(f"PER: Cleared old experiences: {old_size} -> {len(self.memory)}")
    
    def __len__(self):
        return len(self.memory)
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        priorities_array = np.array(list(self.priorities)) if self.priorities else np.array([0])
        return {
            'current_size': len(self.memory),
            'capacity': self.capacity,
            'total_stored': self.total_stored,
            'sample_count': self.sample_count,
            'alpha': self.alpha,
            'beta': self.beta,
            'avg_priority': priorities_array.mean(),
            'max_priority': self.max_priority
        }