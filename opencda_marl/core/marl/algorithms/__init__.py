'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 18:29:58
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/algorithms/__init__.py
Description  : MARL Algorithms Package

This package contains implementations of various reinforcement learning algorithms
for the MARL agent system.

Available algorithms:
- QLearningAlgorithm: Tabular Q-learning for discrete state/action spaces
- DQNAlgorithm: Deep Q-Network for continuous state, discrete action spaces
- TD3Algorithm: Twin Delayed DDPG for continuous state/action spaces (off-policy)
- MAPPOAlgorithm: Multi-Agent PPO for cooperative multi-agent learning (on-policy)
- SACAlgorithm: Soft Actor-Critic with auto-tuning entropy (off-policy)

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from .q_learning import QLearningAlgorithm
from .dqn import DQNAlgorithm
from .td3 import TD3Algorithm
from .mappo import MAPPOAlgorithm
from .sac import SACAlgorithm

__all__ = ['QLearningAlgorithm', 'DQNAlgorithm', 'TD3Algorithm', 'MAPPOAlgorithm', 'SACAlgorithm']