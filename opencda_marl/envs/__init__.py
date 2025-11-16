'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-04 15:46:42
FilePath     : /OpenCDA-MARL/opencda_marl/envs/__init__.py
Description  : Environment package for OpenCDA-MARL.

This package contains environment-related components for MARL training:
- CarlaMonitor: Monitor and manage CARLA world settings
- CarlaSpectator: CARLA camera spectator for GUI
- MARLEnv: Specific MARL environment for OpenCDA-MARL
- EvaluationManager: Modular evaluation system for comprehensive metrics tracking

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''

from .carla_monitor import CarlaMonitor
from .carla_spectator import CarlaSpectator
from .marl_env import MARLEnv
from .sumo_marl_env import SumoMARLEnv
from .evaluation import EvaluationManager
from .evaluation_plots import EvaluationPlotter

__all__ = ['CarlaMonitor', 'CarlaSpectator', 'MARLEnv', 'SumoMARLEnv', 'EvaluationManager', 'EvaluationPlotter']