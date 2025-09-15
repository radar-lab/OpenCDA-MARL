'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-19 17:49:04
FilePath     : /OpenCDA-MARL/opencda_marl/__init__.py
Description  : OpenCDA-MARL: Multi-Agent Reinforcement Learning extension for OpenCDA

Provides scenario management, GUI control, and standard Gym API for RL training.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''

__version__ = "0.1.0"

from .coordinator import MARLCoordinator

__all__ = [
    '__version__',
    'MARLCoordinator'
]