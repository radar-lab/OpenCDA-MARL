'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-19 17:49:04
FilePath     : /OpenCDA-MARL/opencda_marl/__init__.py
Description  : OpenCDA-MARL: Multi-Agent Reinforcement Learning extension for OpenCDA

Provides scenario management, the training lifecycle, and a PettingZoo API for RL training.

Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''

from importlib import import_module

__version__ = "2.2.1"

__all__ = [
    '__version__',
    'MARLCoordinator',
]


def __getattr__(name):
    """Lazily import heavy coordinator dependencies."""
    if name == "MARLCoordinator":
        return import_module("opencda_marl._kernel.lifecycle.coordinator").MARLCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
