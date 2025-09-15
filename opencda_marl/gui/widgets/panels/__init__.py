'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 13:40:43
FilePath     : /OpenCDA-MARL/opencda_marl/gui/widgets/panels/__init__.py
Description  : GUI panels for different setting categories in OpenCDA-MARL.

This package contains modular panel components for the tabbed environment interface:
- SystemPanel: CARLA simulation system parameters
- WeatherPanel: Weather and environment conditions
- TrafficPanel: Traffic manager and vehicle settings
- RewardPanel: Reward settings

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from .system_panel import SystemPanel
from .weather_panel import WeatherPanel
from .traffic_panel import TrafficPanel
from .reward_panel import RewardPanel

__all__ = ['SystemPanel', 'WeatherPanel', 'TrafficPanel', 'RewardPanel']