'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 15:30:00
FilePath     : /OpenCDA-MARL/opencda_marl/core/agents/marl_agent.py
Description  : MARL Agent with multiple RL algorithm support
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from loguru import logger

from opencda_marl.core.agents.vanilla_agent import VanillaAgent


class MARLAgent(VanillaAgent):
    """
    MARL agent implementing speed-only control with multiple RL algorithms.
    """

    def __init__(self, vehicle, carla_map, config_yaml):
        """
        Initialize MARLAgent with selectable reinforcement learning algorithms.

        Args:
            vehicle: CARLA vehicle actor
            carla_map: CARLA HD map
            config_yaml: Configuration containing marl parameters
        """
        # Initialize VanillaAgent base
        super().__init__(vehicle, carla_map, config_yaml)

        logger.debug(
            f"MARLAgent initialized for vehicle {vehicle.id}")

    # --------------------------------------------------------------------- #
    # Main Step Function
    # --------------------------------------------------------------------- #
    def run_step(self, target_speed=None):
        return super().run_step(target_speed)
