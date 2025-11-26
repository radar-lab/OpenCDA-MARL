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

    Key difference from VanillaAgent:
    - When RL provides target_speed: Uses RL speed directly (bypasses BasicAgent's autonomous control)
    - When target_speed is None (warmup): Falls back to VanillaAgent's default behavior
    - Path/steering still controlled by local planner
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
        """
        Execute one step with RL-controlled speed.

        CRITICAL: This method overrides VanillaAgent to ensure RL target_speed
        is actually applied to the vehicle, not ignored by BasicAgent.

        When target_speed is provided by RL:
        - Use RL target_speed directly (RL controls speed)
        - Still get target_location from local planner (agent controls path)

        When target_speed is None (warmup or baseline):
        - Fall back to VanillaAgent's default behavior

        Args:
            target_speed: Speed in km/h from RL algorithm, or None for vanilla behavior

        Returns:
            Tuple[float, carla.Location]: (clamped_speed, target_location)
        """
        # If no RL target speed provided, use vanilla agent behavior
        if target_speed is None:
            return super().run_step(target_speed)

        # Check if agent is properly initialized
        if not self._ego_pos:
            logger.warning("MARLAgent: ego position not set, returning zero speed")
            return 0.0, None

        # Check destination reached
        if self.is_close_to_destination():
            raise StopIteration("Destination reached - simulation complete")

        # RL mode: Use RL-provided target speed directly
        # Get target location from local planner for path following
        local_planner = self.get_local_planner()
        if local_planner and hasattr(local_planner, 'target_waypoint') and local_planner.target_waypoint:
            target_location = local_planner.target_waypoint.transform.location
        else:
            target_location = self._ego_pos.location if self._ego_pos else None

        # Clamp target_speed to safe bounds [0, max_speed]
        clamped_speed = max(0.0, min(self.max_speed, target_speed))

        return clamped_speed, target_location
