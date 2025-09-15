'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-30 10:39:52
FilePath     : /OpenCDA-MARL/opencda_marl/core/safety/marl_safety_manager.py
Description  : MARL minimal safety manager extension
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''

from loguru import logger

from opencda.core.safety.safety_manager import SafetyManager
from .marl_collision_sensor import MARLCollisionSensor


class MARLSafetyManager(SafetyManager):
    """
    Minimal extension of OpenCDA's SafetyManager.
    Only replaces collision sensor and adds collision checking method.
    """

    def __init__(self, cav_world, vehicle, params):
        """
        Initialize MARL safety manager by extending OpenCDA's version.
        """
        super().__init__(cav_world, vehicle, params)

        # Only replace the collision sensor (first sensor) with our version
        self.sensors[0] = MARLCollisionSensor(
            vehicle, params['collision_sensor'])

        logger.debug(f"Vehicle {vehicle.id}: Initialized MARL safety manager")

    def check_collision(self) -> bool:
        """
        Simple collision check method using our enhanced collision sensor.
        """
        is_marl_sensor = isinstance(self.sensors[0], MARLCollisionSensor)
        if len(self.sensors) > 0 and is_marl_sensor:
            return self.sensors[0].check_and_reset()
        return False
