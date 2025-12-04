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

    Key improvements:
    - Uses MARLCollisionSensor with improved safety handling
    - Provides direct access to collision_sensor for safe cleanup
    """

    def __init__(self, cav_world, vehicle, params):
        """
        Initialize MARL safety manager by extending OpenCDA's version.
        """
        super().__init__(cav_world, vehicle, params)

        # Destroy the original collision sensor before replacing
        if len(self.sensors) > 0 and self.sensors[0] is not None:
            try:
                self.sensors[0].destroy()
            except Exception:
                pass

        # Replace the collision sensor (first sensor) with our improved version
        self.sensors[0] = MARLCollisionSensor(
            vehicle, params['collision_sensor'])

        # Keep a direct reference for easy access during cleanup
        self.collision_sensor = self.sensors[0]

        logger.debug(f"Vehicle {vehicle.id}: Initialized MARL safety manager")

    def check_collision(self) -> bool:
        """
        Simple collision check method using our enhanced collision sensor.
        """
        if self.collision_sensor is not None:
            return self.collision_sensor.check_and_reset()
        return False

    def destroy(self):
        """
        Safe destroy sequence for MARL safety manager.
        """
        # Destroy all sensors safely
        for sensor in self.sensors:
            if sensor is not None:
                try:
                    sensor.destroy()
                except Exception:
                    pass

        self.sensors.clear()
        self.collision_sensor = None
