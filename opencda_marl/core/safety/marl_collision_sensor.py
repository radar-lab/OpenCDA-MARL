'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-30 10:39:52
FilePath     : /OpenCDA-MARL/opencda_marl/core/safety/marl_collision_sensor.py
Description  : MARL minimal collision sensor extension
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from opencda.core.safety.sensors import CollisionSensor
from loguru import logger


class MARLCollisionSensor(CollisionSensor):
    """
    Minimal extension of OpenCDA's CollisionSensor.
    Only adds MARL-specific collision checking method.
    """

    def __init__(self, vehicle, params):
        """
        Initialize MARL collision sensor by extending OpenCDA's version.
        """
        super().__init__(vehicle, params)

    def return_status(self):
        """
        Override parent's return_status to NOT reset the collision flag.
        This allows us to check the flag later in our collision detection.
        """
        if self.collided:
            return {'collision': True}
        return {'collision': False}

    def check_and_reset(self) -> bool:
        """
        Check if collision occurred and reset flags for next check.
        Uses parent's collided flag which is set by the original callback.
        """
        # reset parent's collision flag to False
        if self.collided:
            logger.debug("MARL: Collision detected!")
            self.collided = False
            return True
        return False
