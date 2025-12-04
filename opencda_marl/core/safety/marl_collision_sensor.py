'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-30 10:39:52
FilePath     : /OpenCDA-MARL/opencda_marl/core/safety/marl_collision_sensor.py
Description  : MARL minimal collision sensor extension
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import math
import weakref
from collections import deque

import carla

from loguru import logger


class MARLCollisionSensor:
    """
    MARL collision sensor with improved safety handling to prevent Signal 11 crashes.

    Key improvements over base CollisionSensor:
    1. Active flag to prevent callbacks after stop() is called
    2. Vehicle alive check in callback to prevent accessing destroyed actors
    3. Safe destroy sequence with explicit stop before destroy
    """

    def __init__(self, vehicle, params):
        """
        Initialize MARL collision sensor.
        """
        self._active = True  # Flag to prevent callbacks after stop
        self._vehicle = weakref.ref(vehicle)  # Weak ref to vehicle

        world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(),
                                        attach_to=vehicle)

        # Use weak reference to self to avoid circular reference
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: MARLCollisionSensor._on_collision(weak_self, event))

        self.collided = False
        self.collided_frame = -1
        self._history = deque(maxlen=params['history_size'])
        self._threshold = params['col_thresh']

    @staticmethod
    def _on_collision(weak_self, event) -> None:
        self = weak_self()
        if not self:
            return

        # Safety check: Don't process if sensor has been stopped
        if not self._active:
            return

        # Safety check: Don't access vehicle if it's been destroyed
        vehicle = self._vehicle()
        if not vehicle or not vehicle.is_alive:
            return

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history.append((event.frame, intensity))
        if intensity > self._threshold:
            self.collided = True
            self.collided_frame = event.frame

    def return_status(self):
        """
        Return collision status without resetting the flag.
        """
        if self.collided:
            return {'collision': True}
        return {'collision': False}

    def check_and_reset(self) -> bool:
        """
        Check if collision occurred and reset flags for next check.
        """
        if self.collided:
            logger.debug("MARL: Collision detected!")
            self.collided = False
            return True
        return False

    def tick(self, data_dict):
        pass

    def stop(self) -> None:
        """
        Stop the sensor callback (but don't destroy yet).
        Safe to call multiple times.
        """
        self._active = False
        try:
            if hasattr(self, 'sensor') and self.sensor is not None:
                if self.sensor.is_alive and not getattr(self, '_stopped', False):
                    self.sensor.stop()
                    self._stopped = True
        except Exception:
            pass

    def destroy(self) -> None:
        """
        Safe destroy sequence:
        1. Set active flag to False to stop callbacks
        2. Clear history
        3. Stop sensor (if not already stopped)
        4. Destroy sensor
        5. Clear references
        """
        # Step 1: Prevent any more callbacks from being processed
        self._active = False

        # Step 2: Clear history
        self._history.clear()

        # Step 3-4: Stop (if needed) and destroy sensor
        try:
            if hasattr(self, 'sensor') and self.sensor is not None:
                if self.sensor.is_alive:
                    # Only stop if not already stopped
                    if not getattr(self, '_stopped', False):
                        self.sensor.stop()
                        self._stopped = True
                    self.sensor.destroy()
                self.sensor = None
        except Exception:
            pass  # Suppress - sensor may already be destroyed

        # Step 5: Clear vehicle reference
        self._vehicle = None
