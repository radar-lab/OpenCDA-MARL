'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-30 10:39:52
FilePath     : /OpenCDA-MARL/opencda_marl/core/safety/marl_safety_manager.py
Description  : MARL minimal safety manager extension
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import weakref
from collections import deque

from loguru import logger

from opencda.core.safety.sensors import StuckDetector, OffRoadDetector, TrafficLightDector, IMUSensor
from .marl_collision_sensor import MARLCollisionSensor


class MARLSafetyManager:
    """
    MARL-specific safety manager that properly manages sensor lifecycle.

    This is a standalone implementation (not extending SafetyManager) to avoid
    the double-sensor creation issue that causes "sensor went out of scope" warnings.

    Key improvements:
    - Creates sensors directly without parent class creating duplicates
    - Uses MARLCollisionSensor with improved safety handling
    - Proper lifecycle management to prevent orphaned sensors
    """

    def __init__(self, cav_world, vehicle, params):
        """
        Initialize MARL safety manager with proper sensor lifecycle.

        Creates only the sensors we need, avoiding duplicate creation.
        """
        self.vehicle = vehicle
        self.cav_world = weakref.ref(cav_world)()
        self.print_message = params.get('print_message', False)
        self.status_queue = deque(maxlen=params.get('queue_maxlen', 2000))

        # Create sensors directly - no parent class to create duplicates
        self.collision_sensor = MARLCollisionSensor(vehicle, params['collision_sensor'])
        self.imu_sensor = IMUSensor(vehicle)

        # Build sensor list (same structure as SafetyManager for compatibility)
        self.sensors = [
            self.collision_sensor,
            StuckDetector(params['stuck_dector']),
            OffRoadDetector(params['offroad_dector']),
            TrafficLightDector(params['traffic_light_detector'], vehicle),
            self.imu_sensor
        ]

        logger.debug(f"Vehicle {vehicle.id}: Initialized MARL safety manager with {len(self.sensors)} sensors")

    def update_info(self, data_dict) -> dict:
        """Update safety status from all sensors."""
        status_dict = {}
        for sensor in self.sensors:
            sensor.tick(data_dict)
            status_dict.update(sensor.return_status())

        # Store status with timestamp
        if self.cav_world:
            self.status_queue.append((self.cav_world.global_clock, status_dict))

        # Print hazard message if configured
        if self.print_message:
            for key, val in status_dict.items():
                if val:
                    print("Safety Warning from the safety manager:")
                    print(status_dict)
                    break

        return status_dict

    def check_collision(self) -> bool:
        """
        Check if collision occurred and reset for next check.
        """
        if self.collision_sensor is not None:
            return self.collision_sensor.check_and_reset()
        return False

    def destroy(self):
        """
        Safe destroy sequence - destroy all sensors properly.
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
        self.imu_sensor = None
        self.vehicle = None
        self.cav_world = None
