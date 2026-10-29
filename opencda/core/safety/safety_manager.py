"""
The safety manager is used to collect the AV's hazard status and give the
control back to human if necessary
"""
import weakref
from collections import deque

# <OpenCDA-MARL> logger for hardened sensor teardown
from loguru import logger
# </OpenCDA-MARL>

from opencda.core.safety.sensors import CollisionSensor, \
    TrafficLightDector, StuckDetector, OffRoadDetector, IMUSensor


class SafetyManager:
    """
    A class that manages the safety of a given vehicle in a simulation environment.

    Parameters
    ----------
    vehicle: carla.Actor
        The vehicle that the SafetyManager is responsible for.
    params: dict
        A dictionary of parameters that are used to configure the SafetyManager.
    """

    def __init__(self, cav_world, vehicle, params, collision_sensor_cls=None):
        self.vehicle = vehicle
        self.cav_world = weakref.ref(cav_world)()
        self.print_message = params['print_message']
        # <OpenCDA-MARL>
        self._printed_warnings = frozenset()
        # Sensor-actor economy: the safety IMU duplicates the localization
        # IMU and its data deque is only consumed by OpenCDA's comfort
        # evaluator; make it opt-in (params['imu']: default True keeps stock
        # OpenCDA behavior). collision_sensor_cls lets callers install a
        # specialised collision sensor at construction time instead of
        # spawning the stock one and immediately destroying/re-spawning it.
        collision_cls = collision_sensor_cls or CollisionSensor
        use_imu = bool(params.get('imu', True))
        self.imu_sensor = IMUSensor(vehicle) if use_imu else None
        self.status_queue = deque(maxlen=params.get('queue_maxlen', 2000))
        self.sensors = [collision_cls(vehicle, params['collision_sensor']),
                        StuckDetector(params['stuck_dector']),
                        OffRoadDetector(params['offroad_dector']),
                        TrafficLightDector(params['traffic_light_detector'], vehicle)]
        if self.imu_sensor is not None:
            self.sensors.append(self.imu_sensor)
        # </OpenCDA-MARL>

    def update_info(self, data_dict) -> dict:
        status_dict = {}
        for sensor in self.sensors:
            sensor.tick(data_dict)
            status_dict.update(sensor.return_status())
        # store the status every tick
        self.status_queue.append((self.cav_world.global_clock, status_dict))
        # print hazard message
        if self.print_message:
            # <OpenCDA-MARL>
            # Print when the set of triggered warnings changes, not on every
            # tick of a warning that persists (one line per tick for a whole
            # route otherwise).
            warnings = frozenset(
                key for key, val in status_dict.items() if val)
            if warnings and warnings != self._printed_warnings:
                print("Safety Warning from the safety manager:")
                print(status_dict)
            self._printed_warnings = warnings
            # </OpenCDA-MARL>

    def destroy(self):
        # <OpenCDA-MARL>
        # Per-sensor isolation: one sensor's destroy failure must not skip
        # the rest of the list (the list mixes real CARLA sensor actors with
        # pure-python detectors).
        for sensor in self.sensors:
            try:
                sensor.destroy()
            except Exception as exc:
                logger.warning(
                    f"SafetyManager.destroy: {type(sensor).__name__}.destroy "
                    f"failed: {exc}")
        # </OpenCDA-MARL>
        self.sensors.clear()  # Clear references to avoid GC warnings
        self.imu_sensor = None
        self.vehicle = None
