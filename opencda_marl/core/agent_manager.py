'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 17:59:29
FilePath     : /OpenCDA-MARL/opencda_marl/core/agent_manager.py
Description  :
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from collections import deque
from typing import Dict, Any, List, Deque
from loguru import logger
import carla
import uuid

from opencda_marl.core.traffic.events import SpawnEvent
from opencda_marl.core.adapter import MARLVehicleAdapter
from opencda_marl.core.adapter.vehicle_adapter import CollisionException
from opencda_marl.core.events import StepEvent


class MARLAgentManager:
    def __init__(self, config: Dict[str, Any], state: Dict[str, Any],
                 world: carla.World,
                 cav_world):
        self.config = config
        self.state = state
        self.world = world
        self.map = world.get_map()
        self.cav_world = cav_world

        self.debug = config.get("debug", False)

        self._failure_events: Deque[SpawnEvent] = deque()

        # Track spawned vehicles and adapters for cleanup
        self._spawned_vehicles: List[carla.Actor] = []
        self._vehicle_adapters: List['MARLVehicleAdapter'] = []

        # Event log for GUI display
        self._event_logs: List[StepEvent] = []

        self.agent_type = config.get("agent_type", "simple")

        logger.success(
            f"MARLAgentManager initialized (all vehicles are CAV agents using '{self.agent_type}' agent type).")

    # --------------------------------------------------------------------- #
    # Main stepping (called by ScenarioManager)
    # --------------------------------------------------------------------- #
    def step(self, events: List[SpawnEvent], target_speed: Dict[int, float] = {}):
        self._step_all_vms(target_speed)
        self._spawn_vehicles(events)

    def _step_all_vms(self, target_speed: Dict[int, float] = {}):
        remove_indices: List[int] = []

        for i, adapter in enumerate(self._vehicle_adapters):
            try:
                # Handle both int and string keys from MARL manager
                # CARLA uses int actor IDs, but observation extractor may use string keys
                actor_id = adapter.actor_id
                if actor_id in target_speed:
                    agent_target_speed = target_speed[actor_id]
                elif str(actor_id) in target_speed:
                    agent_target_speed = target_speed[str(actor_id)]
                else:
                    agent_target_speed = None
                adapter.step(agent_target_speed)
            except StopIteration as e:
                logger.debug(f"Vehicle {adapter.actor_id} completed: {e}")
                # Store completion event for GUI logging
                event = StepEvent(step=self.state['step'],
                                  event_id=str(uuid.uuid4()),
                                  vehicle_id=adapter.actor_id,
                                  event_type="success")
                self._add_event_log(event)
                remove_indices.append(i)
            except CollisionException as e:
                logger.debug(f"Vehicle {adapter.actor_id} collided: {e}")
                # Store collision event for GUI logging
                event = StepEvent(step=self.state['step'],
                                  event_id=str(uuid.uuid4()),
                                  vehicle_id=adapter.actor_id,
                                  event_type="collision")
                self._add_event_log(event)
                remove_indices.append(i)
            except Exception as e:
                logger.error(f"Failed to step vehicle with adapter {adapter.actor_id} "
                             f"with error: {e}")
                # keep this for traceback
                import traceback
                traceback.print_exc()
                raise e

        for i in reversed(remove_indices):
            self._remove_adapter_by_index(i)

    def _remove_adapter_by_index(self, index: int):
        if 0 <= index < len(self._vehicle_adapters):
            adapter = self._vehicle_adapters[index]
            vehicle = self._spawned_vehicles[index]

            # Clean up the adapter (this will destroy the VehicleManager)
            adapter.destroy()

            # Destroy the CARLA vehicle actor
            if vehicle.is_alive:
                vehicle.destroy()

            # Remove from lists
            self._vehicle_adapters.pop(index)
            self._spawned_vehicles.pop(index)

            logger.debug(f"Successfully removed vehicle {adapter.actor_id}")
    # --------------------------------------------------------------------- #
    # Public methods
    # --------------------------------------------------------------------- #

    def get_queue_count(self) -> int:
        return len(self._failure_events)

    def get_all_observations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get observation data from all active vehicle adapters.

        Returns:
            dict: Dictionary mapping vehicle IDs to their observation data
        """
        observations = {}
        for adapter in self._vehicle_adapters:
            try:
                obs = adapter.get_observation()
                vehicle_id = adapter.actor_id
                observations[vehicle_id] = obs
                observations[vehicle_id]['vehicle_blueprint'] = str(
                    adapter.vehicle.type_id)
            except Exception as e:
                logger.warning(
                    f"Failed to get observation from vehicle {adapter.actor_id}: {e}")
        return observations

    def _add_event_log(self, event: StepEvent):
        """Add event to log for GUI display."""
        self._event_logs.append(event)
        # Keep only last 50 events to prevent memory buildup
        if len(self._event_logs) > 50:
            self._event_logs.pop(0)

    def get_event_logs(self) -> List[StepEvent]:
        """Get event logs for GUI display."""
        events = self._event_logs.copy()
        self._event_logs.clear()  # Clear after retrieving
        return events
    # --------------------------------------------------------------------- #
    # Private methods
    # --------------------------------------------------------------------- #

    def _spawn_vehicles(self, events: List[SpawnEvent]):
        # Combine new events with failure events to avoid modifying the original list
        all_events = []

        # Add failure events first (retry them)
        while self._failure_events:
            all_events.append(self._failure_events.popleft())

        # Then add new events
        all_events.extend(events)

        # spawn all events
        for event in all_events:
            bp = event.blueprint
            if not bp:
                raise ValueError(
                    f"No blueprint provided for {event.vehicle_id}")

            spawn_tf = event.transform
            dest_tf = event.destination
            taget_speed = event.target_speed
            try:
                vehicle = self.world.spawn_actor(bp, spawn_tf)
                adapter = MARLVehicleAdapter(config=self.config,
                                             vehicle=vehicle,
                                             carla_map=self.map,
                                             cav_world=self.cav_world,
                                             agent_type=self.agent_type,
                                             )
                self.visualize_route(spawn_tf, dest_tf)
                adapter.set_destination(
                    spawn_tf.location, dest_tf.location, clean=True)
                adapter.set_target_speed(taget_speed)
                # Initialize ego position after setting destination
                adapter.vm.update_info()
            except Exception as e:
                if 'collision at spawn' in str(e):
                    logger.debug(f"Failed to spawn vehicle with blueprint {bp.id} "
                                 f"at transform {spawn_tf.location.x}, {spawn_tf.location.y}, {spawn_tf.location.z} "
                                 f"with error: {e}")
                    self._failure_events.append(event)
                    logger.debug(f"Queueing failure event: {event.vehicle_id}")
                    continue
                else:
                    raise e

            # Track spawned vehicle and adapter for cleanup
            self._spawned_vehicles.append(vehicle)
            self._vehicle_adapters.append(adapter)
    # --------------------------------------------------------------------- #
    # visualizations
    # --------------------------------------------------------------------- #

    def visualize_route(self, spawn_tf, dest_tf, life_time=10):
        if self.debug:
            self.world.debug.draw_line(
                spawn_tf.location,
                dest_tf.location,
                thickness=0.2,
                color=carla.Color(0, 0, 255),  # Blue
                life_time=life_time
            )

    # --------------------------------------------------------------------- #
    # Clean up
    # --------------------------------------------------------------------- #

    def reset(self):
        """Reset agent manager for new episode."""
        logger.info("Resetting MARLAgentManager for new episode")

        # Clean up existing vehicles and adapters
        self.cleanup()

        # Clear failure event queue
        self._failure_events.clear()

        # Clear event logs
        self._event_logs.clear()

        logger.success("MARLAgentManager reset completed")

    def cleanup(self):
        """Clean up all spawned vehicles and their adapters."""
        logger.info(
            f"Cleaning up {len(self._vehicle_adapters)} vehicle adapters"
            f"and {len(self._spawned_vehicles)} vehicles")
        for _, adapter in enumerate(self._vehicle_adapters):
            adapter.destroy()

        for _, vehicle in enumerate(self._spawned_vehicles):
            if hasattr(vehicle, 'is_alive') and vehicle.is_alive:
                vehicle.destroy()

        # Clear tracking lists
        self._vehicle_adapters.clear()
        self._spawned_vehicles.clear()

        logger.info("MARLAgentManager cleanup completed")
