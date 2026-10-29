'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 17:59:29
FilePath     : /OpenCDA-MARL/opencda_marl/agents/agent_manager.py
Description  :
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from collections import deque
import math
from typing import Dict, Any, List, Deque
from loguru import logger
import carla
import uuid

from opencda.core.plan.global_route_planner import GlobalRoutePlanner
from opencda.core.plan.global_route_planner_dao import GlobalRoutePlannerDAO
from opencda_marl._kernel.traffic.events import SpawnEvent
from opencda_marl._kernel.traffic.population import (
    is_learning_population,
    validate_population_mix,
)
from opencda_marl._kernel.adapter.vehicle_adapter import MARLVehicleAdapter
from opencda_marl._kernel.adapter.vehicle_adapter import CollisionException
from opencda_marl._kernel.adapter.world_snapshot import WorldSnapshot
from opencda_marl.api.components import registered_components
from opencda_marl.api.exceptions import PlannerFailureException
from opencda_marl.api.events import StepEvent
from opencda_marl.api.path_action import decision_destination, decision_is_admissible
from opencda_marl._kernel.lifecycle.actor_registry import ActorRegistry, SpawnCollisionError
from opencda_marl._kernel.lifecycle.sensor_rig import SensorRig


class MARLAgentManager:
    def __init__(self, config: Dict[str, Any], state: Dict[str, Any],
                 world: carla.World,
                 cav_world,
                 client=None,
                 planner=None,
                 intent_provider=None,
                 marl_manager=None):
        self.config = config
        self.state = state
        self.world = world
        self.client = client
        self.map = world.get_map()
        self.cav_world = cav_world

        # Route-intent integration. Both fields stay None for existing
        # speed-only configs so _spawn_vehicles skips the hook entirely
        # and runs the original set_destination path byte-identically.
        self.planner = planner
        self.intent_provider = intent_provider
        self.marl_manager = marl_manager

        self.debug = config.get("debug", False)
        self._route_visualization = config.get("route_visualization", {})
        self._route_arrow_step = max(1, int(self._route_visualization.get("arrow_step", 3)))
        self._route_arrow_thickness = float(self._route_visualization.get("arrow_thickness", 0.18))
        self._route_arrow_size = float(self._route_visualization.get("arrow_size", 0.45))
        self._route_life_time = float(self._route_visualization.get("life_time", 10.0))
        self._route_z_offset = float(self._route_visualization.get("z_offset", 0.9))
        self._route_point_size = float(self._route_visualization.get("point_size", 0.16))
        self._route_planner = self._build_route_planner() if self.debug else None

        self._failure_events: Deque[SpawnEvent] = deque()
        #: Replay vehicles abandoned after MAX_SPAWN_RETRIES. They never
        #: entered, so like the still-queued ones they are unfinished routes
        #: of this episode, not vehicles that never existed.
        self._dropped_spawns: int = 0
        self._spawn_retry_counts: Dict[int, int] = {}

        # Live adapters in spawn order; each adapter owns its vehicle actor
        # (adapter.vehicle), so no parallel actor list is kept.
        self._vehicle_adapters: List['MARLVehicleAdapter'] = []
        # World snapshot of the current (un-ticked) world state, captured by
        # the observation pass and reused by step(); reset once the world moves.
        self._current_snapshot = None
        self._spawn_metadata_by_actor_id: Dict[Any, Dict[str, Any]] = {}
        #: The spawn event per live vehicle: the approach-trigger decision
        #: needs it long after the spawn call returned.
        self._spawn_event_by_actor_id: Dict[Any, Any] = {}

        # Single owner of actor destruction; verifies teardown against the
        # server. agents.lifecycle.strict_reconcile=true makes a detected
        # leak fatal (for runs whose data must not come from a leaking process).
        self.registry = ActorRegistry(
            world,
            client=client,
            strict=bool(config.get("lifecycle", {}).get("strict_reconcile", False)),
        )

        # Event log for GUI display
        self._event_logs: List[StepEvent] = []

        self.agent_type = config.get("agent_type", "simple")
        # Every population's agent is built through the agent registry at
        # its vehicle's spawn, so a mix with an unregistered name (or without
        # the learner) is rejected here, before the first episode starts.
        if config.get("population"):
            validate_population_mix(config.get("population"), self.agent_type,
                                    known_agent_types=registered_components("agent"))

        logger.success(
            f"MARLAgentManager initialized (all vehicles are CAV agents using '{self.agent_type}' agent type).")

    # --------------------------------------------------------------------- #
    # Main stepping (called by ScenarioManager)
    # --------------------------------------------------------------------- #
    def _committed_exit_lanes(self, actor_id, metadata):
        """Exit lanes other vehicles at this junction have already claimed."""
        from opencda_marl.api.route_intent import committed_exit_lane_occupancy

        others = []
        for other_id, other_meta in self._spawn_metadata_by_actor_id.items():
            if other_id == actor_id or not other_meta:
                continue
            row = dict(other_meta)
            row.setdefault("junction_id", row.get("junction_id"))
            others.append(row)
        return committed_exit_lane_occupancy(
            others, junction_id=metadata.get("junction_id"))

    @property
    def decision_zone(self):
        """The route-decision trigger, built on first use (see SUMO twin)."""
        zone = self.__dict__.get("_decision_zone")
        if zone is None:
            from opencda_marl._kernel.policy.decision_zone import build_decision_zone

            zone = build_decision_zone(
                (getattr(self, "config", None) or {}).get("route_intent"))
            self.__dict__["_decision_zone"] = zone
        return zone

    def _process_decision_zone(self):
        """Approach mode: decide for adapters that reached the zone.

        The vehicle is already driving its demand route, so the outcome is
        applied with the planner's own re-routing
        (``set_destination(..., clean=True)``) instead of a spawn-time
        destination.
        """
        zone = self.decision_zone
        if zone.decides_at_spawn or self.intent_provider is None or self.planner is None:
            return 0
        decided = 0
        for adapter in list(self._vehicle_adapters):
            actor_id = adapter.actor_id
            if not zone.is_pending(actor_id):
                continue
            event = self._spawn_event_by_actor_id.get(actor_id)
            if event is None:
                zone.forget(actor_id)
                continue
            if not is_learning_population(
                    getattr(event, "metadata", None), getattr(self, "agent_type", None)):
                zone.forget(actor_id)
                continue
            try:
                center = self._junction_center(actor_id)
                if center is None:
                    distance = adapter._calculate_distance_to_intersection()
                else:
                    position = adapter.vehicle.get_location()
                    distance = math.hypot(position.x - center.x, position.y - center.y)
            except Exception:  # noqa: BLE001 - a vanished actor is not an error
                zone.forget(actor_id)
                continue
            if not zone.due(actor_id, distance):
                continue
            decision = self.intent_provider({
                "event": event,
                "adapter": adapter,
                "planner": self.planner,
                "active_adapters": list(self._vehicle_adapters),
            })
            zone.mark_decided(actor_id)
            if decision is None:
                continue
            decided += 1
            metadata = dict(self._spawn_metadata_by_actor_id.get(actor_id, {}) or {})
            metadata["route_intent"] = decision.as_metadata()
            self._spawn_metadata_by_actor_id[actor_id] = metadata
            adapter.spawn_metadata = metadata
            if not decision_is_admissible(decision):
                logger.warning(
                    "The path decision failed its own invariant at the "
                    f"decision zone for {actor_id}; keeping the demand route.")
                continue
            destination = decision_destination(decision)
            if destination is None:
                # The provider chose to keep the demand route; nothing to re-plan.
                continue
            try:
                adapter.set_destination(
                    adapter.vehicle.get_location(),
                    destination.location,
                    clean=True,
                )
            except Exception as exc:  # noqa: BLE001 - keep the episode alive
                logger.warning(f"decision-zone re-route failed for {actor_id}: {exc}")
        return decided

    def _junction_center(self, actor_id):
        """The assigned junction's map-frame center, when its planner exposes it.

        Legacy planners without junction geometry keep their existing adapter
        distances. Never substitute another nearby junction for a known event.
        """
        event = getattr(self, '_spawn_event_by_actor_id', {}).get(actor_id)
        junction_id = getattr(event, 'junction_id', None)
        getter = getattr(getattr(self, 'planner', None), 'get_junctions', None)
        if junction_id is None or not callable(getter):
            return None
        junctions = getter()
        row = junctions.get(junction_id)
        if row is None:
            row = next((value for key, value in junctions.items()
                        if str(key) == str(junction_id)), None)
        return row.get('center') if row else None

    def step(self, events: List[SpawnEvent], target_speed: Dict[int, float] = {}):
        # Verify last step's teardowns first: the world has ticked since
        # their destroy chains ran, so the server's answer is now real.
        self.registry.sweep_pending()
        # The world has not ticked since the pre-step observation pass, so
        # its snapshot is still exact: reuse it (adapters then skip the
        # redundant update_info — see _update_info_once) instead of paying
        # a second capture per step. A fresh capture only if none exists.
        self._process_decision_zone()
        snapshot = self._current_snapshot
        if snapshot is None and self._vehicle_adapters:
            snapshot = WorldSnapshot.capture(self.world)
        if snapshot is not None:
            snapshot.publish_to_perception()
        try:
            self._step_all_vms(target_speed)
        finally:
            WorldSnapshot.unpublish_from_perception()
            # Controls are applied but the world has NOT ticked yet (the
            # caller ticks next), so this snapshot must not feed perception
            # again — but its positions are still exact for the spawn-point
            # occupancy check below, which would otherwise capture a third
            # time per step whenever spawn events are pending.
            self._current_snapshot = None
        self._spawn_vehicles(events, occupancy_snapshot=snapshot)

    def _step_all_vms(self, target_speed: Dict[int, float] = {}):
        finished: List['MARLVehicleAdapter'] = []

        for adapter in list(self._vehicle_adapters):
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
                if self.marl_manager is not None:
                    try:
                        self.marl_manager.notify_vehicle_terminal(adapter.actor_id)
                    except Exception as exc:
                        logger.warning(f"notify_vehicle_terminal failed: {exc}")
                finished.append(adapter)
            except CollisionException as e:
                logger.debug(f"Vehicle {adapter.actor_id} collided: {e}")
                # Store collision event for GUI logging
                event = StepEvent(step=self.state['step'],
                                  event_id=str(uuid.uuid4()),
                                  vehicle_id=adapter.actor_id,
                                  event_type="collision",
                                  metadata=dict(getattr(adapter, 'collision_details', None) or {}))
                self._add_event_log(event)
                if self.marl_manager is not None:
                    try:
                        self.marl_manager.notify_vehicle_terminal(adapter.actor_id)
                    except Exception as exc:
                        logger.warning(f"notify_vehicle_terminal failed: {exc}")
                finished.append(adapter)
            except PlannerFailureException as e:
                logger.warning(f"Vehicle {adapter.actor_id} planner failure: {e}")
                event = StepEvent(step=self.state['step'],
                                  event_id=str(uuid.uuid4()),
                                  vehicle_id=adapter.actor_id,
                                  event_type="planner_failure",
                                  metadata={"reason": str(e)})
                self._add_event_log(event)
                if self.marl_manager is not None:
                    try:
                        self.marl_manager.notify_vehicle_terminal(adapter.actor_id)
                    except Exception as exc:
                        logger.warning(f"notify_vehicle_terminal failed: {exc}")
                finished.append(adapter)
            except Exception as e:
                logger.warning(f"Vehicle {adapter.actor_id} failed: {e}")
                event = StepEvent(step=self.state['step'],
                                  event_id=str(uuid.uuid4()),
                                  vehicle_id=adapter.actor_id,
                                  event_type="planner_failure",
                                  metadata={"reason": str(e)})
                self._add_event_log(event)
                if self.marl_manager is not None:
                    try:
                        self.marl_manager.notify_vehicle_terminal(adapter.actor_id)
                    except Exception as exc:
                        logger.warning(f"notify_vehicle_terminal failed: {exc}")
                finished.append(adapter)

        # Remove completed/collided vehicles
        for adapter in finished:
            self._remove_adapter(adapter)

    def _remove_adapter(self, adapter: 'MARLVehicleAdapter'):
        """Tear down one vehicle adapter and forget it."""
        if adapter not in self._vehicle_adapters:
            return
        vehicle_id = adapter.actor_id

        # The adapter's destroy chain does the orderly teardown
        # (sensor managers -> vehicle); the registry then verifies
        # against the server and reclaims anything the chain missed.
        try:
            adapter.destroy()
        except Exception as exc:
            logger.warning(f"Adapter destroy failed for vehicle {vehicle_id}: {exc}")
        finally:
            self.registry.finalize_vehicle(vehicle_id)

        self._spawn_metadata_by_actor_id.pop(vehicle_id, None)
        self.__dict__.setdefault("_spawn_event_by_actor_id", {}).pop(vehicle_id, None)
        self.decision_zone.forget(vehicle_id)
        self._vehicle_adapters.remove(adapter)

        logger.debug(f"Successfully removed vehicle {vehicle_id}")

    def _build_route_planner(self):
        sampling_resolution = float(
            self.config.get("vehicle", {})
            .get("behavior", {})
            .get("sample_resolution", 2.0)
        )
        dao = GlobalRoutePlannerDAO(self.map, sampling_resolution)
        planner = GlobalRoutePlanner(dao, self.world, debug=False)
        planner.setup()
        return planner
    # --------------------------------------------------------------------- #
    # Public methods
    # --------------------------------------------------------------------- #

    def get_queue_count(self) -> int:
        return len(self._failure_events)

    def get_unentered_count(self) -> int:
        """Replay vehicles this episode never put on the road.

        A spawn point held by a queue defers its vehicle, and a vehicle
        deferred past the retry budget is abandoned. Either way the demand was
        offered and not served, so both are unfinished routes of this episode
        — the same accounting SUMO applies to its insertion queue. Counting
        them nowhere silently shrinks the denominator exactly when congestion
        is worst, which flatters every rate at high demand.
        """
        return len(self._failure_events) + self._dropped_spawns

    def get_all_observations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get observation data from all active vehicle adapters.

        Returns:
            dict: Dictionary mapping vehicle IDs to their observation data
        """
        observations = {}
        # One world snapshot serves every adapter this pass: a single
        # get_actors + one kinematics read per vehicle, instead of every
        # adapter re-fetching the actor list and each neighbour's state.
        # It is also published to the perception layer, since
        # get_observation() refreshes update_info() (perception detect),
        # and kept for the following step() so the world is read once per
        # tick, not once per consumer.
        snapshot = WorldSnapshot.capture(self.world) if self._vehicle_adapters else None
        self._current_snapshot = snapshot
        if snapshot is not None:
            snapshot.publish_to_perception()
        try:
            self._collect_observations(observations, snapshot)
        finally:
            WorldSnapshot.unpublish_from_perception()
        return observations

    def _collect_observations(self, observations, snapshot):
        for adapter in self._vehicle_adapters:
            try:
                obs = adapter.get_observation(snapshot=snapshot)
                vehicle_id = adapter.actor_id
                if 'error' in obs:
                    # Failed observation (already logged by the adapter):
                    # leave the agent out of this step rather than hand the
                    # extractor a stub it would fill with default values.
                    continue
                center = self._junction_center(vehicle_id)
                if center is not None and 'position_x' in obs and 'position_y' in obs:
                    # Same convention as the SUMO twin: vehicle minus the
                    # assigned junction center, not a vector to a nearby waypoint.
                    rel_x = float(obs['position_x']) - center.x
                    rel_y = float(obs['position_y']) - center.y
                    obs['relative_position_to_intersection'] = {'x': rel_x, 'y': rel_y}
                    obs['distance_to_intersection'] = math.hypot(rel_x, rel_y)
                observations[vehicle_id] = obs
                observations[vehicle_id]['vehicle_blueprint'] = str(
                    adapter.vehicle.type_id)
                metadata = dict(self._spawn_metadata_by_actor_id.get(vehicle_id, {}) or {})
                if metadata:
                    observations[vehicle_id]['entry_direction'] = metadata.get('entry_direction')
                    observations[vehicle_id]['route_type'] = metadata.get('route_type')
                    observations[vehicle_id]['lane_id'] = metadata.get('lane_id')
                    observations[vehicle_id]['preference'] = metadata.get('preference')
                    observations[vehicle_id]['route_intent'] = metadata.get('route_intent')
                    population = metadata.get('population') or self.agent_type
                    observations[vehicle_id]['population'] = population
                    observations[vehicle_id]['learning'] = (
                        str(population) == str(self.agent_type))
                    observations[vehicle_id]['committed_exit_lanes'] = (
                        self._committed_exit_lanes(vehicle_id, metadata))
            except Exception as e:
                logger.warning(
                    f"Failed to get observation from vehicle {adapter.actor_id}: {e}")
        return observations

    def _add_event_log(self, event: StepEvent):
        """Add event to the per-step log (drained by get_event_logs)."""
        self._event_logs.append(event)
        # get_event_logs() drains this list every step, so the cap is only a
        # runaway guard. Dropping must be loud: a silently lost collision or
        # success event corrupts rewards and episode counters (the old cap of
        # 50 could silently lose events in a mass pile-up step).
        if len(self._event_logs) > 1000:
            dropped = self._event_logs.pop(0)
            logger.warning(
                "Event log overflow (>1000 in one step): dropping oldest "
                f"event {dropped.event_id} ({dropped.event_type}) — is "
                "get_event_logs() being drained?")

    def get_event_logs(self) -> List[StepEvent]:
        """Get event logs for GUI display."""
        events = self._event_logs.copy()
        self._event_logs.clear()  # Clear after retrieving
        return events
    # --------------------------------------------------------------------- #
    # Private methods
    # --------------------------------------------------------------------- #

    #: Radius (m) within which a live vehicle *certainly* blocks a spawn.
    #: Deliberately tighter than the server's own bounding-box test so the
    #: pre-check only skips attempts CARLA would reject anyway — it must
    #: never be stricter than the server, or vehicles the replay schedules
    #: would go unspawned (that changed episode composition in testing).
    SPAWN_CLEARANCE_M = 2.0
    #: Runaway guard only. Deferring is normal (a queue can block a spawn
    #: point for a long time); dropping a replay-scheduled vehicle changes
    #: the episode, so this is generous and dropping is logged as an ERROR.
    MAX_SPAWN_RETRIES = 3000
    #: Class-level default so a manager built without ``__init__`` (as the
    #: unit tests build one) still has a tally to add to.
    _dropped_spawns = 0

    @classmethod
    def _spawn_point_occupied(cls, snapshot, spawn_tf) -> bool:
        if snapshot is None or len(snapshot) == 0:
            return False
        loc = spawn_tf.location
        d2 = (snapshot.pos[:, 0] - loc.x) ** 2 + (snapshot.pos[:, 1] - loc.y) ** 2
        return bool((d2 < cls.SPAWN_CLEARANCE_M ** 2).any())

    def _requeue_or_drop(self, event: SpawnEvent, *, reason: str) -> None:
        key = id(event)
        tries = self._spawn_retry_counts.get(key, 0) + 1
        self._spawn_retry_counts[key] = tries
        if tries >= self.MAX_SPAWN_RETRIES:
            self._spawn_retry_counts.pop(key, None)
            self._dropped_spawns += 1
            logger.error(
                f"Dropping spawn event {event.vehicle_id} after {tries} "
                f"deferred attempts ({reason}) — the episode no longer "
                "matches its replay; investigate the blocked spawn point")
            return
        self._failure_events.append(event)
        logger.debug(f"Deferring spawn of {event.vehicle_id} ({reason}), attempt {tries}")

    def _spawn_vehicles(self, events: List[SpawnEvent], occupancy_snapshot=None):
        # Combine new events with failure events to avoid modifying the original list
        all_events = []

        # Add failure events first (retry them)
        while self._failure_events:
            all_events.append(self._failure_events.popleft())

        # Then add new events
        all_events.extend(events)

        # A spawn point occupied by a vehicle that just spawned there stays
        # blocked for many ticks; retrying blindly cost ~120 failing server
        # RPCs per successful spawn. Skip attempts whose spawn point is
        # provably occupied (client-side, from one snapshot), and stop
        # retrying an event after a bounded number of tries.
        occupancy = None
        if all_events and self._vehicle_adapters:
            occupancy = (occupancy_snapshot or self._current_snapshot
                         or WorldSnapshot.capture(self.world))

        # spawn all events
        for event in all_events:
            bp = event.blueprint
            if not bp:
                raise ValueError(
                    f"No blueprint provided for {event.vehicle_id}")

            spawn_tf = event.transform
            dest_tf = event.destination
            target_speed_kmh = event.target_speed
            decision = None
            if occupancy is not None and self._spawn_point_occupied(occupancy, spawn_tf):
                self._requeue_or_drop(event, reason="spawn point occupied")
                continue
            try:
                # Registered inside spawn_vehicle, before anything else can
                # fail, so the registry can always find (and destroy) it.
                vehicle = self.registry.spawn_vehicle(bp, spawn_tf)
            except SpawnCollisionError as e:
                logger.debug(f"Failed to spawn vehicle with blueprint {bp.id} "
                             f"at transform {spawn_tf.location.x}, {spawn_tf.location.y}, {spawn_tf.location.z} "
                             f"with error: {e}")
                self._requeue_or_drop(event, reason="collision at spawn")
                continue

            adapter = None
            try:
                # Mixed autonomy: the spawn event carries this vehicle's
                # population (agent type). Homogeneous runs carry none and
                # every vehicle keeps the configured agent type.
                vehicle_agent_type = str(
                    (event.metadata or {}).get("population") or self.agent_type)
                receives_path_action = is_learning_population(event.metadata, self.agent_type)
                adapter = MARLVehicleAdapter(config=self.config,
                                             vehicle=vehicle,
                                             carla_map=self.map,
                                             cav_world=self.cav_world,
                                             agent_type=vehicle_agent_type,
                                             spawn_metadata=dict(event.metadata or {}),
                                             )
                # One-shot post-spawn route-intent decision. When the
                # intent hook is disabled (existing speed-only configs)
                # this branch is skipped and the original destination is
                # used unchanged.
                if (self.intent_provider is not None and self.planner is not None
                        and receives_path_action
                        and self.decision_zone.decides_at_spawn):
                    decision = self.intent_provider({
                        "event": event,
                        "adapter": adapter,
                        "planner": self.planner,
                        "active_adapters": list(self._vehicle_adapters),
                    })
                    if decision is not None:
                        destination = decision_destination(decision)
                        if decision_is_admissible(decision) and destination is not None:
                            dest_tf = destination
                        elif not decision_is_admissible(decision):
                            logger.warning(
                                "The path decision failed its own invariant for "
                                f"vehicle {event.vehicle_id}; keeping the demand route."
                            )

                self.visualize_route(spawn_tf, dest_tf)
                adapter.set_destination(
                    spawn_tf.location, dest_tf.location, clean=True)
                adapter.set_target_speed(target_speed_kmh)
                # Initialize ego position after setting destination
                adapter.vm.update_info()
            except Exception:
                # The vehicle (and any sensors its managers already
                # attached) must not outlive a failed setup.
                if adapter is not None:
                    try:
                        adapter.destroy()
                    except Exception as exc:
                        logger.warning(
                            f"Adapter destroy failed after a setup failure: {exc}")
                self.registry.abort_spawn(vehicle)
                raise

            # Track spawned vehicle and adapter for cleanup; hand the
            # sensor actors to the registry for verified teardown.
            self.registry.adopt_sensors(
                vehicle.id, SensorRig.from_vehicle_manager(adapter.vm).sensor_ids())
            self._vehicle_adapters.append(adapter)
            spawn_metadata = dict(event.metadata or {})
            if decision is not None:
                spawn_metadata["route_intent"] = decision.as_metadata()
            adapter.spawn_metadata = spawn_metadata
            self._spawn_metadata_by_actor_id[adapter.actor_id] = spawn_metadata
            self._spawn_event_by_actor_id[adapter.actor_id] = event
            if receives_path_action:
                self.decision_zone.register(adapter.actor_id)
    # --------------------------------------------------------------------- #
    # visualizations
    # --------------------------------------------------------------------- #

    def visualize_route(self, spawn_tf, dest_tf, life_time=None):
        if not self.debug:
            return

        life_time = self._route_life_time if life_time is None else life_time
        spawn_loc = spawn_tf.location + carla.Location(z=self._route_z_offset)
        dest_loc = dest_tf.location + carla.Location(z=self._route_z_offset)

        self.world.debug.draw_point(
            spawn_loc,
            size=self._route_point_size,
            color=carla.Color(255, 128, 0),
            life_time=life_time,
        )
        self.world.debug.draw_point(
            dest_loc,
            size=self._route_point_size,
            color=carla.Color(0, 255, 255),
            life_time=life_time,
        )

        if self._route_planner is None:
            self._route_planner = self._build_route_planner()

        try:
            route = self._route_planner.trace_route(spawn_tf.location, dest_tf.location)
        except Exception as exc:
            logger.debug(f"Route visualization trace failed, using direct line fallback: {exc}")
            route = []

        if not route:
            self.world.debug.draw_line(
                spawn_loc,
                dest_loc,
                thickness=self._route_arrow_thickness,
                color=carla.Color(255, 128, 0),
                life_time=life_time,
            )
            return

        for index in range(0, max(0, len(route) - 1), self._route_arrow_step):
            start_wp = route[index][0]
            end_index = min(index + self._route_arrow_step, len(route) - 1)
            end_wp = route[end_index][0]
            start_loc = start_wp.transform.location + carla.Location(z=self._route_z_offset)
            end_loc = end_wp.transform.location + carla.Location(z=self._route_z_offset)
            self.world.debug.draw_arrow(
                start_loc,
                end_loc,
                thickness=self._route_arrow_thickness,
                arrow_size=self._route_arrow_size,
                color=carla.Color(255, 128, 0),
                life_time=life_time,
            )

    # --------------------------------------------------------------------- #
    # Clean up
    # --------------------------------------------------------------------- #

    def reset(self):
        """Reset agent manager for new episode."""
        logger.info("Resetting MARLAgentManager for new episode")

        # Clean up existing vehicles and adapters
        self.cleanup()

        # Clear failure event queue and its retry bookkeeping
        self._failure_events.clear()
        self._spawn_retry_counts.clear()
        self._dropped_spawns = 0
        self._current_snapshot = None

        # Clear event logs
        self._event_logs.clear()

        logger.success("MARLAgentManager reset completed")

    def cleanup(self):
        """Clean up all spawned vehicles and their adapters.

        Phase 1 runs the python-side manager teardown (which stops and
        destroys the sensor actors — without it CARLA logs "sensor object
        went out of the scope but the sensor is still alive"). Phase 2
        batch-destroys the vehicles and reconciles against the server, so
        anything phase 1 missed is reclaimed instead of leaked. Adapter
        handles are dropped only after verification.
        """
        logger.info(
            f"Cleaning up {len(self._vehicle_adapters)} vehicle adapters")

        # Terminal sweep: vehicles still alive at cleanup are timeouts.
        # Notifying before teardown finalizes their pending route-intent
        # transitions with the accumulated reward (parity with the SUMO
        # reset sweep). Without this, timed-out vehicles never reach the
        # route-intent buffer — the head trained only on collided/arrived
        # vehicles (survivorship bias) and _pending grew without bound.
        if self.marl_manager is not None:
            for adapter in self._vehicle_adapters:
                try:
                    self.marl_manager.notify_vehicle_terminal(adapter.actor_id)
                except Exception as exc:
                    logger.warning(
                        "Terminal sweep notify failed for "
                        f"{getattr(adapter, 'actor_id', '?')}: {exc}")

        # Phase 1: detach + sensor-manager teardown; vm handles stay intact
        # so a later failure can still reach every resource.
        for adapter in self._vehicle_adapters:
            vm = getattr(adapter, "vm", None)
            if vm is None:
                continue
            try:
                self.cav_world.remove_vehicle_manager(vm)
            except Exception as exc:
                actor_id = getattr(adapter, "actor_id", "unknown")
                logger.warning(
                    f"Failed to detach vehicle manager for {actor_id}: {exc}")
            rig = SensorRig.from_vehicle_manager(vm)
            for component_name in ("perception_manager",
                                   "localizer",
                                   "safety_manager"):
                component = getattr(vm, component_name, None)
                if component is None:
                    continue
                try:
                    component.destroy()
                except Exception as exc:
                    logger.warning(
                        f"{component_name}.destroy failed during cleanup: {exc}")
            # The vendored managers swallow some of their own failures; a
            # sensor they left alive must be stopped and destroyed through
            # this handle before it is dropped.
            rig.stop_and_destroy()

        # Phase 2: batch vehicle destroy; destruction materializes on the
        # next tick in synchronous mode.
        self.registry.destroy_all()
        try:
            self.world.tick()
        except Exception as e:
            logger.debug(f"World tick after cleanup failed: {e}")

        # Phase 3: post-tick verification — genuine survivors are destroyed
        # and reported (raising under agents.lifecycle.strict_reconcile).
        self.registry.sweep_pending(force=True)

        # Phase 4: drop handles only after verification.
        for adapter in self._vehicle_adapters:
            adapter.vm = None
        self._vehicle_adapters.clear()
        self._spawn_metadata_by_actor_id.clear()
        self.__dict__.setdefault("_spawn_event_by_actor_id", {}).clear()
        self.decision_zone.reset_episode()

        logger.info("MARLAgentManager cleanup completed")
