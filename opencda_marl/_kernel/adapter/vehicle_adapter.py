'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 18:22:20
FilePath     : /OpenCDA-MARL/opencda_marl/adapter/vehicle_adapter.py
Description  :
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from typing import Dict, Any
from loguru import logger
import carla

from opencda.core.common.vehicle_manager import VehicleManager

from opencda_marl._kernel.adapter.vehicle_defaults import get_vehicle_manager_defaults
from opencda_marl._kernel.lifecycle.sensor_rig import SensorRig
from opencda_marl.api.exceptions import CollisionException
from opencda_marl._kernel.sensing.marl_collision_sensor import MARLCollisionSensor

from opencda_marl._kernel.adapter._physics import (
    get_physics_speed_planar_kmh,
    get_physics_state,
    inspect_physics_state,
)
from opencda_marl._kernel.adapter._spatial import (
    calculate_distance_to_intersection,
    calculate_distance_to_front_vehicle,
    classify_lane_position,
    calculate_relative_position_to_intersection,
    get_vehicle_heading_angle,
    calculate_distance_to_destination,
)
from opencda_marl._kernel.adapter._nearby import (
    DEFAULT_NEARBY_DETECTION_RADIUS,
    DEFAULT_MAX_NEARBY_VEHICLES,
    DEFAULT_FEATURES_PER_VEHICLE,
    DEFAULT_MAX_RELATIVE_VELOCITY,
    DEFAULT_MAX_TTC,
    compute_nearby_vehicle_features,
)


class MARLVehicleAdapter:

    def __init__(self, config: Dict[str, Any],
                 vehicle: carla.Actor, carla_map, cav_world,
                 dump_data: bool = False,
                 agent_type: str = "behavior",
                 spawn_metadata: Dict[str, Any] | None = None):
        self.config = config
        self.vm_cfg = self._merge_config(config)
        # Uncomment to print the vehicle config for debugging
        # self._print_vm_cfg()

        self.vehicle = vehicle
        self.actor_id = vehicle.id
        self.carla_map = carla_map
        self.cav_world = cav_world
        self.world = vehicle.get_world()
        self.dump_data = dump_data
        self.agent_type = agent_type

        # Store target speed for monitoring
        self.target_speed = 0.0

        # Load nearby vehicle config from MARL config (or use defaults)
        self._init_nearby_vehicle_config(config)

        # This vehicle's demand as the recording states it (destination,
        # driver parameters, population). The agent factory sees it, so a
        # driver model can be per-vehicle and reproducible across methods.
        self.spawn_metadata = dict(spawn_metadata or {})

        self.vm = self.get_vm()

    def _init_nearby_vehicle_config(self, config: Dict[str, Any]):
        """Initialize nearby vehicle detection parameters from config."""
        # Try to get config from MARL.td3.nearby_vehicle_config
        marl_config = config.get('MARL', {})
        td3_config = marl_config.get('td3', {})
        nearby_config = td3_config.get('nearby_vehicle_config', {})

        # Load values from config with defaults
        self.nearby_detection_radius = nearby_config.get('detection_radius', DEFAULT_NEARBY_DETECTION_RADIUS)
        self.max_nearby_vehicles = nearby_config.get('max_vehicles', DEFAULT_MAX_NEARBY_VEHICLES)
        self.features_per_vehicle = nearby_config.get('features_per_vehicle', DEFAULT_FEATURES_PER_VEHICLE)
        self.max_relative_velocity = nearby_config.get('max_relative_velocity', DEFAULT_MAX_RELATIVE_VELOCITY)
        self.max_ttc = nearby_config.get('max_ttc', DEFAULT_MAX_TTC)

    # --------------------------------------------------------------------- #
    # Public control API
    # --------------------------------------------------------------------- #

    def step(self, target_speed: float = None):
        # Store target speed for monitoring
        if target_speed is not None:
            self.target_speed = target_speed
        else:
            # During warmup (target_speed=None): use current vehicle speed as "target"
            # This reflects what vanilla agent is commanding, for TensorBoard tracking
            try:
                self.target_speed = self._get_physics_speed_planar_kmh()
            except Exception:
                pass  # Keep previous value if velocity unavailable

        self._update_info_once(getattr(self, '_snapshot', None))

        if self.check_collision():
            raise CollisionException(f"Vehicle {self.actor_id} collided")

        control = self.vm.run_step(target_speed)
        self.vm.vehicle.apply_control(control)

    def _update_info_once(self, snapshot) -> None:
        """Run ``vm.update_info()`` at most once per world snapshot.

        Keyed on the snapshot's monotonic ``seq``: the agent manager captures
        one snapshot per step and hands it to both the step loop and the
        observation passes, so all three see the same id and only the first
        recomputes. Without a snapshot (legacy callers) every call refreshes.
        """
        vm = getattr(self, 'vm', None)
        if vm is None or not hasattr(vm, 'update_info'):
            return
        seq = getattr(snapshot, 'seq', None)
        if seq is not None and getattr(self, '_last_update_seq', None) == seq:
            return
        vm.update_info()
        self._last_update_seq = seq

    def set_destination(self, start_location: carla.Location, end_location: carla.Location,
                        clean: bool = False, end_reset: bool = True):
        self.vm.set_destination(start_location, end_location,
                                clean, end_reset)

    def set_target_speed(self, target_speed: float):
        # Duck-typed seam: any agent exposing set_target_speed receives the
        # speed command (vanilla/marl do; behavior agents deliberately not).
        setter = getattr(self.vm.agent, "set_target_speed", None)
        if setter is not None:
            logger.debug(
                f"Setting target speed for vehicle {self.actor_id} to {target_speed}")
            setter(target_speed)

    def check_collision(self) -> bool:
        """
        Check-and-reset the MARL collision flag.

        Reads the MARLCollisionSensor directly (its non-resetting
        return_status() lets update_info() report collision without consuming
        it; check_and_reset() consumes it here). Reading the sensor from the
        adapter replaces the old monkey-patched closure on the safety
        manager, which created a per-vehicle reference cycle that only the
        gc's cycle collector could break.
        """
        try:
            safety_manager = self.vm.safety_manager
            sensors = getattr(safety_manager, 'sensors', None) if safety_manager else None
            if sensors and isinstance(sensors[0], MARLCollisionSensor):
                collided = sensors[0].check_and_reset()
                if collided:
                    self.collision_details = dict(
                        getattr(sensors[0], 'last_collision_details', None) or {})
                return collided
        except Exception as e:
            logger.debug(f"Could not check collision for vehicle {self.actor_id}: {e}")
        return False

    # --------------------------------------------------------------------- #
    # Observation assembly
    # --------------------------------------------------------------------- #

    def get_observation(self, snapshot=None) -> Dict[str, Any]:
        """
        Assemble this vehicle's observation.

        Args:
            snapshot: Optional per-step ``WorldSnapshot`` shared by every
                adapter (one server round trip for the whole step). Without
                it, neighbour features fall back to per-vehicle RPCs.

        Returns:
            dict: Agent observation data
        """
        self._snapshot = snapshot
        try:
            # Refresh perception/localization for this tick — once. Within a
            # synchronous tick the world is frozen, so a second update_info
            # (the step loop and both observation passes used to each run
            # one: 3x per vehicle per step) recomputes identical results.
            self._update_info_once(snapshot)

            obs = {'vehicle_id': self.actor_id}
            physics_state = self._get_physics_state()
            agent_speed = 0.0

            # Use agent getter methods if available
            if hasattr(self.vm, 'agent'):
                agent = self.vm.agent

                if hasattr(agent, 'get_speed'):
                    agent_speed = float(agent.get_speed())
                elif hasattr(agent, '_ego_speed'):
                    agent_speed = float(agent._ego_speed)
                else:
                    agent_speed = 0.0

                # Get position using getter method
                if hasattr(agent, 'get_position'):
                    pos_x, pos_y = agent.get_position()
                    obs['position_x'] = round(pos_x, 1)
                    obs['position_y'] = round(pos_y, 1)
                elif hasattr(agent, '_ego_pos') and agent._ego_pos:
                    obs['position_x'] = round(agent._ego_pos.location.x, 1)
                    obs['position_y'] = round(agent._ego_pos.location.y, 1)
                else:
                    obs['position_x'] = 0.0
                    obs['position_y'] = 0.0

                # Get waypoint buffer size from local planner
                if hasattr(agent, '_local_planner'):
                    planner = agent._local_planner
                    buffer = planner.get_waypoint_buffer()
                    obs['waypoint_buffer_size'] = len(buffer) if buffer else 0
                else:
                    obs['waypoint_buffer_size'] = 0
            else:
                # Fallback values if no agent
                obs['speed'] = 0.0
                obs['position_x'] = 0.0
                obs['position_y'] = 0.0
                obs['lane_id'] = 'N/A'
                obs['waypoint_buffer_size'] = 0

            obs['speed'] = round(physics_state['planar_speed_kmh'], 2)
            obs['physics_speed'] = round(physics_state['planar_speed_kmh'], 2)
            obs['physics_speed_planar'] = round(physics_state['planar_speed_kmh'], 2)
            obs['physics_speed_3d'] = round(physics_state['speed_3d_kmh'], 2)
            obs['vertical_speed'] = round(physics_state['vertical_speed_kmh'], 2)
            obs['agent_speed'] = round(agent_speed, 2)
            obs['tilt_pitch'] = round(physics_state['pitch_deg'], 2)
            obs['tilt_roll'] = round(physics_state['roll_deg'], 2)
            obs['height_above_waypoint'] = round(physics_state['height_delta_m'], 3)

            # Basic status - can be expanded based on agent state
            obs['status'] = self.vehicle.is_alive

            # Add MARL-specific features
            obs['distance_to_intersection'] = self._calculate_distance_to_intersection()
            obs['distance_to_front_vehicle'] = self._calculate_distance_to_front_vehicle()
            obs['lane_position'] = self._classify_lane_position()

            # Add nearby vehicle features for Phase 3 enhancement (35D)
            nearby_features, min_ttc = self._compute_nearby_vehicle_features()
            obs['nearby_vehicles'] = nearby_features  # 35D list (5 vehicles × 7 features)
            obs['min_ttc'] = min_ttc  # For TTC-based reward calculation
            obs['distance_to_destination'] = self._calculate_distance_to_destination()  # For progress reward

            # Add new enhanced DQN features (9D feature set)
            # 1. Relative position to intersection (replaces absolute position)
            rel_x, rel_y = self._calculate_relative_position_to_intersection()
            obs['relative_position_to_intersection'] = {
                'x': round(rel_x, 2),
                'y': round(rel_y, 2)
            }

            # 2. Vehicle heading angle
            obs['heading_angle'] = round(self._get_vehicle_heading_angle(), 3)

            # 3. Raw waypoint buffer count (already available as waypoint_buffer_size)

            # Keep legacy DQN observations for backward compatibility
            # Location dictionary for DQN extractor (now uses relative position)
            obs['location'] = {
                'x': rel_x,
                'y': rel_y
            }

            # Keep velocity for backward compatibility but remove from new feature set
            try:
                velocity = self.vehicle.get_velocity()
                obs['velocity'] = {
                    'x': velocity.x,
                    'y': velocity.y,
                    'z': velocity.z,
                }
            except Exception as e:
                logger.debug(f"Could not get velocity for vehicle {self.actor_id}: {e}")
                obs['velocity'] = {'x': 0.0, 'y': 0.0}

            # Add target speed for monitoring
            obs['target_speed'] = round(self.target_speed, 2)

            # Duck-typed seam: an agent that shields the commanded speed
            # reports what it applied (``applied_speed``) and whether the
            # shield bound (``shield_active``), so the trace can measure
            # how often the safety layer, not the policy, set the speed.
            # The bundled RuleBasedAgent reports its lane hold and merge
            # order (``lane_hold``, ``merge_priority_*``) and its rules'
            # target (``applied_speed``) the same way; it has no shield, so
            # it never reports ``shield_active``.
            shield_state = getattr(getattr(self.vm, 'agent', None), 'shield_state', None)
            if callable(shield_state):
                try:
                    obs.update(dict(shield_state() or {}))
                except Exception as e:  # noqa: BLE001 - diagnostics never break a step
                    logger.debug(f"shield_state failed for vehicle {self.actor_id}: {e}")

            return obs
        except Exception as e:
            # A failed observation must not become training data: without
            # this signal the extractor filled the missing features with
            # defaults and the agent trained on fabricated state, silently.
            logger.warning(
                f"Observation failed for vehicle {self.actor_id}: {e}; "
                "agent is skipped this step")
            return {'vehicle_id': self.actor_id, 'error': str(e)}

    # --------------------------------------------------------------------- #
    # Thin wrappers — delegate to standalone helper functions
    # --------------------------------------------------------------------- #

    def _get_physics_speed_planar_kmh(self) -> float:
        return get_physics_speed_planar_kmh(self.vehicle, self.actor_id)

    def _get_physics_state(self) -> Dict[str, float]:
        return get_physics_state(self.vehicle, self.carla_map, self.actor_id)

    def inspect_physics_state(self, guard_config: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        return inspect_physics_state(
            self.vehicle, self.carla_map, self.actor_id,
            self.target_speed, guard_config)

    def _calculate_distance_to_intersection(self) -> float:
        return calculate_distance_to_intersection(self.vm, self.carla_map, self.actor_id)

    def _calculate_distance_to_front_vehicle(self) -> float:
        snapshot = getattr(self, '_snapshot', None)
        if snapshot is not None:
            return snapshot.front_vehicle_distance(self.actor_id, self.carla_map)
        return calculate_distance_to_front_vehicle(self.vehicle, self.world, self.carla_map, self.actor_id)

    def _classify_lane_position(self) -> int:
        return classify_lane_position(self.vehicle, self.carla_map, self.actor_id)

    def _calculate_relative_position_to_intersection(self) -> tuple:
        return calculate_relative_position_to_intersection(self.vehicle, self.carla_map)

    def _get_vehicle_heading_angle(self) -> float:
        return get_vehicle_heading_angle(self.vehicle)

    def _calculate_distance_to_destination(self) -> float:
        return calculate_distance_to_destination(self.vm, self.vehicle, self.actor_id)

    def _compute_nearby_vehicle_features(self) -> tuple:
        snapshot = getattr(self, '_snapshot', None)
        if snapshot is not None:
            return snapshot.neighbour_features(
                self.actor_id,
                radius=self.nearby_detection_radius,
                k=self.max_nearby_vehicles,
                max_relative_velocity=self.max_relative_velocity,
                max_ttc=self.max_ttc,
                features_per_vehicle=self.features_per_vehicle,
            )
        return compute_nearby_vehicle_features(
            self.vehicle, self.world,
            self.nearby_detection_radius, self.max_nearby_vehicles,
            self.features_per_vehicle, self.max_relative_velocity,
            self.max_ttc)

    # --------------------------------------------------------------------- #
    # Config helpers
    # --------------------------------------------------------------------- #

    def _merge_config(self, config: Dict[str, Any] = {}):
        """
        Deep merge configuration with defaults.
        """
        import copy
        from collections.abc import Mapping

        def deep_merge(dct, merge_dct):
            """Recursive dict merge. Inspired by dict.update(), but for nested dicts."""
            for k, v in merge_dct.items():
                if k in dct and isinstance(dct[k], Mapping) and isinstance(v, Mapping):
                    deep_merge(dct[k], v)
                else:
                    dct[k] = v

        # Start with a deep copy of defaults.
        default_vm_cfg = get_vehicle_manager_defaults()
        merged_cfg = copy.deepcopy(default_vm_cfg)
        deep_merge(merged_cfg, config.get("vehicle", {}))
        return merged_cfg

    def _print_vm_cfg(self):
        import pprint
        print("="*100)
        print("Merged VM Config:")
        pprint.pprint(self.vm_cfg)
        print("="*100)
        print("Default VM Config:")
        pprint.pprint(get_vehicle_manager_defaults())
        print("="*100)

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #

    def get_vm(self):
        """Build the VehicleManager with the MARL collision sensor installed
        at construction (one spawn per sensor — no stock-sensor spawn/destroy
        round trip) and without the redundant safety-manager IMU."""
        try:
            from opencda_marl.api.components import resolve_component

            agent = resolve_component("agent", str(self.agent_type),
                                      config=self.config)(
                self.vehicle, self.carla_map, self._agent_config())
            vm = VehicleManager(
                vehicle=self.vehicle,
                config_yaml=self.vm_cfg,
                application=['single'],
                carla_map=self.carla_map,
                cav_world=self.cav_world,
                agent=agent,
                collision_sensor_cls=MARLCollisionSensor,
            )

            return vm
        except Exception as e:
            logger.error(
                f"Error getting agent for vehicle {self.actor_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _agent_config(self):
        """The agents block as the factory receives it.

        Bit-identical to the configured block unless the vehicle carries
        spawn metadata, in which case a shallow copy adds it under
        ``spawn_metadata`` — the seam a per-vehicle driver model reads.
        """
        metadata = getattr(self, "spawn_metadata", None)
        if not metadata:
            return self.config
        try:
            view = dict(self.config)
        except Exception:  # noqa: BLE001 - an exotic config type keeps its shape
            return self.config
        view["spawn_metadata"] = dict(metadata)
        return view

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_carla_id(self) -> int:
        """Get the CARLA actor ID of this vehicle.

        Returns:
            int: CARLA actor ID
        """
        return self.actor_id

    # --------------------------------------------------------------------- #
    # Cleanup
    # --------------------------------------------------------------------- #

    def destroy(self):
        """
        Destroy the vehicle manager and all its sensors.

        VehicleManager.destroy() is hardened (per-component isolation), so a
        raise here is a genuine teardown failure. It propagates to the
        caller — the ActorRegistry verifies against the server and reclaims
        whatever this chain missed. Swallowing errors here previously made
        every downstream vehicle.destroy() fallback dead code.
        """
        vm = self.vm
        if vm is None:
            return
        try:
            self.cav_world.remove_vehicle_manager(vm)
        except Exception as e:
            logger.warning(f"Failed to detach vehicle manager for "
                           f"{self.actor_id}: {e}")
        rig = SensorRig.from_vehicle_manager(vm)
        try:
            vm.destroy()
        finally:
            # Same net as MARLAgentManager.cleanup: a sensor the vendored
            # teardown left alive is destroyed before the handle is dropped.
            rig.stop_and_destroy()
            self.vm = None
