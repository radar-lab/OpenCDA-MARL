'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 09:39:24
FilePath     : /OpenCDA-MARL/opencda_marl/core/traffic/traffic_manager.py
Description  : 
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import carla
from loguru import logger
from typing import List, Dict, Any
from pathlib import Path

from .events import SpawnEvent
from .flows import TrafficFlow
from .planner import MARLPlanner
from .serializer import EventSerializer


class MARLTrafficManager:
    flows: List[TrafficFlow] = []

    def __init__(self, world: carla.World, config: dict, state: Dict[str, Any], fix_dlt: float = 0.05):
        self.world = world
        self.config = config
        self.state = state

        planner_cfg = self.config.get("planner", {})
        self.planner = MARLPlanner(self.world, planner_cfg)
        self.active_junctions = self.config.get("active_junctions", [])
        self.vbp = self._get_vehicle_blueprints()
        # ---- Flow configuration ----
        self.base_speed = self.config.get('base_speed', 30.0)
        self.fix_dlt = fix_dlt
        self.flows = self._parse_flows()

        # ---- Events ----
        self.mode = config.get('mode', 'live')  # 'record', 'replay', 'live'
        self.replay_file = config.get('replay_file', 'traffic_events.h5')
        self._events: List[SpawnEvent] = []
        self._event_traces: List[List[SpawnEvent]] = []
        self._initialize_events()

        #print(f"DEBUG: events: {len(self._events)}")
        #for i, event in enumerate(self._events):
        #    print(
        #        f"DEBUG: event {i}: {event.spawn_step}, {event.target_speed}, {event.flow_name}, {event.lane_id}, {event.route_id}")
    # --------------------------------------------------------------------- #
    # Public control API
    # --------------------------------------------------------------------- #
    def update(self, current_step: int) -> List[SpawnEvent]:
        spawn_events = []
        # retrive the corresponding events that should spawn at the current step
        count = 0
        for event in self._events:
            if event.spawn_step == current_step:
                spawn_events.append(event)
                self._event_traces.append(event)
                count += 1
            # since we sort the events by spawn_step,
            # we can break the loop early
            if event.spawn_step > current_step:
                break

        #delete the events that have been processed
        self._events = self._events[count:]
        
        return spawn_events
    
    @property
    def events(self) -> List[SpawnEvent]:
        return self._events
    
    @property
    def total_events(self) -> int:
        return len(self._events)
    # --------------------------------------------------------------------- #
    # Private API
    # --------------------------------------------------------------------- #

    def _initialize_events(self):
        """Initialize events based on mode."""
        if self.mode == 'replay':
            self._load_events()
        elif self.mode == 'record':
            self._generate_and_save_events()
        else:  # 'live'
            self._generate_events()

    def _load_events(self):
        """Load events from replay file (supports both JSON and HDF5)."""

        replay_path = Path(self.replay_file)
        if not replay_path.is_absolute():
            replay_path = Path.cwd() / replay_path

        # Validate file first
        validation = EventSerializer.validate_event_file(str(replay_path))
        if not validation['valid']:
            raise FileNotFoundError(
                f"Cannot load replay file {replay_path}: {validation['error']}")

        logger.info(
            f"Loading {validation['format'].upper()} replay file: {replay_path}")
        logger.info(f"File info: {validation['total_events']} events, "
                    f"{validation['file_size_mb']:.1f} MB, version {validation['version']}")

        # Load events using auto-detection
        self._events = EventSerializer.load_events(str(replay_path), self.world)

        if self._events is None:
            raise RuntimeError(f"Failed to load events from {replay_path}")

        logger.success(f"Successfully loaded {len(self._events)} events")

    def _generate_and_save_events(self):
        self._generate_events()

        if self._events:
            save_path = Path(self.replay_file).resolve()
            logger.info(f"Saving {len(self._events)} events to {save_path}")

            # Prepare metadata
            metadata = {
                'flows': [flow.name for flow in self.flows],
                'vbps': [vbp.id for vbp in self.vbp],
                'total_steps': max((flow.end_step for flow in self.flows), default=0),
                'total_events': len(self._events)
            }

            # if save path ends with .h5, save as hdf5, otherwise save as json
            if save_path.suffix == '.h5':
                success = EventSerializer.save_events_to_hdf5(
                    self._events, str(save_path), self.config, metadata)
            else:
                success = EventSerializer.export_events_to_json(
                    self._events, str(save_path))

            if not success:
                logger.error(f"Failed to save events to {save_path}")
            else:
                logger.success(f"Events saved successfully to {save_path}")

    def _generate_events(self):
        for j_id in self.active_junctions:
            for flow in self.flows:
                events = flow._generate_events(self.vbp, self.planner, 
                                               j_id, fix_dlt=self.fix_dlt,
                                               base_speed=self.base_speed)
                self._events.extend(events)
        # sort events by spawn_step
        self._events.sort(key=lambda x: x.spawn_step)

    def _parse_flows(self) -> List[Dict[str, Any]]:
        flow_dicts = self.config.get('flows', [])

        if not flow_dicts:
            logger.warning("No traffic flows configured")
            return []

        # Parse into TrafficFlowConfig objects
        flows = []
        for flow_dict in flow_dicts:
            try:
                flow = TrafficFlow(
                    name=flow_dict['name'],
                    rate_vph=float(flow_dict['rate_vph']),
                    lanes=flow_dict['lanes'],
                    start_step=int(flow_dict['start_step']),
                    end_step=int(flow_dict['end_step']),
                    direction=flow_dict['direction'],
                    speed_variation=float(
                        flow_dict.get('speed_variation', 0.0)),
                    middle_peak=flow_dict.get('middle_peak')
                )
                flows.append(flow)
            except Exception as e:
                raise ValueError(
                    f"Error parsing flow '{flow_dict.get('name', 'unknown')}': {e}")

        return flows

    def _get_vehicle_blueprints(self) -> List[carla.ActorBlueprint]:
        included_types = self.config.get('included_vehicle_types', [])
        excluded_types = self.config.get('excluded_vehicle_types', [])

        all_bps = self.world.get_blueprint_library().filter('vehicle.*')
        vehicle_bps = []

        if included_types:
            # Use ONLY included types (priority)
            for bp in all_bps:
                if any(included in bp.id.lower() for included in included_types):
                    vehicle_bps.append(bp)
        elif excluded_types:
            # Use all EXCEPT excluded types
            for bp in all_bps:
                if not any(excluded in bp.id.lower() for excluded in excluded_types):
                    vehicle_bps.append(bp)
        else:
            # Use all vehicles
            vehicle_bps = list(all_bps)

        logger.success(
            f"Selected {len(vehicle_bps)} vehicle blueprints from {len(all_bps)} total available")
        return vehicle_bps

    # --------------------------------------------------------------------- #
    # Reset and Cleanup
    # --------------------------------------------------------------------- #
    def reset(self):
        """Reset traffic manager for new episode."""
        logger.info("Resetting MARLTrafficManager for new episode")
        
        # Clear existing events and traces
        self._events.clear()
        self._event_traces.clear()
        
        # Re-initialize events based on mode
        self._initialize_events()
        
        logger.success(f"MARLTrafficManager reset completed with {len(self._events)} new events")
    
    def cleanup(self):
        if hasattr(self.planner, 'cleanup'):
            self.planner.cleanup()
        self.world = None
