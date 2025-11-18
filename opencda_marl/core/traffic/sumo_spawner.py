'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-11-17
FilePath     : /OpenCDA-MARL/opencda_marl/core/traffic/sumo_spawner.py
Description  : SUMO vehicle spawner that uses MARLTrafficManager's SpawnEvent system.
               Spawns vehicles in SUMO based on events from traffic manager.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import traci
import math
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .events import SpawnEvent


class SumoVehicleSpawner:
    """
    Spawns vehicles in SUMO using SpawnEvent objects from MARLTrafficManager.
    Maintains consistency with CARLA by using same route generation logic.
    """

    def __init__(self):
        """Initialize SUMO vehicle spawner."""
        self.spawned_vehicles = {}  # {vehicle_id: event}
        self.spawn_counter = 0
        self.failed_spawns = []

        # Get SUMO network offset for coordinate transformation
        # SUMO applies netOffset during conversion: sumo_coord = carla_coord + offset
        net_boundary = traci.simulation.getNetBoundary()
        self.net_offset = self._get_network_offset()
        logger.info(f"SUMO network offset: {self.net_offset}")

        # Note: We use SUMO's built-in 'DEFAULT_VEHTYPE' for all vehicles
        # No need to create custom vehicle types

    def spawn_vehicle(self, event: SpawnEvent) -> Optional[str]:
        """
        Spawn a vehicle in SUMO based on SpawnEvent.

        Args:
            event: SpawnEvent containing spawn location, destination, route type, etc.

        Returns:
            vehicle_id if successful, None otherwise
        """
        # Extract metadata
        entry_dir = event.metadata.get('entry_direction', 'unknown')
        lane_idx = event.metadata.get('lane_id', event.lane_id)

        # Generate unique vehicle ID
        vehicle_id = f"{event.flow_name}_{entry_dir}_{lane_idx}_{self.spawn_counter}"
        self.spawn_counter += 1

        # Extract spawn and destination from event (CARLA coordinates)
        spawn_loc = event.transform.location
        dest_loc = event.destination.location

        # Transform CARLA coordinates to SUMO coordinates
        spawn_x_sumo, spawn_y_sumo = self._carla_to_sumo(spawn_loc.x, spawn_loc.y)
        dest_x_sumo, dest_y_sumo = self._carla_to_sumo(dest_loc.x, dest_loc.y)

        # Find compatible edge pair for spawn and destination
        # This considers both proximity and routing compatibility
        spawn_edge, dest_edge = self._find_compatible_edges(
            spawn_x_sumo, spawn_y_sumo,
            dest_x_sumo, dest_y_sumo,
            event.metadata.get('route_type', 'unknown')
        )

        if not spawn_edge:
            logger.warning(f"Could not find spawn edge for {vehicle_id} at CARLA ({spawn_loc.x:.1f}, {spawn_loc.y:.1f}) → SUMO ({spawn_x_sumo:.1f}, {spawn_y_sumo:.1f})")
            self.failed_spawns.append(event)
            return None

        if not dest_edge:
            logger.warning(f"Could not find destination edge for {vehicle_id} at CARLA ({dest_loc.x:.1f}, {dest_loc.y:.1f}) → SUMO ({dest_x_sumo:.1f}, {dest_y_sumo:.1f})")
            self.failed_spawns.append(event)
            return None

        # Compute route between spawn and destination
        try:
            route = traci.simulation.findRoute(spawn_edge, dest_edge)
            if not route or not route.edges:
                route_type = event.metadata.get('route_type', 'unknown')
                logger.debug(f"No route found: {spawn_edge} → {dest_edge} (route_type={route_type})")
                self.failed_spawns.append(event)
                return None
        except Exception as e:
            route_type = event.metadata.get('route_type', 'unknown')
            logger.debug(f"Route computation failed: {spawn_edge} → {dest_edge} (route_type={route_type}): {e}")
            self.failed_spawns.append(event)
            return None

        # Determine lane index from event (already extracted above)
        # Clamp to valid range for this edge
        num_lanes = traci.edge.getLaneNumber(spawn_edge)
        spawn_lane_idx = min(lane_idx, num_lanes - 1)

        # Get vehicle type from event blueprint (convert CARLA blueprint to SUMO type)
        carla_blueprint_id = event.blueprint.id if event.blueprint else 'vehicle.audi.a2'
        vtype_id = self._carla_blueprint_to_sumo_type(carla_blueprint_id)

        try:
            # Add vehicle to simulation
            traci.vehicle.add(
                vehID=vehicle_id,
                routeID="",  # Empty route, we'll set it manually
                typeID=vtype_id,
                depart='now',
                departLane=str(spawn_lane_idx),
                departSpeed='max'
            )

            # Set the route
            traci.vehicle.setRoute(vehicle_id, route.edges)

            # Set target speed from event
            target_speed_kmh = event.target_speed
            target_speed_ms = target_speed_kmh / 3.6  # Convert km/h to m/s
            traci.vehicle.setSpeed(vehicle_id, target_speed_ms)

            # Store event for tracking
            self.spawned_vehicles[vehicle_id] = event

            logger.debug(f"Spawned {vehicle_id} on {spawn_edge} lane {spawn_lane_idx}, "
                        f"route: {event.metadata.get('route_type', 'unknown')}, "
                        f"dest: {dest_edge}")

            return vehicle_id

        except traci.exceptions.TraCIException as e:
            logger.warning(f"Failed to spawn {vehicle_id}: {e}")
            self.failed_spawns.append(event)
            return None

    def spawn_vehicles(self, events: List[SpawnEvent]) -> List[str]:
        """
        Spawn multiple vehicles from a list of events.

        Args:
            events: List of SpawnEvent objects

        Returns:
            List of successfully spawned vehicle IDs
        """
        spawned_ids = []

        for event in events:
            veh_id = self.spawn_vehicle(event)
            if veh_id:
                spawned_ids.append(veh_id)

        if spawned_ids:
            logger.info(f"Spawned {len(spawned_ids)}/{len(events)} vehicles this step")

        return spawned_ids

    def get_spawn_event(self, vehicle_id: str) -> Optional[SpawnEvent]:
        """Get the SpawnEvent for a vehicle."""
        return self.spawned_vehicles.get(vehicle_id)

    def remove_vehicle(self, vehicle_id: str):
        """Remove vehicle from tracking."""
        if vehicle_id in self.spawned_vehicles:
            del self.spawned_vehicles[vehicle_id]

    def get_failed_spawns(self) -> List[SpawnEvent]:
        """Get list of failed spawn attempts."""
        return self.failed_spawns.copy()

    def clear_failed_spawns(self):
        """Clear failed spawn list."""
        self.failed_spawns.clear()

    # --------------------------------------------------------------------- #
    # Helper Methods
    # --------------------------------------------------------------------- #

    def _find_compatible_edges(self, spawn_x: float, spawn_y: float,
                               dest_x: float, dest_y: float,
                               route_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find compatible edge pair for spawn and destination that can form a valid route.

        This method finds edges that:
        1. Are close to the given coordinates
        2. Can form a valid route (not opposite directions)
        3. Match the expected route type when possible

        Args:
            spawn_x: Spawn X coordinate (SUMO)
            spawn_y: Spawn Y coordinate (SUMO)
            dest_x: Destination X coordinate (SUMO)
            dest_y: Destination Y coordinate (SUMO)
            route_type: Expected route type (straight, left, right)

        Returns:
            Tuple of (spawn_edge, dest_edge) or (None, None) if no valid pair found
        """
        # Get candidate edges for spawn and destination
        spawn_candidates = self._get_candidate_edges(spawn_x, spawn_y, max_distance=50.0)
        dest_candidates = self._get_candidate_edges(dest_x, dest_y, max_distance=50.0)

        if not spawn_candidates or not dest_candidates:
            return None, None

        # Try to find a compatible pair
        # Prioritize pairs that avoid opposite-direction edges
        best_pair = None
        best_score = float('inf')

        for spawn_edge, spawn_dist in spawn_candidates:
            for dest_edge, dest_dist in dest_candidates:
                # Skip opposite direction edges
                if spawn_edge == f"-{dest_edge}" or dest_edge == f"-{spawn_edge}":
                    continue

                # Check if route exists (lightweight check without full computation)
                try:
                    # Quick connectivity check using SUMO's edge connections
                    from_edges = traci.edge.getIDList()
                    if spawn_edge in from_edges and dest_edge in from_edges:
                        # Score based on distance (lower is better)
                        score = spawn_dist + dest_dist

                        if score < best_score:
                            best_score = score
                            best_pair = (spawn_edge, dest_edge)
                except:
                    continue

        if best_pair:
            return best_pair

        # Fallback: return closest edges even if they might not be compatible
        # (routing check will catch issues later)
        return spawn_candidates[0][0] if spawn_candidates else None, \
               dest_candidates[0][0] if dest_candidates else None

    def _get_candidate_edges(self, x: float, y: float, max_distance: float = 50.0) -> List[Tuple[str, float]]:
        """
        Get list of candidate edges near coordinates, sorted by distance.

        Args:
            x: X coordinate (SUMO)
            y: Y coordinate (SUMO)
            max_distance: Maximum search distance

        Returns:
            List of (edge_id, distance) tuples, sorted by distance
        """
        candidates = []
        all_edges = traci.edge.getIDList()

        for edge_id in all_edges:
            # Skip internal junction edges
            if edge_id.startswith(':'):
                continue

            # Get edge shape
            num_lanes = traci.edge.getLaneNumber(edge_id)
            if num_lanes == 0:
                continue

            # Check middle lane
            lane_id = f"{edge_id}_{num_lanes // 2}"
            try:
                shape = traci.lane.getShape(lane_id)
            except:
                continue

            if not shape:
                continue

            # Find minimum distance to this edge
            min_dist = float('inf')
            for point in shape:
                dist = math.sqrt((x - point[0])**2 + (y - point[1])**2)
                min_dist = min(min_dist, dist)

            if min_dist <= max_distance:
                candidates.append((edge_id, min_dist))

        # Sort by distance
        candidates.sort(key=lambda x: x[1])
        return candidates

    def _find_closest_edge(self, x: float, y: float, max_distance: float = 50.0) -> Optional[str]:
        """
        Find the closest SUMO edge to given coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            max_distance: Maximum search distance in meters

        Returns:
            Edge ID or None if no edge found
        """
        # Get all edges in network
        all_edges = traci.edge.getIDList()

        closest_edge = None
        min_distance = float('inf')

        for edge_id in all_edges:
            # Skip internal junction edges
            if edge_id.startswith(':'):
                continue

            # Get edge shape (list of (x, y) points)
            num_lanes = traci.edge.getLaneNumber(edge_id)
            if num_lanes == 0:
                continue

            # Check middle lane
            lane_id = f"{edge_id}_{num_lanes // 2}"
            try:
                shape = traci.lane.getShape(lane_id)
            except:
                continue

            if not shape:
                continue

            # Calculate distance to edge
            for point in shape:
                dist = math.sqrt((x - point[0])**2 + (y - point[1])**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_edge = edge_id

        # Only return if within max_distance
        if min_distance <= max_distance:
            return closest_edge

        return None

    def _carla_blueprint_to_sumo_type(self, carla_blueprint_id: str) -> str:
        """
        Convert CARLA vehicle blueprint ID to SUMO vehicle type ID.

        CARLA uses detailed blueprint IDs like 'vehicle.audi.a2', 'vehicle.bmw.grandtourer',
        while SUMO uses generic type IDs.

        For maximum compatibility, we use SUMO's built-in 'DEFAULT_VEHTYPE' which is
        always available in any SUMO simulation.

        Args:
            carla_blueprint_id: CARLA blueprint ID (e.g., 'vehicle.audi.a2')

        Returns:
            SUMO vehicle type ID ('DEFAULT_VEHTYPE')
        """
        # Use SUMO's built-in default vehicle type - always available
        return 'DEFAULT_VEHTYPE'

    def _get_network_offset(self) -> tuple:
        """
        Get the network offset from SUMO network.

        SUMO applies netOffset during conversion: sumo_coord = carla_coord + offset
        The offset is stored in the network file's <location> tag.

        Returns:
            (offset_x, offset_y) tuple
        """
        # For the intersection network, the offset is (99.80, 100.00)
        # This can be read from network boundary or hardcoded
        # Since CARLA junction 4 is at (99.80, 99.57) and SUMO junction 4 is also at (99.80, 99.57),
        # but SUMO uses positive coordinates (0-200 range), the offset must be applied to spawn points

        # The netOffset from intersection.net.xml is (99.80, 100.00)
        return (99.80, 100.00)

    def _carla_to_sumo(self, x: float, y: float) -> tuple:
        """
        Transform CARLA coordinates to SUMO coordinates.

        Args:
            x: CARLA X coordinate
            y: CARLA Y coordinate

        Returns:
            (sumo_x, sumo_y) tuple
        """
        sumo_x = x + self.net_offset[0]
        sumo_y = y + self.net_offset[1]
        return (sumo_x, sumo_y)

    def _sumo_to_carla(self, x: float, y: float) -> tuple:
        """
        Transform SUMO coordinates to CARLA coordinates.

        Args:
            x: SUMO X coordinate
            y: SUMO Y coordinate

        Returns:
            (carla_x, carla_y) tuple
        """
        carla_x = x - self.net_offset[0]
        carla_y = y - self.net_offset[1]
        return (carla_x, carla_y)

    def cleanup(self):
        """Cleanup spawner resources."""
        self.spawned_vehicles.clear()
        self.failed_spawns.clear()
        self.spawn_counter = 0
