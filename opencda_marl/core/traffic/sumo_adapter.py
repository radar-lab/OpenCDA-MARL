'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-11-17
FilePath     : /OpenCDA-MARL/opencda_marl/core/traffic/sumo_adapter.py
Description  : Adapter to make SUMO network compatible with MARLPlanner.
               Converts SUMO network topology to CARLA-like waypoint structure.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import math
import traci
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from omegaconf import DictConfig


# Mock CARLA classes for compatibility
class Location:
    """Mock of carla.Location with arithmetic operations."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        """Add two locations."""
        return Location(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """Subtract two locations."""
        return Location(self.x - other.x, self.y - other.y, self.z - other.z)

    def __repr__(self):
        return f"Location(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


class Rotation:
    """Mock of carla.Rotation."""

    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)

    def __repr__(self):
        return f"Rotation(pitch={self.pitch:.2f}, yaw={self.yaw:.2f}, roll={self.roll:.2f})"


class Transform:
    """Mock of carla.Transform."""

    def __init__(self, location: Location = None, rotation: Rotation = None):
        self.location = location if location else Location()
        self.rotation = rotation if rotation else Rotation()

    def __repr__(self):
        return f"Transform({self.location}, {self.rotation})"


class Vector3D:
    """Mock of carla.Vector3D."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Vector3D(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


class BoundingBox:
    """Mock of carla.BoundingBox."""

    def __init__(self, location: Location = None, extent: Vector3D = None):
        self.location = location if location else Location()
        self.extent = extent if extent else Vector3D()

    def __repr__(self):
        return f"BoundingBox({self.location}, {self.extent})"


# Mock CARLA enums for compatibility
class LaneType:
    """Mock of carla.LaneType"""
    Driving = 1
    Sidewalk = 2
    Shoulder = 3
    Biking = 4
    Parking = 5
    Any = 0


class LaneChange:
    """Mock of carla.LaneChange"""
    NONE = 0
    Right = 1
    Left = 2
    Both = 3


class SumoWaypoint:
    """
    CARLA-like waypoint wrapper for SUMO edges/lanes.
    Provides same interface as carla.Waypoint for MARLPlanner compatibility.

    IMPORTANT: All positions are stored in CARLA coordinate system for consistency.
    SUMO coordinates are converted to CARLA coordinates using netOffset (99.8, 100.0).
    """

    # Network offset for coordinate conversion (SUMO applies this during OpenDRIVE conversion)
    NET_OFFSET = (99.80, 100.00)

    def __init__(self, edge_id: str, lane_index: int, position: Tuple[float, float],
                 road_id: int, lane_id: int, is_junction: bool = False, junction_id: str = None):
        self.edge_id = edge_id
        self.lane_index = lane_index

        # Convert SUMO coordinates to CARLA coordinates
        # CARLA coords = SUMO coords - offset
        carla_x = position[0] - self.NET_OFFSET[0]
        carla_y = position[1] - self.NET_OFFSET[1]

        self.position = (carla_x, carla_y)  # Store in CARLA coords!
        self.road_id = road_id  # Use edge_id hash for compatibility
        self.lane_id = lane_id  # SUMO lane index
        self.is_junction = is_junction
        self.junction_id = junction_id  # SUMO junction ID if near junction
        self.lane_type = LaneType.Driving  # All SUMO lanes are driving lanes

        # CARLA compatibility attributes
        self.s = 0.0  # Distance along road (used for _is_same_lane_group)
        self.section_id = 0  # SUMO doesn't have sections, use 0 for all

        # Create transform using CARLA coordinates
        self.transform = Transform(
            location=Location(x=carla_x, y=carla_y, z=0.0),
            rotation=Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

    def get_left_lane(self):
        """Get left lane if exists."""
        # In SUMO, lane 0 is rightmost, increasing index goes left
        try:
            edge_lanes = traci.edge.getLaneNumber(self.edge_id)
        except:
            return None

        if self.lane_index < edge_lanes - 1:
            new_lane_id = f"{self.edge_id}_{self.lane_index + 1}"
            try:
                shape = traci.lane.getShape(new_lane_id)
            except:
                return None

            if shape and len(shape) >= 2:
                # IMPORTANT: Use same position index as current waypoint
                # If current waypoint is at end (-1), use end of adjacent lane too
                # This ensures all lanes have coordinates at the same longitudinal position
                position = self.position  # Current waypoint position in CARLA coords

                # Convert to SUMO coords to match the longitudinal position
                current_sumo_y = position[1] + self.NET_OFFSET[1]

                # Find point in adjacent lane shape closest to current Y position
                # For simplicity, use end point if current is near end, else mid point
                if abs(current_sumo_y - shape[-1][1]) < 1.0:
                    # Current waypoint is at end, use end of adjacent lane
                    lane_point = shape[-1]
                else:
                    # Current waypoint is at mid, use mid of adjacent lane
                    lane_point = shape[len(shape) // 2]

                return SumoWaypoint(
                    self.edge_id, self.lane_index + 1, lane_point,
                    self.road_id, self.lane_id + 1, self.is_junction, self.junction_id
                )
        return None

    def get_right_lane(self):
        """Get right lane if exists."""
        if self.lane_index > 0:
            new_lane_id = f"{self.edge_id}_{self.lane_index - 1}"
            try:
                shape = traci.lane.getShape(new_lane_id)
            except:
                return None

            if shape and len(shape) >= 2:
                # IMPORTANT: Use same position index as current waypoint
                position = self.position  # Current waypoint position in CARLA coords

                # Convert to SUMO coords to match the longitudinal position
                current_sumo_y = position[1] + self.NET_OFFSET[1]

                # Find point in adjacent lane shape closest to current Y position
                if abs(current_sumo_y - shape[-1][1]) < 1.0:
                    # Current waypoint is at end, use end of adjacent lane
                    lane_point = shape[-1]
                else:
                    # Current waypoint is at mid, use mid of adjacent lane
                    lane_point = shape[len(shape) // 2]

                return SumoWaypoint(
                    self.edge_id, self.lane_index - 1, lane_point,
                    self.road_id, self.lane_id - 1, self.is_junction, self.junction_id
                )
        return None

    @property
    def lane_change(self):
        """Determine if lane changes are allowed (for MARLPlanner lane ordering)."""
        edge_lanes = traci.edge.getLaneNumber(self.edge_id)

        if edge_lanes == 1:
            return LaneChange.NONE
        elif self.lane_index == 0:
            return LaneChange.Left  # Rightmost - can only change left
        elif self.lane_index == edge_lanes - 1:
            return LaneChange.Right  # Leftmost - can only change right
        else:
            return LaneChange.Both  # Middle - can change both ways

    def get_junction(self):
        """Get junction if this waypoint is at/near a junction."""
        if not self.is_junction or not self.junction_id:
            return None

        return SumoJunction(self.junction_id)

    def next(self, distance: float):
        """
        Get waypoints ahead of this waypoint along the lane.

        Args:
            distance: Distance in meters to move along the lane

        Returns:
            List of SumoWaypoint objects (CARLA compatibility - returns list)
        """
        lane_id = f"{self.edge_id}_{self.lane_index}"

        try:
            # Get lane shape (list of points)
            shape = traci.lane.getShape(lane_id)
            if not shape or len(shape) < 2:
                return []

            # Current position is middle of lane, compute new position
            # For simplicity, return waypoint at end of edge (where it connects to next edge)
            end_point = shape[-1]

            # Check if there's a next edge connected
            to_junction_id = traci.edge.getToJunction(self.edge_id)
            if not to_junction_id:
                return []

            # Get edges leaving this junction
            all_edges = traci.edge.getIDList()
            next_waypoints = []

            for edge_id in all_edges:
                if edge_id.startswith(':'):
                    continue

                from_junction = traci.edge.getFromJunction(edge_id)
                if from_junction == to_junction_id and edge_id != self.edge_id:
                    # This edge leaves our junction
                    num_lanes = traci.edge.getLaneNumber(edge_id)

                    # Create waypoint for same lane index if exists
                    lane_idx = min(self.lane_index, num_lanes - 1)
                    next_lane_id = f"{edge_id}_{lane_idx}"
                    next_shape = traci.lane.getShape(next_lane_id)

                    if next_shape and len(next_shape) >= 2:
                        # Waypoint at start of next edge
                        start_point = next_shape[0]
                        next_to_junction = traci.edge.getToJunction(edge_id)

                        wp = SumoWaypoint(
                            edge_id, lane_idx, start_point,
                            hash(edge_id) % 10000, -lane_idx - 1,
                            is_junction=(next_to_junction is not None and next_to_junction != ''),
                            junction_id=next_to_junction if next_to_junction else None
                        )
                        next_waypoints.append(wp)

            return next_waypoints

        except Exception as e:
            logger.debug(f"Failed to get next waypoint for {self.edge_id}_{self.lane_index}: {e}")
            return []

    def previous(self, distance: float):
        """
        Get waypoints behind this waypoint along the lane.

        IMPORTANT: This method now interpolates along the SAME lane shape
        to provide proper spawn point distribution across lanes.

        Args:
            distance: Distance in meters to move backwards along the lane

        Returns:
            List of SumoWaypoint objects (CARLA compatibility - returns list)
        """
        lane_id = f"{self.edge_id}_{self.lane_index}"

        try:
            # Get lane shape (list of points)
            shape = traci.lane.getShape(lane_id)
            if not shape or len(shape) < 2:
                return []

            # Calculate total lane length
            lane_length = traci.lane.getLength(lane_id)

            # Current position in CARLA coords - convert to SUMO to match shape
            current_sumo_x = self.position[0] + self.NET_OFFSET[0]
            current_sumo_y = self.position[1] + self.NET_OFFSET[1]

            # Find current position along the lane shape
            # For simplicity, if we're near the end, assume we're at the end
            # Otherwise, assume we're at the start
            current_distance_from_start = 0.0
            if len(shape) == 2:
                # Simple 2-point lane - calculate distance from start
                start_x, start_y = shape[0]
                end_x, end_y = shape[-1]
                total_dist = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                current_dist = math.sqrt((current_sumo_x - start_x)**2 + (current_sumo_y - start_y)**2)
                current_distance_from_start = current_dist
            else:
                # Multi-point lane - find closest segment
                for i in range(len(shape) - 1):
                    x1, y1 = shape[i]
                    x2, y2 = shape[i + 1]
                    # Simple check: if we're close to end point, we're at the end
                    if abs(current_sumo_x - x2) < 1.0 and abs(current_sumo_y - y2) < 1.0:
                        # Sum distances up to this segment
                        for j in range(i + 1):
                            x_a, y_a = shape[j]
                            x_b, y_b = shape[j + 1]
                            current_distance_from_start += math.sqrt((x_b - x_a)**2 + (y_b - y_a)**2)
                        break

            # Calculate target distance from start (moving backwards)
            target_distance_from_start = current_distance_from_start - distance

            # Clamp to valid range
            target_distance_from_start = max(0.0, min(lane_length, target_distance_from_start))

            # Interpolate position along lane shape
            if len(shape) == 2:
                # Simple linear interpolation for 2-point lanes
                start_x, start_y = shape[0]
                end_x, end_y = shape[-1]
                total_dist = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                if total_dist > 0:
                    ratio = target_distance_from_start / total_dist
                    new_x = start_x + ratio * (end_x - start_x)
                    new_y = start_y + ratio * (end_y - start_y)
                else:
                    new_x, new_y = start_x, start_y
            else:
                # Multi-point interpolation
                accumulated_dist = 0.0
                new_x, new_y = shape[0]  # Default to start
                for i in range(len(shape) - 1):
                    x1, y1 = shape[i]
                    x2, y2 = shape[i + 1]
                    segment_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                    if accumulated_dist + segment_length >= target_distance_from_start:
                        # Target is in this segment
                        remaining = target_distance_from_start - accumulated_dist
                        if segment_length > 0:
                            ratio = remaining / segment_length
                            new_x = x1 + ratio * (x2 - x1)
                            new_y = y1 + ratio * (y2 - y1)
                        else:
                            new_x, new_y = x1, y1
                        break

                    accumulated_dist += segment_length

            # Create new waypoint at interpolated position
            new_position_sumo = (new_x, new_y)
            wp = SumoWaypoint(
                self.edge_id, self.lane_index, new_position_sumo,
                self.road_id, self.lane_id, self.is_junction, self.junction_id
            )
            return [wp]  # Return as list for CARLA compatibility

        except Exception as e:
            logger.debug(f"Failed to interpolate previous waypoint for {self.edge_id}_{self.lane_index}: {e}")
            # Fallback: return empty to avoid breaking the planner
            return []


class SumoJunction:
    """
    CARLA-like junction wrapper for SUMO junctions.
    Provides same interface as carla.Junction for MARLPlanner compatibility.

    IMPORTANT: Junction center is stored in CARLA coordinate system for consistency.
    """

    # Network offset for coordinate conversion (same as SumoWaypoint)
    NET_OFFSET = (99.80, 100.00)

    def __init__(self, junction_id: str):
        self.id = int(junction_id) if junction_id.isdigit() else hash(junction_id) % 10000
        self.junction_id = junction_id

        # Get junction position and shape (in SUMO coordinates)
        pos = traci.junction.getPosition(junction_id)
        shape = traci.junction.getShape(junction_id)

        # Calculate bounding box in SUMO coordinates first
        if shape:
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            center_x_sumo = sum(xs) / len(xs)
            center_y_sumo = sum(ys) / len(ys)
            extent_x = (max(xs) - min(xs)) / 2
            extent_y = (max(ys) - min(ys)) / 2
        else:
            center_x_sumo, center_y_sumo = pos
            extent_x, extent_y = 10.0, 10.0  # Default size

        # Convert junction center to CARLA coordinates
        center_x_carla = center_x_sumo - self.NET_OFFSET[0]
        center_y_carla = center_y_sumo - self.NET_OFFSET[1]

        # Create CARLA-like bounding box using CARLA coordinates
        self.bounding_box = BoundingBox(
            location=Location(x=center_x_carla, y=center_y_carla, z=0.0),
            extent=Vector3D(x=extent_x, y=extent_y, z=1.0)
        )

    def get_waypoints(self, lane_type=None):
        """
        Get all entry/exit waypoint pairs for this junction.
        Returns list of tuples: (entry_waypoint, exit_waypoint)

        Args:
            lane_type: Filter by lane type (only Driving is supported for SUMO)
        """
        # SUMO only has driving lanes, so lane_type filter has no effect
        connections = []

        # Get all incoming edges to this junction
        incoming_edges = []
        all_edges = traci.edge.getIDList()

        for edge_id in all_edges:
            # Skip internal edges
            if edge_id.startswith(':'):
                continue

            # Check if this edge leads to our junction
            to_node = traci.edge.getToJunction(edge_id)
            if to_node == self.junction_id:
                incoming_edges.append(edge_id)

        # Get all outgoing edges from this junction
        outgoing_edges = []
        for edge_id in all_edges:
            if edge_id.startswith(':'):
                continue

            from_node = traci.edge.getFromJunction(edge_id)
            if from_node == self.junction_id:
                outgoing_edges.append(edge_id)

        # Create waypoint pairs for all connections
        for entry_edge in incoming_edges:
            num_lanes = traci.edge.getLaneNumber(entry_edge)
            for lane_idx in range(num_lanes):
                lane_id = f"{entry_edge}_{lane_idx}"
                shape = traci.lane.getShape(lane_id)
                if not shape:
                    continue

                # Entry waypoint at end of incoming edge
                entry_pos = shape[-1]
                entry_wp = SumoWaypoint(
                    entry_edge, lane_idx, entry_pos,
                    hash(entry_edge) % 10000, -lane_idx - 1, False
                )

                # Find connected outgoing edges
                for exit_edge in outgoing_edges:
                    exit_num_lanes = traci.edge.getLaneNumber(exit_edge)
                    for exit_lane_idx in range(exit_num_lanes):
                        exit_lane_id = f"{exit_edge}_{exit_lane_idx}"
                        exit_shape = traci.lane.getShape(exit_lane_id)
                        if not exit_shape:
                            continue

                        # Exit waypoint at start of outgoing edge
                        exit_pos = exit_shape[0]
                        exit_wp = SumoWaypoint(
                            exit_edge, exit_lane_idx, exit_pos,
                            hash(exit_edge) % 10000, -exit_lane_idx - 1, False
                        )

                        connections.append((entry_wp, exit_wp))

        return connections


class ActorBlueprint:
    """Mock of carla.ActorBlueprint."""

    def __init__(self, blueprint_id: str):
        self.id = blueprint_id

    def __repr__(self):
        return f"ActorBlueprint(id='{self.id}')"


class BlueprintLibrary:
    """Mock of carla.BlueprintLibrary."""

    def __init__(self):
        # Create some default vehicle blueprints for SUMO
        self.blueprints = [
            ActorBlueprint('vehicle.audi.a2'),
            ActorBlueprint('vehicle.tesla.model3'),
            ActorBlueprint('vehicle.volkswagen.t2'),
            ActorBlueprint('vehicle.bmw.grandtourer'),
            ActorBlueprint('vehicle.toyota.prius'),
        ]

    def filter(self, wildcard: str):
        """Filter blueprints by wildcard pattern."""
        if wildcard == 'vehicle.*':
            return self.blueprints
        # Simple pattern matching
        filtered = [bp for bp in self.blueprints if wildcard.replace('*', '') in bp.id]
        return filtered

    def find(self, blueprint_id: str):
        """Find blueprint by exact ID."""
        for bp in self.blueprints:
            if bp.id == blueprint_id:
                return bp
        # If not found, create a new one dynamically
        new_bp = ActorBlueprint(blueprint_id)
        self.blueprints.append(new_bp)
        return new_bp

    def __iter__(self):
        return iter(self.blueprints)

    def __len__(self):
        return len(self.blueprints)


class SumoWorld:
    """
    CARLA-like world wrapper for SUMO simulation.
    Provides same interface as carla.World for MARLPlanner compatibility.
    """

    def __init__(self):
        self._map = SumoMap()
        self._blueprint_library = BlueprintLibrary()

        # Mock debug interface (MARLPlanner uses this for visualization)
        self.debug = type('Debug', (), {
            'draw_point': lambda *args, **kwargs: None,
            'draw_box': lambda *args, **kwargs: None,
            'draw_line': lambda *args, **kwargs: None,
            'draw_string': lambda *args, **kwargs: None,
        })()

    def get_map(self):
        """Get the SUMO map (CARLA World interface)."""
        return self._map

    def get_blueprint_library(self):
        """Get the blueprint library (CARLA World interface)."""
        return self._blueprint_library


class SumoMap:
    """
    CARLA-like map wrapper for SUMO network.
    Provides same interface as carla.Map for MARLPlanner compatibility.
    """

    def generate_waypoints(self, distance: float = 2.0) -> List[SumoWaypoint]:
        """
        Generate waypoints along all edges in the network.

        Args:
            distance: Spacing between waypoints (meters)

        Returns:
            List of SumoWaypoint objects
        """
        waypoints = []

        # Get all edges in network
        all_edges = traci.edge.getIDList()

        for edge_id in all_edges:
            # Skip internal junction edges
            if edge_id.startswith(':'):
                continue

            # Get number of lanes
            num_lanes = traci.edge.getLaneNumber(edge_id)

            # Create waypoints for each lane
            for lane_idx in range(num_lanes):
                lane_id = f"{edge_id}_{lane_idx}"
                shape = traci.lane.getShape(lane_id)

                if not shape or len(shape) < 2:
                    continue

                # Sample point at middle of lane
                mid_idx = len(shape) // 2
                mid_point = shape[mid_idx]

                # Check if this edge leads to a junction
                to_junction_id = traci.edge.getToJunction(edge_id)
                is_junction = (to_junction_id is not None and to_junction_id != '')

                wp = SumoWaypoint(
                    edge_id, lane_idx, mid_point,
                    road_id=hash(edge_id) % 10000,
                    lane_id=-lane_idx - 1,  # Negative like CARLA
                    is_junction=is_junction,
                    junction_id=to_junction_id if is_junction else None
                )
                waypoints.append(wp)

        logger.debug(f"Generated {len(waypoints)} waypoints from SUMO network")
        return waypoints


class SumoMARLPlanner:
    """
    SUMO-compatible version of MARLPlanner that uses SUMO network topology.
    Provides same interface as MARLPlanner but works with SUMO TraCI.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize SUMO MARL planner.

        Args:
            config: Configuration dictionary with planner parameters
        """
        self.config = config
        self.world = SumoWorld()
        self.map = self.world.get_map()

        # Initialize junctions
        self._junctions = self._get_junctions()

        logger.success(f"SumoMARLPlanner initialized with {len(self._junctions)} junctions")

    def _get_junctions(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze SUMO network and extract all junctions with routes.
        Uses same logic as MARLPlanner but with SUMO data.
        """
        from opencda_marl.core.traffic.planner import MARLPlanner

        # Create temporary MARLPlanner instance with SUMO world adapter
        # This reuses all the route generation logic
        temp_planner = MARLPlanner(self.world, self.config)

        # Return the junctions (MARLPlanner does all the heavy lifting)
        return temp_planner._junctions

    def get_junctions(self) -> Dict[int, Dict[str, Any]]:
        """Get all junctions."""
        return self._junctions

    def get_routes(self, junction_id: int) -> Dict[int, Dict]:
        """Get all routes for a junction."""
        if junction_id not in self._junctions:
            return {}
        return self._junctions[junction_id]["routes"]

    def get_route(self, junction_id: int, route_id: int) -> Optional[Dict]:
        """Get specific route by ID."""
        routes = self.get_routes(junction_id)
        return routes.get(route_id)

    def cleanup(self):
        """Cleanup resources."""
        self._junctions.clear()
