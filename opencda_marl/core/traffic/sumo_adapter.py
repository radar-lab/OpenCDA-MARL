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
    """

    def __init__(self, edge_id: str, lane_index: int, position: Tuple[float, float],
                 road_id: int, lane_id: int, is_junction: bool = False, junction_id: str = None):
        self.edge_id = edge_id
        self.lane_index = lane_index
        self.position = position  # (x, y)
        self.road_id = road_id  # Use edge_id hash for compatibility
        self.lane_id = lane_id  # SUMO lane index
        self.is_junction = is_junction
        self.junction_id = junction_id  # SUMO junction ID if near junction
        self.lane_type = LaneType.Driving  # All SUMO lanes are driving lanes

        # CARLA compatibility attributes
        self.s = 0.0  # Distance along road (used for _is_same_lane_group)
        self.section_id = 0  # SUMO doesn't have sections, use 0 for all

        # Create transform using proper mock classes
        self.transform = Transform(
            location=Location(x=position[0], y=position[1], z=0.0),
            rotation=Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

    def get_left_lane(self):
        """Get left lane if exists."""
        # In SUMO, lane 0 is rightmost, increasing index goes left
        edge_lanes = traci.edge.getLaneNumber(self.edge_id)
        if self.lane_index < edge_lanes - 1:
            new_lane_id = f"{self.edge_id}_{self.lane_index + 1}"
            shape = traci.lane.getShape(new_lane_id)
            if shape:
                mid_point = shape[len(shape) // 2]
                return SumoWaypoint(
                    self.edge_id, self.lane_index + 1, mid_point,
                    self.road_id, self.lane_id + 1, self.is_junction, self.junction_id
                )
        return None

    def get_right_lane(self):
        """Get right lane if exists."""
        if self.lane_index > 0:
            new_lane_id = f"{self.edge_id}_{self.lane_index - 1}"
            shape = traci.lane.getShape(new_lane_id)
            if shape:
                mid_point = shape[len(shape) // 2]
                return SumoWaypoint(
                    self.edge_id, self.lane_index - 1, mid_point,
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

            # Current position is middle of lane, compute new position
            # For simplicity, return waypoint at start of edge (where previous edge connects)
            start_point = shape[0]

            # Check if there's a previous edge connected
            from_junction_id = traci.edge.getFromJunction(self.edge_id)
            if not from_junction_id:
                return []

            # Get edges entering this junction
            all_edges = traci.edge.getIDList()
            prev_waypoints = []

            for edge_id in all_edges:
                if edge_id.startswith(':'):
                    continue

                to_junction = traci.edge.getToJunction(edge_id)
                if to_junction == from_junction_id and edge_id != self.edge_id:
                    # This edge enters our junction
                    num_lanes = traci.edge.getLaneNumber(edge_id)

                    # Create waypoint for same lane index if exists
                    lane_idx = min(self.lane_index, num_lanes - 1)
                    prev_lane_id = f"{edge_id}_{lane_idx}"
                    prev_shape = traci.lane.getShape(prev_lane_id)

                    if prev_shape and len(prev_shape) >= 2:
                        # Waypoint at end of previous edge
                        end_point = prev_shape[-1]
                        prev_from_junction = traci.edge.getFromJunction(edge_id)

                        wp = SumoWaypoint(
                            edge_id, lane_idx, end_point,
                            hash(edge_id) % 10000, -lane_idx - 1,
                            is_junction=(to_junction is not None and to_junction != ''),
                            junction_id=to_junction if to_junction else None
                        )
                        prev_waypoints.append(wp)

            return prev_waypoints

        except Exception as e:
            logger.debug(f"Failed to get previous waypoint for {self.edge_id}_{self.lane_index}: {e}")
            return []


class SumoJunction:
    """
    CARLA-like junction wrapper for SUMO junctions.
    Provides same interface as carla.Junction for MARLPlanner compatibility.
    """

    def __init__(self, junction_id: str):
        self.id = int(junction_id) if junction_id.isdigit() else hash(junction_id) % 10000
        self.junction_id = junction_id

        # Get junction position and shape
        pos = traci.junction.getPosition(junction_id)
        shape = traci.junction.getShape(junction_id)

        # Calculate bounding box
        if shape:
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            extent_x = (max(xs) - min(xs)) / 2
            extent_y = (max(ys) - min(ys)) / 2
        else:
            center_x, center_y = pos
            extent_x, extent_y = 10.0, 10.0  # Default size

        # Create CARLA-like bounding box using proper mock classes
        self.bounding_box = BoundingBox(
            location=Location(x=center_x, y=center_y, z=0.0),
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
