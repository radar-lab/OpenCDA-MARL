'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-11-17
FilePath     : /OpenCDA-MARL/opencda_marl/traffic/sumo_adapter.py
Description  : Adapter to make SUMO network compatible with MARLPlanner.
               Converts SUMO network topology to CARLA-like waypoint structure.
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import math
import zlib
from opencda_marl._kernel.traffic._sumo_lib import traci  # libsumo by default; OPENCDA_SUMO_LIB=traci to fall back
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from omegaconf import DictConfig

try:
    import carla as _carla
except Exception:  # pragma: no cover - fallback for CARLA-free tooling
    _carla = None


def _stable_id(name: str) -> int:
    """CARLA-style integer id for a SUMO edge or junction name.

    ``hash()`` of a string differs per process (PYTHONHASHSEED), and the planner
    breaks ties by ``road_id``, so a frozen cell gave different results run to run.
    """
    return zlib.crc32(name.encode())


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
    Driving = _carla.LaneType.Driving if _carla is not None else 1
    Sidewalk = _carla.LaneType.Sidewalk if _carla is not None else 2
    Shoulder = _carla.LaneType.Shoulder if _carla is not None else 3
    Biking = _carla.LaneType.Biking if _carla is not None else 4
    Parking = _carla.LaneType.Parking if _carla is not None else 5
    Any = _carla.LaneType.Any if _carla is not None else 0


class Color:
    """Mock of carla.Color (debug drawing only; inert on SUMO)."""

    def __init__(self, r=0, g=0, b=0, a=255):
        self.r, self.g, self.b, self.a = r, g, b, a


class LaneChange:
    """Mock of carla.LaneChange"""
    NONE = _carla.LaneChange.NONE if _carla is not None else 0
    Right = _carla.LaneChange.Right if _carla is not None else 1
    Left = _carla.LaneChange.Left if _carla is not None else 2
    Both = _carla.LaneChange.Both if _carla is not None else 3


#: The SUMO<->CARLA frame bridge the shims convert through. The SUMO
#: environment publishes its calibrated bridge here before the planner
#: builds any waypoint (see SumoMARLEnv._calibrate_coordinates); with none
#: published, the legacy constant offset of the bundled intersection applies.
_COORDINATE_BRIDGE = None


def set_coordinate_bridge(bridge) -> None:
    """Publish the frame bridge every shim position goes through."""
    global _COORDINATE_BRIDGE
    _COORDINATE_BRIDGE = bridge


def coordinate_bridge():
    return _COORDINATE_BRIDGE


def _sumo_to_carla_xy(x: float, y: float):
    bridge = _COORDINATE_BRIDGE
    if bridge is not None:
        return bridge.sumo_to_carla(x, y)
    return (float(x) - SumoWaypoint.NET_OFFSET[0], float(y) - SumoWaypoint.NET_OFFSET[1])


def _carla_to_sumo_xy(x: float, y: float):
    """CARLA -> SUMO through the published bridge, the inverse of _sumo_to_carla_xy."""
    bridge = coordinate_bridge()
    if bridge is not None:
        return bridge.carla_to_sumo(x, y)
    return (float(x) + SumoWaypoint.NET_OFFSET[0], float(y) + SumoWaypoint.NET_OFFSET[1])


def _split_lane_id(lane_id: str):
    edge_id, index = lane_id.rsplit("_", 1)
    return edge_id, int(index)


def _lane_progress(shape, x: float, y: float) -> float:
    """Distance along the lane shape of the point closest to (x, y)."""
    best, best_s, acc = float("inf"), 0.0, 0.0
    for (x1, y1), (x2, y2) in zip(shape, shape[1:]):
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg <= 1e-9:
            continue
        t = max(0.0, min(1.0, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (seg * seg)))
        d = math.hypot(x - (x1 + t * (x2 - x1)), y - (y1 + t * (y2 - y1)))
        if d < best:
            best, best_s = d, acc + t * seg
        acc += seg
    return best_s


def _lane_point(shape, s: float):
    """The point *s* metres along the lane shape (clamped to its ends)."""
    acc = 0.0
    for (x1, y1), (x2, y2) in zip(shape, shape[1:]):
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg <= 1e-9:
            continue
        if acc + seg >= s:
            r = max(0.0, (s - acc) / seg)
            return (x1 + r * (x2 - x1), y1 + r * (y2 - y1))
        acc += seg
    return (float(shape[-1][0]), float(shape[-1][1]))


def _link_field(link, name: str, index: int):
    """A field of a lane link, from libsumo's object or traci's tuple."""
    if hasattr(link, name):
        return getattr(link, name)
    return link[index] if isinstance(link, (tuple, list)) and len(link) > index else None


_PREDECESSORS = None


def _predecessor_lane(lane_id: str):
    """The lane a walk continues on before the start of *lane_id* (the straightest incoming link)."""
    global _PREDECESSORS
    if _PREDECESSORS is None:
        index = {}
        for edge_id in traci.edge.getIDList():
            for lane_index in range(traci.edge.getLaneNumber(edge_id)):
                source = f"{edge_id}_{lane_index}"
                try:
                    links = traci.lane.getLinks(source)
                except Exception:  # noqa: BLE001 - a lane without links is not an error
                    continue
                for link in links:
                    straight = str(_link_field(link, "direction", 6) or "") in ("s", "S")
                    internal = _link_field(link, "approachedInternal", 4)
                    approached = _link_field(link, "approachedLane", 0)
                    if internal:
                        index.setdefault(internal, []).append((0 if straight else 1, source))
                        index.setdefault(approached, []).append((0 if straight else 1, internal))
                    elif approached:
                        index.setdefault(approached, []).append((0 if straight else 1, source))
        _PREDECESSORS = {lane: sorted(set(v))[0][1] for lane, v in index.items() if v}
    return _PREDECESSORS.get(lane_id)


def _successor_lane(lane_id: str):
    """The lane a walk continues on past the end of *lane_id*.

    The straightest link is preferred; its internal (junction) lane comes first
    when there is one, then the lane it approaches. None at a dead end.
    """
    links = traci.lane.getLinks(lane_id)
    if not links:
        return None
    best = sorted(links, key=lambda l: 0 if str(_link_field(l, "direction", 6) or "") in ("s", "S") else 1)[0]
    return _link_field(best, "approachedInternal", 4) or _link_field(best, "approachedLane", 0) or None


class SumoWaypoint:
    """
    CARLA-like waypoint wrapper for SUMO edges/lanes.
    Provides same interface as carla.Waypoint for MARLPlanner compatibility.

    IMPORTANT: All positions are stored in CARLA coordinate system for consistency.
    SUMO coordinates are converted through the published frame bridge
    (``set_coordinate_bridge``), or the legacy netOffset (99.8, 100.0).
    """

    # Legacy network offset (the bundled intersection's netOffset), used only
    # when no bridge has been published.
    NET_OFFSET = (99.80, 100.00)

    def __init__(self, edge_id: str, lane_index: int, position: Tuple[float, float],
                 road_id: int, lane_id: int, is_junction: bool = False, junction_id: str = None):
        self.edge_id = edge_id
        self.lane_index = lane_index

        # Convert SUMO coordinates to CARLA coordinates through the bridge
        carla_x, carla_y = _sumo_to_carla_xy(position[0], position[1])

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

    def _adjacent_lane(self, offset: int):
        """This waypoint's neighbour on the lane ``offset`` places to the left (+1) or right (-1),
        at the same fraction of the lane's length. SUMO's lane 0 is the rightmost."""
        try:
            edge_lanes = traci.edge.getLaneNumber(self.edge_id)
        except Exception:  # noqa: BLE001
            return None
        new_index = self.lane_index + offset
        if not 0 <= new_index < edge_lanes:
            return None
        try:
            shape = traci.lane.getShape(f"{self.edge_id}_{self.lane_index}")
            other = traci.lane.getShape(f"{self.edge_id}_{new_index}")
        except Exception:  # noqa: BLE001
            return None
        if not shape or len(shape) < 2 or not other or len(other) < 2:
            return None
        sx, sy = _carla_to_sumo_xy(self.position[0], self.position[1])
        length = float(traci.lane.getLength(f"{self.edge_id}_{self.lane_index}")) or 1.0
        other_length = float(traci.lane.getLength(f"{self.edge_id}_{new_index}"))
        fraction = max(0.0, min(1.0, _lane_progress(shape, sx, sy) / length))
        point = _lane_point(other, fraction * other_length)
        return SumoWaypoint(self.edge_id, new_index, point, self.road_id, -new_index - 1,
                            self.is_junction, self.junction_id)

    def get_left_lane(self):
        """The lane to the left at the same longitudinal position, if any."""
        return self._adjacent_lane(+1)

    def get_right_lane(self):
        """The lane to the right at the same longitudinal position, if any."""
        return self._adjacent_lane(-1)

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
        """Waypoints *distance* metres ahead along the lane, continuing onto the
        lane this one links to when the distance runs past the lane's end.

        Returns a list (CARLA compatibility). The current position goes back to
        SUMO through the published bridge, so progress is measured on the lane's
        own shape. The earlier version converted with the bundled intersection's
        constant offset and clamped at the lane's end: on a calibrated map every
        walk landed on an edge end, so a route's destination sat at the next
        junction, however short or long the exit edge was, instead of the
        configured offset past the exit.
        """
        try:
            lane_id = f"{self.edge_id}_{self.lane_index}"
            edge_id, lane_index = self.edge_id, self.lane_index
            shape = traci.lane.getShape(lane_id)
            if not shape or len(shape) < 2:
                return []
            sx, sy = _carla_to_sumo_xy(self.position[0], self.position[1])
            length = float(traci.lane.getLength(lane_id))
            target = _lane_progress(shape, sx, sy) + float(distance)
            hops = 0
            while target > length + 1e-6 and hops < 8:
                successor = _successor_lane(lane_id)
                if successor is None:
                    target = length                       # a dead end: stop at the end of this lane
                    break
                target -= length
                lane_id = successor
                edge_id, lane_index = _split_lane_id(lane_id)
                shape = traci.lane.getShape(lane_id)
                if not shape or len(shape) < 2:
                    return []
                length = float(traci.lane.getLength(lane_id))
                hops += 1
            x, y = _lane_point(shape, max(0.0, min(length, target)))
            if edge_id == self.edge_id:
                road_id, lane_ref, is_junction, junction = self.road_id, self.lane_id, self.is_junction, self.junction_id
            else:
                internal = edge_id.startswith(":")
                road_id, lane_ref, is_junction = _stable_id(edge_id), -lane_index - 1, internal
                junction = edge_id[1:].rsplit("_", 2)[0] if internal else None
            return [SumoWaypoint(edge_id, lane_index, (x, y), road_id, lane_ref, is_junction, junction)]
        except Exception as e:  # noqa: BLE001 - the planner's walk must not die on one lane
            logger.debug(f"Failed to get next waypoint for {self.edge_id}_{self.lane_index}: {e}")
            return []

    def previous(self, distance: float):
        """Waypoints *distance* metres behind this one along the lane, continuing
        onto the lane that leads into it when the distance runs past the start.

        Returns a list (CARLA compatibility). Progress is measured through the
        published bridge on the lane's own shape, like ``next``; the earlier
        version used the bundled intersection's constant offset and stopped at
        the lane's start, so a spawn point further before a junction than its
        approach lane is long sat at the start of that short lane.
        """
        try:
            lane_id = f"{self.edge_id}_{self.lane_index}"
            edge_id, lane_index = self.edge_id, self.lane_index
            shape = traci.lane.getShape(lane_id)
            if not shape or len(shape) < 2:
                return []
            sx, sy = _carla_to_sumo_xy(self.position[0], self.position[1])
            target = _lane_progress(shape, sx, sy) - float(distance)
            hops = 0
            while target < -1e-6 and hops < 8:
                predecessor = _predecessor_lane(lane_id)
                if predecessor is None:
                    target = 0.0                          # nothing leads here: stop at the start
                    break
                lane_id = predecessor
                edge_id, lane_index = _split_lane_id(lane_id)
                shape = traci.lane.getShape(lane_id)
                if not shape or len(shape) < 2:
                    return []
                target += float(traci.lane.getLength(lane_id))
                hops += 1
            length = float(traci.lane.getLength(lane_id))
            x, y = _lane_point(shape, max(0.0, min(length, target)))
            if edge_id == self.edge_id:
                road_id, lane_ref, is_junction, junction = self.road_id, self.lane_id, self.is_junction, self.junction_id
            else:
                internal = edge_id.startswith(":")
                road_id, lane_ref, is_junction = _stable_id(edge_id), -lane_index - 1, internal
                junction = edge_id[1:].rsplit("_", 2)[0] if internal else None
            return [SumoWaypoint(edge_id, lane_index, (x, y), road_id, lane_ref, is_junction, junction)]
        except Exception as e:  # noqa: BLE001 - the planner's walk must not die on one lane
            logger.debug(f"Failed to interpolate previous waypoint for {self.edge_id}_{self.lane_index}: {e}")
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
        self.id = int(junction_id) if junction_id.isdigit() else _stable_id(junction_id)
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

        # Convert junction center to CARLA coordinates through the bridge
        center_x_carla, center_y_carla = _sumo_to_carla_xy(center_x_sumo, center_y_sumo)

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
                    _stable_id(entry_edge), -lane_idx - 1, False
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
                            _stable_id(exit_edge), -exit_lane_idx - 1, False
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
                    road_id=_stable_id(edge_id),
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
        from opencda_marl._kernel.traffic.traffic_manager import _resolve_planner

        # Resolve the registered route planner (default: the built-in MARL
        # planner) against the SUMO world adapter — the same component key
        # (`planner.type`) selects it on both backends.
        planner_cls = _resolve_planner(self.config or {})
        temp_planner = planner_cls(self.world, self.config)

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
