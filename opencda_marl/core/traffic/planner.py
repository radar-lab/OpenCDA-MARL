'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 21:38:54
FilePath     : /OpenCDA-MARL/opencda_marl/core/traffic/planner.py
Description  : MARL traffic planner with comprehensive lane-to-lane route generation.
               Generates all possible routes through intersections for MARL training.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import carla
import random
from typing import List, Tuple, Dict, Any, Optional
from omegaconf import DictConfig
from loguru import logger

from .utils import move_along_lane, wp_to_transform, wp_direction


class MARLPlanner:
    """
    MARL Traffic Planner for generating comprehensive junction routes.
    
    This planner analyzes all junctions in the CARLA world and generates
    all possible lane-to-lane routes for MARL traffic simulation.
    """

    def __init__(self, world: carla.World, config: DictConfig):
        """
        Initialize the MARL planner.
        
        Args:
            world: CARLA world instance
            config: Configuration dictionary with planner parameters
        """
        self.world = world
        self.config = config
        self.map = self.world.get_map()

        seed = self.config.get("seed", 42)
        random.seed(seed)

        # initialize junctions
        self._junctions = self._get_junctions()
        
        # visualize junctions
        self.visualization()
        
        logger.success(f"MARLPlanner initialized with {len(self._junctions)} junctions")

    # --------------------------------------------------------------------- #
    # Private API
    # --------------------------------------------------------------------- #
    def _get_planner_config(self) -> Dict[str, float]:
        """Get planner configuration parameters."""
        return {
            'safe_distance': float(self.config.get("distance", 5.0)),
            'spawn_offset': float(self.config.get("spawn_offset", 2)),
            'dest_offset': float(self.config.get("dest_offset", 2)),
            'z_lift': float(self.config.get("spawn_z_lift", 0.3)),
            'wp_step': float(self.config.get("wp_step", 1.0)),
        }

    def _collect_junction_waypoints(self, junction: carla.Junction) -> Tuple[List[carla.Waypoint], List[carla.Waypoint]]:
        """Collect all unique entry and exit waypoints with all their lanes."""
        connections = junction.get_waypoints(carla.LaneType.Driving)
        
        entry_waypoints = []
        exit_waypoints = []
        
        for entry_wp, exit_wp in connections:
            # Check if we already have this road group for entry
            if not any(self._is_same_lane_group(entry_wp, e) for e in entry_waypoints):
                # Get all lanes at this road position
                entry_lanes = self._get_all_lanes_at_road(entry_wp)
                entry_waypoints.extend(entry_lanes)
            
            # Check if we already have this road group for exit
            if not any(self._is_same_lane_group(exit_wp, e) for e in exit_waypoints):
                exit_lanes = self._get_all_lanes_at_road(exit_wp)
                exit_waypoints.extend(exit_lanes)
        
        return entry_waypoints, exit_waypoints

    def _generate_junction_routes(self, entry_groups: Dict[str, List[carla.Waypoint]], 
                                  exit_groups: Dict[str, List[carla.Waypoint]], 
                                  config: Dict[str, float]) -> Tuple[Dict[int, Dict], Dict[str, List[carla.Waypoint]], Dict[str, List[carla.Waypoint]]]:
        """Generate all possible lane-to-lane combinations for the junction."""
        # Pre-calculate spawn waypoints for each direction for efficient lane indexing
        spawn_groups = {}
        for entry_dir, entry_lanes in entry_groups.items():
            spawn_lanes = []
            for lane_wp in entry_lanes:
                spawn_lane_wp = move_along_lane(
                    lane_wp, distance=-(config['spawn_offset'] * config['safe_distance']), 
                    step=config['wp_step'])
                spawn_lanes.append(spawn_lane_wp)
            spawn_groups[entry_dir] = spawn_lanes
        
        # Pre-calculate destination waypoints for each direction (exit waypoints + offset)
        dest_groups = {}
        for exit_dir, exit_lanes in exit_groups.items():
            dest_lanes = []
            for lane_wp in exit_lanes:
                dest_lane_wp = move_along_lane(
                    lane_wp, distance=+(config['dest_offset'] * config['safe_distance']), 
                    step=config['wp_step'])
                dest_lanes.append(dest_lane_wp)
            dest_groups[exit_dir] = dest_lanes
        
        conn_info = {}
        route_id = 0
        allow_uturn = self.config.get("allow_uturn", False)
        for entry_dir, entry_lanes in entry_groups.items():
            for exit_dir, exit_lanes in exit_groups.items():
                # Generate route for each lane combination
                for i, entry_wp in enumerate(entry_lanes):
                    for exit_wp in exit_lanes:
                        # Use pre-calculated spawn waypoint
                        spawn_wp = spawn_groups[entry_dir][i]
                        dest_wp = move_along_lane(
                            exit_wp, distance=+(config['dest_offset'] * config['safe_distance']), 
                            step=config['wp_step'])

                        # Build transforms with a small z lift
                        spawn_tf = wp_to_transform(spawn_wp, config['z_lift'])
                        dest_tf = wp_to_transform(dest_wp)

                        # Determine route type for debugging
                        if entry_dir == self._opposite_direction(exit_dir):
                            route_type = "straight"
                        elif (entry_dir, exit_dir) in [('north','east'), ('east','south'), ('south','west'), ('west','north')]:
                            route_type = "right"  
                        elif (entry_dir, exit_dir) in [('north','west'), ('west','south'), ('south','east'), ('east','north')]:
                            route_type = "left"
                        elif (entry_dir, exit_dir) in [('north','north'), ('east','east'), ('south','south'), ('west','west')]:
                            if allow_uturn:
                                route_type = "U-turn"
                            else:
                                continue
                        else:
                            route_type = "unknown"

                        
                        # Get lane indices using pre-calculated spawn waypoints for reliable lane detection
                        spawn_lane_indices = self._get_lane_indices(spawn_groups[entry_dir])
                        exit_lane_indices = self._get_lane_indices(exit_lanes)
                        
                        # Use the spawn waypoint we already have
                        entry_lane_idx = spawn_lane_indices[spawn_wp]
                        exit_lane_idx = exit_lane_indices[exit_wp]
                        
                        conn_info[route_id] = {
                            "route_id": route_id,
                            "entry_wp": entry_wp,
                            "exit_wp": exit_wp,
                            "spawn_wp": spawn_wp,
                            "dest_wp": dest_wp,
                            "spawn_tf": spawn_tf,
                            "dest_tf": dest_tf,
                            "entry_direction": entry_dir,
                            "exit_direction": exit_dir,
                            "entry_lane_idx": entry_lane_idx,  # 0, 1, 2... (position-based)
                            "exit_lane_idx": exit_lane_idx,    # 0, 1, 2... (position-based)
                            "route_type": route_type,
                        }
                        route_id += 1
        
        return conn_info, spawn_groups, dest_groups

    def _build_junction_data(self, junction: carla.Junction, 
                            entry_groups: Dict[str, List[carla.Waypoint]], 
                            exit_groups: Dict[str, List[carla.Waypoint]], 
                            routes: Dict[int, Dict],
                            spawn_groups: Dict[str, List[carla.Waypoint]],
                            dest_groups: Dict[str, List[carla.Waypoint]]) -> Dict[str, Any]:
        """Build the complete junction data structure."""
        bbox = junction.bounding_box
        return {
            "center": bbox.location,
            "bbox": bbox,
            "extent": bbox.extent,
            "routes": routes,
            "entry_groups": entry_groups,
            "exit_groups": exit_groups,
            "spawn_groups": spawn_groups,
            "dest_groups": dest_groups,
            "total_routes": len(routes),
        }

    def _get_junctions(self) -> Dict[int, Dict[str, Any]]:
        config = self._get_planner_config()

        waypoints = self.map.generate_waypoints(distance=config['safe_distance'])
        seen = set()
        junctions = {}

        for wp in waypoints:
            if not wp.is_junction:
                continue
            j = wp.get_junction()
            if j.id in seen:
                continue
            seen.add(j.id)

            # Process junction waypoints and routes
            entry_waypoints, exit_waypoints = self._collect_junction_waypoints(j)
            entry_groups = self._group_waypoints_by_direction(entry_waypoints, j.bounding_box.location)
            exit_groups = self._group_waypoints_by_direction(exit_waypoints, j.bounding_box.location)
            routes, spawn_groups, dest_groups = self._generate_junction_routes(entry_groups, exit_groups, config)
            
            junctions[j.id] = self._build_junction_data(j, entry_groups, exit_groups, routes, spawn_groups, dest_groups)
        return junctions

    def _get_all_lanes_at_road(self, waypoint: carla.Waypoint) -> List[carla.Waypoint]:
        """Get all driving lanes at the same road position."""
        lanes = []
        # Find leftmost lane first
        current = waypoint
        while True:
            left_wp = current.get_left_lane()
            if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                current = left_wp
            else:
                break
        
        # Collect all lanes from left to right
        lanes.append(current)
        while True:
            right_wp = current.get_right_lane()
            if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                lanes.append(right_wp)
                current = right_wp
            else:
                break
        
        return lanes

    def _is_same_lane_group(self, wp1: carla.Waypoint, wp2: carla.Waypoint) -> bool:
        """Check if two waypoints belong to the same road group."""
        return (wp1.road_id == wp2.road_id and 
                wp1.section_id == wp2.section_id and
                abs(wp1.s - wp2.s) < 5.0)  # Within 5m considered same position

    def _group_waypoints_by_direction(self, waypoints: List[carla.Waypoint], 
                                      center: carla.Location) -> Dict[str, List[carla.Waypoint]]:
        """Group waypoints by direction relative to junction center."""
        groups = {'north': [], 'south': [], 'east': [], 'west': []}
        for wp in waypoints:
            direction = wp_direction(wp, center)
            groups[direction].append(wp)
        return groups

    def _get_unique_lane_key(self, waypoint: carla.Waypoint, direction: str) -> str:
        """Create unique lane identifier combining direction, road_id, and lane_id."""
        return f"{direction}_{waypoint.road_id}_{waypoint.lane_id}"
    
    def _get_lane_position_score(self, waypoint: carla.Waypoint) -> int:
        """Get lane position score (lower = more left) using CARLA's lane change property."""
        lane_change = waypoint.lane_change
        
        if lane_change == carla.LaneChange.Right:
            return 0  # Leftmost - can only change right
        elif lane_change == carla.LaneChange.Both:
            return 1  # Middle - can change both ways
        elif lane_change == carla.LaneChange.Left:
            return 2  # Rightmost - can only change left
        else:  # carla.LaneChange.None
            # Single lane or restricted - use lane_id as fallback
            # More negative lane_id = further right in CARLA
            return 1 if waypoint.lane_id == -1 else (0 if waypoint.lane_id < -1 else 2)

    def _get_lane_indices(self, waypoints: List[carla.Waypoint]) -> Dict[carla.Waypoint, int]:
        """Map waypoints to lane indices (0, 1, 2...) using lane change properties for ordering."""
        # Group by unique (road_id, lane_id) to identify unique physical lanes
        unique_lanes = {}
        for wp in waypoints:
            lane_key = (wp.road_id, wp.lane_id)
            if lane_key not in unique_lanes:
                unique_lanes[lane_key] = wp
        
        # Sort lanes by position score (leftmost first), then by road_id, then by lane_id for consistency
        sorted_lanes = sorted(unique_lanes.values(), 
                             key=lambda wp: (self._get_lane_position_score(wp), wp.road_id, wp.lane_id))
        
        # Create index mapping: assign sequential indices 0, 1, 2...
        lane_indices = {}
        for global_idx, sorted_wp in enumerate(sorted_lanes):
            # Map all waypoints with same (road_id, lane_id) to this index
            for wp in waypoints:
                if wp.road_id == sorted_wp.road_id and wp.lane_id == sorted_wp.lane_id:
                    lane_indices[wp] = global_idx

        return lane_indices

    def _opposite_direction(self, direction: str) -> str:
        """Get opposite direction."""
        opposites = {'north': 'south', 'south': 'north', 
                     'east': 'west', 'west': 'east'}
        return opposites.get(direction, direction)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_junctions(self) -> Dict[int, Dict[str, Any]]:
        return self._junctions

    def get_random_route(self, junction_id: int, visualize: bool = False) -> Optional[Dict]:
        routes = self.get_routes(junction_id)
        if not routes:
            return None
        route_id = random.choice(list(routes.keys()))
        if visualize:
            self._visualize_points([routes[route_id]['spawn_wp'].transform.location], 10.0)
        return routes[route_id]
    
    def get_random_route_by_type(self, junction_id: int, route_type: str, visualize: bool = False) -> Optional[Dict]:
        routes = self.get_routes(junction_id)
        if not routes:
            return None
        route_ids = [route_id for route_id, route in routes.items() if route['route_type'] == route_type]
        if not route_ids:
            return None
        route_id = random.choice(route_ids)
        if visualize:
            self._visualize_points([routes[route_id]['spawn_wp'].transform.location], 10.0)
        return routes[route_id]
    
    def get_random_route_by_direction(self, junction_id: int, direction: str, visualize: bool = False) -> Optional[Dict]:
        routes = self.get_routes(junction_id)
        if not routes:
            return None
        route_ids = [route_id for route_id, route in routes.items() if route['entry_direction'] == direction]
        if not route_ids:
            return None
        route_id = random.choice(route_ids)
        if visualize:
            self._visualize_points([routes[route_id]['spawn_wp'].transform.location], 10.0)
        return routes[route_id]

    def get_random_routes(self, junction_id: int, num: int, 
                          route_type: str = None, direction: str = None,
                          visualize: bool = False) -> List[Dict]:
        routes = []
        for i in range(num):
            if route_type:
                route = self.get_random_route_by_type(junction_id, route_type, visualize)
            elif direction:
                route = self.get_random_route_by_direction(junction_id, direction, visualize)
            else:
                route = self.get_random_route(junction_id, visualize)
            if route:
                routes.append(route)
        return routes

    def get_routes(self, junction_id: int) -> Dict[int, Dict]:
        return self._junctions[junction_id]["routes"]

    def get_route(self, junction_id: int, route_id: int) -> Optional[Dict]:
        routes = self.get_routes(junction_id)
        if route_id not in routes:
            return None
        return routes[route_id]
   
    def cleanup(self):
        self._junctions.clear()
        self.world = None
        self.map = None

    # --------------------------------------------------------------------- #
    # Visualization
    # --------------------------------------------------------------------- #
    def _visualize_points(self, points: List[carla.Location], life_time: float):
        """Draw points."""
        for point in points:
            self.world.debug.draw_point(
                point + carla.Location(z=0.5),
                size=0.1,
                color=carla.Color(255, 0, 255),  # MAGENTA
                life_time=life_time
            )

    def _visualize_junction_center(self, center: carla.Location, life_time: float):
        """Draw junction center point."""
        self.world.debug.draw_point(
            center + carla.Location(z=0.5),
            size=0.3,
            color=carla.Color(255, 0, 0),  # RED
            life_time=life_time
        )

    def _visualize_junction_bbox(self, center: carla.Location, extent: carla.Vector3D, life_time: float):
        """Draw junction bounding box."""
        self.world.debug.draw_box(
            carla.BoundingBox(center, extent),
            carla.Rotation(),
            thickness=0.1,
            color=carla.Color(0, 255, 0),  # GREEN
            life_time=life_time
        )

    def _visualize_junction_extent(self, center: carla.Location, extent: carla.Vector3D, 
                                  spawn_offset: float, dest_offset: float, safe_distance: float, life_time: float):
        """Draw extended junction box."""
        offset = max(spawn_offset, dest_offset) * safe_distance
        new_extent = carla.Vector3D(
            extent.x + offset, extent.y + offset, extent.z)

        self.world.debug.draw_box(
            carla.BoundingBox(center, new_extent),
            carla.Rotation(),
            thickness=0.1,
            color=carla.Color(255, 0, 0),  # RED
            life_time=life_time
        )

    def _visualize_lane_labels(self, spawn_groups: Dict[str, List[carla.Waypoint]], 
                              dest_groups: Dict[str, List[carla.Waypoint]],
                              life_time: float):
        """Draw lane index labels using spawn waypoints for reliable lane detection."""
        
        # Label spawn lanes since these have reliable lane change properties
        for direction, spawn_waypoints in spawn_groups.items():
            if not spawn_waypoints:
                continue
                
            spawn_lane_indices = self._get_lane_indices(spawn_waypoints)
            
            # Track which lanes we've already labeled using unique keys
            labeled_lanes = set()
            
            for spawn_wp in spawn_waypoints:
                lane_idx = spawn_lane_indices[spawn_wp]
                unique_key = self._get_unique_lane_key(spawn_wp, direction)
                
                if unique_key not in labeled_lanes:
                    labeled_lanes.add(unique_key)
                    
                    loc = spawn_wp.transform.location + carla.Location(z=2.0)
                    text = f"{direction[0].upper()}{lane_idx} {spawn_wp.lane_id}"  # e.g., "N0", "N1", "N2"
                    
                    self.world.debug.draw_string(
                        loc, text, draw_shadow=False,
                        color=carla.Color(0, 255, 0),  # GREEN for entry
                        life_time=life_time * 10000
                    )

            for direction, dest_waypoints in dest_groups.items():
                if not dest_waypoints:
                    continue
                    
                dest_lane_indices = self._get_lane_indices(dest_waypoints)
                for dest_wp in dest_waypoints:
                    lane_idx = dest_lane_indices[dest_wp]
                    unique_key = self._get_unique_lane_key(dest_wp, direction)
                    
                    if unique_key not in labeled_lanes:
                        labeled_lanes.add(unique_key)
                    
                    loc = dest_wp.transform.location + carla.Location(z=2.0)
                    text = f"D{direction[0].upper()}{lane_idx} {dest_wp.lane_id}"  # e.g., "DN0", "DN1", "DN2"
                    
                    self.world.debug.draw_string(
                        loc, text, draw_shadow=False,
                        color=carla.Color(255, 255, 0),  # YELLOW for destinations
                        life_time=life_time * 10000
                    )

    def _visualize_junction_text(self, junction_id: int, center: carla.Location, 
                                extent: carla.Vector3D, routes_count: int,
                                spawn_groups: Dict[str, List[carla.Waypoint]],
                                dest_groups: Dict[str, List[carla.Waypoint]],
                                life_time: float):
        """Draw junction text labels and lane IDs."""
        # Existing junction info text
        label_location = carla.Location(x=center.x, y=center.y, z=0.5)
        label_text = [
            f"Intersection {junction_id}",
            f"Routes: {routes_count}",
            f"Center: [{center.x:.1f}, {center.y:.1f}, {center.z:.1f}]",
            f"Size: {extent.x:.1f}x{extent.y:.1f}m"
        ]

        for i, text in enumerate(label_text):
            text_location = label_location + carla.Location(x=i*2)
            self.world.debug.draw_string(
                text_location,
                text,
                draw_shadow=False,
                color=carla.Color(0, 255, 255),  # CYAN
                life_time=life_time
            )
        
        # Add lane labels
        self._visualize_lane_labels(spawn_groups, dest_groups, life_time)

    def _visualize_route(self, route: Dict, vis_cfg: Dict, life_time: float):
        """Visualize a single route with all its waypoints."""
        entry_wp = route['entry_wp']
        exit_wp = route['exit_wp']
        spawn_wp = route['spawn_wp']
        dest_wp = route['dest_wp']

        # Draw route line
        if vis_cfg.get("route_line", False):
            self.world.debug.draw_line(
                spawn_wp.transform.location,
                dest_wp.transform.location,
                thickness=0.1,
                color=carla.Color(0, 255, 0),  # GREEN
                life_time=life_time
            )

        # Draw entry waypoint
        if vis_cfg.get("entry_wp", False):
            self.world.debug.draw_point(
                entry_wp.transform.location + carla.Location(z=0.5),
                size=0.1,
                color=carla.Color(0, 255, 0),  # GREEN
                life_time=life_time
            )

        # Draw exit waypoint
        if vis_cfg.get("exit_wp", False):
            self.world.debug.draw_point(
                exit_wp.transform.location + carla.Location(z=0.5),
                size=0.1,
                color=carla.Color(0, 0, 255),  # BLUE
                life_time=life_time
            )

        # Draw spawn waypoint
        if vis_cfg.get("spawn_wp", False):
            self.world.debug.draw_point(
                spawn_wp.transform.location + carla.Location(z=0.5),
                size=0.1,
                color=carla.Color(255, 0, 255),  # MAGENTA
                life_time=life_time
            )

        # Draw destination waypoint
        if vis_cfg.get("dest_wp", False):
            self.world.debug.draw_point(
                dest_wp.transform.location + carla.Location(z=0.5),
                size=0.1,
                color=carla.Color(0, 0, 0),  # BLACK
                life_time=life_time
            )

    def _visualize_junction(self, junction_id: int, junction_data: Dict, vis_cfg: Dict, 
                           spawn_offset: float, dest_offset: float, safe_distance: float, life_time: float):
        """Visualize all components of a single junction."""
        center = junction_data["center"]
        extent = junction_data["extent"]
        routes = junction_data["routes"]

        # Draw junction components
        if vis_cfg.get("junction_center", False):
            self._visualize_junction_center(center, life_time)

        if vis_cfg.get("bbox", False):
            self._visualize_junction_bbox(center, extent, life_time)

        if vis_cfg.get("extent_box", False):
            self._visualize_junction_extent(center, extent, spawn_offset, dest_offset, safe_distance, life_time)

        if vis_cfg.get("text", False):
            self._visualize_junction_text(
                junction_id, center, extent, len(routes),
                junction_data["spawn_groups"], junction_data["dest_groups"], life_time
            )

        # Draw all routes
        for route in routes.values():
            self._visualize_route(route, vis_cfg, life_time)

    def visualization(self):
        """Visualize all junctions and their routes."""
        config = self._get_planner_config()
        vis_cfg = self.config.get("visualize", {})
        life_time = vis_cfg.get("life_time", 10.0)

        for junction_id, junction_data in self._junctions.items():
            self._visualize_junction(
                junction_id, junction_data, vis_cfg,
                config['spawn_offset'], config['dest_offset'], 
                config['safe_distance'], life_time
            )
