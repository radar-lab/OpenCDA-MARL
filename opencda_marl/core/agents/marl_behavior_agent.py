'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-31 16:15:24
FilePath     : /OpenCDA-MARL/opencda_marl/core/agents/marl_behavior_agent.py
Description  : MARL Behavior Agent - Simplified Route Following

This agent provides a simple approach to vehicle behavior:
- Follow the initial scheduled route without replanning
- Only actions: slow down, stop, or yield to avoid collisions
- No complex path generation or lane changes

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from opencda.core.plan.behavior_agent import BehaviorAgent


class MARLBehaviorAgent(BehaviorAgent):
    """
    Simplified MARL behavior agent that follows scheduled routes.
    
    Key principles:
    - Follow initial route waypoints without replanning
    - Only collision avoidance actions: slow down, stop, yield
    - Simple, predictable behavior
    
    Parameters
    ----------
    vehicle : carla.Vehicle
        The ego vehicle instance in CARLA
    
    carla_map : carla.Map
        The CARLA map instance
        
    config_yaml : dict
        The configuration dictionary for the behavior agent
    """
    
    def __init__(self, vehicle, carla_map, config_yaml):
        # Call parent constructor
        super().__init__(vehicle, carla_map, config_yaml)
        
        # Simple tracking variables
        self.scheduled_waypoints = []  # Store initial route
        self.destination_location = None  # Store destination
        
    def set_destination(self, start_location, end_location, clean=False, end_reset=True, clean_history=False):
        """
        Set destination without allowing parent to shift the start waypoint.
        This prevents the vehicle from starting in the wrong lane.
        """
        # Store destination for reference
        self.destination_location = end_location
        
        # Clean buffers if requested
        if clean:
            self.get_local_planner().get_waypoints_queue().clear()
            self.get_local_planner().get_trajectory().clear()
            self.get_local_planner().get_waypoint_buffer().clear()
        if clean_history:
            self.get_local_planner().get_history_buffer().clear()

        # Use exact start waypoint without shifting - prevents wrong lane start
        self.start_waypoint = self._map.get_waypoint(start_location)
        
        # Set end waypoint
        end_waypoint = self._map.get_waypoint(end_location)
        if end_reset:
            self.end_waypoint = end_waypoint

        # Generate route trace from exact start position
        route_trace = self._trace_route(self.start_waypoint, end_waypoint)
        
        # Store initial route if not set
        if self.initial_global_route is None:
            self.initial_global_route = route_trace

        # Set the global plan in local planner
        self._local_planner.set_global_plan(route_trace, clean)
        
        # Store the initial waypoints for simple following
        waypoint_buffer = self.get_local_planner().get_waypoint_buffer()
        self.scheduled_waypoints = list(waypoint_buffer)
        
    def run_step(self, target_speed=None, collision_detector_enabled=True, lane_change_allowed=True):
        """
        Simple run step: follow route, slow down for obstacles.
        
        Returns
        -------
        target_speed : float
            The target speed for the vehicle
        target_location : carla.Location or None
            The target location (None if stopping)
        """
        # Basic setup
        ego_vehicle_loc = self._ego_pos.location
        
        # Check if close to destination (within 5 meters)
        if self.destination_location:
            distance_to_dest = ego_vehicle_loc.distance(self.destination_location)
            if distance_to_dest < 5.0:
                raise StopIteration("Destination reached")
        
        # Check if route is completed (no waypoints left)
        waypoint_buffer = self.get_local_planner().get_waypoint_buffer()
        if len(waypoint_buffer) == 0:
            raise StopIteration("Route completed - no more waypoints")
        
        # Traffic light management
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        if self.traffic_light_manager(ego_vehicle_wp) != 0:
            return 0, None  # Stop for red light
        
        # Simple obstacle detection and collision avoidance
        obstacle_vehicle, distance = self._get_closest_obstacle()
        
        # Default speed
        if target_speed is None:
            target_speed = self.max_speed - self.speed_lim_dist
        
        # Simple collision avoidance: slow down or stop if obstacle is close
        if obstacle_vehicle and distance < 15.0:  # 15 meter detection range
            if distance < 5.0:
                # Very close - stop
                target_speed = 0
            elif distance < 10.0:
                # Close - slow down significantly
                target_speed = min(target_speed, 20)  # 20 km/h
            else:
                # Moderate distance - reduce speed
                target_speed = min(target_speed, target_speed * 0.7)
        
        # Generate path for local planner
        rx, ry, rk = self.generate_path()
        return self._local_planner.run_step(rx, ry, rk, target_speed=target_speed)
    
    def generate_path(self):
        """
        Safe path generation with fallback to prevent deque errors.
        
        Returns
        -------
        tuple
            (rx, ry, rk) - path coordinates and curvatures for local planner
        """
        try:
            # Check if vehicle is stationary or very slow at start to prevent PID warnings
            vehicle_speed = 3.6 * self._vehicle.get_velocity().length()  # Convert to km/h
            
            if vehicle_speed < 1.0:  # Very slow or stationary
                # Generate minimal path to avoid PID controller issues
                return self._generate_minimal_path()
            
            # Check if close to destination - avoid generating path to prevent PID warning
            if self.destination_location:
                ego_vehicle_loc = self._ego_pos.location
                distance_to_dest = ego_vehicle_loc.distance(self.destination_location)
                if distance_to_dest < 2.0:
                    # Generate minimal path when very close to destination
                    return self._generate_minimal_path()
            
            # Check if waypoint buffer has enough waypoints
            waypoint_buffer = self.get_local_planner().get_waypoint_buffer()
            
            # If buffer is too small, use simple forward path
            if len(waypoint_buffer) <= 1:
                return self._generate_simple_forward_path()
            
            # Try parent's path generation
            return super().generate_path()
            
        except Exception:
            # If parent's generation fails, use simple forward path
            return self._generate_simple_forward_path()
    
    def _generate_minimal_path(self):
        """
        Generate minimal path when very close to destination to avoid PID warnings.
        Uses vehicle's orientation for proper direction.
        Ensures first point is ahead of vehicle to prevent division by zero in PID controller.
        
        Returns
        -------
        tuple
            (rx, ry, rk) - minimal path coordinates and curvatures
        """
        import math
        
        ego_transform = self._ego_pos
        current_location = ego_transform.location
        current_yaw = math.radians(ego_transform.rotation.yaw)
        
        # Generate 3 points in vehicle's forward direction 
        # Start at 0.5m ahead to prevent PID controller division by zero
        rx, ry, rk = [], [], []
        for i in range(3):
            distance = (i + 1) * 0.5  # Start 0.5m ahead, then 1.0m, 1.5m
            x = current_location.x + distance * math.cos(current_yaw)
            y = current_location.y + distance * math.sin(current_yaw)
            
            rx.append(x)
            ry.append(y)
            rk.append(0.0)
        
        return rx, ry, rk
    
    def _generate_simple_forward_path(self):
        """
        Generate a path using available waypoints or simple forward projection.
        Ensures minimum distance to prevent PID controller issues.
        
        Returns
        -------
        tuple
            (rx, ry, rk) - path coordinates and curvatures
        """
        import math
        
        # Get current position
        ego_transform = self._ego_pos
        current_location = ego_transform.location
        
        # Try to use waypoints from buffer if available
        waypoint_buffer = self.get_local_planner().get_waypoint_buffer()
        
        rx, ry, rk = [], [], []
        
        if len(waypoint_buffer) > 0:
            # Use actual waypoints for better alignment
            for waypoint, _ in list(waypoint_buffer)[:min(10, len(waypoint_buffer))]:
                waypoint_location = waypoint.transform.location
                # Check minimum distance to vehicle to prevent PID issues
                distance_to_vehicle = current_location.distance(waypoint_location)
                
                # Only add waypoint if it's at least 0.3m from vehicle
                if distance_to_vehicle > 0.3:
                    rx.append(waypoint_location.x)
                    ry.append(waypoint_location.y)
                    rk.append(0.0)  # Simple curvature
            
            # If no valid waypoints or need more points, extend with forward projection
            if len(rx) < 3:
                current_yaw = math.radians(ego_transform.rotation.yaw)
                start_distance = 0.5  # Start at least 0.5m ahead
                
                for i in range(3):
                    distance = start_distance + (i * 1.0)  # 0.5m, 1.5m, 2.5m
                    x = current_location.x + distance * math.cos(current_yaw)
                    y = current_location.y + distance * math.sin(current_yaw)
                    
                    rx.append(x)
                    ry.append(y)
                    rk.append(0.0)
            
            # Extend to 10 points if we have fewer
            elif len(rx) < 10 and len(rx) > 0:
                # Use last waypoint for extension
                last_x, last_y = rx[-1], ry[-1]
                # Estimate direction from last two points if possible
                if len(rx) >= 2:
                    dx = rx[-1] - rx[-2]
                    dy = ry[-1] - ry[-2]
                    last_yaw = math.atan2(dy, dx)
                else:
                    last_yaw = math.radians(ego_transform.rotation.yaw)
                
                for i in range(len(rx), 10):
                    distance = (i - len(rx) + 1) * 2.0  # 2 meters apart
                    x = last_x + distance * math.cos(last_yaw)
                    y = last_y + distance * math.sin(last_yaw)
                    
                    rx.append(x)
                    ry.append(y)
                    rk.append(0.0)
        else:
            # Fallback to simple forward projection if no waypoints
            current_yaw = math.radians(ego_transform.rotation.yaw)
            for i in range(10):
                distance = (i + 1) * 1.0  # Start 1m ahead, 2m, 3m, etc.
                x = current_location.x + distance * math.cos(current_yaw)
                y = current_location.y + distance * math.sin(current_yaw)
                
                rx.append(x)
                ry.append(y)
                rk.append(0.0)
        
        return rx, ry, rk

    def _get_closest_obstacle(self):
        """
        Find the closest obstacle ahead of the vehicle.
        
        Returns
        -------
        tuple
            (closest_vehicle, distance) or (None, float('inf')) if no obstacles
        """
        try:
            # Get vehicles in front (simplified detection)
            ego_transform = self._ego_pos
            closest_vehicle = None
            min_distance = float('inf')
            
            # Check all objects for obstacles
            if hasattr(self, 'objects') and self.objects:
                for obj_id, obj_info in self.objects.items():
                    if 'location' in obj_info:
                        obj_location = obj_info['location']
                        distance = ego_transform.location.distance(obj_location)
                        
                        # Simple check: is it ahead and close?
                        if distance < min_distance and distance > 1.0:  # Ignore very close objects
                            # Simple forward direction check
                            import math
                            dx = obj_location.x - ego_transform.location.x
                            dy = obj_location.y - ego_transform.location.y
                            ego_yaw = math.radians(ego_transform.rotation.yaw)
                            
                            # Project onto vehicle's forward direction
                            forward_dist = dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw)
                            
                            if forward_dist > 0:  # Object is ahead
                                min_distance = distance
                                closest_vehicle = obj_info
            
            return closest_vehicle, min_distance
            
        except Exception:
            # If obstacle detection fails, assume no obstacles
            return None, float('inf')