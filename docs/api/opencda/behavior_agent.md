# BehaviorAgent API

`BehaviorAgent` handles autonomous driving decisions through hierarchical planning: global route planning and local trajectory generation with obstacle avoidance.

| Component | Purpose | Configuration |
|-----------|---------|---------------|
| `global_planner` | Route planning | A* algorithm, road topology |
| `local_planner` | Trajectory generation | Obstacle avoidance, dynamics |
| `behavior_state` | Decision making | Traffic rules, safety checks |
| `run_step()` | Motion planning | Target speed, waypoint output |

## Planning Architecture

**What it does**: Hierarchical planning from global route to local trajectory execution

=== "Global Planning"
    ```python
    def set_destination(self, start_location, destination, clean=True):
        """
        Compute global route using A* algorithm
        
        Args:
            start_location: carla.Location starting point
            destination: carla.Location target destination
            clean: bool to clear previous waypoints
        """
        if clean:
            self._global_plan = []
            self._local_planner.reset_route()
        
        # Use CARLA's global planner with A*
        self._global_plan = self._global_planner.trace_route(
            start_location, destination
        )
        
        # Set local planner route
        self._local_planner.set_global_plan(self._global_plan)
    ```

=== "Local Planning"
    ```python
    def run_step(self, target_speed=None):
        """
        Execute local planning and control
        
        Args:
            target_speed: Optional speed override (m/s)
            
        Returns:
            tuple: (target_speed, target_waypoint)
        """
        # Check for emergency situations
        hazard_detected = self._is_vehicle_hazard(self._world.get_actors())
        
        if hazard_detected:
            return self._emergency_stop()
        
        # Normal planning
        if target_speed is None:
            target_speed = self._behavior.max_speed
        
        # Get next waypoint from local planner
        target_waypoint = self._local_planner.run_step(
            target_speed=target_speed,
            debug=self._debug
        )
        
        return target_speed, target_waypoint
    ```

=== "State Management"
    ```python
    def update_information(self, ego_pos, ego_speed, objects):
        """
        Update agent with latest perception and localization
        
        Args:
            ego_pos: carla.Transform current position
            ego_speed: carla.Vector3D current velocity
            objects: dict detected objects from perception
        """
        # Update vehicle state
        self._ego_vehicle_pos = ego_pos
        self._ego_vehicle_speed = ego_speed
        
        # Update detected objects
        self._detected_vehicles = objects.get('vehicles', [])
        self._detected_traffic_lights = objects.get('traffic_lights', [])
        
        # Update behavior state based on environment
        self._update_behavior_state()
    ```

**Planning Levels**: Global (route) → Local (trajectory) → Control (steering/throttle)

## Behavior States

**What it does**: Implements traffic rules and safety behaviors for different driving scenarios

=== "Lane Following"
    ```python
    class LaneFollowingBehavior:
        def __init__(self, target_speed=30.0, max_deceleration=10.0):
            self.target_speed = target_speed  # km/h
            self.max_deceleration = max_deceleration
            self.safety_distance = 2.0  # meters
        
        def plan_speed(self, lead_vehicle_distance, lead_vehicle_speed):
            """
            Adaptive cruise control for lane following
            """
            if lead_vehicle_distance < self.safety_distance:
                # Emergency braking
                return max(0, lead_vehicle_speed - 5.0)
            elif lead_vehicle_distance < 2 * self.safety_distance:
                # Follow mode
                return min(self.target_speed, lead_vehicle_speed)
            else:
                # Free driving
                return self.target_speed
    ```

=== "Lane Changing"
    ```python
    def execute_lane_change(self, direction='left'):
        """
        Safe lane change execution
        
        Args:
            direction: 'left' or 'right'
        """
        # Check blind spots
        if not self._is_lane_change_safe(direction):
            return False
        
        # Get target lane waypoints
        target_waypoints = self._get_lane_change_waypoints(direction)
        
        # Update local planner with new waypoints
        self._local_planner.set_temporary_route(target_waypoints)
        
        # Set lane change state
        self._behavior_state = 'lane_changing'
        return True
    
    def _is_lane_change_safe(self, direction):
        """Check for vehicles in target lane"""
        for vehicle in self._detected_vehicles:
            # Check lateral and longitudinal clearance
            lateral_distance = self._calculate_lateral_distance(vehicle)
            longitudinal_distance = self._calculate_longitudinal_distance(vehicle)
            
            if (lateral_distance < 3.5 and  # Lane width
                abs(longitudinal_distance) < 10.0):  # Safety buffer
                return False
        return True
    ```

=== "Traffic Light Handling"
    ```python
    def handle_traffic_lights(self):
        """
        Traffic light compliance behavior
        """
        for traffic_light in self._detected_traffic_lights:
            distance = self._calculate_distance(traffic_light.location)
            
            if distance < 50.0:  # Detection range
                if traffic_light.state == 'Red':
                    # Stop before intersection
                    self._local_planner.set_target_speed(0.0)
                    return 'stopping'
                elif traffic_light.state == 'Yellow' and distance < 10.0:
                    # Too close to stop safely
                    return 'proceeding'
                elif traffic_light.state == 'Green':
                    return 'proceeding'
        
        return 'normal'
    ```

**Behaviors**: Lane keeping, adaptive cruise control, lane changes, intersection navigation

## Trajectory Generation

**What it does**: Creates smooth, dynamically feasible trajectories from waypoints

=== "Local Planner Interface"
    ```python
    class LocalPlanner:
        def __init__(self, vehicle, carla_map, config):
            self._vehicle = vehicle
            self._map = carla_map
            
            # Planning parameters
            self._dt = config['dt']  # Planning timestep
            self._target_speed = config['target_speed']
            self._buffer_size = config['buffer_size']
            
            # Waypoint buffer
            self._waypoints_queue = deque(maxlen=self._buffer_size)
            
        def run_step(self, target_speed, debug=False):
            """
            Generate next waypoint for vehicle control
            
            Returns:
                carla.Waypoint: Next target waypoint
            """
            # Purge completed waypoints
            self._purge_waypoints()
            
            # Add new waypoints if needed
            if len(self._waypoints_queue) < self._buffer_size // 2:
                self._compute_next_waypoints()
            
            # Return next waypoint
            if self._waypoints_queue:
                return self._waypoints_queue[0][0]  # (waypoint, road_option)
            else:
                return None
    ```

=== "Obstacle Avoidance"
    ```python
    def _compute_next_waypoints(self):
        """
        Generate waypoints with obstacle avoidance
        """
        # Get base waypoints from global plan
        base_waypoints = self._get_global_plan_waypoints()
        
        # Check for obstacles
        obstacles = self._detect_obstacles()
        
        if obstacles:
            # Generate avoidance waypoints
            avoidance_waypoints = self._generate_avoidance_path(
                base_waypoints, obstacles
            )
            waypoints = avoidance_waypoints
        else:
            waypoints = base_waypoints
        
        # Add to queue with road options
        for waypoint in waypoints:
            road_option = self._get_road_option(waypoint)
            self._waypoints_queue.append((waypoint, road_option))
    
    def _generate_avoidance_path(self, base_waypoints, obstacles):
        """
        Generate smooth path around obstacles using spline interpolation
        """
        # Identify avoidance points
        avoidance_points = []
        for obstacle in obstacles:
            # Calculate lateral offset
            offset = self._calculate_avoidance_offset(obstacle)
            avoidance_point = self._apply_lateral_offset(
                base_waypoints[0], offset
            )
            avoidance_points.append(avoidance_point)
        
        # Generate smooth trajectory
        smooth_waypoints = self._interpolate_smooth_path(
            base_waypoints[0], avoidance_points, base_waypoints[-1]
        )
        
        return smooth_waypoints
    ```

=== "Speed Planning"
    ```python
    def plan_velocity_profile(self, waypoints, obstacles):
        """
        Generate velocity profile considering obstacles and dynamics
        """
        velocity_profile = []
        
        for i, waypoint in enumerate(waypoints):
            # Base speed from global plan
            base_speed = self._target_speed
            
            # Adjust for obstacles
            obstacle_speed = self._compute_obstacle_speed_limit(
                waypoint, obstacles
            )
            
            # Adjust for road curvature
            curvature_speed = self._compute_curvature_speed_limit(
                waypoint, waypoints[max(0, i-2):i+3]
            )
            
            # Take minimum for safety
            planned_speed = min(base_speed, obstacle_speed, curvature_speed)
            velocity_profile.append(planned_speed)
        
        # Smooth velocity profile
        smooth_profile = self._smooth_velocity_profile(velocity_profile)
        return smooth_profile
    ```

**Features**: Smooth trajectory generation, obstacle avoidance, velocity planning

## Integration Examples

!!! example 
    Common ways to use BehaviorAgent:

    === "Basic Navigation"
        ```python
        # Create behavior agent
        behavior_agent = BehaviorAgent(
            vehicle=carla_vehicle,
            carla_map=carla_map,
            config_yaml=config['behavior']
        )
        
        # Set destination
        start = vehicle.get_location()
        destination = carla.Location(x=100, y=50, z=0)
        behavior_agent.set_destination(start, destination)
        
        # Planning loop
        while not behavior_agent.done():
            # Update with perception data
            behavior_agent.update_information(ego_pos, ego_speed, detected_objects)
            
            # Get planning commands
            target_speed, target_waypoint = behavior_agent.run_step()
            
            # Execute control
            control = controller.run_step(target_speed, target_waypoint)
            vehicle.apply_control(control)
        ```

    === "Custom Behavior"
        ```python
        # Extend with custom behavior
        class AggressiveBehaviorAgent(BehaviorAgent):
            def __init__(self, vehicle, carla_map, config):
                super().__init__(vehicle, carla_map, config)
                self.max_speed = 50.0  # Higher speed limit
                self.safety_distance = 1.5  # Closer following
            
            def _is_lane_change_safe(self, direction):
                # More aggressive gap acceptance
                min_gap = 8.0  # Reduced from 10.0
                return super()._is_lane_change_safe_with_gap(direction, min_gap)
        ```

    === "MARL Integration"
        ```python
        # Use behavior agent in MARL environment
        class MARLBehaviorAgent(BehaviorAgent):
            def run_step_with_action(self, rl_action):
                """
                Execute planning with RL action input
                
                Args:
                    rl_action: dict with 'target_speed', 'lane_change_request'
                """
                target_speed = rl_action.get('target_speed', None)
                
                # Handle lane change requests
                if rl_action.get('lane_change_request'):
                    direction = rl_action['lane_change_direction']
                    self.execute_lane_change(direction)
                
                return self.run_step(target_speed=target_speed)
        ```

**Related**: 

- [VehicleManager](vehicle_manager.md) 
- [LocalizationManager](localization_manager.md) 
- [PerceptionManager](perception_manager.md)