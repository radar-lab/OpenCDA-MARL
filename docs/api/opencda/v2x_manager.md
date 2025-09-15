# V2XManager API

`V2XManager` enables Vehicle-to-Everything (V2X) communication for cooperative autonomous driving through information sharing and coordination between CAVs.

| Component | Purpose | Configuration |
|-----------|---------|---------------|
| `communication_range` | V2X range limit | Distance-based filtering |
| `message_passing` | Info sharing | Position, velocity, intentions |
| `noise_simulation` | Realistic errors | Position noise, timing delays |
| `platoon_support` | Coordination | Leader-follower communication |

## Communication Setup

**What it does**: Configures V2X communication with realistic range limits and noise models

=== "Initialization"
    ```python
    def __init__(self, cav_world, config_yaml, vehicle_id):
        """
        Initialize V2X communication manager
        
        Args:
            cav_world: Global CAV registry
            config_yaml: V2X configuration parameters
            vehicle_id: Unique vehicle identifier
        """
        # Communication parameters
        self.communication_range = config_yaml.get('communication_range', 35.0)  # meters
        self.loc_noise = config_yaml.get('loc_noise', 0.0)  # position noise
        self.time_delay = config_yaml.get('time_delay', 0.1)  # latency (seconds)
        
        # Vehicle identification
        self.vid = vehicle_id
        self.cav_world = cav_world
        
        # Communication state
        self.ego_pos = None
        self.ego_speed = None
        self.neighbors = {}  # Nearby CAVs
        self.platoon_id = None
    ```

=== "YAML Configuration"
    ```yaml
    v2x:
      communication_range: 35.0    # V2X range in meters
      loc_noise: 0.1              # Position uncertainty (meters)
      time_delay: 0.1             # Communication latency (seconds)
      message_loss_rate: 0.05     # Packet loss probability
      bandwidth_limit: 10.0       # Mbps capacity limit
      
      # Platoon-specific settings
      platoon:
        leader_range: 50.0        # Extended range for platoon leader
        update_frequency: 10      # Hz for platoon coordination
    ```

=== "Noise Models"
    ```python
    def apply_communication_noise(self, position, velocity):
        """
        Simulate realistic V2X communication errors
        """
        # Position noise (GPS/localization uncertainty)
        if self.loc_noise > 0:
            noise_x = np.random.normal(0, self.loc_noise)
            noise_y = np.random.normal(0, self.loc_noise)
            position = carla.Location(
                x=position.x + noise_x,
                y=position.y + noise_y,
                z=position.z
            )
        
        # Velocity noise
        if self.loc_noise > 0:
            vel_noise = np.random.normal(0, self.loc_noise * 0.1)
            velocity = carla.Vector3D(
                x=velocity.x + vel_noise,
                y=velocity.y + vel_noise,
                z=velocity.z
            )
        
        return position, velocity
    ```

**Benefits**: Realistic communication simulation, configurable range and noise

## Message Broadcasting

**What it does**: Handles information sharing between vehicles within communication range

=== "State Broadcasting"
    ```python
    def update_info(self, ego_pos, ego_speed):
        """
        Update vehicle state and broadcast to neighbors
        
        Args:
            ego_pos: carla.Transform current position
            ego_speed: carla.Vector3D current velocity
        """
        # Apply communication noise
        noisy_pos, noisy_speed = self.apply_communication_noise(
            ego_pos.location, ego_speed
        )
        
        # Update local state
        self.ego_pos = carla.Transform(
            location=noisy_pos,
            rotation=ego_pos.rotation
        )
        self.ego_speed = noisy_speed
        
        # Broadcast to nearby vehicles
        self.broadcast_state()
        
        # Update neighbor information
        self.update_neighbors()
    ```

=== "Neighbor Discovery"
    ```python
    def search(self):
        """
        Discover nearby CAVs within communication range
        
        Returns:
            dict: {vehicle_id: distance} for neighbors
        """
        neighbors = {}
        
        # Get all vehicle managers from CAV world
        all_vehicles = self.cav_world.get_all_vehicle_managers()
        
        for vid, vehicle_manager in all_vehicles.items():
            if vid == self.vid:
                continue  # Skip self
            
            # Calculate distance
            other_pos = vehicle_manager.localizer.get_ego_pos()
            distance = self._calculate_distance(self.ego_pos, other_pos)
            
            # Check if within communication range
            if distance <= self.communication_range:
                neighbors[vid] = {
                    'distance': distance,
                    'position': other_pos,
                    'velocity': vehicle_manager.localizer.get_ego_speed(),
                    'last_update': time.time()
                }
        
        self.neighbors = neighbors
        return neighbors
    ```

=== "Message Types"
    ```python
    class V2XMessage:
        def __init__(self, sender_id, message_type, content, timestamp):
            self.sender_id = sender_id
            self.message_type = message_type  # 'position', 'intention', 'emergency'
            self.content = content
            self.timestamp = timestamp
            self.received_time = None
    
    def broadcast_state(self):
        """Broadcast current vehicle state"""
        message = V2XMessage(
            sender_id=self.vid,
            message_type='position',
            content={
                'position': self.ego_pos,
                'velocity': self.ego_speed,
                'heading': self.ego_pos.rotation.yaw,
                'acceleration': self.get_acceleration()
            },
            timestamp=time.time()
        )
        
        # Send to all neighbors
        for neighbor_id in self.neighbors.keys():
            self.send_message(neighbor_id, message)
    
    def broadcast_intention(self, intention):
        """Broadcast driving intention (lane change, merging, etc.)"""
        message = V2XMessage(
            sender_id=self.vid,
            message_type='intention',
            content={
                'intention': intention,  # 'lane_change_left', 'merging', etc.
                'target_location': self.get_target_location(),
                'confidence': 0.9
            },
            timestamp=time.time()
        )
        
        self.broadcast_to_neighbors(message)
    ```

**Features**: State broadcasting, neighbor discovery, intention sharing

## Platoon Communication

**What it does**: Specialized communication protocols for platoon coordination

=== "Platoon Formation"
    ```python
    def join_platoon(self, platoon_id, role='follower'):
        """
        Join a vehicle platoon
        
        Args:
            platoon_id: str unique platoon identifier
            role: 'leader' or 'follower'
        """
        self.platoon_id = platoon_id
        self.platoon_role = role
        
        # Extended communication range for platoons
        if role == 'leader':
            self.communication_range = max(
                self.communication_range, 50.0
            )
        
        # Register with platoon
        platoon_message = V2XMessage(
            sender_id=self.vid,
            message_type='platoon_join',
            content={
                'platoon_id': platoon_id,
                'role': role,
                'capabilities': self.get_vehicle_capabilities()
            },
            timestamp=time.time()
        )
        
        self.broadcast_to_platoon(platoon_message)
    
    def in_platoon(self):
        """Check if vehicle is in an active platoon"""
        return self.platoon_id is not None
    ```

=== "Leader-Follower Protocol"
    ```python
    def update_platoon_coordination(self):
        """
        Handle platoon-specific communication
        """
        if not self.in_platoon():
            return
        
        if self.platoon_role == 'leader':
            # Broadcast platoon commands
            platoon_command = {
                'target_speed': self.get_target_speed(),
                'formation': self.get_desired_formation(),
                'next_maneuver': self.get_planned_maneuver()
            }
            
            message = V2XMessage(
                sender_id=self.vid,
                message_type='platoon_command',
                content=platoon_command,
                timestamp=time.time()
            )
            
            self.broadcast_to_platoon(message)
        
        else:  # follower
            # Send status to leader
            follower_status = {
                'following_distance': self.get_following_distance(),
                'readiness': self.assess_maneuver_readiness(),
                'health_status': self.get_system_health()
            }
            
            message = V2XMessage(
                sender_id=self.vid,
                message_type='platoon_status',
                content=follower_status,
                timestamp=time.time()
            )
            
            self.send_to_platoon_leader(message)
    ```

=== "Coordination Examples"
    ```python
    def execute_coordinated_lane_change(self, direction):
        """
        Coordinate lane change across entire platoon
        """
        if self.platoon_role == 'leader':
            # Initiate coordinated maneuver
            maneuver_plan = {
                'maneuver_type': 'lane_change',
                'direction': direction,
                'execution_time': time.time() + 3.0,  # 3 second prep
                'sequence': self.calculate_lane_change_sequence()
            }
            
            message = V2XMessage(
                sender_id=self.vid,
                message_type='coordinated_maneuver',
                content=maneuver_plan,
                timestamp=time.time()
            )
            
            self.broadcast_to_platoon(message)
            return True
        
        return False  # Only leader can initiate
    ```

**Features**: Leader-follower protocols, coordinated maneuvers, formation control

## Integration Examples

!!! example "Usage Patterns"
    Common ways to use V2XManager:

    === "Basic V2X Communication"
        ```python
        # Create V2X manager
        v2x_manager = V2XManager(
            cav_world=cav_world,
            config_yaml=config['v2x'],
            vehicle_id=vehicle_id
        )
        
        # In simulation loop
        while True:
            # Update vehicle state
            v2x_manager.update_info(ego_pos, ego_speed)
            
            # Discover neighbors
            neighbors = v2x_manager.search()
            
            # Use neighbor information for cooperative planning
            if neighbors:
                cooperative_plan = plan_with_neighbors(neighbors)
            else:
                cooperative_plan = plan_independently()
        ```

    === "Platoon Coordination"
        ```python
        # Join platoon as follower
        v2x_manager.join_platoon('platoon_001', role='follower')
        
        # Coordination loop
        while v2x_manager.in_platoon():
            # Update platoon communication
            v2x_manager.update_platoon_coordination()
            
            # Get platoon commands
            if v2x_manager.has_platoon_command():
                command = v2x_manager.get_latest_platoon_command()
                target_speed = command['target_speed']
                formation = command['formation']
                
                # Execute platoon behavior
                behavior_agent.execute_platoon_behavior(target_speed, formation)
        ```

    === "Cooperative Perception"
        ```python
        # Share perception data via V2X
        class CooperativePerceptionManager(PerceptionManager):
            def __init__(self, vehicle, config, ml_manager, v2x_manager):
                super().__init__(vehicle, config, ml_manager)
                self.v2x_manager = v2x_manager
            
            def detect_with_cooperation(self, ego_pos):
                # Local detection
                local_objects = super().detect(ego_pos)
                
                # Get shared detections from neighbors
                neighbors = self.v2x_manager.get_neighbors()
                shared_objects = []
                
                for neighbor_id, neighbor_info in neighbors.items():
                    if neighbor_info.get('shared_detections'):
                        shared_objects.extend(neighbor_info['shared_detections'])
                
                # Fuse local and shared detections
                fused_objects = self.fuse_detections(local_objects, shared_objects)
                
                # Share local detections
                self.v2x_manager.broadcast_detections(local_objects)
                
                return fused_objects
        ```

**Related**: 

- [VehicleManager](vehicle_manager.md) 
- [CavWorld](cav_world.md) 
- [BehaviorAgent](behavior_agent.md)