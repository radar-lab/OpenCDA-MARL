# LocalizationManager API

`LocalizationManager` provides accurate vehicle positioning through sensor fusion of GNSS, IMU, and Kalman filtering for noise-resistant localization.

| Component | Purpose | Configuration |
|-----------|---------|---------------|
| `gnss_sensor` | GPS positioning | Noise levels, update rate |
| `imu_sensor` | Acceleration/rotation | Sensor calibration |
| `kalman_filter` | State estimation | Process/measurement noise |
| `update()` | Position estimation | Real-time sensor fusion |

## Sensor Configuration

**What it does**: Configures GNSS and IMU sensors with realistic noise models

=== "GNSS Setup"
    ```python
    def __init__(self, vehicle, config_yaml, carla_map):
        # GNSS configuration
        gnss_config = config_yaml['gnss']
        self.gnss_noise = gnss_config.get('noise_alt_stddev', 0.1)
        self.gnss_bias = gnss_config.get('noise_lat_bias', 0.0)
        
        # Create GNSS sensor
        self.gnss_sensor = GnssSensor(
            vehicle=vehicle,
            noise_alt_stddev=self.gnss_noise,
            noise_lat_bias=self.gnss_bias,
            noise_lon_bias=gnss_config.get('noise_lon_bias', 0.0),
            noise_seed=gnss_config.get('noise_seed', 0)
        )
    ```

=== "IMU Setup"
    ```python
    # IMU sensor for acceleration and angular velocity
    imu_config = config_yaml.get('imu', {})
    self.imu_sensor = ImuSensor(
        vehicle=vehicle,
        noise_accel_stddev_x=imu_config.get('noise_accel_stddev_x', 0.0),
        noise_accel_stddev_y=imu_config.get('noise_accel_stddev_y', 0.0),
        noise_accel_stddev_z=imu_config.get('noise_accel_stddev_z', 0.0),
        noise_gyro_stddev_x=imu_config.get('noise_gyro_stddev_x', 0.0),
        noise_gyro_stddev_y=imu_config.get('noise_gyro_stddev_y', 0.0),
        noise_gyro_stddev_z=imu_config.get('noise_gyro_stddev_z', 0.0)
    )
    ```

=== "YAML Configuration"
    ```yaml
    sensing:
      localization:
        activate: true
        dt: 0.1  # Kalman filter timestep
        gnss:
          noise_alt_stddev: 0.1    # Altitude noise (meters)
          noise_lat_bias: 0.0      # Latitude bias
          noise_lon_bias: 0.0      # Longitude bias
          noise_seed: 0            # Random seed
        imu:
          noise_accel_stddev_x: 0.0  # X acceleration noise
          noise_accel_stddev_y: 0.0  # Y acceleration noise
          noise_accel_stddev_z: 0.0  # Z acceleration noise
    ```

**Benefits**: Realistic sensor noise simulation, configurable uncertainty levels

## Kalman Filtering

**What it does**: Provides optimal state estimation for position and velocity with noise handling

=== "Filter Initialization"
    ```python
    def initialize_kalman_filter(self):
        """
        Initialize extended Kalman filter for localization
        State vector: [x, y, yaw, x_dot, y_dot, yaw_dot]
        """
        # State transition model (constant velocity)
        dt = self.dt
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])
        
        # Process noise covariance
        self.Q = np.eye(6) * 0.1
        
        # Measurement noise covariance (GNSS + IMU)
        self.R = np.diag([
            self.gnss_noise**2,  # x position
            self.gnss_noise**2,  # y position
            0.01,                # yaw
            0.1,                 # x velocity
            0.1,                 # y velocity
            0.01                 # yaw rate
        ])
    ```

=== "State Prediction"
    ```python
    def predict_state(self):
        """
        Prediction step of Kalman filter
        """
        # Predict state
        self.state_pred = self.F @ self.state
        
        # Predict covariance
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return self.state_pred
    ```

=== "Measurement Update"
    ```python
    def update_measurement(self, gnss_data, imu_data):
        """
        Update step with GNSS and IMU measurements
        """
        # Construct measurement vector
        z = np.array([
            gnss_data.latitude,   # Converted to x
            gnss_data.longitude,  # Converted to y
            imu_data.compass,     # Yaw angle
            self.vehicle.get_velocity().x,  # x velocity
            self.vehicle.get_velocity().y,  # y velocity
            imu_data.gyroscope.z  # Yaw rate
        ])
        
        # Measurement residual
        y = z - self.H @ self.state_pred
        
        # Kalman gain
        S = self.H @ self.P_pred @ self.H.T + self.R
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.state = self.state_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P_pred
    ```

**Features**: Extended Kalman filter, multi-sensor fusion, uncertainty quantification

## Position Estimation

**What it does**: Provides real-time position and velocity estimates with confidence intervals

=== "Main Update Loop"
    ```python
    def update(self):
        """
        Main localization update cycle
        """
        # Get sensor measurements
        gnss_data = self.gnss_sensor.get_data()
        imu_data = self.imu_sensor.get_data()
        
        if gnss_data is not None and imu_data is not None:
            # Prediction step
            self.predict_state()
            
            # Measurement update
            self.update_measurement(gnss_data, imu_data)
            
            # Convert to CARLA coordinates
            self.ego_pos = self.convert_to_carla_location(self.state[:3])
            self.ego_speed = carla.Vector3D(
                x=self.state[3],
                y=self.state[4],
                z=0.0
            )
    ```

=== "Position Access"
    ```python
    def get_ego_pos(self):
        """
        Get current estimated position
        
        Returns:
            carla.Transform: Vehicle position with uncertainty
        """
        if hasattr(self, 'ego_pos'):
            return self.ego_pos
        else:
            # Fallback to ground truth if no estimate available
            return self.vehicle.get_transform()
    
    def get_ego_speed(self):
        """
        Get current estimated velocity
        
        Returns:
            carla.Vector3D: Vehicle velocity
        """
        if hasattr(self, 'ego_speed'):
            return self.ego_speed
        else:
            return self.vehicle.get_velocity()
    ```

=== "Coordinate Conversion"
    ```python
    def convert_to_carla_location(self, state_pos):
        """
        Convert from filter coordinates to CARLA world coordinates
        """
        transform = carla.Transform(
            location=carla.Location(
                x=float(state_pos[0]),
                y=float(state_pos[1]),
                z=self.vehicle.get_location().z
            ),
            rotation=carla.Rotation(
                yaw=float(np.degrees(state_pos[2]))
            )
        )
        return transform
    ```

## Integration Examples

!!! example "Usage Patterns"
    Common ways to use LocalizationManager:

    === "Basic Localization"
        ```python
        # Create localization manager
        localization_manager = LocalizationManager(
            vehicle=carla_vehicle,
            config_yaml=config['sensing']['localization'],
            carla_map=carla_map
        )
        
        # In simulation loop
        while True:
            # Update localization estimates
            localization_manager.update()
            
            # Get current position and velocity
            ego_pos = localization_manager.get_ego_pos()
            ego_speed = localization_manager.get_ego_speed()
            
            # Use in other subsystems
            perception_manager.detect(ego_pos)
            behavior_agent.update_information(ego_pos, ego_speed, objects)
        ```

    === "Noise Sensitivity Analysis"
        ```python
        # Test different noise levels
        noise_levels = [0.0, 0.1, 0.5, 1.0]
        
        for noise in noise_levels:
            config['sensing']['localization']['gnss']['noise_alt_stddev'] = noise
            
            localization_manager = LocalizationManager(vehicle, config, carla_map)
            
            # Run scenario and collect position accuracy metrics
            position_errors = []
            for step in range(1000):
                localization_manager.update()
                estimated_pos = localization_manager.get_ego_pos()
                ground_truth = vehicle.get_transform()
                error = calculate_position_error(estimated_pos, ground_truth)
                position_errors.append(error)
        ```

    === "Custom Sensor Fusion"
        ```python
        # Extend with additional sensors
        class ExtendedLocalizationManager(LocalizationManager):
            def __init__(self, vehicle, config_yaml, carla_map):
                super().__init__(vehicle, config_yaml, carla_map)
                
                # Add wheel odometry
                self.wheel_odometry = WheelOdometrySensor(vehicle)
                
            def update_measurement(self, gnss_data, imu_data):
                # Include wheel odometry in measurement
                odom_data = self.wheel_odometry.get_data()
                
                # Extended measurement vector
                z_extended = np.append(z_standard, [
                    odom_data.velocity_x,
                    odom_data.velocity_y
                ])
                
                # Update with extended measurements
                super().update_measurement_extended(z_extended)
        ```

**Related**: 

- [VehicleManager](vehicle_manager.md) 
- [PerceptionManager](perception_manager.md) 
- [BehaviorAgent](behavior_agent.md)