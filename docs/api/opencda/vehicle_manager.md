# VehicleManager API

`VehicleManager` coordinates individual CAVs by orchestrating all subsystems: perception, localization, planning, control, and V2X communication.

| Component            | Type                  | Purpose                              |
| -------------------- | --------------------- | ------------------------------------ |
| `vid`                | `str`                 | Unique vehicle identifier (UUID)     |
| `perception_manager` | `PerceptionManager`   | Sensor fusion and ML detection       |
| `localizer`          | `LocalizationManager` | Position tracking with Kalman filter |
| `agent`              | `BehaviorAgent`       | Path planning and decision making    |
| `controller`         | `ControlManager`      | PID-based vehicle control            |
| `v2x_manager`        | `V2XManager`          | Inter-vehicle communication          |

## Initialization

**What it does**: Creates and configures all vehicle subsystems from YAML configuration

=== "Constructor"
    ```python
    def __init__(self, vehicle, config_yaml, application, 
                 carla_map, cav_world, current_time='', data_dumping=False):
        
        # Unique vehicle identifier
        self.vid = str(uuid.uuid1())
        self.vehicle = vehicle
        self.carla_map = carla_map
        
        # Initialize subsystem managers
        self.v2x_manager = V2XManager(cav_world, config_yaml['v2x'], self.vid)
        self.localizer = LocalizationManager(vehicle, config_yaml['sensing']['localization'], carla_map)
        self.perception_manager = PerceptionManager(
            vehicle, config_yaml['sensing']['perception'], cav_world.ml_manager, data_dumping)
        self.agent = BehaviorAgent(vehicle, carla_map, config_yaml['behavior'])
        self.controller = ControlManager(config_yaml['controller'])
        
        # Register with CAV world
        cav_world.update_vehicle_manager(self)
    ```

=== "Configuration"
    ```yaml
    # YAML configuration structure
    vehicle:
      sensing:
        perception: { activate: true, camera: {...}, lidar: {...} }
        localization: { activate: true, dt: 0.1, gnss: {...} }
      behavior:
        local_planner: { buffer_size: 10, trajectory_update_freq: 10 }
        behavior_agent: { behavior: normal, ignore_traffic_light: false }
      controller:
        type: pid
        args: { k_p: 1.0, k_i: 0.05, k_d: 0.1 }
      v2x:
        communication_range: 35.0
        loc_noise: 0.0
    ```

**Benefits**: Modular architecture, configuration-driven setup, automatic registration

## Core Methods

**What it does**: Orchestrates subsystem updates and vehicle control in simulation loop

=== "Information Update"
    ```python
    def update_info(self):
        # 1. Update localization (position, velocity)
        self.localizer.update()
        ego_pos = self.localizer.get_ego_pos()
        ego_speed = self.localizer.get_ego_speed()
        
        # 2. Update perception (object detection)
        detected_objects = self.perception_manager.detect(ego_pos)
        
        # 3. Update V2X communication
        self.v2x_manager.update_info(ego_pos, ego_speed)
        
        # 4. Update agent with new information
        self.agent.update_information(ego_pos, ego_speed, detected_objects)
        
        # 5. Update controller state
        self.controller.update_info(ego_pos, ego_speed)
    ```

=== "Control Execution"
    ```python
    def run_step(self, target_speed=None):
        # Get planned trajectory from behavior agent
        target_speed, target_waypoint = self.agent.run_step(target_speed)
        
        # Generate control command
        control = self.controller.run_step(target_speed, target_waypoint)
        
        return control
    ```

=== "Navigation & Cleanup"
    ```python
    # Set navigation destination
    def set_destination(self, start_location, destination, clean=True):
        self.agent.set_destination(start_location, destination, clean)
    
    # Safe destruction
    def destroy(self):
        self.perception_manager.destroy()
        self.localizer.destroy()
        if self.vehicle:
            self.vehicle.destroy()
    ```

**Execution Order**: Update info → Run step → Apply control (critical for proper coordination)

## Integration Examples

!!! example "Usage Patterns"
    Common integration patterns for VehicleManager:

=== "Basic Simulation"
    ```python
    # Standard simulation loop
    while True:
        # Update vehicle information
        vehicle_manager.update_info()
        
        # Get control commands
        control = vehicle_manager.run_step()
        
        # Apply to CARLA vehicle
        vehicle_manager.vehicle.apply_control(control)
        
        # Check if destination reached
        if vehicle_manager.agent.done():
            break
    ```

=== "Platoon Coordination"
    ```python
    # VehicleManager in platoon formation
    if vehicle_manager.v2x_manager.in_platoon():
        # Get platoon commands
        platoon_info = vehicle_manager.v2x_manager.get_platoon_info()
        target_speed = platoon_info.leader_speed
        
        # Execute platoon control
        control = vehicle_manager.run_step(target_speed=target_speed)
    else:
        # Independent operation
        control = vehicle_manager.run_step()
    ```

**Design Features**: Modular subsystems, configuration-driven behavior, extensible architecture

**Related**: 

- [CavWorld](cav_world.md) 
- [PerceptionManager](perception_manager.md) 
- [BehaviorAgent](behavior_agent.md) 
- [V2XManager](v2x_manager.md)