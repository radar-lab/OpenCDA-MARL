# Traffic Generation in OpenCDA

!!! info "Complete Documentation"
    For comprehensive traffic generation documentation, please visit the [Official OpenCDA Documentation](https://opencda-documentation.readthedocs.io/en/latest/).

OpenCDA supports two main approaches for traffic generation:

1. **CARLA Traffic Manager**: Native CARLA traffic simulation
2. **SUMO Co-simulation**: External traffic simulation with SUMO

## CARLA Traffic Manager

The CARLA Traffic Manager provides built-in traffic simulation capabilities.

### Configuration

Configure traffic in your YAML file under `carla_traffic_manager`:

```yaml
carla_traffic_manager:
  sync_mode: true
  global_distance: 5.0
  global_speed_perc: -100
  auto_lane_change: false
  random: false
  vehicle_list:
    - spawn_position: [100, 100, 0.3, 0, 0, 0]
    - spawn_position: [150, 100, 0.3, 0, 0, 0]
```

### Parameters

- **sync_mode**: Synchronization with CARLA world
- **global_distance**: Minimum distance between vehicles (meters)
- **global_speed_perc**: Speed adjustment (-100 = 60km/h for 30km/h limit)
- **auto_lane_change**: Enable/disable automatic lane changes
- **random**: Randomize vehicle models and colors
- **vehicle_list**: Specific spawn positions and configurations

### Vehicle Spawning Methods

#### Method 1: Explicit Positions

```yaml
vehicle_list:
  - spawn_position: [x, y, z, roll, yaw, pitch]
  - spawn_position: [x2, y2, z2, roll2, yaw2, pitch2]
```

#### Method 2: Range-based Spawning

```yaml
vehicle_list: ~
range:
  - [x_min, x_max, y_min, y_max, x_step, y_step, vehicle_count]
```

## SUMO Traffic Management (Co-simulation)

SUMO provides more sophisticated traffic simulation capabilities.

### Prerequisites

- SUMO installation
- SUMO configuration files (.sumocfg, .net.xml, .rou.xml)

### Configuration

```yaml
sumo:
  port: 8813
  host: "127.0.0.1"
  gui: true
  client_order: 1
  step_length: 0.05
```

### SUMO Files Structure

```
scenario_folder/
├── Town06.sumocfg    # Main configuration
├── Town06.net.xml    # Road network
└── Town06.rou.xml    # Route definitions
```

### Co-simulation Process

1. **Initialize**: Start both CARLA and SUMO
2. **Synchronize**: Coordinate timesteps between simulators
3. **Exchange**: Share vehicle states and commands
4. **Update**: Apply changes in both environments

## Traffic Flow Configuration

### Basic Traffic Flow

```yaml
carla_traffic_manager:
  global_distance: 4.0
  global_speed_perc: -200 # 50km/h
  vehicle_list:
    - spawn_position: [-100, 4.8, 0.3, 0, 0, 0]
    - spawn_position: [-150, 4.8, 0.3, 0, 0, 0]
```

### Mixed Traffic Scenarios

```yaml
carla_traffic_manager:
  vehicle_list:
    - spawn_position: [-100, 4.8, 0.3, 0, 0, 0]
      vehicle_speed_perc: -150 # Individual speed setting
    - spawn_position: [-150, 8.3, 0.3, 0, 0, 0]
      vehicle_speed_perc: -250
```

### Dense Traffic

```yaml
carla_traffic_manager:
  global_distance: 2.0
  vehicle_list: ~
  range:
    - [-500, 500, -20, 20, 25, 4, 50] # 50 vehicles in area
```

## Advanced Features

### Traffic Light Management

```yaml
carla_traffic_manager:
  ignore_lights_percentage: 20 # 20% ignore traffic lights
```

### Dynamic Behavior

- **Aggressive driving**: Lower global_distance, higher speeds
- **Cautious driving**: Higher global_distance, lower speeds
- **Mixed behavior**: Individual vehicle configurations

## Performance Considerations

### CARLA Traffic Manager

- **Pros**: Easy setup, integrated with CARLA
- **Cons**: Limited behavioral complexity
- **Best for**: Simple scenarios, testing CAV algorithms

### SUMO Co-simulation

- **Pros**: Realistic traffic patterns, advanced behaviors
- **Cons**: Complex setup, higher computational cost
- **Best for**: Large-scale scenarios, traffic flow studies

## Best Practices

1. **Start Simple**: Use CARLA Traffic Manager for initial testing
2. **Gradual Complexity**: Add more vehicles incrementally
3. **Performance Monitoring**: Watch simulation frame rate
4. **Scenario Matching**: Choose traffic density based on scenario needs

## Troubleshooting

### Common Issues

- **Low frame rate**: Reduce vehicle count or disable visualizations
- **Collisions**: Increase global_distance
- **Unrealistic behavior**: Switch to SUMO co-simulation
- **Sync issues**: Check step_length consistency

### Debug Tips

- Use CARLA spectator to observe traffic
- Enable traffic manager debug mode
- Monitor vehicle spawn success
- Check for conflicting spawn positions

For more detailed information, refer to the [Official OpenCDA Documentation](https://opencda-documentation.readthedocs.io/en/latest/).
