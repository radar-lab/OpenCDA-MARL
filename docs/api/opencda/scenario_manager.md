# ScenarioManager API

`ScenarioManager` handles simulation setup, CARLA world configuration, and orchestrates vehicle spawning for OpenCDA scenarios.

| Component                  | Purpose                 | Configuration            |
| -------------------------- | ----------------------- | ------------------------ |
| `create_vehicle_manager()` | Individual CAV spawning | YAML vehicle configs     |
| `create_platoon_manager()` | Platoon formation       | Platoon-specific configs |
| `create_traffic_carla()`   | Background traffic      | CARLA TrafficManager     |
| `tick()`                   | Simulation stepping     | World synchronization    |

## Vehicle Creation

**What it does**: Spawns and configures CAVs based on YAML scenario definitions

=== "Individual CAVs"
    ```python
    def create_vehicle_manager(self, application, map_helper=None, data_dump=False):
        """
        Create individual Connected Automated Vehicles
        
        Args:
            application: List of applications ['platooning', 'cooperative_perception']
            map_helper: HD map helper for navigation
            data_dump: Enable sensor data collection
        
        Returns:
            List[VehicleManager]: Spawned vehicle managers
        """
        vehicle_managers = []
        
        for vehicle_config in self.scenario_params['vehicles']:
            # Spawn CARLA vehicle
            carla_vehicle = self.spawn_vehicle_carla(vehicle_config)
            
            # Create vehicle manager with subsystems
            vm = VehicleManager(
                vehicle=carla_vehicle,
                config_yaml=vehicle_config,
                application=application,
                carla_map=self.carla_map,
                cav_world=self.cav_world,
                data_dumping=data_dump
            )
            
            vehicle_managers.append(vm)
        
        return vehicle_managers
    ```

=== "Platoon Formation"
    ```python
    def create_platoon_manager(self, map_helper=None, data_dump=False):
        """
        Create coordinated vehicle platoons
        
        Returns:
            List[PlatooningManager]: Platoon managers
        """
        platoon_managers = []
        
        for platoon_config in self.scenario_params['platoons']:
            # Create member vehicles
            member_vehicles = []
            for member_config in platoon_config['members']:
                carla_vehicle = self.spawn_vehicle_carla(member_config)
                vm = VehicleManager(
                    vehicle=carla_vehicle,
                    config_yaml=member_config,
                    application=['platooning'],
                    carla_map=self.carla_map,
                    cav_world=self.cav_world
                )
                member_vehicles.append(vm)
            
            # Create platoon coordinator
            platoon_manager = PlatooningManager(
                vehicle_manager_list=member_vehicles,
                config_yaml=platoon_config,
                carla_map=self.carla_map
            )
            
            platoon_managers.append(platoon_manager)
        
        return platoon_managers
    ```

=== "Background Traffic"
    ```python
    def create_traffic_carla(self):
        """
        Spawn background traffic using CARLA TrafficManager
        
        Returns:
            tuple: (TrafficManager, List[carla.Vehicle])
        """
        traffic_config = self.scenario_params.get('traffic', {})
        num_vehicles = traffic_config.get('num_vehicles', 50)
        
        # Get vehicle blueprints
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        
        # Spawn vehicles at random locations
        spawn_points = self.carla_map.get_spawn_points()
        traffic_vehicles = []
        
        for i in range(num_vehicles):
            blueprint = random.choice(vehicle_blueprints)
            spawn_point = random.choice(spawn_points)
            
            vehicle = self.world.spawn_actor(blueprint, spawn_point)
            traffic_vehicles.append(vehicle)
        
        # Configure traffic manager
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.global_percentage_speed_difference(30.0)
        
        # Enable autopilot for all traffic vehicles
        for vehicle in traffic_vehicles:
            vehicle.set_autopilot(True, traffic_manager.get_port())
        
        return traffic_manager, traffic_vehicles
    ```

**Configuration**: Vehicle spawning locations, sensor configurations, platoon formations from YAML

## Simulation Control

**What it does**: Manages CARLA world settings and simulation timing

=== "World Setup"
    ```python
    def __init__(self, scenario_params, apply_ml=False, xodr_path=None, cav_world=None):
        # CARLA connection
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        # Load custom map if provided
        if xodr_path:
            self.world = self.client.load_world(xodr_path)
        else:
            self.world = self.client.get_world()
        
        # Configure world settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        
        # Set weather conditions
        weather = scenario_params.get('weather', 'ClearNoon')
        self.world.set_weather(getattr(carla.WeatherParameters, weather))
    ```

=== "Simulation Loop"
    ```python
    def tick(self):
        """
        Advance simulation by one step
        
        Returns:
            carla.WorldSnapshot: Current world state
        """
        # Synchronous tick
        snapshot = self.world.tick()
        
        # Optional: Data collection
        if self.data_dumping:
            self.collect_simulation_data(snapshot)
        
        return snapshot
    ```

=== "Cleanup"
    ```python
    def destroy(self):
        """
        Clean shutdown of simulation
        """
        # Reset world settings
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        # Destroy spawned actors
        for actor in self.spawned_actors:
            if actor.is_alive:
                actor.destroy()
    ```

**Features**: Synchronous simulation, custom weather, HD map loading, actor lifecycle management

## Integration Examples

!!! example 
    Common ways to use ScenarioManager:

    === "Basic Setup"
        ```python
        # Load scenario configuration
        scenario_params = load_yaml('configs/platoon_scenario.yaml')
        
        # Create CAV world and scenario manager
        cav_world = CavWorld(apply_ml=True)
        scenario_manager = ScenarioManager(
            scenario_params=scenario_params,
            apply_ml=True,
            xodr_path='maps/Town06.xodr',
            cav_world=cav_world
        )
        
        # Spawn vehicles and platoons
        single_cavs = scenario_manager.create_vehicle_manager(['platooning'])
        platoons = scenario_manager.create_platoon_manager()
        traffic_manager, bg_vehicles = scenario_manager.create_traffic_carla()
        ```

    === "Custom MARL Setup"
        ```python
        # MARL-specific scenario setup
        scenario_manager = ScenarioManager(marl_scenario_params, apply_ml=True)
        
        # Create agents for MARL training
        agents = scenario_manager.create_vehicle_manager(['marl_training'])
        
        # Custom reward collection
        for i, agent in enumerate(agents):
            agent.reward_collector = MARLRewardCollector(agent_id=i)
        ```

    === "Data Collection"
        ```python
        # Enable comprehensive data dumping
        scenario_manager = ScenarioManager(
            scenario_params, 
            apply_ml=True, 
            data_dumping=True
        )
        
        vehicles = scenario_manager.create_vehicle_manager(
            application=['platooning'], 
            data_dump=True
        )
        
        # Automatic sensor data recording to disk
        ```

**Integration Points**: CARLA world, HD maps, traffic simulation, data collection

**Related**: 

- [CavWorld](cav_world.md) 
- [VehicleManager](vehicle_manager.md) 
- [PerceptionManager](perception_manager.md)