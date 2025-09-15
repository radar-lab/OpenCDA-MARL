# Vehicle Adapter API

!!! info "Implementation Status"
    The Vehicle Adapter is **fully implemented** and provides the core bridge between OpenCDA's VehicleManager and MARL multi-agent control systems.

The MARLVehicleAdapter is the central component that enables individual vehicles to be controlled by different agent types while preserving OpenCDA's proven autonomous driving capabilities. It supports multiple controller types and provides seamless integration for reinforcement learning agents.

```text
MARLVehicleAdapter
├── Controller Management  # Multiple controller types (behavior, vanilla, rule_based, rl_agent)
├── External Control       # Target speed control for RL agents
├── Observation System     # Vehicle state extraction for RL
├── Collision Handling     # Instant destruction for clean experiments
└── OpenCDA Integration    # Wraps VehicleManager functionality
```

## Core Classes

=== "MARLVehicleAdapter"

    The main adapter class that bridges OpenCDA VehicleManager with multi-agent control.

    ```python
    class MARLVehicleAdapter:
        """
        Adapter that bridges OpenCDA VehicleManager with MARL agents.
        
        Provides unified interface for:
        - Multiple controller types (autonomous vs external control)
        - RL observation extraction from vehicle state
        - External action integration (target speed control)
        - Collision detection with instant destruction
        - OpenCDA compatibility preservation
        """
    ```

=== "Constructor"

    ```python
    def __init__(
        self,
        blueprint: carla.ActorBlueprint,
        transform: carla.Transform, 
        world: carla.World,
        config: Dict,
        cav_world: CavWorld,
        destination: carla.Transform = None,
        marl_id: str = None
    ):
        """
        Initialize MARL vehicle adapter.
        
        Parameters
        ----------
        blueprint : carla.ActorBlueprint
            Vehicle blueprint for CARLA spawning
        transform : carla.Transform
            Initial spawn transform
        world : carla.World
            CARLA world instance
        config : dict
            Vehicle configuration
        cav_world : CavWorld
            CAV world for coordination
        destination : carla.Transform, optional
            Vehicle destination (for route planning)
        marl_id : str, optional
            Unique MARL agent identifier
        """
    ```

### Controller Types

The adapter supports multiple controller types for different use cases:

=== "Available Controllers"

    ```python
    CONTROLLER_TYPES = {
        'behavior': "OpenCDA BehaviorAgent (standard autonomous driving)", 
        'vanilla': "VanillaAgent (enhanced collision avoidance)",
        'rule_based': "RuleBasedAgent (3-stage intersection rules)",
        'marl': "MARL-controlled agent (PPO/SAC/TD3 - Planned)"
    }
    ```

=== "Controller Configuration"

    ```yaml
    # Set controller type in configuration
    agents:
      vehicle:
        controller: "rule_based"  # Choose controller type
        
      # Controller-specific settings
      rule_based:
        junction_approach_distance: 70.0
        cautious_speed: 20.0
        time_headway: 2.0
        
      vanilla:
        intersection_safety_multiplier: 2.0
        multi_vehicle_ttc: true
    ```

### Key Methods

=== "Vehicle Control"

    ```python
    def run_step(self, target_speed: Optional[float] = None) -> carla.VehicleControl:
        """
        Execute one control step.
        
        Parameters
        ----------
        target_speed : float, optional
            Target speed for external control (km/h). If provided, uses external
            speed control instead of autonomous control.
            
        Returns
        -------
        control : carla.VehicleControl
            Vehicle control command (throttle, brake, steering)
            
        Example
        -------
        # Autonomous control
        control = adapter.run_step()
        
        # External speed control (for RL)
        control = adapter.run_step(target_speed=25.0)  # 25 km/h
        """
    ```

=== "Observation Extraction"

    ```python
    def get_observation(self) -> Dict[str, Any]:
        """
        Get vehicle observation for RL agents.
        
        Returns
        -------
        observation : dict
            Vehicle state observation including:
            - position: Vehicle world coordinates
            - velocity: Vehicle velocity vector
            - rotation: Vehicle orientation (yaw, pitch, roll)
            - speed: Speed magnitude in m/s
            - acceleration: Current acceleration
            - nearby_vehicles: Surrounding vehicle information (planned)
            
        Example
        -------
        obs = adapter.get_observation()
        print(f"Position: {obs['position']}")
        print(f"Speed: {obs['speed']:.1f} m/s")
        """
    ```

=== "Collision Handling"

    ```python
    def destroy(self) -> None:
        """
        Destroy vehicle actor and clean up resources.
        
        Called automatically when collision is detected and instant_destruction
        is enabled. Performs complete cleanup of:
        - CARLA vehicle actor
        - OpenCDA VehicleManager
        - Collision sensors
        - Internal state
        
        Note
        ----
        This method is typically called automatically by the collision sensor
        callback, not manually by user code.
        """
    ```

=== "State Management"

    ```python
    @property
    def is_alive(self) -> bool:
        """Check if vehicle is still active (not destroyed)."""
        
    @property 
    def vehicle_id(self) -> int:
        """Get CARLA vehicle actor ID."""
        
    @property
    def marl_id(self) -> str:
        """Get unique MARL agent identifier."""
        
    def update_destination(self, destination: carla.Transform) -> None:
        """Update vehicle destination for route planning."""
    ```

## Usage Examples

=== "Basic Adapter Setup"

    ```python
    from opencda_marl.core.adapters.vehicle_adapter import MARLVehicleAdapter
    from opencda.core.common.cav_world import CavWorld
    import carla
    
    # Connect to CARLA
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    
    # Get vehicle blueprint
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    
    # Define spawn location
    spawn_point = carla.Transform(
        carla.Location(x=10.0, y=20.0, z=0.5),
        carla.Rotation(yaw=0.0)
    )
    
    # Create CAV world
    cav_world = CavWorld(apply_ml=False)
    
    # Create adapter
    adapter = MARLVehicleAdapter(
        blueprint=vehicle_bp,
        transform=spawn_point,
        world=world,
        config=config,
        cav_world=cav_world,
        marl_id="agent_001"
    )
    ```

=== "Autonomous Driving Mode"

    ```python
    # Run vehicle autonomously using OpenCDA
    for step in range(100):
        # Autonomous control (no external actions)
        control = adapter.run_step()
        
        # Vehicle automatically follows OpenCDA planning
        print(f"Step {step}: Vehicle running autonomously")
        
        # Check if vehicle is still active
        if not adapter.is_alive:
            print("Vehicle destroyed (collision detected)")
            break
    ```

=== "External Speed Control (RL)"

    ```python
    # Control vehicle with target speed (for RL agents)
    target_speeds = [20.0, 25.0, 30.0, 15.0]  # km/h
    
    for speed in target_speeds:
        # External speed control
        control = adapter.run_step(target_speed=speed)
        
        # Get current vehicle state
        obs = adapter.get_observation()
        current_speed = obs['speed'] * 3.6  # Convert to km/h
        
        print(f"Target: {speed:.1f} km/h, Current: {current_speed:.1f} km/h")
        
        # Wait for simulation step
        world.tick()
    ```

=== "RL Training Integration"

    ```python
    # Integration with RL training loop
    def collect_experience(adapter, rl_agent):
        """Collect experience for RL training."""
        
        # Get observation
        observation = adapter.get_observation()
        
        # Get action from RL agent
        action = rl_agent.get_action(observation)
        
        # Convert RL action to target speed
        target_speed = action * 30.0  # Scale to 0-30 km/h
        
        # Execute action
        control = adapter.run_step(target_speed=target_speed)
        
        # Get next observation
        next_observation = adapter.get_observation()
        
        # Calculate reward (example)
        reward = calculate_reward(observation, action, next_observation)
        
        # Check termination
        done = not adapter.is_alive
        
        return {
            'observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_observation,
            'done': done
        }
    ```

=== "Multi-Agent Coordination"

    ```python
    # Manage multiple vehicle adapters
    adapters = {}
    
    # Create multiple adapters
    for i in range(4):
        adapter = MARLVehicleAdapter(
            blueprint=vehicle_bp,
            transform=spawn_points[i],
            world=world,
            config=config,
            cav_world=cav_world,
            marl_id=f"agent_{i:03d}"
        )
        adapters[f"agent_{i:03d}"] = adapter
    
    # Coordinate multiple agents
    for step in range(500):
        # Get actions for all agents
        actions = {}
        for agent_id, adapter in adapters.items():
            if adapter.is_alive:
                observation = adapter.get_observation()
                actions[agent_id] = get_agent_action(observation)
        
        # Execute actions
        for agent_id, action in actions.items():
            adapter = adapters[agent_id]
            if adapter.is_alive:
                target_speed = action * 35.0  # Scale action
                adapter.run_step(target_speed=target_speed)
        
        # Remove destroyed vehicles
        adapters = {
            agent_id: adapter 
            for agent_id, adapter in adapters.items() 
            if adapter.is_alive
        }
        
        if not adapters:
            print("All vehicles destroyed")
            break
    ```

## Configuration Integration

=== "Basic Configuration"

    ```yaml
    # Vehicle adapter configuration
    agents:
      vehicle:
        controller: "vanilla"           # Controller type
        safety_manager:
          instant_destruction: true    # Enable instant destruction on collision
          collision_sensor:
            col_thresh: 1              # Collision detection threshold
            history_size: 30           # Collision history buffer
            
        # OpenCDA components (preserved)
        sensing:
          perception:
            activate: false            # Use server-side perception
        localization:
          activate: false              # Use server-side localization
    ```

=== "Controller-Specific Settings"

    ```yaml
    agents:
      # VanillaAgent configuration
      vanilla:
        intersection_safety_multiplier: 2.0  # Safety boost at intersections
        multi_vehicle_ttc: true             # Track multiple vehicle TTC
        max_vehicles_to_track: 5            # TTC tracking limit
        lateral_check_distance: 15.0        # Lateral conflict detection
        
      # RuleBasedAgent configuration  
      rule_based:
        junction_approach_distance: 70.0    # Junction detection distance
        junction_conflict_distance: 50.0    # Conflict check distance
        cautious_speed: 20.0                # Speed when conflict detected
        time_headway: 2.0                   # Car following time headway
        
      # BehaviorAgent configuration (OpenCDA standard)
      behavior:
        max_speed: 35                       # Maximum speed (km/h)
        safety_time: 3                      # TTC safety threshold
        ignore_traffic_light: false         # Respect traffic lights
    ```

## Performance Characteristics

| Aspect | OpenCDA Mode | External Control Mode |
|--------|-------------|----------------------|
| **Control Authority** | Full autonomous | Speed-only external |
| **Planning** | OpenCDA BehaviorAgent | Simplified speed control |
| **Safety** | Full safety manager | Collision detection only |
| **Performance** | Production-ready | Suitable for RL training |
| **Flexibility** | Limited to OpenCDA | Adaptable to any RL agent |

## Integration Points

=== "MARLAgentManager Integration"

    ```python
    # Used by MARLAgentManager for multi-agent coordination
    class MARLAgentManager:
        def step(self, external_actions=None):
            results = {}
            
            for agent_id, adapter in self.adapters.items():
                if adapter.is_alive:
                    # Apply external action or autonomous control
                    if external_actions and agent_id in external_actions:
                        control = adapter.run_step(external_actions[agent_id])
                    else:
                        control = adapter.run_step()
                    
                    results[agent_id] = {
                        'control': control,
                        'observation': adapter.get_observation()
                    }
            
            return results
    ```

=== "MARLEnvironment Integration"

    ```python
    # Used by MARLEnvironment for RL training
    class MARLEnvironment:
        def _get_observations(self):
            observations = {}
            
            for agent_id, adapter in self.scenario_manager.agents.items():
                if adapter.is_alive:
                    observations[agent_id] = adapter.get_observation()
                    
            return observations
            
        def step(self, actions):
            # Apply actions through adapters
            for agent_id, action in actions.items():
                adapter = self.scenario_manager.agents[agent_id]
                if adapter.is_alive:
                    adapter.run_step(target_speed=action)
    ```

---

**Location**: `opencda_marl/core/adapters/vehicle_adapter.py`