# OpenCDA-MARL API Overview

OpenCDA-MARL extends the OpenCDA framework with Multi-Agent Reinforcement Learning capabilities for autonomous driving research. This API documentation covers the MARL-specific components and interfaces built on top of the OpenCDA foundation.

!!! success "Implementation Status"
    **Phase 1 Foundation: 95% Complete** - Core architecture implemented and ready for RL agent integration.

### Architecture & Implementation

```text
OpenCDA-MARL/
├── core/
│   ├── coordinator.py          # Central MARL orchestrator
│   ├── adapters/
│   │   └── map_adapter.py      # OpenCDA-MARL map bridge
│   └── world/
│       └── map_manager.py      # MARL map management
├── scenarios/
│   ├── scenario_builder.py    # Factory for scenario creation
│   ├── scenario_manager.py    # Enhanced scenario management
│   └── templates/              # Scenario templates
│       ├── intersection_template.py  # ✅ Implemented
│       ├── highway_template.py       # 📋 Placeholder
│       └── parking_template.py       # 📋 Placeholder
├── envs/
│   ├── base_env.py            # Abstract Gym environment
│   ├── multi_agent_env.py     # Multi-agent Gym wrapper
│   └── single_agent_env.py    # Single-agent placeholder
├── gui/
│   ├── step_controller.py     # Step-by-step debugging
│   ├── observation_viewer.py  # Real-time observation display
│   └── widgets/               # Reusable GUI components
└── configs/                   # MARL-specific configurations
```

| Component                                   | Status          | Description                             |
| ------------------------------------------- | --------------- | --------------------------------------- |
| **[Coordinator](coordinator.md)**           | ✅ Complete      | Central orchestrator for MARL execution |
| **[Scenario System](scenario.md)**          | ✅ Complete      | Template-based scenario generation      |
| **[Map Adapter](adapters/map_adapter.md)**  | ✅ Complete      | OpenCDA-MARL map integration bridge     |
| **[Environment Interface](environment.md)** | ✅ Complete      | Gym-compatible multi-agent environment  |
| **[GUI Components](#)**                     | ✅ Complete      | Visual debugging and control interface  |
| **[Vehicle Adapter](#)**                    | 🔄 Next Priority | RL-OpenCDA vehicle bridge               |
| **[Agent Infrastructure](#)**               | 📋 Planned       | RL algorithm implementations            |
| **[Training Pipeline](#)**                  | 📋 Planned       | Distributed training infrastructure     |


## Design Philosophy

=== "1. Non-Intrusive Extension"
    - OpenCDA core remains unchanged
    - MARL components operate as optional extensions
    - Backward compatibility maintained
    - Adapter pattern for seamless integration

=== "2. Modular Architecture"
    - Clear separation between OpenCDA and MARL components
    - Coordinator-based orchestration
    - Template-based scenario generation
    - Standard Gym environment interface

=== "3. Research-Focused"
    - Support for multiple MARL algorithms (planned)
    - Flexible experiment configuration via OmegaConf
    - Comprehensive callback system for extensibility
    - Visual debugging tools for development

## Core Implementation

### 1. MARL Coordinator

The central orchestrator provides unified control over multi-agent scenarios:

```python
from opencda_marl.core.coordinator import MARLCoordinator, ExecutionMode
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load('configs/marl/intersection.yaml')

# Create coordinator with GUI mode
coordinator = MARLCoordinator(
    config=config,
    mode=ExecutionMode.GUI_DEBUG,
    enable_gui=True
)

# Initialize all components
coordinator.initialize()

# Run GUI interface for debugging
coordinator.run_gui_mode()
```

### 2. Scenario System

Template-based scenario generation with factory pattern:

```python
from opencda_marl.scenarios.scenario_builder import ScenarioBuilder
from opencda.core.common.cav_world import CavWorld

# Create scenario builder
builder = ScenarioBuilder()

# Build scenario from configuration
cav_world = CavWorld(apply_ml=False)
scenario_manager = builder.build_from_config(config, cav_world)

# Or build programmatically
config = builder.build_intersection_scenario(
    num_agents=4,
    intersection_type='4way',
    spawn_strategy='balanced'
)
```

### 3. Environment Interface

Standard Gym-compatible multi-agent environment:

```python
from opencda_marl.envs.multi_agent_env import MARLGymEnv

# Create Gym environment
env = MARLGymEnv(
    scenario_manager=scenario_manager,
    config=config,
    render_mode='human',
    max_episode_steps=500
)

# Standard Gym interface
observations, info = env.reset()
observations, rewards, dones, truncated, info = env.step(actions)
```

### 4. Map Integration

Hybrid system combining OpenCDA map loading with MARL coordination:

```python
from opencda_marl.core.adapters.map_adapter import MARLMapAdapter
import carla

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# Create map adapter
map_adapter = MARLMapAdapter(config, client)

# Get coordinated spawn points for MARL scenarios
spawn_points = map_adapter.get_marl_spawn_points(
    num_agents=4, 
    strategy='balanced'
)

# Get OpenCDA-compatible spawn points
opencda_spawns = map_adapter.get_opencda_spawn_points(num_agents=4)

# Create per-vehicle OpenCDA MapManagers
for vehicle in vehicles:
    map_manager = map_adapter.create_opencda_map_manager(
        vehicle=vehicle,
        config=config.map_manager
    )
```

## Current Capabilities

=== "Intersection Management"

    Complete multi-agent intersection scenarios ready for RL integration:

    ```python
    # Create intersection scenario
    coordinator = MARLCoordinator(config=config, mode=ExecutionMode.TRAINING)
    coordinator.initialize()

    # Run training episodes (placeholder agents currently)
    results = coordinator.run_training(num_episodes=100)

    # Or use GUI for debugging
    coordinator = MARLCoordinator(config=config, mode=ExecutionMode.GUI_DEBUG)
    coordinator.run_gui_mode()
    ```

=== "Multiple Execution Modes"

    Four execution modes for different research needs:

    ```python
    # 1. GUI debugging mode
    coordinator = MARLCoordinator(config, ExecutionMode.GUI_DEBUG, enable_gui=True)
    coordinator.run_gui_mode()

    # 2. Training mode
    coordinator = MARLCoordinator(config, ExecutionMode.TRAINING)
    results = coordinator.run_training(num_episodes=1000)

    # 3. Evaluation mode
    coordinator = MARLCoordinator(config, ExecutionMode.EVALUATION)
    metrics = coordinator.run_evaluation(num_episodes=100)

    # 4. Interactive CLI mode
    coordinator = MARLCoordinator(config, ExecutionMode.DEMO)
    coordinator.run_interactive_cli()
    ```

=== "Template-Based Scenario Generation"

    Extensible scenario system with parameter validation:

    ```python
    # Get available scenario types
    builder = ScenarioBuilder()
    available = builder.get_available_scenarios()      # ['intersection', 'highway', 'parking']
    implemented = builder.get_implemented_scenarios()  # ['intersection']

    # Create scenarios programmatically
    intersection_config = builder.build_intersection_scenario(
        num_agents=6,
        intersection_type='4way',
        traffic_density='high',
        weather_conditions='rainy',
        spawn_strategy='conflict'  # Create challenging scenarios
    )
    ```

## Planned Components

=== "🔄 Vehicle Adapter (Next Priority)"

    Bridge between OpenCDA VehicleManager and RL agents:

    ```python
    # Planned vehicle adapter interface
    from opencda_marl.core.adapters.vehicle_adapter import VehicleAdapter

    # Bridge OpenCDA VehicleManager with RL agents
    adapter = VehicleAdapter(
        vehicle_manager=opencda_vehicle,
        config=agent_config
    )

    # Convert RL actions to vehicle commands
    vehicle_action = adapter.action_to_vehicle_command(rl_action)
    observation = adapter.get_observation_from_vehicle()
    ```

=== "📋 RL Algorithm Integration (Planned)"

    Integration with standard RL libraries:

    ```python
    # Planned RL integration
    from opencda_marl.agents import PPOAgent
    from stable_baselines3 import PPO

    # Create RL agents
    agent = PPOAgent(
        env=env,
        policy='MlpPolicy',
        learning_rate=3e-4
    )

    # Training with coordinator
    coordinator.set_agent_manager(agent)
    results = coordinator.run_training(total_timesteps=1000000)
    ```

    === "📋 Advanced MARL Features (Future)"

    Communication and coordination capabilities:

    ```python
    # Planned communication interface
    from opencda_marl.communication import V2XCommunication

    # Enable agent communication
    comm_system = V2XCommunication(
        range=50.0,
        message_size=64,
        protocol='broadcast'
    )

    coordinator.enable_communication(comm_system)
    ```

## Configuration System

=== "Current Configuration Support"

    MARL uses OmegaConf for flexible configuration management:

    ```yaml
    # configs/marl/intersection.yaml
    scenario_type: intersection

    map:
    name: intersection
    safe_distance: 5.0
    spawn_offset: 2
    dest_offset: 2

    world:
    sync_mode: true
    client_port: 2000
    fixed_delta_seconds: 0.05
    weather:
        sun_altitude_angle: 15
        cloudiness: 0

    marl:
    agents:
        num_agents: 4
        agent_type: random
    environment:
        observation_type: vector
        action_type: discrete
        reward_function: safety_efficiency
    coordination:
        enable_communication: true
        spawn_strategy: balanced
    ```

=== "Integration with OpenCDA"

    Seamless integration with existing OpenCDA configurations:

    ```python
    from omegaconf import OmegaConf

    # Load base OpenCDA config
    base_config = OmegaConf.load('opencda/scenario_testing/config_yaml/default.yaml')

    # Load MARL extensions
    marl_config = OmegaConf.load('configs/marl/intersection.yaml')

    # Merge configurations
    config = OmegaConf.merge(base_config, marl_config)

    # Use with coordinator
    coordinator = MARLCoordinator(config=config)
    ```

=== "Command-Line Interface"

    Access MARL functionality through extended `opencda.py`:

    ```bash
    # Basic MARL scenario execution
    python opencda.py -t intersection -v 0.9.15 --marl

    # GUI debugging mode
    python opencda.py -t intersection -v 0.9.15 --marl --gui

    # Training mode (when RL agents are implemented)
    python opencda.py -t intersection -v 0.9.15 --marl --train

    # Interactive CLI mode
    python opencda.py -t intersection -v 0.9.15 --marl --interactive
    ```