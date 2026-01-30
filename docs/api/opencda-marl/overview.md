# OpenCDA-MARL API Overview

OpenCDA-MARL extends the OpenCDA framework with Multi-Agent Reinforcement Learning capabilities for autonomous driving research. This API documentation covers the MARL-specific components and interfaces built on top of the OpenCDA foundation.

!!! success "Implementation Status"
    **Core Framework: Complete** - Architecture, RL algorithms, training infrastructure, and evaluation tools are all implemented and ready for research use.

### Architecture & Implementation

```text
opencda_marl/
├── coordinator.py             # Central MARL orchestrator
├── core/
│   ├── agent_manager.py       # Vehicle spawning & adapter management
│   ├── adapter/
│   │   └── vehicle_adapter.py # Vehicle-agent bridge
│   ├── agents/                # Agent implementations (factory pattern)
│   ├── marl/                  # RL algorithms & training infrastructure
│   │   ├── marl_manager.py    # Algorithm orchestrator
│   │   ├── extractor.py       # Observation feature extraction
│   │   ├── metrics.py         # Training metrics & CSV export
│   │   ├── checkpoint.py      # Model checkpoint management
│   │   └── algorithms/        # TD3, DQN, Q-Learning, MAPPO, SAC
│   ├── safety/                # Collision detection & avoidance
│   └── traffic/               # Traffic flow management & replay
├── envs/
│   ├── marl_env.py            # Custom CARLA RL environment
│   ├── sumo_marl_env.py       # SUMO-only environment
│   ├── evaluation.py          # Episode evaluation
│   └── cross_agent_evaluator.py
├── gui/                       # PySide6 Qt dashboard
├── scenarios/                 # Scenario templates & management
└── configs/                   # MARL-specific YAML configurations
```

| Component                                              | Status     | Description                              |
| ------------------------------------------------------ | ---------- | ---------------------------------------- |
| **[Coordinator](coordinator.md)**                      | ✅ Complete | Central orchestrator for MARL execution  |
| **[Scenario System](scenario.md)**                     | ✅ Complete | Template-based scenario generation       |
| **[Agent Manager](agent_manager.md)**                  | ✅ Complete | Multi-agent lifecycle management         |
| **[Map Manager](map_manager.md)**                      | ✅ Complete | Junction-based spawn point generation    |
| **[Vehicle Adapter](adapters/vehicle_adapter.md)**     | ✅ Complete | RL-OpenCDA vehicle bridge                |
| **[Environment Interface](environment.md)**            | ✅ Complete | Custom MARL environment (non-Gym)        |
| **[Training Infrastructure](training.md)**             | ✅ Complete | Algorithms, checkpoints, metrics         |
| **[Traffic Configuration](traffic_config.md)**         | ✅ Complete | Flow-based traffic generation            |
| **[Benchmark System](benchmark.md)**                   | ✅ Complete | Cross-agent performance comparison       |


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
    - Custom environment with direct CARLA integration

=== "3. Research-Focused"
    - Five RL algorithms (TD3, DQN, Q-Learning, MAPPO, SAC)
    - Flexible experiment configuration via OmegaConf
    - Comprehensive callback system for extensibility
    - TensorBoard integration and convergence detection

## Core Implementation

### 1. MARL Coordinator

The central orchestrator provides unified control over multi-agent scenarios:

```python
from opencda_marl.coordinator import MARLCoordinator

# Load configuration
config = OmegaConf.merge(
    OmegaConf.load("configs/marl/default.yaml"),
    OmegaConf.load("configs/marl/td3_simple_v4.yaml")
)

# Create and initialize coordinator
coordinator = MARLCoordinator(config=config)
coordinator.initialize()

# Run training
coordinator.run()

# Or run with GUI
coordinator.run_gui_mode()
```

### 2. Scenario System

Template-based scenario generation with factory pattern:

```python
from opencda_marl.scenarios.scenario_builder import ScenarioBuilder

# Build scenario from configuration
builder = ScenarioBuilder()
scenario_manager = builder.build_from_config(config, cav_world)
```

### 3. Environment Interface

Custom MARL environment with direct CARLA integration:

```python
from opencda_marl.envs.marl_env import MARLEnv

# Created automatically by coordinator
env = MARLEnv(scenario_manager, config=marl_config)

# Step cycle: observe → act → reward → learn
# Handled internally by coordinator.run()
```

### 4. RL Algorithms

All algorithms share a common `BaseAlgorithm` interface:

```python
from opencda_marl.core.marl.marl_manager import MARLManager

# Algorithm selected based on config MARL.algorithm
manager = MARLManager(config)
action = manager.select_action(observations, ego_id, training=True)
manager.store_transition(obs, ego_id, action, reward, next_obs, done)
losses = manager.update()
```

## Configuration System

=== "Configuration Loading"

    MARL uses OmegaConf for flexible configuration management:

    ```yaml
    # configs/marl/default.yaml - Base configuration
    meta:
      simulator: "carla"

    world:
      sync_mode: true
      client_port: 2000

    scenario:
      max_steps: 2400
      max_episodes: 500

    MARL:
      algorithm: "td3"
      state_dim: 8
      action_dim: 1
      training: true

    agents:
      agent_type: "marl"
    ```

=== "Algorithm-Specific Config"

    ```yaml
    # configs/marl/td3_simple_v4.yaml - Overrides default.yaml
    MARL:
      algorithm: "td3"
      state_dim: 8
      td3:
        learning_rate_actor: 0.0001
        learning_rate_critic: 0.001
        exploration_noise: 0.3
        warmup_steps: 1000
    ```

=== "Command-Line Interface"

    ```bash
    # MARL training with specific algorithm config
    python opencda.py -t td3_simple_v4 --marl

    # With GUI visualization
    python opencda.py -t td3_simple_v4 --marl --gui

    # Baseline agent testing
    python opencda.py -t vanilla --marl
    python opencda.py -t rule_based --marl
    ```
