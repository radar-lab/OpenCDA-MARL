# OpenCDA-MARL Architecture

OpenCDA-MARL extends the original OpenCDA framework to support Multi-Agent Reinforcement Learning (MARL) for cooperative autonomous driving. This document describes the current architecture implementation.

!!! info "Development Status"
    OpenCDA-MARL is in early development (v0.1.0-alpha). The system currently focuses on intersection scenarios with multiple agent types and basic RL algorithms. This is research-oriented code designed for experimentation rather than production use.

## Directory Structure

This is the directory structure of the OpenCDA-MARL project.

<details>
<summary>Directory Structure</summary>
```sh
OpenCDA-MARL/
├── docs/                       # Documentation
├── opencda/                    # Original OpenCDA core (preserved)
│   ├── assets/                 # Maps and resources
│   ├── co_simulation/          # SUMO integration
│   ├── core/                   # Core modules
│   │   ├── actuation/          # Control algorithms
│   │   ├── application/        # Cooperative driving apps
│   │   ├── common/             # Base classes and V2X
│   │   ├── map/                # HD Map management
│   │   ├── plan/               # Planning algorithms
│   │   └── sensing/            # Perception and localization
│   ├── customize/              # User customizations
│   └── scenario_testing/       # Scenario scripts and configs
│
├── opencda_marl/               # MARL extensions (implemented)
│   ├── coordinator.py          # Main MARL orchestrator
│   ├── core/                   # Core MARL components
│   │   ├── adapter/            # OpenCDA integration adapters
│   │   │   └── vehicle_adapter.py     # Vehicle-agent bridge
│   │   ├── agents/             # Agent implementations
│   │   │   ├── agent_factory.py       # Agent factory
│   │   │   ├── basic_agent.py         # Base agent
│   │   │   ├── marl_agent.py          # MARL agent base
│   │   │   ├── marl_behavior_agent.py # Behavior agent
│   │   │   ├── vanilla_agent.py       # Safety agent
│   │   │   └── rule_based_agent.py    # Rule-based agent
│   │   ├── marl/               # MARL algorithms
│   │   │   └── algorithms/     # RL algorithm implementations
│   │   │       ├── q_learning.py      # Q-Learning
│   │   │       ├── dqn.py             # Deep Q-Network
│   │   │       └── td3.py             # Twin Delayed DDPG
│   │   ├── plan/               # Planning components
│   │   ├── safety/             # Safety management
│   │   └── traffic/            # Traffic utilities
│   ├── envs/                   # MARL environments
│   │   ├── marl_env.py         # Main MARL environment
│   │   ├── carla_monitor.py    # CARLA monitoring
│   │   ├── carla_spectator.py  # Camera control
│   │   └── evaluation.py       # Evaluation system
│   ├── gui/                    # GUI system
│   │   ├── dashboard.py        # Main dashboard
│   │   ├── observation_viewer.py # Agent observations
│   │   └── step_controller.py  # Simulation control
│   ├── scenarios/              # MARL scenario management
│   │   └── scenario_manager.py # Enhanced ScenarioManager
│   │
│   └── assets/                 # MARL-specific assets
│       └── maps/               # Custom intersection maps
│           ├── intersection.xodr      # Intersection map file
│           └── intersection.fbx       # Intersection 3D model
│
├── configs/                    # NEW: Unified configuration
│   ├── opencda/                # Original OpenCDA configs
│   └── marl/                   # MARL-specific configs
│
└── scripts/                    # Installation and setup scripts
```
</details>

## Architecture Overview

OpenCDA-MARL follows a 3-layer architecture that preserves OpenCDA's core functionality while adding MARL capabilities through adapter interfaces.

![OpenCDA-MARL Architecture](../images/OpenCDA_MARL_architecture.png)

### Layer 1: OpenCDA Core

Fully preserved OpenCDA components including CARLA integration, physics simulation, sensor systems (RGB, LiDAR, GPS), vehicle management, V2X communication, and scenario management. This layer remains unchanged from the original OpenCDA framework.

### Layer 2: MARL Adapter Interface

The bridge layer between OpenCDA and MARL algorithms. Key components include the MARLCoordinator (main orchestrator), MARLEnvironment (custom RL environment with CARLA integration), Vehicle Adapters (bridge OpenCDA vehicles with MARL agents), GUI System (dashboard with visualization), and Evaluation System (cross-agent performance comparison).

### Layer 3: Algorithm Implementation

Contains both baseline agents and RL algorithms. Baseline agents include Rule-based Agent (3-stage intersection navigation), Behavior Agent (OpenCDA standard driving), and Vanilla Agent (enhanced safety). RL algorithms include Q-Learning (discrete state/action with Q-tables), DQN (Deep Q-Network with neural networks), and TD3 (Twin Delayed DDPG for continuous control). Supports intersection scenarios with custom XODR maps and traffic replay patterns.

## Configuration System

The configuration system uses OmegaConf for YAML merging:

```python
# Load MARL configurations
if opt.marl:
    default_yaml = "configs/marl/default.yaml"
    config_yaml = f"configs/marl/{opt.test_scenario}.yaml"

# Direct OmegaConf merge
config = OmegaConf.merge(
    OmegaConf.load(default_yaml),
    OmegaConf.load(config_yaml)
)
```

## Execution Flow

```bash
# Basic MARL intersection scenario
python opencda.py -t intersection --marl

# MARL with GUI visualization
python opencda.py -t intersection --marl --gui

# Quick test with pixi
pixi run marl-quick-test
pixi run marl-quick-test-gui
```

1. **Configuration Loading**: Load MARL-specific YAML configuration
2. **MARLCoordinator**: Create main orchestrator with merged config
3. **Environment Setup**: Initialize CARLA world and MARL environment
4. **Agent Creation**: Setup agents based on configuration
5. **Execution Mode**: Start GUI visualization or automated simulation
