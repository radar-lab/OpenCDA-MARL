# MARL Framework Overview

## Introduction

The Multi-Agent Reinforcement Learning (MARL) framework in OpenCDA-MARL enables researchers to train and evaluate cooperative autonomous driving policies in realistic simulation environments.

## Planned Changes

!!! warning "Upcoming Changes"
    The following architectural changes are planned for future releases:

    | Change                  | Priority | Timeline  | Breaking       |
    | ----------------------- | -------- | --------- | -------------- |
    | Modular sensor system   | High     | Aug 2025  | ⚠️ Potentially |
    | Plugin-based algorithms | Medium   | Aug 2025  | ❌ No          |
    | Async communication     | Low      | Aug 2025  | ❌ No          |
    | Distributed simulation  | Low      | TBD       | ⚠️ Potentially |

## Key Components

### Current Implementation (Phase 3)

✅ **Custom MARL Environment**: Direct integration without Gym complexity, perfect fit for CARLA's dynamic vehicle nature

✅ **Baseline Agent Suite**:

- **behavior**: OpenCDA BehaviorAgent for standard autonomous driving
- **vanilla**: Enhanced collision avoidance with multi-vehicle TTC tracking  
- **rule_based**: 3-stage intersection rules (approach, follow, cruise)

✅ **Benchmark Comparison System**: Automated testing across different traffic scenarios with comprehensive metrics

✅ **Traffic Scenario Presets**: 

- **safe** (~5% collision rate): Conservative parameters for smooth flow
- **balanced** (~10% collision rate): Moderate complexity with balanced conflicts
- **aggressive** (~20% collision rate): High density with frequent conflicts

**Planned Components (Phase 3)**

- 🚧 **MARL Agent Policies**: PPO, SAC, TD3 implementations (stub policies exist)
- 🚧 **Advanced Reward Functions**: Cooperative reward shaping
- 🚧 **Model Persistence**: Checkpointing and model loading

### Architecture Integration

The MARL framework extends OpenCDA's existing infrastructure:

- **MARLEnvironment**: Custom training environment with direct CARLA integration
- **MARLScenarioManager**: Coordinates world, agents, and traffic management
- **MARLVehicleAdapter**: Bridges OpenCDA VehicleManager with multi-agent control
- **Configuration System**: YAML-based with dynamic override capabilities
- See [Architecture](../architecture.md) for detailed information

### Training & Evaluation

- **MARLTrainer**: Integrated training and evaluation capabilities
- **Benchmark Testing**: Automated comparison across agents and scenarios  
- **Performance Metrics**: Throughput (vpm), success rate, collision rate
- See [Training](training.md) for complete workflows

## Use Cases

1. **Cooperative Perception**: Training agents to share perception data
2. **Platoon Control**: Learning optimal platoon behaviors
3. **Intersection Management**: Coordinating vehicles at intersections
4. **Highway Merging**: Learning cooperative merging strategies
