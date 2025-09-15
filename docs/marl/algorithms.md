# MARL Algorithms

## Overview

This section describes the Multi-Agent Reinforcement Learning algorithms implemented in
OpenCDA-MARL.

## Implemented Algorithms

### Independent Learning

-   **DQN (Deep Q-Network)**: Each agent learns independently
-   **PPO (Proximal Policy Optimization)**: Stable policy gradient method
-   **SAC (Soft Actor-Critic)**: Off-policy algorithm with entropy regularization

### Centralized Training with Decentralized Execution (CTDE)

-   **QMIX**: Monotonic value function factorization
-   **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient
-   **MAPPO**: Multi-Agent PPO with centralized value function

### Communication-based Methods

-   **CommNet**: Learnable communication between agents
-   **MAAC**: Multi-Agent Actor-Critic with attention mechanism

## Algorithm Selection Guide

Choose algorithms based on your scenario requirements:

| Scenario        | Recommended Algorithm | Reason                              |
| --------------- | --------------------- | ----------------------------------- |
| Platooning      | MADDPG                | Continuous control, stability       |
| Intersection    | QMIX                  | Discrete actions, credit assignment |
| Highway Merging | MAPPO                 | Mixed cooperation/competition       |

## Implementation Details

Each algorithm is implemented with:

-   Configurable hyperparameters
-   Tensorboard logging
-   Model checkpointing
-   Evaluation metrics

## Custom Algorithms

To implement your own algorithm:

```python
# Example structure will be added
```

## Performance Benchmarks

Benchmark results on standard scenarios:

-   Training convergence rates
-   Sample efficiency
-   Computational requirements
