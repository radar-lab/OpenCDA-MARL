# MARL Training Infrastructure

## Overview

!!! success "Implementation Status"
    The Training Infrastructure is **fully implemented** with five RL algorithms, checkpoint management, metrics tracking, TensorBoard integration, and convergence detection.

The MARL Training Infrastructure provides the complete pipeline for training and evaluating reinforcement learning agents in OpenCDA-MARL scenarios.

```text
Training Infrastructure
├── MARLManager           # Algorithm orchestrator
├── BaseAlgorithm         # Common algorithm interface
├── Algorithms            # TD3, DQN, Q-Learning, MAPPO, SAC
├── ObservationExtractor  # Feature extraction for RL
├── CheckpointManager     # Model saving/loading
├── TrainingMetrics       # Episode statistics & CSV export
├── Replay Buffers        # Smart, Prioritized, Rollout
└── TensorBoard           # Training visualization
```

## Supported Algorithms

| Algorithm | Type | Status | Description |
|-----------|------|--------|-------------|
| **TD3** | Continuous, Off-policy | ✅ Implemented | Twin Delayed DDPG with LSTM encoder |
| **DQN** | Discrete, Off-policy | ✅ Implemented | Deep Q-Network with target network |
| **Q-Learning** | Discrete, Tabular | ✅ Implemented | Tabular Q-Learning with state bins |
| **MAPPO** | Continuous, On-policy | ✅ Implemented | Multi-Agent PPO with GAE |
| **SAC** | Continuous, Off-policy | ✅ Implemented | Soft Actor-Critic with entropy tuning |

## Core Classes

=== "MARLManager"

    The algorithm orchestrator that selects and manages the active RL algorithm.

    ```python
    from opencda_marl.core.marl.marl_manager import MARLManager

    manager = MARLManager(config)

    # Select action based on observations
    action = manager.select_action(
        multi_agent_obs=observations,
        ego_agent_id="agent_001",
        training=True
    )

    # Store experience
    manager.store_transition(obs, ego_id, action, reward, next_obs, done)

    # Update algorithm (returns loss dict)
    losses = manager.update()

    # Get training statistics
    info = manager.get_training_info()
    ```

=== "BaseAlgorithm"

    Abstract base class that all algorithms implement:

    ```python
    from opencda_marl.core.marl.algorithms.base_algorithm import BaseAlgorithm

    class BaseAlgorithm(ABC):
        """Common interface for all RL algorithms."""

        def select_action(self, state, training=True) -> float: ...
        def store_transition(self, state, action, reward, next_state, done): ...
        def update(self) -> Dict[str, float]: ...
        def reset_episode(self): ...
        def get_training_info(self) -> Dict: ...
        def save(self, path: str): ...
        def load(self, path: str): ...
    ```

    Built-in features: TensorBoard logging, convergence detection, episode metrics.

=== "ObservationExtractor"

    Converts raw CARLA vehicle data into normalized observation vectors:

    ```python
    from opencda_marl.core.marl.extractor import ObservationExtractor

    extractor = ObservationExtractor(config)
    obs_vector = extractor.extract(vehicle_data)
    ```

    Supported features: relative position, heading, speed, distance to intersection,
    distance to front vehicle, lane position, waypoint buffer, min TTC, distance to destination.

## Algorithm Details

=== "TD3"

    The primary algorithm with LSTM multi-agent context encoding.

    ```python
    # Architecture: 8D ego state + LSTM-encoded context → speed action [0, 65 km/h]
    # Actor: [8+256, 1024, 1024, 512, 256] → LayerNorm → tanh
    # Critic: Twin Q-networks for reduced overestimation
    ```

    Key features:

    - LSTM encoder for processing multi-agent observations
    - LayerNorm before tanh to prevent gradient vanishing
    - Delayed policy updates (every 2 steps)
    - Target policy smoothing with noise
    - Exploration noise decay (0.3 → 0.05)
    - Warmup phase (1000 steps of vanilla agent)
    - SmartReplayBuffer or PrioritizedReplayBuffer (configurable)

=== "DQN"

    Discrete action deep Q-network.

    ```python
    # Network: [state_dim, 64, 32] → Q-values for discrete speed actions
    # Actions: Predefined speed levels (e.g., [0, 5, 8, 12, 15] m/s)
    ```

    Key features: Epsilon-greedy exploration, target network (updated every 100 steps), gradient clipping.

=== "MAPPO"

    Multi-Agent Proximal Policy Optimization.

    Key features: Generalized Advantage Estimation (GAE), Gaussian actor for continuous actions, rollout buffer, clipped surrogate objective.

=== "SAC"

    Soft Actor-Critic with automatic entropy tuning.

    Key features: Entropy-regularized objective, automatic temperature (alpha) tuning, twin Q-networks, reparameterization trick.

## Checkpoint Management

=== "CheckpointManager"

    ```python
    from opencda_marl.core.marl.checkpoint import CheckpointManager

    checkpoint_mgr = CheckpointManager(config)

    # Save checkpoints (automatic in training)
    checkpoint_mgr.save(algorithm, episode=100, reward=350.0)

    # Load checkpoints
    checkpoint_mgr.load(algorithm, mode="latest")  # or "best"
    ```

    Saves three types: `latest_checkpoint.pth`, `best_checkpoint.pth`, `episode_XXXX.pth`

=== "Directory Structure"

    ```text
    checkpoints/
    └── td3_simple_v4/
        ├── latest_checkpoint.pth
        ├── best_checkpoint.pth
        ├── episode_0100.pth
        └── metadata.json
    ```

## Metrics & Logging

=== "TrainingMetrics"

    ```python
    from opencda_marl.core.marl.metrics import TrainingMetrics

    metrics = TrainingMetrics(config)
    ```

    Tracks per episode: total reward, success/collision rates, average speed, target speed gap, throughput, near-miss count, TTC violations.

    Exports to CSV in `metrics_history/` directory.

=== "TensorBoard"

    Metrics logged to TensorBoard:

    - **Loss/critic**, **Loss/actor**: Training losses
    - **Q_values/Q1_mean**, **Q_values/Q2_mean**: Value estimates
    - **Gradients/critic_pre_clip**, **Gradients/actor_pre_clip**: Gradient norms
    - **TD3/exploration_noise**: Current noise level
    - **Learning/reward_moving_avg**: Smoothed reward trend
    - **Safety/near_miss_count**, **Safety/ttc_violation_rate**: Safety metrics

    ```bash
    # View training progress
    pixi run tensorboard
    # or: tensorboard --logdir=runs
    ```

=== "Convergence Detection"

    Automatic convergence detection checks:

    - Reward CV < 15% over 10-episode window
    - Success rate CV < 20%
    - Collision rate improving (second half ≤ first half × 1.1)
    - Minimum 20 episodes before checking

## Replay Buffers

=== "SmartReplayBuffer"

    Default off-policy buffer with recency bias.

    - Pre-allocated numpy arrays for O(1) random access
    - Circular buffer with O(1) push
    - Sampling: 50% recent experiences (last 20%) + 50% diverse
    - Pre-emptive clearing at 90% capacity

=== "PrioritizedReplayBuffer"

    Optional TD-error prioritized sampling.

    - Alpha (priority exponent): 0.6
    - Beta annealing: 0.4 → 1.0 for importance sampling
    - Enabled via config: `td3.use_per: true`

=== "RolloutBuffer"

    On-policy buffer for MAPPO.

    - Stores complete episodes for GAE computation
    - Cleared after each policy update

## Usage

=== "Training"

    ```bash
    # Train with TD3 (default)
    python opencda.py -t td3_simple_v4 --marl

    # Train with DQN
    python opencda.py -t dqn --marl

    # Train with SAC
    python opencda.py -t sac --marl
    ```

=== "Evaluation"

    ```yaml
    # Set training: false in config to run evaluation
    MARL:
      training: false
    ```

    The CheckpointManager will load the best model automatically when training is disabled.

=== "Configuration Example"

    ```yaml
    # configs/marl/td3_simple_v4.yaml
    MARL:
      algorithm: "td3"
      state_dim: 8
      action_dim: 1
      training: true
      td3:
        learning_rate_actor: 0.0001
        learning_rate_critic: 0.001
        batch_size: 256
        memory_size: 100000
        gamma: 0.99
        tau: 0.005
        exploration_noise: 0.3
        noise_decay: 0.998
        min_noise: 0.05
        warmup_steps: 1000
        lstm_hidden: 256
    ```

---

- **Location**: `opencda_marl/core/marl/`
- **Algorithms**: `opencda_marl/core/marl/algorithms/`
