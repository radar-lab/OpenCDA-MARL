# MARL Environment API

!!! success "Implementation Status"
    The MARL Environment is **fully implemented** as a custom environment (non-Gym) that provides direct integration with CARLA and OpenCDA. It manages the full RL loop: observation extraction via ObservationExtractor, action selection via MARLManager, reward calculation, and algorithm updates.

The MARL Environment (`MARLEnv`) provides a clean, direct interface for multi-agent reinforcement learning, designed specifically for CARLA's dynamic vehicle nature without Gym dependency.

```text
MARLEnv
├── Episode Management      # reset_episode(), step execution
├── Observation System      # ObservationExtractor for feature vectors
├── Reward Calculation      # Multi-objective configurable rewards
├── Event Tracking          # StepEvent-based collision/completion tracking
└── Training Integration    # MARLManager for algorithm updates
```

## Core Classes

=== "MARLEnv"

    Custom MARL environment with direct CARLA integration.

    ```python
    class MARLEnv:
        """
        Custom MARL Environment for OpenCDA.

        Provides direct RL training loop without Gym dependency.
        Integrates with MARLScenarioManager for simulation control
        and MARLManager for algorithm orchestration.
        """
    ```

=== "Constructor"

    ```python
    def __init__(self, scenario_manager: MARLScenarioManager, config: Dict = {}):
        """
        Initialize MARL Environment.

        Parameters
        ----------
        scenario_manager : MARLScenarioManager
            The scenario manager providing CARLA simulation control
        config : dict
            MARL configuration (reward params, training settings, etc.)
        """
    ```

## Key Methods

=== "Episode Management"

    ```python
    def reset_episode(self):
        """
        Reset for new episode.

        Resets scenario manager, clears event logs,
        and prepares for new training episode.
        """

    def step(self):
        """
        Execute one environment step.

        Runs the full RL loop:
        1. Scenario manager step (vehicle control, traffic)
        2. Collect observations via ObservationExtractor
        3. Calculate rewards from step events
        4. MARLManager action selection and algorithm update

        Returns internally managed state; use getter methods
        for observations, rewards, and events.
        """
    ```

=== "Observation System"

    ```python
    def get_observations(self) -> Dict:
        """
        Get observations for all active agents.

        Returns observations extracted by ObservationExtractor
        with configurable feature types.

        Returns
        -------
        observations : dict
            Agent observations (agent_id -> observation vector)
        """

    def get_training_info(self) -> Dict[str, Any]:
        """
        Get current training information from MARLManager.

        Returns algorithm-specific training stats
        (losses, exploration noise, Q-values, etc.)
        """
    ```

=== "Reward System"

    ```python
    def get_current_step_rewards(self) -> Dict[str, float]:
        """
        Get rewards from the current step.

        Returns
        -------
        rewards : dict
            Agent rewards (agent_id -> reward value)
        """

    def get_reward_params(self) -> Dict:
        """
        Get current reward function parameters.

        Returns the configurable reward weights and values.
        """
    ```

    Multi-objective reward components (configurable via YAML):

    | Component     | Default Value | Description                           |
    | ------------- | ------------- | ------------------------------------- |
    | Collision     | -500          | Terminal penalty on collision          |
    | Success       | +400          | Terminal reward on reaching destination |
    | Step penalty  | -0.5          | Per-step cost for efficiency          |
    | Speed bonus   | +1.0          | Reward for maintaining target speed   |
    | Progress      | scaled        | Based on distance to destination      |
    | Stop penalty  | -3.0          | Penalty for stopped vehicles          |
    | Yielding      | +1.0          | Reward for yielding to obstacles      |

=== "Event Tracking"

    ```python
    def get_current_events(self) -> List:
        """
        Get step events from current simulation step.

        Returns
        -------
        events : list[StepEvent]
            Events including collisions, completions, timeouts
        """

    def get_episode_metrics(self) -> Dict:
        """
        Get aggregated episode metrics.

        Returns metrics like total reward, collision count,
        success count, average speed, etc.
        """
    ```

## Observation Features

The `ObservationExtractor` supports 9 configurable feature types:

| Feature                    | Description                          |
| -------------------------- | ------------------------------------ |
| `rel_x`, `rel_y`          | Relative position to ego             |
| `heading`                  | Vehicle orientation (radians)        |
| `speed`                    | Current velocity                     |
| `distance_to_intersection` | Remaining distance to junction       |
| `distance_to_front`        | Distance to nearest vehicle ahead    |
| `lane_position`            | Lateral position in lane             |
| `waypoint_buffer`          | Next waypoint distance               |
| `min_ttc`                  | Minimum time-to-collision            |
| `distance_to_destination`  | Remaining route distance             |

## Usage Examples

=== "Basic Setup"

    ```python
    from opencda_marl.envs.marl_env import MARLEnv

    # Created automatically by coordinator, but can be used directly
    env = MARLEnv(scenario_manager, config=marl_config)

    # Training loop
    env.reset_episode()
    for step in range(max_steps):
        env.step()

        observations = env.get_observations()
        rewards = env.get_current_step_rewards()
        events = env.get_current_events()
    ```

=== "With Coordinator (Recommended)"

    ```python
    from opencda_marl.coordinator import MARLCoordinator

    # Coordinator creates and manages MARLEnv internally
    coordinator = MARLCoordinator(config=config)
    coordinator.initialize()

    # MARLEnv is accessible via coordinator
    coordinator.run()  # Handles full training loop
    ```

=== "Custom Reward Configuration"

    ```yaml
    # Configure rewards in YAML
    rewards:
      collision_penalty: -500
      success_reward: 400
      step_penalty: -0.5
      speed_bonus: 1.0
      stop_penalty: -3.0
      yielding_bonus: 1.0
    ```

## Configuration Integration

=== "Environment Configuration"

    ```yaml
    # MARL environment settings in config
    scenario:
      max_steps: 2400              # Maximum steps per episode
      max_episodes: 500            # Maximum training episodes

    MARL:
      algorithm: "td3"             # RL algorithm selection
      state_dim: 8                 # Observation vector dimension
      action_dim: 1                # Action dimension (speed control)
      training: true               # Training vs evaluation mode
    ```

---

**Location**: `opencda_marl/envs/marl_env.py`
