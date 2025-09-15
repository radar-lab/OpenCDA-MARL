# Phase 2: Custom Environment & Baseline Agents

This phase focused on creating a custom MARL environment and implementing baseline agents for benchmarking, replacing the original Gym-based approach with a direct CARLA integration solution.

| Component                   | Status | Description                                                 |
| --------------------------- | ------ | ----------------------------------------------------------- |
| **Custom MARL Environment** | ✅      | Direct CARLA integration without Gym complexity             |
| **MARLTrainer**             | ✅      | Training and evaluation workflows                           |
| **Baseline Agent Suite**    | ✅      | Three agent implementations (behavior, vanilla, rule_based) |
| **Benchmark System**        | ✅      | Automated testing across traffic scenarios                  |
| **Traffic Scenarios**       | ✅      | Standardized presets (safe, balanced, aggressive)           |
| **Performance Metrics**     | ✅      | VPM-based throughput calculation                            |

## Architecture Migration

=== "Removed: Gym-based Environment"

    ```python
    # OLD v0.1.x approach (removed)
    class MARLGymEnv(gym.Env):
        def __init__(self, scenario_manager):
            self.scenario_manager = scenario_manager
            # Complex agent ID mapping issues
            self.agent_mapping = {}  # Source of bugs
    ```

=== "New: Custom MARL Environment"

    ```python
    # NEW v0.2.0 approach (implemented)
    class MARLEnvironment:
    def __init__(self, scenario_manager, config):
        self.scenario_manager = scenario_manager
        # Direct integration, no ID mapping complexity
        
    def reset_episode(self) -> Dict[str, Any]:
        # Clean episode management
        
    def step(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        # Direct CARLA integration
    ```

## Custom MARL Environment


=== "Direct CARLA Integration"

    - **No Gym Wrapper**: Eliminates agent ID mapping complexity
    - **Dynamic Vehicles**: Perfect fit for CARLA's dynamic vehicle nature
    - **Clean Interface**: Simplified training and evaluation workflows
    - **Better Performance**: No wrapper overhead

=== "Episode Management"

    ```python
    # Episode workflow
    episode_info = environment.reset_episode()
    
    for step in range(max_steps):
        step_info = environment.step(actions)
        
        if step_info.get('done') or step_info.get('truncated'):
            break
    ```

=== "Observation System"

    **10D Observation Vector per Agent**:

    - Position (x, y, z)
    - Velocity (x, y, z) 
    - Rotation (pitch, yaw, roll)
    - Speed magnitude

=== "Multi-objective Reward Function"

    - **Progress Reward**: `min(speed / 10.0, 1.0)`
    - **Safety Reward**: `+1.0` if distance > 5m, `-5.0` if < 2m
    - **Efficiency Reward**: `+2.0` for ideal speed range (5-15 m/s)

## Baseline Agent Suite

### Agent Implementations

=== "BehaviorAgent (OpenCDA Standard)"

    ```python
    # Standard OpenCDA autonomous driving
    controller_type: "behavior"
    
    Features:
    - Full OpenCDA BehaviorAgent pipeline
    - Perception, planning, and control integration
    - Proven baseline for comparison
    - Production-ready performance
    ```

=== "VanillaAgent (Enhanced Safety)"

    ```python
    # Enhanced collision avoidance
    controller_type: "vanilla"
    
    Key Features:
    - Multi-vehicle TTC (Time-to-Collision) tracking
    - Intersection safety multipliers
    - Lateral conflict detection
    - Predictive collision avoidance
    
    Configuration:
    vanilla:
      intersection_safety_multiplier: 2.0
      multi_vehicle_ttc: true
      max_vehicles_to_track: 5
    ```

=== "RuleBasedAgent (3-Stage Rules)"

    ```python
    # Rule-based intersection management
    controller_type: "rule_based"
    
    Three-Stage Logic:
    1. Junction Management: Approach and conflict detection
    2. Car Following: Time headway control  
    3. Cruising: Target speed maintenance
    
    Configuration:
    rule_based:
      junction_approach_distance: 70.0
      cautious_speed: 20.0
      time_headway: 2.0
    ```

### Agent Factory System

```python
# Dynamic agent creation based on configuration
from opencda_marl.core.agents.agent_factory import AgentFactory

def create_agent(agent_type, vehicle, config):
    if agent_type == "behavior":
        return create_behavior_agent(vehicle, config)
    elif agent_type == "vanilla": 
        return create_vanilla_agent(vehicle, config)
    elif agent_type == "rule_based":
        return create_rule_based_agent(vehicle, config)
```

## Benchmark System

### Automated Testing Infrastructure

=== "BenchmarkComparator"
    **Location**: `test/marl/test_benchmark_comparison.py`
    
    **Features**:
    - Dynamic YAML configuration override
    - Multi-agent, multi-scenario testing
    - Comprehensive performance metrics
    - Results export and visualization

=== "Performance Metrics"

    ```python
    # Key metrics calculated
    throughput_vpm = vehicles_completed / (evaluation_time_seconds / 60)
    success_rate = (completed_vehicles / total_vehicles) * 100
    collision_rate = (collided_vehicles / total_vehicles) * 100
    ```

### Usage Examples

=== "CLI Testing"

    ```bash
    # Test all agents with all scenarios
    python test/marl/test_benchmark_comparison.py --all-agents --all-scenarios
    
    # Test specific combinations
    python test/marl/test_benchmark_comparison.py \
        --agents behavior vanilla rule_based \
        --scenarios balanced aggressive \
        --timeout 300
    ```

=== "Results Analysis"

    ```python
    # Example benchmark results
    {
        "safe_behavior": {
            "throughput_vpm": 33.8,
            "success_rate": 86.1,
            "collision_rate": 5.6,
            "total_vehicles": 167
        },
        "balanced_vanilla": {
            "throughput_vpm": 37.1,
            "success_rate": 83.3,
            "collision_rate": 13.6,
            "total_vehicles": 270
        }
    }
    ```

## MARLTrainer Integration

### Training Workflows

=== "Basic Training"
    ```python
    from opencda_marl.core.marl_environment import MARLEnvironment, MARLTrainer
    
    # Create environment
    environment = MARLEnvironment(scenario_manager, config)
    trainer = MARLTrainer(environment)
    
    # Single episode
    episode_stats = trainer.train_episode()
    
    # Multiple episodes  
    results = trainer.train_multiple_episodes(num_episodes=100)
    ```

=== "Evaluation"
    ```python
    # Evaluation workflow
    evaluation_results = trainer.evaluate(
        num_episodes=50,
        agents=baseline_agents
    )
    
    # Results summary
    print(f"Average Reward: {evaluation_results['avg_reward']:.3f}")
    print(f"Success Rate: {evaluation_results['success_rate']:.3f}")
    ```

## Traffic Configuration System

### Dynamic Override

=== "Configuration Structure"

    ```yaml
    # Base configuration in configs/marl/default.yaml
    traffic_scenarios:
      safe:
        rate_vph: 200
        min_headway_s: 3.5
        safety_time: 4.0
        emergency_param: 0.5
        cautious_speed: 25.0
        spawn_num: 1
        strategy: "balanced"
    ```

=== "Runtime Override"

    ```python
    # Dynamic configuration modification
    def create_modified_config(base_config, scenario):
        override_params = {
            'scenario.traffic_manager.rate_vph': 600,
            'scenario.traffic_manager.min_headway_s': 1.0,
            'agents.agent_behavior.safety_time': 2.0
        }
        
        return apply_overrides(base_config, override_params)
    ```

## Key Benefits Achieved

=== "Resolved Issues from v0.1.x"

    - **Agent ID Mapping**: Eliminated complex dynamic/static ID mapping
    - **Vehicle Disappearing**: Fixed vehicles disappearing due to ID conflicts
    - **Gym Overhead**: Removed unnecessary abstraction layers
    - **Debugging Complexity**: Direct access to all simulation components

=== "Performance Improvements"

    - **Setup Overhead**: Minimal vs high ID mapping complexity
    - **Step Performance**: Direct CARLA access vs wrapper overhead
    - **Agent Management**: Natural string IDs vs dynamic mapping
    - **Debugging**: Direct component access vs multiple abstraction layers

=== "Enhanced Capabilities"

    - **Baseline Comparison**: Three agent types for benchmarking
    - **Traffic Scenarios**: Standardized complexity levels
    - **Automated Testing**: Comprehensive benchmark system
    - **Performance Metrics**: VPM-based throughput calculation



---

**Next**: [Phase 3 - MARL Agents](phase3_marl.md)