# MARL Training and Evaluation Guide

## Quick Start

=== "Basic MARL Demo"

    ```bash
    # Run intersection scenario with MARL environment
    python opencda.py -t intersection --marl

    # Demo mode with specific agent types
    python opencda.py -t intersection --marl --demo --controller rule_based
    ```

=== "Available Controllers"

  - `behavior`: OpenCDA BehaviorAgent (standard autonomous driving)
  - `vanilla`: Enhanced collision avoidance with multi-vehicle TTC
  - `rule_based`: 3-stage intersection rules (approach, follow, cruise)
  - `placeholder`: Development/testing controller

## Benchmark Comparison

=== "Automated Testing"

    ```bash
    # Test all agents with all traffic scenarios
    python test/marl/test_benchmark_comparison.py --all-agents --all-scenarios

    # Specific agents and scenarios
    python test/marl/test_benchmark_comparison.py --agents behavior vanilla --scenarios balanced

    # Quick test with custom parameters
    python test/marl/test_benchmark_comparison.py --agents rule_based --scenarios safe --timeout 60
    ```

=== "Traffic Scenario Presets"

    | Scenario       | Rate (vph) | Headway (s) | Safety Time | Expected Collisions | Description             |
    | -------------- | ---------- | ----------- | ----------- | ------------------- | ----------------------- |
    | **safe**       | 200        | 3.5         | 4.0         | ~5%                 | Conservative parameters |
    | **balanced**   | 400        | 2.0         | 3.0         | ~10%                | Moderate complexity     |
    | **aggressive** | 600        | 1.0         | 2.0         | ~20%                | High density conflicts  |

    **Performance Metrics**

    - **Throughput (vpm)**: Vehicles successfully crossing intersection per minute
    - **Success Rate (%)**: Percentage of vehicles reaching destination  
    - **Collision Rate (%)**: Percentage of vehicles involved in collisions
    - **Total Vehicles**: Actual vehicles spawned during episode

## Training Environment

=== "Custom MARLEnvironment"

    ```python
    from opencda_marl.core.marl_environment import MARLEnvironment, MARLTrainer

    # Initialize environment
    environment = MARLEnvironment(scenario_manager, config)

    # Training workflow
    trainer = MARLTrainer(environment)
    episode_stats = trainer.train_episode(agents)
    results = trainer.train_multiple_episodes(num_episodes=100)
    ```

=== "Training Loop Structure"

    1. **Episode Reset**: `environment.reset_episode()` 
    2. **Action Generation**: Agent policies or autonomous control
    3. **Environment Step**: `environment.step(actions)` 
    4. **Reward Calculation**: Multi-objective reward function
    5. **Termination Check**: Collision, completion, or timeout

## Configuration System

=== "Traffic Flow Tuning"

    ```yaml
    # Override default traffic parameters
    traffic_scenarios:
      custom:
        description: "Custom scenario"
        rate_vph: 300          # Vehicles per hour
        min_headway_s: 2.5     # Following distance
        safety_time: 3.5       # TTC safety threshold
        emergency_param: 0.45  # Emergency braking sensitivity
        cautious_speed: 22.0   # Rule-based agent speed (km/h)
        spawn_num: 1           # Vehicles per spawn event
        strategy: "balanced"   # Spawn distribution strategy
    ```

=== "Agent Behavior Configuration"

    ```yaml
    agents:
      # VanillaAgent configuration
      vanilla:
        intersection_safety_multiplier: 2.0
        multi_vehicle_ttc: true
        max_vehicles_to_track: 5
      
      # RuleBasedAgent configuration  
      rule_based:
        junction_approach_distance: 70.0
        cautious_speed: 20.0
        time_headway: 2.0
    ```

## Evaluation Workflows

=== "Single Agent Testing"

    ```bash
    # Test specific agent type with timeout
    python test_baseline_agents.py --agent behavior --timeout 60
    python test_baseline_agents.py --agent vanilla --timeout 60
    python test_baseline_agents.py --agent rule_based --timeout 60
    ```

    The benchmark system automatically:
    - Modifies intersection.yaml with different traffic parameters
    - Runs each agent for 300-second episodes
    - Parses coordinator output for metrics
    - Generates comparison visualizations
    - Exports JSON results with metadata

=== "Results Interpretation"

    ```python
    # Example benchmark results
    {
        "safe_behavior": {
            "throughput_vpm": 33.8,
            "success_rate": 86.1,
            "collision_rate": 5.6,
            "total_vehicles": 167
        }
    }
    ```

=== "Advanced Configuration"

    **Parameter Interaction Effects**

    - **rate_vph ↔ min_headway_s**: Higher rate with lower headway = more conflicts
    - **safety_time ↔ emergency_param**: Lower values = more aggressive behavior
    - **spawn_num ↔ strategy**: Multiple spawns with "conflict" = maximum challenge

    **Custom Scenarios**

    ```python
    # Dynamic configuration override
    import yaml
    config_override = {
        "rate_vph": 450,
        "min_headway_s": 1.5,
        "safety_time": 2.5,
        "strategy": "conflict"
    }
    ```