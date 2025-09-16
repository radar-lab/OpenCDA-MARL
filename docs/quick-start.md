# Quick Start Guide

Get started with OpenCDA for cooperative driving automation research and OpenCDA-MARL for multi-agent reinforcement learning scenarios.

---

## OpenCDA Scenarios

**What it provides**: Pre-configured benchmark scenarios for cooperative driving research

```bash
pixi run start -t <scenario_name> [--apply_ml] [--record]
```

**Parameters**:

- `-t`: Scenario name (must have matching `.py` in `configs/opencda/scenario_testing/` and `.yaml` in `configs/opencda/config_yaml/`)
- `--apply_ml`: Enable ML models (requires PyTorch)
- `--record`: Record simulation for replay

**Note**: Version argument (`-v`) has been removed. OpenCDA now uses CARLA 0.9.15 exclusively.

---

### Single Vehicle Tests

=== "Highway Navigation"
    ```bash
    # Basic 2-lane highway test (no ML required)
    pixi run start -t single_2lanefree_carla
    ```

    **Features**:

    - 100 km/h target speed
    - Safe traffic interaction
    - Localization, planning, control modules active
    - Perception disabled by default (no PyTorch needed)
    
    ![Highway Test](images/single_2lanefree_carla.gif)

=== "Urban with ML"
    ```bash
    # Town06 with YOLOv8/YOLOv5 detection
    pixi run start -t single_town06_carla --apply_ml
    ```

    **Features**:

    - Full perception pipeline with ML
    - Urban environment navigation
    - Camera-LiDAR fusion
    - **Requires**: PyTorch, CUDA (recommended)
    
    ![Town06 ML](images/single_town06_carla_2.gif)

=== "SUMO Co-simulation"
    ```bash
    # Town06 with SUMO traffic generation
    pixi run start -t single_town06_cosim --apply_ml
    ```

    **Features**:

    - SUMO-generated traffic flow
    - More realistic traffic patterns
    - ML-based 3D object detection
    - **Requires**: PyTorch, SUMO
    
    ![SUMO Cosim](images/town06_cosim.gif)

**Tip**: Bounding boxes in non-ML mode come from CARLA ground truth. With `--apply_ml`, they're from the detection model.

---

### Cooperative Driving Tests

=== "Platoon Stability"
    ```bash
    # Test platoon stability under speed changes
    pixi run start -t platoon_stability_2lanefree_carla
    ```

    **Features**:
    - 4-vehicle platoon
    - Dynamic speed changes
    - Time gap maintenance
    - Stability verification

=== "Platoon Joining"
    ```bash
    # Cooperative merge and platoon joining
    pixi run start -t platoon_joining_2lanefree_carla
    ```

    **Features**:
    - Mainline platoon with traffic
    - Cooperative merging from ramp
    - V2X communication
    - Real-time formation adjustment
    
    ![Platoon Joining](images/platoon_joining_2lanefree.gif)

=== "Platoon Back-Join"
    ```bash
    # Join platoon from behind with ML
    pixi run start -t platoon_joining_town06_carla --apply_ml
    ```

    **Features**:
    - Overtaking maneuvers
    - YOLOv8/YOLOv5 detection
    - Complex urban scenario
    - **Requires**: PyTorch
    
    ![Back Join](images/platoon_joining_town06.gif)

=== "SUMO Platoon"
    ```bash
    # Platoon with SUMO co-simulation
    pixi run start -t platoon_joining_2lanefree_cosim
    ```

    **Features**:
    - SUMO traffic integration
    - Realistic traffic patterns
    - **Requires**: SUMO
    
    ![SUMO Platoon](images/platoon_joining_cosim.gif)

---

### Advanced Scenarios

=== "Cooperative Perception"
    ```bash
    # V2X-enabled perception sharing
    pixi run start -t cooperception_cavs_town05 --apply_ml
    ```

    **Features**:
    - Shared object detection via V2X
    - Extended perception range
    - Occlusion handling
    - Multi-vehicle fusion

=== "Intersection Management"
    ```bash
    # Cooperative intersection navigation
    pixi run start -t intersection_town05 --apply_ml
    ```

    **Features**:
    - Traffic light compliance
    - V2I communication
    - Conflict resolution
    - Safety validation

**Performance**: Most scenarios run at 20 FPS simulation time on modern GPUs

---

### Configuration & Customization

**What you can modify**: Scenarios, vehicle behaviors, perception models

=== "YAML Configuration"
    ```yaml
    # Example: Modify perception settings
    vehicle:
      sensing:
        perception:
          activate: true  # Enable ML perception
          camera:
            fov: 100
            image_size_x: 800
            image_size_y: 600
        localization:
          gnss:
            noise_alt_stddev: 0.1  # Add GPS noise
    ```

=== "Custom ML Models"
    ```python
    # Add custom detection model
    from opencda.customize.ml_libs.ml_manager import MLManager

    class CustomDetector(MLManager):
        def __init__(self):
            super().__init__()
            self.model = load_your_model()
        
        def detect(self, image):
            return self.model(image)
    ```

=== "Scenario Creation"
    ```python
    # Create new scenario
    from opencda.scenario_testing.scenario_manager import ScenarioManager

    def custom_scenario():
        scenario_manager = ScenarioManager(config, apply_ml=True)
        
        # Spawn vehicles
        cavs = scenario_manager.create_vehicle_manager(['custom'])
        
        # Run simulation
        while True:
            scenario_manager.tick()
            # Custom logic here
    ```

**Tips**:

- Check `configs/opencda/scenario_testing/config_yaml/` for configuration examples
- Use `opencda/customize/` for custom implementations
- See [YAML Configuration Guide](opencda/yaml_define.md) for all options

---

## OpenCDA-MARL

### MARL Environment Setup

**What it adds**: Multi-Agent Reinforcement Learning capabilities for training cooperative driving policies

!!! info "Development Status"
    OpenCDA-MARL is currently in Phase 1 development, refer to [Phase 1](marl/architecture.md) for more details. The following sections show planned usage patterns.

=== "Basic Environment"
    ```python
    # PLACEHOLDER: MARL environment initialization
    from opencda_marl import MARLEnvironment

    # Create multi-agent environment
    env = MARLEnvironment(
        scenario="highway_merge",
        num_agents=4,
        observation_type="camera_lidar",
        reward_type="cooperative"
    )
    
    # Environment follows Gym interface
    obs = env.reset()
    actions = policy.get_actions(obs)
    next_obs, rewards, dones, info = env.step(actions)
    ```

=== "Training Configuration"
    ```yaml
    # PLACEHOLDER: MARL training configuration
    marl:
      environment:
        scenario: platoon_formation
        num_agents: 6
        max_steps: 1000

      training:
        algorithm: PPO
        batch_size: 4096
        learning_rate: 3e-4
        num_workers: 8
      
      rewards:
        safety_weight: 0.4
        efficiency_weight: 0.3
        cooperation_weight: 0.3
    ```

=== "Custom Scenarios"
    ```python
    # PLACEHOLDER: Custom MARL scenario
    from opencda_marl.scenarios import MARLScenario

    class CustomCooperativeScenario(MARLScenario):
        def __init__(self, config):
            super().__init__(config)
            # Custom initialization
        
        def reset(self):
            # Scenario-specific reset logic
            pass
        
        def compute_reward(self, agent_id, action, next_state):
            # Custom reward shaping
            pass
    ```

---

### MARL Training Examples

=== "Single-Agent Baseline"
    ```python
    # PLACEHOLDER: Single agent training
    # Content to be added
    ```

=== "Multi-Agent Training"
    ```python
    # PLACEHOLDER: Multi-agent cooperative training
    # Content to be added
    ```

=== "Distributed Training"
    ```python
    # PLACEHOLDER: Ray-based distributed training
    # Content to be added
    ```

**Note**: MARL features are under active development. Check the [MARL documentation](marl/architecture.md) for updates.

---

### MARL Evaluation

=== "Policy Evaluation"
    ```python
    # PLACEHOLDER: Evaluate trained policies
    # Content to be added
    ```

=== "Benchmark Comparison"
    ```python
    # PLACEHOLDER: Compare against baselines
    # Content to be added
    ```

---

## Troubleshooting

| Issue                       | Solution                                           |
| --------------------------- | -------------------------------------------------- |
| **CARLA connection failed** | Ensure CARLA 0.9.15 server is running on port 2000 |
| **PyTorch not found**       | Install with `pixi install` or `pip install torch` |
| **CUDA out of memory**      | Reduce batch size or use CPU mode                  |
| **Low FPS**                 | Try `--quality-level=Low` for CARLA server         |
| **SUMO errors**             | Verify SUMO installation and PATH settings         |

---

## More Information

- **OpenCDA Users**: Explore [scenario testing](opencda/scenarios.md) and [API documentation](api/opencda/overview.md)
- **MARL Researchers**: Check [MARL overview](marl/architecture.md) and [training guide](marl/training.md)
- **Contributors**: See [contributing guide](contributing.md)
- **Related Documentation**:
    - [Installation](installation.md)
    - [OpenCDA](opencda/core.md)
    - [API Reference](api/opencda/overview.md)
    - [OpenCDA Official Documentation](https://opencda.readthedocs.io/en/latest/)
    - [Carla Official Documentation](https://carla.readthedocs.io/en/latest/)
