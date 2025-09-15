# OpenCDA API Overview

!!! info "Complete API Documentation" 
    For comprehensive OpenCDA API documentation, visit the [Official OpenCDA Documentation](https://opencda-documentation.readthedocs.io/en/latest/modules.html).

| Component                                      | Purpose                                | Status      |
| ---------------------------------------------- | -------------------------------------- | ----------- |
| [CavWorld](cav_world.md)                       | Global CAV registry & shared ML models | ✅ Enhanced |
| [ScenarioManager](scenario_manager.md)         | Simulation setup & vehicle spawning    | ✅ Stable   |
| [VehicleManager](vehicle_manager.md)           | Individual vehicle coordination        | ✅ Enhanced |
| [PerceptionManager](perception_manager.md)     | Sensor fusion & object detection       | ✅ YOLOv8   |
| [LocalizationManager](localization_manager.md) | Position tracking & filtering          | ✅ Stable   |
| [BehaviorAgent](behavior_agent.md)             | Path planning & decision making        | ✅ Stable   |
| [V2XManager](v2x_manager.md)                   | Vehicle-to-vehicle communication       | ✅ Stable   |

## Key Updates

=== "ML Improvements" 
    - **YOLOv8 Integration**: ~20% performance improvement with automatic YOLOv5 fallback
    - **Thread Safety**: Fixed GIL issues in multi-threaded scenarios
    - **Memory Efficiency**: Shared model instances across all vehicles
    - **Compatibility**: Maintained existing camera-lidar fusion interfaces

=== "System Changes" 
    - **Version Management**: Centralized in `opencda/__init__.py` 
    - **CARLA Version**: Fixed to 0.9.15 for consistency 
    - **Package Manager**: Pixi-based environment management 
    - **Documentation**: Enhanced API structure with examples

---

## Core Workflow

**What it does**: Orchestrates simulation setup, vehicle management, and execution loop

=== "Basic Setup"
    ```python 
    from opencda.scenario_testing.utils.customized_map_api import customized_map_helper

    # Load configuration and create CAV world
    scenario_params = load_yaml(config_yaml)
    cav_world = CavWorld(apply_ml=True)

    # Initialize scenario manager
    scenario_manager = ScenarioManager(
        scenario_params, apply_ml=True,
        xodr_path=xodr_path, cav_world=cav_world
    )
    ```

=== "Vehicle Creation" 
    ```python 
    # Create individual CAVs and platoons 
    single_cavs = scenario_manager.create_vehicle_manager(['platooning']) 
    platoons = scenario_manager.create_platoon_manager()

    # Add background traffic
    traffic_manager, bg_vehicles = scenario_manager.create_traffic_carla()
    ```

=== "Simulation Loop" 
    ```python 
    while True: 
        scenario_manager.tick()

        # Update platoons
        for platoon in platoons:
            platoon.update_information()
            platoon.run_step()

        # Update individual CAVs
        for cav in single_cavs:
            if not cav.v2x_manager.in_platoon():
                cav.update_info()
                control = cav.run_step()
                cav.vehicle.apply_control(control)
    ```


=== "Design Principles"

    | Principle                | Implementation                            | Benefits                                 |
    | ------------------------ | ----------------------------------------- | ---------------------------------------- |
    | **Manager Pattern**      | Dedicated manager for each subsystem      | Clear responsibilities, easy testing     |
    | **Configuration-Driven** | YAML configs with inheritance             | Flexible scenarios, easy experimentation |
    | **Plugin Architecture**  | Custom algorithms in `opencda/customize/` | Extensible without core changes          |

---

## Integration Guide

!!! example "Quick Start" 
    Get started with OpenCDA in 4 steps:
    - Installation
        ```bash
        # Install with Pixi 
        pixi install 
        pixi shell 
        ```
    - Quick Test
        ```bash
        # Verify installation 
        pixi run quick-test 
        ```
    - Basic Scenario
        ```python
        # Run simple scenario 
        python opencda.py -t single_2lanefree_carla 
        ```
    - Custom Development
        ```python
        # Extend with custom algorithms 
        from opencda.customize.ml_libs import MyCustomDetector 
        cav_world = CavWorld(apply_ml=True, custom_ml=MyCustomDetector()) 
        ```

**Integration Points**: CARLA 0.9.15, SUMO co-simulation, V2X communication, PyTorch ML models
