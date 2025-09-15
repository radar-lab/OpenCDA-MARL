# CavWorld API

`CavWorld` serves as the global registry for CAVs and manages shared ML models. Prevents duplicate
model loading and enables efficient resource utilization.

| Attribute               | Type        | Purpose                     |
| ----------------------- | ----------- | --------------------------- |
| `vehicle_id_set`        | `set`       | CARLA vehicle ID tracking   |
| `_vehicle_manager_dict` | `dict`      | VehicleManager registry     |
| `_platooning_dict`      | `dict`      | PlatooningManager registry  |
| `ml_manager`            | `MLManager` | Shared YOLOv8/YOLOv5 models |

---

## Core Methods

**What it does**: Manages vehicle registration and shared ML model access

=== "Initialization" 
    ```python 
    # Standard simulation 
    cav_world = CavWorld(apply_ml=False)

    # ML-enabled with shared models
    cav_world = CavWorld(apply_ml=True)
    ```

=== "Vehicle Management" 
    ```python 
    # Register vehicle (automatic during creation)
    cav_world.update_vehicle_manager(vehicle_manager)

    # Retrieve by ID
    vm = cav_world.get_vehicle_manager(vehicle_id)

    # Remove from registry
    cav_world.remove_vehicle_manager(vehicle_id)
    ```

=== "ML Model Access"
    ```python 
    # Access shared models from any component 
    if cav_world.ml_manager: 
        detections = cav_world.ml_manager.object_detector(images) 
    ```

## ML Integration

=== "Model Selection"
    **What Changed**: Automatic YOLOv8/YOLOv5 selection with shared instance
    ```python 
    def _initialize_ml_models(self): 
        try: 
            # Prefer YOLOv8 (better performance) 
            from ultralytics import YOLO 
            detector = YOLO('yolov8m.pt') 
            use_v8 = True 
        except ImportError: 
            # Fallback to YOLOv5 (compatibility) 
            detector = torch.hub.load('ultralytics/yolov5', 'yolov5m') 
            use_v8 = False 
        return MLManager(detector, use_v8) 
    ```

=== "Simulation Setup" 
    ```python 
    # Create CAV world with ML support 
    cav_world = CavWorld(apply_ml=True)

    # Create scenario manager with shared world
    scenario_manager = ScenarioManager(
        scenario_params, apply_ml=True, cav_world=cav_world
    )

    # Vehicles automatically register during creation
    vehicles = scenario_manager.create_vehicle_manager(['platooning'])
    ```

=== "Custom ML Models" 
    ```python 
    # Extend with custom models 
    class CustomMLManager(MLManager): 
        def __init__(self): 
            super().__init__() 
            self.custom_detector = load_custom_model() 
    cav_world = CavWorld(apply_ml=True)
    cav_world.ml_manager = CustomMLManager()
    ```

=== "Thread Safety" 
    ```python 
    # Thread-safe vehicle management 
    with cav_world.vehicle_lock: 
        cav_world.update_vehicle_manager(vehicle_manager)

    # Safe model access from multiple threads
    detections = cav_world.ml_manager.object_detector(images)
    ```

**Related**: 

- [VehicleManager](vehicle_manager.md) 
- [PerceptionManager](perception_manager.md) 
- [ScenarioManager](scenario_manager.md)
