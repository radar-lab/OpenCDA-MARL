# PerceptionManager API

`PerceptionManager` handles sensor fusion and object detection using multi-view cameras, LiDAR, and
ML models (YOLOv8/YOLOv5).

| Component               | Status     | Purpose                               |
| ----------------------- | ---------- | ------------------------------------- |
| Multi-Camera Array      | ✅ Active  | Front, left, right RGB cameras        |
| LiDAR Sensor            | ✅ Active  | Distance filtering and validation     |
| YOLOv8 Detection        | 🔄 Updated | Primary ML model with YOLOv5 fallback |
| Traffic Light Detection | ✅ Active  | Ground truth from CARLA server        |
| Thread Safety           | 🔧 Fixed   | Robust cv2.waitKey error handling     |

---

## Sensor Configuration

**What it manages**: Multi-sensor setup with ML-based object detection

=== "Sensor Array" 
    ```python 
    # Multi-view camera setup 
    self.rgb_camera = [
        CameraSensor(vehicle,'front'), 
        CameraSensor(vehicle, 'right'), 
        CameraSensor(vehicle, 'left') 
    ]

    # LiDAR for distance validation
    self.lidar = LidarSensor(vehicle, config['lidar'])

    # Optional semantic LiDAR for data collection
    if data_dump:
        self.semantic_lidar = SemanticLidarSensor(vehicle, config['lidar'])
    ```

=== "ML Integration" 
    ```python 
    # Shared ML manager from CavWorld 
    self.ml_manager = ml_manager 

    # YOLOv8/YOLOv5 
    self.activate = config['activate'] 

    # Detection state
    self.objects = {'vehicles': [], 'traffic_lights': []}
    ```

=== "YAML Config"
    ```yaml 
    perception: 
        activate: true 
        camera: 
            number: 3 # front, left, right 
            fov: 100 
            image_size_x: 800 
            image_size_y: 600 
            lidar: 
                channels: 64 
                range: 100.0 
                points_per_second: 56000 
    ```

## Detection Pipeline

**What it does**: Multi-modal object detection with ML models and sensor fusion

=== "Main Detection" 
    ```python 
    def detect(self, ego_pos): 
        objects = {'vehicles': [], 'traffic_lights': []}

        if self.activate:
            # ML-based detection (YOLOv8/YOLOv5)
            objects = self.activate_mode(objects)
        else:
            # Ground truth from CARLA (perfect accuracy)
            objects = self.deactivate_mode(objects)

        return objects
    ```

=== "ML Detection Flow" 
    ```python 
    def activate_mode(self, objects): 
        # Process each camera view for camera in self.rgb_camera: 
        if camera.image is not None: 
            # YOLOv8/YOLOv5 inference 
            detections = self.ml_manager.object_detector(camera.image) 
            vehicles = self.process_detections(detections, camera)
            objects['vehicles'].extend(vehicles)

        # LiDAR distance filtering
        objects['vehicles'] = self.lidar_filter(objects['vehicles'])

        # Add traffic lights (ground truth)
        objects = self.retrieve_traffic_lights(objects)
        return objects
    ```

=== "Output Format"
    ```python 
    # Standardized detection format 
    { 
        'vehicles': [ { 
            'location': carla.Location, 
            'confidence': float, 
            'bounding_box': [x1, y1, x2, y2], 
            'distance': float # from LiDAR 
            } 
        ], 
        'traffic_lights': [ { 
            'location': carla.Location, 
            'state': 'Red'|'Yellow'|'Green' 
            }
        ]
    } 
    ```

## Detection Modes

**What it supports**: Flexible detection modes for different use cases

=== "ML Mode (activate=True)" 

    - **Multi-Camera Fusion**: Front, left, right camera processing 
    - **YOLOv8/YOLOv5**: Automatic model selection with fallback 
    - **LiDAR Validation**: Distance filtering and range checking 
    - **Thread Safety**: Robust error handling for concurrent access 
    - **Use Case**: Realistic ML-based perception for training/evaluation

=== "Ground Truth Mode (activate=False)" 

    - **Perfect Accuracy**: CARLA server ground truth
    - **Zero Latency**: No ML inference overhead 
    - **Debugging**: Ideal for algorithm development 
    - **Use Case**: Baseline comparison, rapid prototyping

=== "Traffic Light Detection" 
    ```python 
    def retrieve_traffic_lights(self, objects): 
        tl_list = world.get_actors().filter('traffic.traffic_light*')

        for tl in tl_list:
            if self.dist(tl) < 50:  # 50m range
                traffic_light = TrafficLight(
                    tl.get_location(),
                    tl.get_state()  # Red/Yellow/Green
                )
                objects['traffic_lights'].append(traffic_light)
    ```

=== "Configuration"
    Set `activate: true/false` in YAML for ML vs ground truth mode

## YOLOv8 Technical Details

**What's new**: Enhanced ML pipeline with automatic model selection and compatibility

=== "Model Selection Logic"
    ```python 
    # Automatic YOLOv8/YOLOv5 selection in MLManager 
    try: 
        from ultralytics import YOLO 
        self.object_detector = YOLO('yolov8m.pt') 
        self.use_v8 = True 
    except ImportError: 
        self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5m') 
        self.use_v8 = False 
    ```

=== "Format Compatibility" 
    ```python 
    def process_detections(self, results, camera): 
        if self.use_v8: 
            # Convert YOLOv8 → YOLOv5 format for compatibility 
            boxes = results[0].boxes 
            xyxy = boxes.xyxy.cpu() 
            conf = boxes.conf.cpu().unsqueeze(1) 
            cls = boxes.cls.cpu().unsqueeze(1) 
            detections = torch.cat([xyxy, conf, cls], dim=1) 
        else: 
            # YOLOv5 format already compatible 
            detections = results.xyxy[0].cpu()

        return self.camera_lidar_fusion(detections, camera)
    ```

=== "Thread Safety Fix"
    ```python 
    # Fixed cv2.waitKey threading issues 
    try: 
        cv2.waitKey(1) 
    except Exception: 
        # Continue without waiting to prevent GIL hangs 
        pass 
    ```

## Integration Examples

!!! example "Usage Patterns" 
    Common ways to use PerceptionManager in scenarios:

    === "Basic Detection" 
        ```python 
        # Create perception manager 
        perception_manager = PerceptionManager(
            vehicle=carla_vehicle, config_yaml=config['sensing']['perception'], ml_manager=cav_world.ml_manager
        )

        # Detection in simulation loop
        ego_pos = vehicle.get_transform()
        detected_objects = perception_manager.detect(ego_pos)

        # Process results
        for vehicle_detection in detected_objects['vehicles']:
            distance = vehicle_detection['distance']
            confidence = vehicle_detection['confidence']
        ```

    === "Custom ML Models" 
        ```python 
        # Extend MLManager for custom detection class
        class CustomMLManager(MLManager): 
            def __init__(self): 
                super().__init__() 
                self.custom_detector = load_custom_model()

                def detect_custom_objects(self, image):
                    return self.custom_detector(image)

            # Use with PerceptionManager
            cav_world.ml_manager = CustomMLManager()
        ```

    === "Performance Monitoring" 
        ```python 
        # Monitor detection performance 
        import time
        start_time = time.time()
        objects = perception_manager.detect(ego_pos)
        detection_time = time.time() - start_time

        print(f"Detection took {detection_time*1000:.1f}ms")
        print(f"Found {len(objects['vehicles'])} vehicles")
        ```

**Related**: 

- [CavWorld](cav_world.md) 
- [VehicleManager](vehicle_manager.md) 
- [BehaviorAgent](behavior_agent.md)
