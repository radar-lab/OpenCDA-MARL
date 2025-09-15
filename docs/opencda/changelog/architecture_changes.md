# Architecture Changes

!!! info "Navigation"
    TOC is limited to main sections for better readability. Use Ctrl+F to search for specific changes.

| Change                            | Date     | Impact | Breaking | Status      |
| --------------------------------- | -------- | ------ | -------- | ----------- |
| Object Detection Model Upgrade   | Aug 2025 | Low    | ❌ No    | ✅ Complete |
| Version Management Centralization | Aug 2025 | Low    | ✅ Yes   | ✅ Complete |
| MARL Extension Separation         | Aug 2025 | Low    | ❌ No    | ✅ Complete |
| Documentation Reorganization      | Aug 2025 | Low    | ❌ No    | ✅ Complete |

---

## Object Detection Architecture

**What Changed**: Upgraded from YOLOv5 to YOLOv8 with improved threading and compatibility

=== "Model Selection Logic"

    ```python
    # opencda/customize/ml_libs/ml_manager.py
    def __init__(self):
        try:
            # Prefer YOLOv8 (ultralytics package)
            from ultralytics import YOLO
            self.object_detector = YOLO('yolov8m.pt')
            self.use_v8 = True
        except ImportError:
            # Fallback to YOLOv5 (torch hub)
            self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5m')
            self.use_v8 = False
    ```

=== "Output Format Compatibility"

    ```python
    # Automatic conversion for camera-lidar fusion
    if self.use_v8:
        # YOLOv8 format: separate arrays (xyxy, conf, cls)
        xyxy = boxes.xyxy.cpu()
        conf = boxes.conf.cpu().unsqueeze(1)
        cls = boxes.cls.cpu().unsqueeze(1)
        # Convert to YOLOv5 format: [x1, y1, x2, y2, confidence, class_id]
        yolo_v5_format = torch.cat([xyxy, conf, cls], dim=1)
    ```

=== "Threading Improvements"

    ```python
    # Fixed GIL issues in perception_manager.py
    try:
        cv2.waitKey(1)
    except Exception as e:
        # Continue without waiting in case of threading issues
        pass
    ```

**Impact**: None - fully backward compatible  
**Benefits**: Better performance, improved accuracy, thread safety, modern dependency management

---

## Version Management System

**What Changed**: Removed `opencda/version.py`, centralized version info in `opencda/__init__.py`

=== "Before & After"

    ```python 
    # Before (will fail) 
    from opencda.version import version

    # After (required)
    from opencda import __version__, CARLA_VERSION
    ```

=== "New Structure"

    ```python 
    # opencda/__init__.py 
    
    __version__ = "0.1.3" 
    CARLA_VERSION = "0.9.15" 
    ```

**Impact**: All code importing version information needs updating  
**Benefits**: Simplified version management, single source of truth

---

## Module Organization

**What Changed**: Created separate `opencda_marl/` directory for MARL components

=== "Directory Structure"

    ```bash
    opencda/ # Core OpenCDA (unchanged) 
    ├── core/ ├── scenario_testing/
    └── ...

    opencda_marl/      # MARL extensions (new)
    ├── envs/          # Gym environments
    ├── agents/        # MARL algorithms
    ├── adapters/      # OpenCDA bridges
    └── configs/       # MARL configs
    ```

=== "Integration"

    ```python
    # Traditional OpenCDA (unchanged)
    from opencda.core.sensing import VehicleManager

    # With MARL extension (optional)
    from opencda_marl.adapters import VehicleAdapter
    vm = VehicleManager(vehicle, config, adapter=adapter)
    ```

**Impact**: No impact on existing OpenCDA code  
**Benefits**: Clear separation, parallel development, optional MARL integration

---

## Documentation Organization

**What Changed**: Structured docs to separate OpenCDA core from MARL extension

| Component      | Location        | Purpose                        |
| -------------- | --------------- | ------------------------------ |
| OpenCDA Core   | `docs/opencda/` | Original OpenCDA documentation |
| MARL Extension | `docs/marl/`    | MARL-specific guides and API   |
| API Reference  | `docs/api/`     | Combined API documentation     |

**Impact**: Documentation moved to new locations  
**Benefits**: Clear separation, easier maintenance, parallel development

---

[← Back to Changelog](index.md)
