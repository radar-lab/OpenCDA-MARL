# API Changes

!!! info "Navigation" 
    TOC is limited to main sections for better readability. Use Ctrl+F to search for specific changes.

| Change                         | Date     | Impact | Breaking | Status      |
| ------------------------------ | -------- | ------ | -------- | ----------- |
| YOLOv5 to YOLOv8 Upgrade       | Aug 2025 | Low    | ❌ No    | ✅ Complete |
| Removed CARLA Version Argument | Aug 2025 | Medium | ✅ Yes   | ✅ Complete |
| Version Import Changes         | Aug 2025 | Low    | ✅ Yes   | ✅ Complete |

---

## Machine Learning Models

**What Changed**: Upgraded from YOLOv5 to YOLOv8 for object detection with automatic fallback

| Component                | Status       | Replacement/Improvement           |
| ------------------------ | ------------ | --------------------------------- |
| YOLOv5 object detection  | ✅ Supported | YOLOv8 with YOLOv5 fallback      |
| torch.hub loading        | ✅ Supported | ultralytics package (preferred)  |
| Detection output format  | ✅ Compatible| Automatic format conversion       |

=== "Usage (No Changes Required)"

    ```python
    # Your existing code works unchanged
    from opencda.customize.ml_libs.ml_manager import MLManager
    
    ml_manager = MLManager()  # Automatically uses YOLOv8 if available
    detection = ml_manager.object_detector(images)
    ```

=== "Benefits"

    - **Better Performance**: YOLOv8 is ~20% faster than YOLOv5
    - **Higher Accuracy**: Improved detection accuracy on COCO dataset
    - **Modern API**: Uses ultralytics package (already in dependencies)
    - **Thread Safety**: Fixed GIL threading issues with cv2.waitKey
    - **Automatic Fallback**: Falls back to YOLOv5 if YOLOv8 unavailable

=== "Technical Details"

    ```python
    # Automatic model selection
    try:
        from ultralytics import YOLO
        self.object_detector = YOLO('yolov8m.pt')  # YOLOv8 medium
        self.use_v8 = True
    except ImportError:
        self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5m')
        self.use_v8 = False
    
    # Output format compatibility maintained
    # YOLOv8 results automatically converted to YOLOv5 format for camera-lidar fusion
    ```

**Impact**: None - fully backward compatible  
**Benefits**: Better performance, improved accuracy, modern codebase, thread safety

---

## Command Line Interface

**What Changed**: Removed `-v/--carla_version` argument, CARLA version now fixed to 0.9.15

| API                        | Status     | Replacement                       |
| -------------------------- | ---------- | --------------------------------- |
| `opencda.version` module   | ❌ Removed | `from opencda import __version__` |
| Command line `-v` argument | ❌ Removed | Fixed to CARLA 0.9.15             |

=== "Before & After"

    ```bash
    # Before (will fail)
    python opencda.py -t single_2lanefree_carla -v 0.9.12
    python opencda.py -t platoon_joining_2lanefree_carla -v 0.9.11

    # After (required)
    python opencda.py -t single_2lanefree_carla
    python opencda.py -t platoon_joining_2lanefree_carla
    ```

=== "Migration Steps"

    - Remove `-v` or `--carla_version` from all scripts
    - Update CI/CD pipelines
    - Update documentation examples

    **Impact**: All command line scripts need updating
    **Benefits**: Simplified interface, consistency across scenarios, reduced compatibility issues

---

## Function Signatures

**What Changed**: Version import moved from separate module to main package

=== "Import Changes"

    ```python
    # Before (will fail)
    from opencda.version import version

    # After (required)
    from opencda import __version__, CARLA_VERSION
    ```

=== "Available Constants"

    ```python
    from opencda import __version__, CARLA_VERSION

    print(f"OpenCDA version: {__version__}")      # "0.1.3"
    print(f"CARLA version: {CARLA_VERSION}")      # "0.9.15"
    ```

**Impact**: All code importing version information needs updating  
**Benefits**: Cleaner package structure, centralized version management

---

## Compatibility & Stability

!!! success "Stability Commitment" 
    - **Core OpenCDA APIs**: Remain stable with backward compatibility
    - **MARL APIs**: Experimental, may change in future versions
    - **Breaking Changes**: Always documented with migration guides

| Component      | Stability       | Versioning          |
| -------------- | --------------- | ------------------- |
| Core OpenCDA   | ✅ Stable       | Semantic versioning |
| MARL Extension | ⚠️ Experimental | May change          |
| CLI Interface  | ✅ Stable       | Documented changes  |

---

[← Back to Changelog](index.md)
