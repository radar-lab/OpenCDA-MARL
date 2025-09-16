# Frequently Asked Questions

## Installation Issues

### Q: Pixi environment issues

**A:** Try these steps:

```bash
# Clean pixi cache
pixi clean

# Reinstall environment
pixi install --force-reinstall
```

## Common Issues

### Q: CARLA Connection Error

**A:** Ensure CARLA server is running, check firewall settings, and verify CARLA_HOME environment variable.

### Q: CUDA compatibility issues

**A:** Ensure you have:

- Install NVIDIA drivers (if using NVIDIA GPU)
- Verify CUDA 12.8 installation
- Check GPU memory availability
- PyTorch compiled with correct CUDA version

## OpenCDA-MARL Integration

### Q: How to add new MARL algorithms?

**A:** Check the [MARL Framework](marl/overview.md) documentation and [API Reference](api/opencda-marl/overview.md).

## OpenCDA Integration

### Q: Perception module not working

**A:** Check:

- `activate: true` in perception config
- PyTorch installation with `--apply_ml` flag
- Camera/LiDAR sensor configuration

### Q: torch.cuda.amp deprecation warnings and boolean flag errors

**A:** If you see warnings about deprecated `torch.cuda.amp` or errors like `Expected 'device_type' of type 'str', got: '<class 'bool'>'`, this is due to PyTorch API changes.

**Solution:** Replace the YOLOv5 common.py file with the fixed version:

1. Copy `docs/files/common.py` from this repository
2. Replace your cached YOLOv5 file at:

   ```bash
   C:/Users/[username]/.cache/torch/hub/ultralytics_yolov5_master/models/common.py
   ```

3. This fixes both the deprecated API warnings and the boolean flag error

The fixed version uses the new `torch.amp.autocast` API with proper device type and enabled parameters.

### Q: How to create custom scenarios?

**A:** See the [YAML Configuration Guide](opencda/yaml_define.md) for detailed instructions.

## Getting Help

Still having issues?

- Check the [Original OpenCDA Documentation](https://opencda-documentation.readthedocs.io/en/latest/)
- Open an issue on [GitHub](https://github.com/radar-lab/opencda-marl/issues)
- Review the [Contributing Guide](contributing.md)
