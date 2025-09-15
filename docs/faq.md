# Frequently Asked Questions

## Installation Issues

### Q: CARLA installation fails on Windows

**A:** Make sure you have:

- Windows 10/11 64-bit
- Latest Visual C++ Redistributables
- Sufficient disk space (>20GB)
- Download CARLA 0.9.15 from official releases

### Q: Pixi environment issues

**A:** Try these steps:

```bash
# Clean pixi cache
pixi clean

# Reinstall environment
pixi install --force-reinstall
```

### Q: CUDA compatibility issues

**A:** Ensure you have:

- CUDA 12.8 installed
- Compatible GPU driver
- PyTorch compiled with correct CUDA version

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

## MARL Training Issues

### Q: Training runs but no learning occurs

**A:** Check:

- Learning rate (try 1e-4 to 1e-3)
- Reward function implementation
- Environment reset logic
- Agent observation space

### Q: Multi-agent coordination fails

**A:** Verify:

- Communication range settings
- V2X manager configuration
- Agent spawn positions
- Scenario YAML configuration

## OpenCDA Integration

### Q: OpenCDA scenarios don't work with MARL

**A:** Ensure:

- Proper YAML configuration
- Compatible OpenCDA version
- CARLA server running
- Correct scenario parameters

### Q: Perception module not working

**A:** Check:

- `activate: true` in perception config
- PyTorch installation with `--apply_ml` flag
- Camera/LiDAR sensor configuration

## Performance Issues

### Q: Simulation runs slowly

**A:** Try:

- Reduce number of agents
- Disable unnecessary visualizations
- Use synchronous mode
- Optimize CARLA settings

### Q: Memory usage too high

**A:** Consider:

- Smaller replay buffer size
- Batch processing
- Reduce observation dimensions
- Clean unused variables

## Configuration Questions

### Q: How to create custom scenarios?

**A:** See the [YAML Configuration Guide](opencda/yaml_define.md) for detailed instructions.

### Q: How to add new MARL algorithms?

**A:** Check the [MARL Framework](marl/overview.md) documentation and [API Reference](api/opencda-marl/overview.md).

## Getting Help

Still having issues?

- Check the [Original OpenCDA Documentation](https://opencda-documentation.readthedocs.io/en/latest/)
- Open an issue on [GitHub](https://github.com/lgcyaxi/opencda-marl/issues)
- Review the [Contributing Guide](contributing.md)
