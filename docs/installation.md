# Installation Guide

## Prerequisites

- **Windows 10/11** (64-bit)
- **Python 3.10**
- **CUDA 12.8** (for GPU acceleration)
- **CARLA 0.9.15**
- **Git**

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/opencda-marl.git
cd opencda-marl
```

## Step 2: Install Pixi

If you don't have pixi installed:

```bash
# Install pixi (Windows)
powershell -c "irm https://pixi.sh/install.ps1 | iex"
```

## Step 3: Setup Environment

```bash
# Install all dependencies
pixi install

# Test the installation
pixi run quick-test
```

## Step 4: Install CARLA

1. Download CARLA 0.9.15 from [CARLA Releases](https://github.com/carla-simulator/carla/releases/tag/0.9.15)
2. Extract to a folder (e.g., `D:\Applications\CARLA_0.9.15`)
3. Set environment variable:

   ```bash
   $env:CARLA_HOME = "D:\Applications\CARLA_0.9.15"
   ```

## Step 5: Verify Installation

```bash
# Run CARLA server (in one terminal)
cd $env:CARLA_HOME
.\CarlaUE4.exe

# Run OpenCDA test (in another terminal)
pixi run quick-test
```

## Development Environment

For documentation development:

```bash
# Activate docs environment
pixi shell -e docs

# Serve documentation locally
pixi run -e docs docs-serve
```

## Troubleshooting

### Common Issues

**CARLA Connection Error:**

- Ensure CARLA server is running
- Check firewall settings
- Verify CARLA_HOME environment variable

**GPU Issues:**

- Install NVIDIA drivers
- Verify CUDA 12.8 installation
- Check GPU memory availability

**Python Dependencies:**

- Use `pixi clean` and `pixi install` to refresh dependencies
- Ensure Python 3.10 is being used

### Getting Help

1. Check the [FAQ](faq.md)
2. Search existing [Issues](https://github.com/lgcyaxi/opencda-marl/issues)
3. Create a new issue with detailed error information
