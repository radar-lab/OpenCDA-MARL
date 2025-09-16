# Installation Guide

## Prerequisites

- **Windows 10/11** (64-bit)
- **Python 3.10**
- **CUDA 12.8** (for GPU acceleration)
- [**CARLA 0.9.15**](https://github.com/carla-simulator/carla/releases)
- **Git**

## Step 1: Clone Repository

```bash
git clone https://github.com/radar-lab/opencda-marl.git
cd opencda-marl
```

## Step 2: Install Pixi

If you don't have pixi installed:

```bash
# Install pixi (Windows)
powershell -c "irm https://pixi.sh/install.ps1 | iex"

# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | sh
```

## Step 3: Setup Environment

```bash
# Find compatible pixi.toml file from the dependencies folder
# For example, if you are using AMD GPU, you should use the pixi.toml file 
# in the dependencies/pixi/pixi_Linux_ROCm.toml

# Install all dependencies
pixi install

# Test the installation
pixi run marl-quick-test
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
# Or manually start the server by clicking the CarlaUE4.exe file

# Run OpenCDA test (in another terminal)
pixi run marl-quick-test
```

## Development Environment

For documentation development:

```bash
# Activate docs environment
pixi shell -e docs

# Serve documentation locally
pixi run -e docs docs-serve
```

## Getting Help

1. Check the [FAQ](faq.md)
2. Search existing [Issues](https://github.com/radar-lab/opencda-marl/issues)
3. Create a new issue with detailed error information
