# Platform & Dependencies Changes

!!! info "Navigation" 
    TOC is limited to main sections for better readability. Use Ctrl+F to search for specific changes.

| Change                         | Date     | Impact | Breaking       | Status      |
| ------------------------------ | -------- | ------ | -------------- | ----------- |
| Conda → Pixi Migration         | Aug 2025 | High   | ✅ Yes         | ✅ Complete |
| Python 3.10.x Requirement      | Aug 2025 | Medium | ⚠️ Potentially | ✅ Complete |
| CARLA 0.9.15 Standardization   | Aug 2025 | Medium | ✅ Yes         | ✅ Complete |
| Enhanced Windows Support       | Aug 2025 | Low    | ❌ No          | ✅ Complete |
| Documentation System Migration | Aug 2025 | Low    | ❌ No          | ✅ Complete |

---

## Package Management

**What Changed**: Replaced Conda/Anaconda with Pixi package manager

=== "Before & After"

    ```bash
    # Before (Conda)
    conda env create -f environment.yml
    conda activate opencda
    conda install new-package

    opencda conda install new-package

    # After (Pixi)
    pixi install
    pixi shell  # or pixi run <command>
    pixi add new-package
    ```

=== "Migration Steps"

    1. Install Pixi: `curl -fsSL https://pixi.sh/install.sh | bash`
    2. Remove conda environment: `conda env remove -n opencda`
    3. Run: `pixi install`
    4. Use `pixi shell` or `pixi run` for commands

=== "Benefits" 
    - **Performance**: 10x faster environment creation 
    - **Reliability**: Better dependency resolution
    - **Features**: Built-in task runner
    - **Compatibility**: Cross-platform support
    - **Reproducibility**: Lock files and consistent environments

**Impact**: Complete environment rebuild required  
**Files**: `pixi.toml` replaces `environment.yml`

---

## Python & Runtime

**What Changed**: Standardized on Python 3.10.x for Windows/CUDA compatibility

=== "Compatibility Matrix"

    | Component | Version | Notes |
    |-----------|---------|-------|
    | Python | 3.10.11 | Required |
    | CUDA | 12.8 | For GPU support |
    | CARLA | 0.9.15 | Fixed version |
    | PyTorch | 2.0+ | CUDA 12.8 compatible |

=== "Platform Support"

    | OS | Python | CUDA | Status |
    |----|--------|------|--------|
    | Windows 11 | 3.10.11 | 12.8 | ✅ Fully Supported |
    | Windows 10 | 3.10.11 | 12.8 | ✅ Fully Supported |
    | Ubuntu 22.04 | 3.10.x | 12.x | ✅ Fully Supported |
    | Ubuntu 20.04 | 3.10.x | 11.x | ⚠️ Limited Support |
    | macOS | 3.10.x | N/A | ❌ Not Supported |

    **Impact**: May require Python version upgrade
    **Benefits**: Consistent Windows support, CUDA 12.8 compatibility

---

## CARLA Integration & Dependencies

**What Changed**: 

1. Fixed CARLA version to 0.9.15 with pre-built wheels
2. Updated major packages for Python 3.10 and MARL support
3. Migrated from Sphinx to MkDocs for documentation
   
=== "Installation" 
    ```bash 
    # Automatic via Pixi (recommended) 
    pixi install

    # Manual installation
    pip install dependencies/whl/carla-0.9.15-cp310-cp310-win_amd64.whl
    ```

=== "File Structure"
    ```bash 
    dependencies/whl/ 
    ├── carla-0.9.15-cp310-cp310-win_amd64.whl 
    ├── carla-0.9.15-cp310-cp310-linux_x86_64.whl 
    └── ...
    
    docs/ # New MkDocs documentation 
    ├── opencda/ # OpenCDA core docs 
    ├── marl/ # MARL extension docs 
    └── api/ # API references
    ```

    **Impact**: All scenarios now use CARLA 0.9.15 consistently & documentation is now in MkDocs
    **Benefits**: Reduced compatibility issues, simplified deployment & better documentation

=== "Key Dependencies"
    ```toml 
    # pixi.toml 
    [dependencies] 
    python = "3.10.11" 
    carla = "0.9.15" # via local wheel 
    pytorch = ">=2.0" # CUDA 12.8 compatible 
    numpy = "<2.0" # Compatibility constraint 
    opencv-python = ">=4.8" 
    omegaconf = ">=2.3" 
    ray = ">=2.0" # For MARL 
    ```
    **Impact**: Automatic installation via Pixi  
    **Benefits**: MARL support, better GPU performance, modern tooling

=== "Development Tools"
    ```bash 
    # Old (Sphinx)
    cd docs && make html

    # New (MkDocs)
    pixi run docs-serve    # Live preview at localhost:8000
    pixi run docs-build    # Build static site
    pixi run docs-deploy   # Deploy to GitHub Pages
    ```

---

## Installation Guide

!!! example "Clean Installation" 
    Follow these steps for a fresh installation

=== "Quick Start" 
    ```bash 
    # 1. Install 
    Pixi curl -fsSL https://pixi.sh/install.sh | bash
    # 2. Clone repository
    git clone https://github.com/radar-lab/opencda-marl.git
    cd opencda-marl

    # 3. Install dependencies
    pixi install

    # 4. Activate environment
    pixi shell

    # 5. Verify installation
    pixi run quick-test
    ```

=== "Docker Alternative" 
    ```bash 
    # Build image 
    docker build -t opencda-marl .

    # Run with GPU support
    docker run --gpus all opencda-marl
    ```

=== "Windows PowerShell" 
    ```powershell 
    # 1. Install Pixi 
    iwr -useb https://pixi.sh/install.ps1 | iex

    # 2. Follow same steps as Quick Start
    pixi install
    pixi shell
    pixi run quick-test
    ```

---

[← Back to Changelog](index.md)
