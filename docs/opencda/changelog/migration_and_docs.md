# Migration Guide & Documentation Changes

!!! info "Navigation" 
    TOC is limited to main sections for better readability. Use Ctrl+F to search for specific changes.

| Change                      | Date     | Impact | Breaking | Status      |
| --------------------------- | -------- | ------ | -------- | ----------- |
| Package Manager Migration   | Aug 2025 | High   | ✅ Yes   | ✅ Complete |
| Documentation System        | Aug 2025 | Low    | ❌ No    | ✅ Complete |
| Version Import Changes      | Aug 2025 | Low    | ✅ Yes   | ✅ Complete |
| Command Line Simplification | Aug 2025 | Medium | ✅ Yes   | ✅ Complete |

---

## Quick Migration Steps

**What Changed**:

1. Complete environment and tooling migration from Conda to Pixi
2. Migrated from Sphinx to MkDocs with Material theme

=== "Compatibility Matrix"

    | Component     | Supported       | Removed       | Notes               |
    | ------------- | --------------- | ------------- | ------------------- |
    | Python 3.10.x | ✅ Required     | 3.7-3.9       | Use 3.10.11         |
    | CARLA 0.9.15  | ✅ Only version | 0.9.11-0.9.14 | Fixed version       |
    | Pixi          | ✅ Required     | Conda         | New package manager |
    | MkDocs        | ✅ Current      | Sphinx        | New documentation   |

=== "Migration Checklist"

    !!! warning "Required Actions" 
        Complete all items before using OpenCDA-MARL:

        -   [ ] Install Pixi: `curl -fsSL https://pixi.sh/install.sh | bash`
        -   [ ] Remove Conda environment: `conda env remove -n opencda`
        -   [ ] Update imports: Find/replace `opencda.version` → `opencda`
        -   [ ] Update scripts: Remove all `-v` arguments
        -   [ ] Use new docs: `pixi run docs-serve`

=== "Environment Migration"

    ```bash
    # 1. Install Pixi
    curl -fsSL https://pixi.sh/install.sh | bash
    # Linux/macOS
    # OR
    # Windows PowerShell
    iwr -useb https://pixi.sh/install.ps1 | iex

    # 2. Clone and setup
    git clone https://github.com/radar-lab/opencda-marl.git
    cd opencda-marl
    pixi install
    pixi shell

    # 3. Verify installation
    pixi run quick-test
    ```

=== "Documentation System"

    **What Changed**: Migrated from Sphinx to MkDocs with Material theme

    ```bash
    # Old (Sphinx)
    cd docs && make html

    # New (MkDocs)
    pixi run docs-serve    # Live preview at localhost:8000
    pixi run docs-build    # Build static site
    pixi run docs-deploy   # Deploy to GitHub Pages
    ```

=== "File Structure"

    ```bash
    docs/ # New MkDocs documentation
    ├── opencda/ # OpenCDA core docs
    ├── marl/ # MARL extension docs
    └── api/ # API references
    ```

    **Impact**: New documentation commands and structure
    **Benefits**: Modern UI, better navigation, faster builds, dark mode, search

=== "Common Issues & Solutions"

    | Issue                                                    | Solution                                       |
    | -------------------------------------------------------- | ---------------------------------------------- |
    | `ModuleNotFoundError: No module named 'opencda.version'` | Update to `from opencda import __version__`    |
    | `error: unrecognized arguments: -v`                      | Remove `-v` or `--carla_version` from commands |
    | `conda: command not found`                               | Use `pixi` instead of `conda`                  |
    | Documentation not building                               | Use `pixi run docs-serve` instead of Sphinx    |


---

## Rollback Plan

!!! warning "Emergency Rollback"
    If you need to revert to original OpenCDA:

    === "Quick Rollback" 
        ```bash 
        # 1. Clone original repository 
        git clone https://github.com/ucla-mobility/OpenCDA.git opencda-original cd opencda-original

        # 2. Use original environment
        conda env create -f environment.yml
        conda activate opencda
        ```

    === "Documentation" 
        Original documentation: https://opencda-documentation.readthedocs.io/

---

[← Back to Changelog](index.md)
