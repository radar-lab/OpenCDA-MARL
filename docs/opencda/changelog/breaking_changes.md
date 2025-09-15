# Breaking Changes

!!! danger "Breaking Changes Summary" 
    Critical changes that require immediate action when migrating from original OpenCDA.

| Change          | Impact    | Action Required       | Status      |
| --------------- | --------- | --------------------- | ----------- |
| Package Manager | 🔴 High   | Reinstall environment | ✅ Complete |
| Version Module  | 🟡 Medium | Update imports        | ✅ Complete |
| Command Line    | 🟡 Medium | Remove `-v` argument  | ✅ Complete |
| CARLA Version   | 🟡 Medium | Fixed to 0.9.15       | ✅ Complete |

---

### Critical Changes

**What Changed**: Core infrastructure and interface changes

=== "Environment Setup" 
    ```bash 
    # Before (will fail) 
    conda activate opencda

    # After (required)
    pixi shell
    ```

=== "Code Imports" 
    ```python 
    # Before (will fail) 
    from opencda.version import version

    # After (required)
    from opencda import __version__, CARLA_VERSION
    ```

=== "Command Line" 
    ```bash 
    # Before (will fail) 
    python opencda.py -t scenario -v 0.9.12

    # After (required)
    python opencda.py -t scenario
    ```

**Impact**: Complete migration required for all users  
**Timeline**: Immediate action needed

---

[← Back to Changelog](index.md)
