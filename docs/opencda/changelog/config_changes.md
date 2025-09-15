# Configuration Changes

!!! info "Navigation"
    TOC is limited to main sections for better readability. Use Ctrl+F to search for specific changes.

| Change                  | Date      | Impact | Breaking | Status      |
| ----------------------- | --------- | ------ | -------- | ----------- |
| Config File Relocation  | Aug 2025  | Low    | ❌ No    | ✅ Complete |
| CARLA Version Fixed     | Aug 2025  | Medium | ✅ Yes   | ✅ Complete |
| Default Template System | Inherited | Low    | ❌ No    | ✅ Complete |

---

## Configuration File Organization

**What Changed**: OpenCDA YAML configuration files relocated for better organization

=== "New Structure"

    ```bash
    configs/
    ├── opencda/              # OpenCDA scenario configs (moved here)
    │   ├── default.yaml
    │   ├── single_2lanefree_carla.yaml
    │   ├── platoon_joining_2lanefree_carla.yaml
    │   └── ...
    └── marl/                 # MARL configs (future)
        └── ...
    ```

=== "Previous Location"

    ```bash
    opencda/scenario_testing/config_yaml/  # Old location
    ```

**Impact**: Update paths when loading configuration files  
**Benefits**: Clean separation between OpenCDA and MARL configurations

---

## Core Changes

**What Changed**: CARLA version removed from YAML configurations, now fixed in code

=== "YAML Updates" 

    ```yaml 
    # Before (remove this) 
    carla: 
      version: 0.9.12 # Remove this line
    client_port: 2000

    # After (current)
    carla:
    client_port: 2000
    # version is now fixed to 0.9.15 in code
    ```

=== "File Structure"

    ```bash 
    opencda/scenario_testing/config_yaml/ 
    # OpenCDA configs (unchanged) 
    opencda_marl/configs/ 
    # MARL configs (future) 
    scenarios/ 
    # MARL-specific scenarios 
    ```
    **Impact**: Remove CARLA version from existing YAML files  
    **Benefits**: Consistent CARLA version across all scenarios

---

## Configuration Guidelines

!!! tip "Best Practices" 
    When working with OpenCDA configurations:

    -   **OpenCDA Core**: Use existing configuration system (unchanged)
    -   **MARL Extension**: Will use separate config structure in `opencda_marl/configs/`
    -   **Inheritance**: MARL configs may inherit from OpenCDA base configs
   
    **Migration**: Only remove `carla.version` from YAML files if present

---

[← Back to Changelog](index.md)
