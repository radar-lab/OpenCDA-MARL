# Phase 1: Foundation Setup

This guide documents the completed MARL foundation for OpenCDA-MARL.

| Task | Description                        | Status |
| ---- | ---------------------------------- | ------ |
| 1.1  | Create MARL directory structure    | ✅     |
| 1.2  | Create **init**.py files           | ✅     |
| 1.3  | Move Custom Maps to MARL Directory | ✅     |
| 1.4  | Fix Maps for CARLA Compatibility   | ✅     |
| 1.5  | Test Map Loading                   | ✅     |
| 1.6  | Implement Scenario Template System | ✅     |
| 1.7  | Create MARL Coordinator            | ✅     |
| 1.8  | Implement Gym Environment Interface| ✅     |
| 1.9  | Create Map Adapter Layer           | ✅     |
| 1.10 | Implement GUI Components           | ✅     |

## Phase 1 Tasks

Phase 1 has been successfully completed with all foundation components implemented and tested.

=== "XODR Map Georeference"

    If you have custom XODR map files, place them in the MARL maps directory.

    Create and run the map fixer utility:

    **Fix maps automatically:**

    ```bash
    # with cdata (default)
    pixi run python opencda_marl/core/world/xodr_fixer.py

    # without cdata
    pixi run python opencda_marl/core/world/xodr_fixer.py --use-cdata=False
    ```

=== "Understanding Map Fixes"

    **Why Map Fixing is Needed:**

    -   CARLA requires georeference tags in XODR files
    -   The georeference content must be in CDATA format
    -   Missing or malformed georeference causes warnings: `"cannot parse georeference: ''"`

    **What the Fixer Does:**

    1. **Adds missing georeference tags** as children of the `<header>` element
    2. **Wraps georeference content in CDATA** format for proper parsing
    3. **Creates backups** before modifying files
    4. **Validates** the fixed maps

    **Before and After:**

    ```xml
    <!-- Before (causes warnings) -->
    <header>
    <revMinor>1</revMinor>
    <revMajor>1</revMajor>
    <name>intersection</name>
    <!-- Missing geoReference -->
    </header>

    <!-- After (CARLA compatible) -->
    <header>
    <revMinor>1</revMinor>
    <revMajor>1</revMajor>
    <name>intersection</name>
    <geoReference><![CDATA[+lat_0=4.9000000000000000e+1 +lon_0=8.0000000000000000e+0]]></geoReference>
    </header>
    ```

=== "Test Map Loading"

    Created a test script to verify map loading works correctly.

    **Run the test script:**

    ```bash
    # Make sure CARLA is running first
    pixi run python test/marl/test_map_loading.py
    # or
    pixi run marl-loadmap
    
    # Test complete foundation
    pixi run marl-quick-test      # Basic MARL scenario test
    pixi run marl-gui-debug       # GUI debugging mode
    ```

=== "Scenario Templates and Builder"

    | Task | Description                        | Status |
    | ---- | ---------------------------------- | ------ |
    | 2.1  | Create base scenario template      | ✅     |
    | 2.2  | Implement intersection template    | ✅     |
    | 2.3  | Create scenario builder factory    | ✅     |
    | 2.4  | Add placeholder templates          | ✅     |

    **Completed Components**:
    
    - `opencda_marl/scenarios/templates/base_template.py` - Abstract template interface
    - `opencda_marl/scenarios/templates/intersection_template.py` - Full intersection implementation
    - `opencda_marl/scenarios/templates/highway_template.py` - Placeholder for future
    - `opencda_marl/scenarios/templates/parking_template.py` - Placeholder for future
    - `opencda_marl/scenarios/scenario_builder.py` - Factory pattern implementation

=== "MARL Coordinator"

    | Task | Description                        | Status |
    | ---- | ---------------------------------- | ------ |
    | 3.1  | Implement coordinator architecture | ✅     |
    | 3.2  | Add execution modes                | ✅     |
    | 3.3  | Create callback system             | ✅     |
    | 3.4  | Integrate with scenario builder    | ✅     |

    **Implementation Details**:
    
    - `opencda_marl/core/coordinator.py` - Central MARL orchestrator
    - Supports GUI, training, evaluation, and CLI modes
    - Manages scenario → environment → agent flow
    - Comprehensive callback system for extensibility

=== "Base Environment and Gym Interface"

    | Task | Description                        | Status |
    | ---- | ---------------------------------- | ------ |
    | 4.1  | Create abstract base environment   | ✅     |
    | 4.2  | Implement multi-agent Gym wrapper  | ✅     |
    | 4.3  | Add single-agent placeholder       | ✅     |
    | 4.4  | Standard Gym API compliance        | ✅     |

    **Gym Integration**:
    
    - `opencda_marl/envs/base_env.py` - Abstract Gym environment base
    - `opencda_marl/envs/multi_agent_env.py` - Multi-agent implementation
    - `opencda_marl/envs/single_agent_env.py` - Placeholder for future
    - Standard `reset()`, `step()`, `render()` interface

=== "Map Adapter Layer"

    | Task | Description                        | Status |
    | ---- | ---------------------------------- | ------ |
    | 5.1  | Implement map adapter architecture | ✅     |
    | 5.2  | OpenCDA compatibility layer        | ✅     |
    | 5.3  | MARL coordination features         | ✅     |
    | 5.4  | Visualization and debugging        | ✅     |

    **Integration Bridge**:
    
    - `opencda_marl/core/adapters/map_adapter.py` - Main adapter class
    - Combines OpenCDA map loading with MARL spawn coordination
    - Preserves backward compatibility with OpenCDA
    - Multi-agent spawn strategies (balanced, conflict, random)

=== "GUI Components"

    | Task | Description                        | Status |
    | ---- | ---------------------------------- | ------ |
    | 6.1  | Create step controller GUI         | ✅     |
    | 6.2  | Implement observation viewer       | ✅     |
    | 6.3  | Add reusable widget components     | ✅     |
    | 6.4  | Fix PyQt lifecycle management      | ✅     |

    **GUI System**:
    
    - `opencda_marl/gui/step_controller.py` - Step-by-step debugging interface
    - `opencda_marl/gui/observation_viewer.py` - Real-time observation display
    - `opencda_marl/gui/widgets/` - Reusable components (InfoPanel, LogWidget, etc.)
    - Proper PyQt object lifecycle management


## Status Summary

**Completed Major Components**:

- **Map Management**: Full map loading, adaptation, and coordination system
- **Scenario System**: Template-based scenario generation with intersection focus
- **Execution Framework**: Coordinator with multiple execution modes
- **Environment Interface**: Gym-compatible multi-agent environment  
- **GUI System**: Step-by-step debugging and observation visualization
- **Integration Layer**: Adapters for OpenCDA compatibility
