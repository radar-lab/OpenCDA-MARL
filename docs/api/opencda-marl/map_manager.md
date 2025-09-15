# MARL Map Manager API

The `MARLMapManager` extends OpenCDA's map management with MARL-specific features for multi-agent scenarios, handling custom maps, junction-based spawn points, and visualization.

| Component          | Status | Description                                     |
| ------------------ | ------ | ----------------------------------------------- |
| Map Loading        | ✅      | Load built-in CARLA maps or custom XODR files   |
| Registry System    | ✅      | Hardcoded MAP_REGISTRY for available maps       |
| Junction Detection | ✅      | Advanced junction analysis with connections     |
| MARL Spawn Points  | ✅      | Junction-based spawn/destination generation     |
| Visualization      | ✅      | Enhanced junction and spawn point visualization |
| Custom Maps        | ✅      | Runtime registration of custom XODR maps        |

## Core Implementation

=== "Initialization"

    ```python
    from omegaconf import DictConfig
    import carla
    from opencda_marl.core.world.map_manager import MARLMapManager
    
    # Initialize with detailed config
    config = DictConfig({
        "map": {
            "name": "intersection",
            "safe_distance": 5.0,      # Base spacing for waypoints
            "spawn_offset": 2.0,       # Multiplier for upstream spawn distance
            "dest_offset": 2.0,        # Multiplier for downstream dest distance
            "spawn_z_lift": 0.3,       # Z lift to avoid ground collision
            "wp_step": 1.0             # Stepping granularity along lanes
        }
    })
    
    client = carla.Client("localhost", 2000)
    map_manager = MARLMapManager(config, client)
    ```

=== "Map Registry"

    ```python
    # Uses hardcoded MAP_REGISTRY from opencda_marl.core.world
    MAP_REGISTRY = {
        "intersection": {
            "type": "marl",
            "description": "4-way intersection for MARL training",
            "xodr_path": "opencda_marl/assets/maps/intersection.xodr",
            "fbx_path": "opencda_marl/assets/maps/intersection.fbx",  # Optional
            "built_in": False  # Auto-detected if available in CARLA
        },
        "Town05": {
            "type": "carla",
            "description": "CARLA built-in town",
            "built_in": True
        }
    }
    ```

=== "Internal Structure"

    ```python
    class MARLMapManager:
        def __init__(self, config: DictConfig, client: carla.Client):
            self.config = config.get("map", {})
            self.client = client
            self.world = None
            self.map = None
            self.map_registry = MAP_REGISTRY.copy()
            
            # Enhanced metadata with junction connections
            self._meta = {
                "map_name": None,
                "map_type": None,
                "junctions": [],      # Junction data with connections
                "spawn_points": []    # Generated from junction connections
            }
            
            # Automatic initialization
            self._verify_registry()
            self._initialize_map()
    ```

## Map Loading

=== "load_map()"

    ```python
    def load_map(self, map_name: str) -> carla.World:
        """Load a map for MARL training."""
        
        is_built_in = self.map_registry[map_name].get("built_in")
        
        if is_built_in:
            # Load built-in CARLA map
            world = self.client.load_world(map_name)
        else:
            # Load custom XODR map using OpenCDA utilities
            xodr_path = self.map_registry[map_name].get("xodr_path")
            world = load_customized_world(xodr_path, self.client)
        
        self.world = world
        self.map = world.get_map()
        
        # Update metadata
        self._meta["map_name"] = map_name
        self._meta["map_type"] = self.map_registry[map_name].get("type")
        
        logger.success(f"✓ Map '{map_name}' loaded successfully")
        return self.world
    ```

=== "Registry Verification"

    ```python
    def _verify_registry(self):
        """Verify and auto-detect maps in registry."""
        
        available_maps = self.client.get_available_maps()
        
        for name, info in self.map_registry.items():
            # Validate required fields
            if not info.get("type"):
                raise ValueError(f"Map {name} has no type")
            if not info.get("description"):
                raise ValueError(f"Map {name} has no description")
            
            # Verify custom map files exist
            if not info.get("built_in") and info.get("xodr_path"):
                if not os.path.exists(info["xodr_path"]):
                    raise FileNotFoundError(f"XODR file not found: {info['xodr_path']}")
            
            # Auto-detect CARLA built-in maps
            if self._discover_carla_maps(name):
                self.map_registry[name]["built_in"] = True
        
        logger.success(f"✓ Registry ({len(self.map_registry)} maps) verification completed")
    ```

=== "Custom Map Registration"

    ```python
    def register_custom_map(self, name: str, xodr_path: str, **kwargs):
        """Register a custom map at runtime."""
        
        from opencda_marl.core.world import register_map
        
        register_map(name, xodr_path, **kwargs)
        logger.info(f"Registered custom map: {name} from {xodr_path}")
    ```

## Junction-Based Spawn System

=== "Junction Detection"

    ```python
    def _get_junctions(self):
        """Collect junctions with spawn/dest points from connections."""
        
        # Configuration parameters
        safe_distance = float(self.config.get("safe_distance", 5.0))
        spawn_offset = float(self.config.get("spawn_offset", 2))    # Upstream multiplier
        dest_offset = float(self.config.get("dest_offset", 2))      # Downstream multiplier
        z_lift = float(self.config.get("spawn_z_lift", 0.3))       # Avoid ground collision
        step_size = float(self.config.get("wp_step", 1.0))         # Lane stepping
        
        waypoints = self.map.generate_waypoints(distance=safe_distance)
        seen = set()
        junctions = []
        
        for wp in waypoints:
            if not wp.is_junction:
                continue
            j = wp.get_junction()
            if j.id in seen:
                continue
            seen.add(j.id)
            
            # Get junction properties
            ...
            
            # Analyze junction connections
            connections = j.get_waypoints(carla.LaneType.Driving)
            conn_info = []
            
            for (entry_wp, exit_wp) in connections:
                # Calculate spawn point upstream from entry
                # Calculate destination downstream from exit
                # Create transforms with Z lift
                # Snap to drivable lanes for safety
                ...
                
                conn_info.append({
                    "entry_wp": entry_wp,
                    "exit_wp": exit_wp,
                    ...
                })
            
            junctions.append({
                "id": j.id,
                ...
            })
        
        self._meta["junctions"] = junctions
    ```

=== "MARL Spawn Generation"

    ```python
    def _marl_spawn_init(self):
        """Generate spawn points from junction connections."""
        
        self._meta["spawn_points"] = []
        for j in self._meta["junctions"]:
            for i, c in enumerate(j["connections"]):
                self._meta["spawn_points"].append({
                    "id": f"j{j['id']}_conn{i}",
                    ...
                })
    ```

## Spawn Point Management

=== "Get Spawn Points"

    ```python
    def get_spawn_points(self, num: int = None, dest: bool = False, detail: bool = False):
        """Get spawn points with optional destinations."""
        
        spawn_points = self._meta["spawn_points"]
        
        if num is not None:
            num = min(num, len(spawn_points))
            assert num >= 0, "num must be greater than or equal to 0"
            spawn_points = spawn_points[:num]
        
        if detail:
            # Return full spawn point data with IDs
            return spawn_points
        else:
            if dest:
                # Return (transform, destination) tuples
                return [(sp["transform"], sp["dest"]) for sp in spawn_points]
            else:
                # Return (transform, None) tuples for compatibility
                return [(sp["transform"], None) for sp in spawn_points]
    ```

=== "Random Spawn Points"

    ```python
    def get_random_spawn_points(self, num: int = 1, detail: bool = False):
        """Get random spawn points for agents."""
        
        available = self.get_spawn_points(detail=detail)
        
        if len(available) >= num:
            return random.sample(available, num)
        else:
            # Allow repetition if not enough unique points
            return [random.choice(available) for _ in range(num)]
    ```

## Visualization

=== "Enhanced Junction Visualization"

    ```python
    def draw_junction_centers(self, life_time=0):
        """Draw detailed junction information."""
        
        for junction in self._meta["junctions"]:
            center = junction['center']
            extent = junction['extent']
            
            # Draw junction center
            self.world.debug.draw_point(
                center + carla.Location(z=0.5),
                size=0.3,
                color=get_color("RED"),
                life_time=life_time
            )
            
            # Draw bounding box
            ...
            # Draw expanded spawn/dest area
            ...
            # Draw connection lanes
            ...
            # Draw informative text labels
            ...
        
    ```

=== "Spawn Point Visualization"

    ```python
    def draw_spawn_points(self, life_time=0):
        """Draw spawn and destination points."""
        
        for spawn_point in self._meta["spawn_points"]:
            # Draw spawn point
            self.world.debug.draw_point(
                ...
            )
            
            # Draw destination point
            if spawn_point["dest"] is not None:
                self.world.debug.draw_point(
                    ...
                )
    ```

## Map Information

=== "Available Maps"

    ```python
    def get_available_maps(self):
        """Get dictionary of all available maps from registry."""
        return self.map_registry
    
    def list_maps(self):
        """Print formatted list of available maps."""
        
        print("\nAvailable MARL Maps (from Registry):")
        print("=" * 120)
        print(f"{'Name':15} | {'Type':12} | {'Built-in':8} | {'FBX':8} | {'Description'}")
        print("-" * 120)
        
        for name, info in self.map_registry.items():
            has_fbx = "✓" if info.get("fbx_path") and os.path.exists(info["fbx_path"]) else "✗"
            built_in = "✓" if info.get("built_in") else "✗"
            
            print(f"{name:15} | {info['type']:12} | {built_in:^8} | {has_fbx:^8} | {info['description']}")
    ```

=== "Map Metadata"

    ```python
    def get_info(self) -> Dict[str, Any]:
        """Get metadata information about the loaded map."""
        
        return self._meta
        # Returns:
        # {
        #     "map_name": "intersection",
        #     "map_type": "marl",
        #     "junctions": [
        #         {
        #             "id": 1,
        #             "center": carla.Location,
        #             "connections": [
        #                 {
        #                     "entry_wp": carla.Waypoint,
        #                     "exit_wp": carla.Waypoint,
        #                     "spawn_tf": carla.Transform,
        #                     "dest_tf": carla.Transform
        #                 }
        #             ]
        #         }
        #     ],
        #     "spawn_points": [
        #         {
        #             "id": "j1_conn0",
        #             "transform": carla.Transform,
        #             "dest": carla.Transform
        #         }
        #     ]
        # }
    ```

## Configuration

=== "Advanced YAML Configuration"

    ```yaml
    # configs/marl_training.yaml
    map:
      name: "intersection"        # Map name from registry
      safe_distance: 5.0         # Base spacing for waypoint generation (m)
      spawn_offset: 2.0           # Upstream spawn distance multiplier
      dest_offset: 2.0            # Downstream destination multiplier  
      spawn_z_lift: 0.3           # Z elevation to avoid ground collision (m)
      wp_step: 1.0                # Lane stepping granularity (m)
    
    # Map registry configuration
    MAP_REGISTRY = {
        "intersection": {
            "type": "marl",
            "description": "4-way intersection with junction-based spawns",
            "xodr_path": "opencda_marl/assets/maps/intersection.xodr",
            "fbx_path": "opencda_marl/assets/maps/intersection.fbx",
            "built_in": False
        }
    }
    ```

=== "Integration with OpenCDA"

    ```python
    from opencda.scenario_testing.utils.sim_api import ScenarioManager
    from opencda_marl.core.world.map_manager import MARLMapManager
    
    # Initialize MARL map manager with advanced config
    config = DictConfig({
        "map": {
            "name": "Town05",
            "safe_distance": 5.0,
            "spawn_offset": 2.0,
            "dest_offset": 2.0
        }
    })
    
    map_manager = MARLMapManager(config, client)
    
    # Get spawn points with destinations for MARL agents
    spawn_dest_pairs = map_manager.get_spawn_points(num=4, dest=True)
    
    # Create vehicles using OpenCDA with destination goals
    scenario_manager = ScenarioManager(scenario_params)
    vehicles = []
    
    for i, (spawn_tf, dest_tf) in enumerate(spawn_dest_pairs):
        vehicle = scenario_manager.create_single_cav(
            spawn_transform=spawn_tf,
            config_override={
                "agent_id": f"agent_{i}",
                "destination": dest_tf
            }
        )
        vehicles.append(vehicle)
    ```

## API Summary

| Method                                           | Description                        | Returns                       |
| ------------------------------------------------ | ---------------------------------- | ----------------------------- |
| `load_map(map_name)`                             | Load a map from registry           | `carla.World`                 |
| `get_spawn_points(num, dest, detail)`            | Get spawn points with destinations | `list[tuple]` or `list[dict]` |
| `get_random_spawn_points(num, detail)`           | Get random spawn points            | `list[tuple]` or `list[dict]` |
| `register_custom_map(name, xodr_path, **kwargs)` | Register custom map                | `None`                        |
| `get_available_maps()`                           | Get all maps in registry           | `dict`                        |
| `list_maps()`                                    | Print formatted map list           | `None`                        |
| `get_info()`                                     | Get detailed map metadata          | `dict`                        |
| `draw_junction_centers(life_time)`               | Visualize junctions with details   | `None`                        |
| `draw_spawn_points(life_time)`                   | Visualize spawn and dest points    | `None`                        |
| `cleanup()`                                      | Clean up resources                 | `None`                        |

## Comparison with OpenCDA MapManager

=== "Key Differences"

    | Feature            | OpenCDA MapManager               | MARL MapManager                     |
    | ------------------ | -------------------------------- | ----------------------------------- |
    | **Purpose**        | BEV rasterization for perception | Junction-based spawn generation     |
    | **Initialization** | Requires vehicle instance        | Uses config and client only         |
    | **Map Focus**      | Lane-level rasterization         | Junction connection analysis        |
    | **Spawn Points**   | Uses CARLA default spawns        | Generates from junction connections |
    | **Visualization**  | BEV maps with OpenCV             | 3D debug drawings in CARLA          |
    | **Configuration**  | Vehicle-centric parameters       | MARL scenario parameters            |

=== "Integration Benefits"

    - **Complementary**: MARL manager for spawn planning, OpenCDA for perception
    - **Consistent**: Both use CARLA map data and OpenCDA utilities
    - **Flexible**: Can use MARL spawns with OpenCDA vehicles
    - **Scalable**: Junction-based approach works for any map size

## Related Documentation

- **[OpenCDA Map Manager](../opencda/overview.md)**: Core rasterization and perception utilities
- **[MARL Overview](../../marl/overview.md)**: MARL framework documentation  
- **[OpenCDA Integration](../../opencda/core.md)**: Integration patterns with OpenCDA