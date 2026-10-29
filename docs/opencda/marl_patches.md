# OpenCDA Patch Ledger

This page tracks code we intentionally modify under the upstream `opencda/` tree for OpenCDA-MARL. Every row is a delta against UCLA OpenCDA `main` at commit `72b17e7b`.

The purpose is simple:

- make rebases against upstream OpenCDA easier
- make local framework patches easy to audit
- keep a human-readable explanation next to the code comments

## Tracking Rule

When OpenCDA-MARL changes code under `opencda/`:

1. Wrap the modified block with `OpenCDA-MARL` comments in the source file.
2. Add or update an entry on this page with the file and the reason.
3. Keep the patch as small and local as possible.

Example source markers:

```python
# <OpenCDA-MARL>
# reason for the patch
...
# </OpenCDA-MARL>
```

## Current Patch Inventory

| File | Purpose |
| --- | --- |
| `opencda/core/safety/sensors.py` | Retains the accepted collision event's frame, other actor ID/type and impulse for terminal-event diagnostics. The impulse threshold and collision decision are unchanged. |
| `opencda/scenario_testing/utils/customized_map_api.py`, `sim_api.py` | Disable pedestrian navigation generation for vehicle-only OpenDRIVE worlds, preventing repeated world reloads from accumulating RecastBuilder jobs. `world.pedestrian_navigation: true` enables it for scenarios that need AI walkers; road geometry and vehicle physics settings are unchanged. `ScenarioManager` takes `carla_version` as a keyword defaulting to `opencda.CARLA_VERSION` (the scenario scripts no longer pass it), reads the client host, port, timeout and fixed delta from the world config with the stock values as defaults, and starts background autopilots on the Traffic Manager's own port. `build_world_with_retry` runs the OpenDRIVE generation (xodr branch) or `client.load_world` (town branch) with the client timeout raised to `max(world.timeout, 90 s)` (`WORLD_BUILD_TIMEOUT`), then restores `world.timeout`, also on error; it does not retry (a retry queues behind the still-running first request), and every error, a time-out included, propagates. `is_client_timeout` detects the RPC time-out. The town branch re-raises a time-out instead of reporting the map as missing. |
| `opencda/core/actuation/control_manager.py` | Exposes controller debug info to MARL-side agents and overlays. |
| `opencda/core/actuation/pid_controller.py` | Stores controller debug info for overlays/TensorBoard and guards zero-length or non-finite lateral target vectors to avoid runtime warnings near route completion. The longitudinal PID uses the longitudinal error buffer and gains (UCLA used the lateral ones), which changes the stock throttle and brake response. |
| `opencda/core/common/v2x_manager.py` | Skips the O(N^2) nearby-CAV `search()` when V2X is disabled (its only consumer is platooning); one `get_ego_pos` per neighbour instead of two; bounds `ego_dynamic_trace` (`v2x.trace_maxlen`, default 5000) instead of growing per step for the vehicle's lifetime. |
| `opencda/core/common/vehicle_manager.py` | Preserves MARL-side custom agent injection instead of always constructing stock `BehaviorAgent`. Hardens `destroy()`: per-manager isolation so a sensor-manager failure can no longer skip `vehicle.destroy()` (actor leak). Passes `collision_sensor_cls` through to `SafetyManager`. Selects the platooning agent on `'platooning' in application`, the value the platooning scenarios pass and the check stock OpenCDA makes; the fork had narrowed it to `'platoon'`, which never matches, so a platoon member silently ran the plain `BehaviorAgent`. |
| `opencda/core/safety/safety_manager.py` | Prints a warning only when the set of triggered warnings changes, not on every tick it persists (`print_message`). Hardens `destroy()`: per-sensor isolation so one sensor's failure cannot skip the rest; adds the loguru import for the warnings. Accepts `collision_sensor_cls` (install a specialised collision sensor at construction instead of spawn-then-swap) and an opt-out `params['imu']` for the duplicate safety IMU (default `True` = stock behavior). |
| `opencda/customize/ml_libs/ml_manager.py` | Detector swapped to YOLOv8 (`ultralytics`) from the upstream YOLOv5 hub download; weights resolve to `dependencies/models/` (override `OPENCDA_ML_WEIGHTS_DIR`) instead of downloading into the current directory. Only loaded when a scenario runs with `--apply_ml` and activates perception. |
| `opencda/core/plan/behavior_agent.py` | Keeps extra route-trace debugging hooks used during MARL planner/behavior inspection. A reached destination raises `StopIteration` instead of calling `sys.exit`, so the scenario decides how to end, and the global route planner is built with the world and the debug flag. |
| `opencda/core/plan/global_route_planner.py` | Adds richer graph/path debug controls used to inspect route-only and junction-only planner behavior. |
| `opencda/core/plan/global_route_planner_dao.py` | `get_topology` appends sampled segments for driving lanes missing from CARLA's own topology. This changes the route graph of every planner, not only MARL debugging. The editor-generated 'All Rights Reserved' header line is removed. |
| `opencda/core/map/map_manager.py` | Guards `destroy()`'s process-global `cv2.destroyAllWindows()` behind the visualize flag (it ran once per destroyed vehicle). Builds the topology / traffic-light / lane tables (`_build_static_map_info`) only when `activate` is true: they feed the BEV rasterizer alone, and building them cost ~0.2 s per vehicle spawn. Stock behaviour when active. |
| `opencda/core/sensing/localization/localization_debug_helper.py` | The per-step `matplotlib.use('TkAgg')` in the animation hook is process-global: it switched the backend for the whole run (and every later test in a pytest session). An explicit `MPLBACKEND` now wins and a failed switch is non-fatal, so headless servers and the test session keep `Agg`. Stock behaviour when `MPLBACKEND` is unset and Tk is available. |
| `opencda/core/plan/planer_debug_helper.py` | Adds route-trace debug helpers used by MARL-side planner inspection. The `world` constructor argument is optional (`draw_point` is a no-op without it) so the upstream `PlanDebugHelper(actor_id)` signature and `PlatoonDebugHelper` keep working. |
| `opencda/core/application/platooning/platoon_debug_helper.py`, `platoon_behavior_agent.py` | Pass the world through to the route-trace helper (the subclass previously called the patched constructor with the upstream arity and raised at construction). |
| `opencda/core/sensing/perception/perception_manager.py` | Includes stability fixes for lazy visualizer creation and wait-loop handling in MARL workflows. Hardens `destroy()`: None-safe sensor access, per-sensor isolation, and reference nulling against repeated/racing teardown. Adds an optional class-level per-step world snapshot (`set_world_snapshot`/`clear_world_snapshot`): in plain (no lidar / no data-dump) mode `deactivate_mode` builds obstacles from it instead of `get_actors` + per-vehicle distance + ~6 queries per obstacle (was ~2,500 round trips per step at 30 vehicles). Stock path unchanged when no snapshot is set. |
| `opencda/core/sensing/perception/obstacle_vehicle.py` | Adds `ObstacleVehicle.from_snapshot`, an interface-equivalent constructor from cached kinematics used by the perception snapshot path. |
| `opencda/LICENSE` | UCLA's Academic Software License text, moved here unchanged from `docs/LICENSE` so it sits beside the vendored core; `THIRD_PARTY_NOTICES.md` reproduces it. |
| `opencda/__init__.py`, `opencda/version.py` (removed) | The version string lives in the package `__init__` (UCLA's `version.py` is removed). `CARLA_VERSION` defaults to `0.9.16`; `CARLA_VERSION_CANDIDATE`, `CARLA_VERSION_EXPERIMENTAL` and `SUPPORTED_CARLA_VERSIONS` are declared but not read anywhere. The editor-generated 'All Rights Reserved' header line is removed. |

## Inherited from OpenCDA-MARL 1.0.0

These deltas were already in the 1.0.0 release, before the ledger existed, and carry no `OpenCDA-MARL` markers. They are listed so the baseline above is complete.

| File | Purpose |
| --- | --- |
| `opencda/core/common/cav_world.py` | Adds `remove_vehicle_manager`, which drops a destroyed vehicle's manager and id from the world's registries. |
| `opencda/core/sensing/localization/localization_manager.py` | GNSS and IMU sensors get their own `destroy()`, and the manager's teardown tolerates an already-destroyed sensor. `use_kalman` (default `true`) can bypass the Kalman filter; an inactive manager still feeds ground truth to the debug helper; the GNSS-to-ENU conversion uses the map's reference altitude instead of 0. |
| `opencda/core/application/platooning/platooning_manager.py` | The platoon evaluation tolerates members whose debug helper has no time-gap or distance-gap lists (non-platoon agents) and reports 0 for an empty list. |
| `opencda/co_simulation/sumo_integration/bridge_helper.py`, `sumo_simulation.py`, `vtypes.json` | SUMO default vehicle types map to a CARLA blueprint at debug level instead of a warning; `DEFAULT_VEHTYPE` is a passenger type; the net-file lookup uses a valid XPath and `sumolib` directly; a SUMO GUI starts running (`--start`) instead of waiting for the play button. |
| `opencda/scenario_testing/evaluations/evaluate_manager.py` | Plots are cut to a common length (no dimension mismatch between route, IMU and safety logs), and every figure is closed after saving. |
| `opencda/core/map/map_drawing.py`, `opencda/core/sensing/perception/o3d_lidar_libs.py`, `opencda/core/sensing/perception/sensor_transformation.py` | `np.int` becomes `np.int32` (removed from NumPy); the lidar fusion reads `scipy.stats.mode` results of either shape. |
| `opencda/scenario_testing/*.py` (the 13 scenario scripts) | Follow the `ScenarioManager` signature without `opt.version`. `single_2lanefree_carla`, `single_town06_carla` and `platoon_joining_2lanefree_carla` end on `StopIteration` (a vehicle at its destination); those, `v2xp_online_carla` and the co-simulation scripts clean up only what was created; the co-simulation scripts pin `carla_version='0.9.15'`. |
| `opencda/scenario_testing/config_yaml/` (removed) | The scenario YAMLs moved to the engine's bundled `opencda` config family (`opencda_marl/_bundled/configs/opencda/`); a checkout-level `configs/opencda/` still overrides them. |
| `opencda/assets/2lane_freeway_complete/README.md` | Markdown formatting only. |

## Current Guidance

- Prefer implementing new MARL behavior in `opencda_marl/` first.
- Only patch `opencda/` when the change is truly shared infrastructure.
- If a patch exists only for one benchmark or one scenario family, keep it out of `opencda/`.
- If a patch becomes obsolete after upstream sync or refactor, remove both the source comments and the entry here.
