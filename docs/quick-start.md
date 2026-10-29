# Quick Start

Run a bundled learner untouched, watch it live, then change one thing.
[Installation](installation.md) comes first — though for the SUMO path
below that is just the repository and `pixi install`, no CARLA involved.

## 1. One training run

```bash
pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl
```

That is PCMAPPO learning speed control at a signal-free intersection on
SUMO — headless, around 10 ms per step, nothing to start beforehand.
Starting the run starts the **web monitor** and prints its URL
(`http://127.0.0.1:8765`): losses and episode summaries as they happen,
and a *Traffic view* drawing the intersection schematically with the
learning agents in it. Checkpoints, metrics and evaluation summaries land
under `artifacts/` ([Run Experiments](run-experiments.md) maps every
output).

Swap the profile to change the experiment — `route_learn_path_pcmappo`
adds the path action, `marl/intersection/rule_based --marl` runs a
no-learner baseline on CARLA. `opencda_marl/_bundled/configs/<family>/<scenario>/`
lists what ships — the profiles are package data, so a pip install has
exactly the same ones.

## 2. As a library

The engine is a wheel with the SUMO scenario family bundled, so an
environment with the dependencies runs scenarios without a repository
checkout:

```bash
pip install <opencda-marl wheel or checkout>            # SUMO runs out of the box
pip install "<opencda-marl checkout>[carla]"            # + libraries for CARLA runs from that checkout
```

There are two builds. `pixi run wheel --public` builds the **public wheel**
(`dist/public/opencda_marl-*.whl`), which leaves out everything UCLA's licence
covers (the vendored OpenCDA core and what is derived from it): it runs SUMO,
the library, the plugins and the baselines, and it is the only build a package
index may carry. `pixi run wheel` builds the **full wheel**, with the core, for
installing from Git (`pip install git+https://github.com/radar-lab/OpenCDA-MARL.git@v2.2.1`);
it is never uploaded. The wheel's own CLI and environment port drive SUMO
only, and CARLA needs a `git clone` of the repository, which brings the core. `[carla]` installs the libraries CARLA scenarios need when they run
from a repository checkout (`python opencda.py -t <ref> --marl`), because
those scenarios read their maps from the working tree; `carla` itself
comes with your simulator. The faster `libsumo` binding is optional:
without it SUMO runs over TraCI, about 13× slower per step, and the
fallback warning names the matching `pip install libsumo==<SUMO version>`.

```bash
opencda-marl run -t sumo-marl/intersection/route_speed_only_pcmappo
```

```bash
opencda-marl list          # registered algorithms, and from where
```

The bundled configs are the last config root searched: `--config-root DIR`
(repeatable) adds your own in front, and a profile placed there inherits
the family default and scenario base from the bundle. That is the whole
mechanism an external research package uses to ship its experiments — see
[Add an Algorithm](marl/custom_algorithms.md).

The wheel also carries the framework paper's **baseline checkpoints**:
`MARL.training.load_checkpoint: ral/sac_300vph` resolves inside the
package, and `opencda_marl/_bundled/checkpoints/ral/manifest.json` lists
every entry with the profile and recording it pairs with — a fixed
reference point for comparing your own learner against.

## 3. Change one thing

Configuration merges three layers — the family's `default.yaml`, the
scenario's `base.yaml`, then the profile — and a profile states only what
it changes (a section left empty, such as `outputs:`, inherits the lower
layers' block rather than clearing it). A new experiment is a new profile:
it inherits the scenario (map, traffic, SUMO settings; the SUMO base runs
headless) but names its learner's action space, objective and observation
features, because those belong to the profile, not to the scenario:

```yaml
# my_configs/sumo-marl/intersection/my_quick.yaml
# (the family default and the scenario base are inherited from the bundle)
scenario:
  simulation:
    max_episodes: 5

MARL:
  algorithm: "pcmasac"
  reward_profile: "competitive_intersection_v2"
  objective_mode: "competitive_ordering_v1"
  state_dim: 11                 # 9 features + 2 preference entries
  action_dim: 1
  action_space:
    mode: speed_only
  preference:
    profile: "balanced"
    assignment: "entry_direction"
    injection: "state_augmentation"
    dim: 2
  pcmasac:
    features:
      rel_x: 1
      rel_y: 1
      position_x: 1
      position_y: 1
      lane_position: 1
      heading_angle: 1
      dist_to_intersection: 1
      dist_to_front_vehicle: 1
      waypoint_buffer: 1
```

```bash
pixi run python opencda.py -t sumo-marl/intersection/my_quick --marl --marl-config-root my_configs
```

The scenario's base traffic with the observation of
`route_speed_only_pcmappo`, a different learner (PCMASAC with its default
hyperparameters), five episodes. Every knob the layers
expose — action space, rewards, traffic, outputs — is mapped in
[Architecture](marl/architecture.md); when the thing you want to change
is not a knob but a component, that page's seam tables say which kind to
register, and [Replace a Component](marl/replace_component.md) is the
door: the mechanism on one page, a worked example per kind behind it.
