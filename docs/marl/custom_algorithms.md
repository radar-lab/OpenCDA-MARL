# Add an Algorithm

The framework owns the pipeline — observation extraction, reward shaping,
route legality masks, episode lifecycle, checkpointing, metrics. An
algorithm owns only the decision. This page shows how to plug one in.

You never edit a file under `opencda_marl/` to add an algorithm. If you find
yourself doing so, that is a bug in the seam — please report it.

## Two integration levels

| Level | Who owns the training loop | Use when |
| --- | --- | --- |
| **Environment port** — `OpenCDAMARLParallelEnv` | You | You want to drive the benchmark with *any* RL code (SB3, RLlib, CleanRL, your own loop) against a standard obs-in / speed-out interface |
| **Policy port** — `Policy` + `register_algorithm` | The framework | You want the framework's training infrastructure: matrix launcher, checkpoints, the metrics store and monitor, route-intent heads |

The built-in PCMA algorithms are level-2 citizens; both levels see identical
observations, rewards, and execution, which is what keeps results comparable.

## Level 1: the ParallelEnv port

A PettingZoo-style parallel environment: observations out, target speeds in,
framework-computed rewards back — with the per-component breakdown, objective
vectors, and preference vectors in `infos` so you can re-shape rewards in
your own loop without forking the pipeline.

The engine installs as a library — the wheel bundles the `sumo-marl`
config family, the lite traffic replay and the intersection SUMO network,
so this runs without a repository checkout; torch is installed as a
dependency but this path never imports it:

```bash
pixi run wheel --public                  # builds dist/public/opencda_marl-*.whl, without the UCLA-licensed paths
pip install dist/public/opencda_marl-*.whl   # SUMO runs out of the box
```

The full build, which also carries the vendored OpenCDA core, installs from Git
(`pip install git+https://github.com/radar-lab/OpenCDA-MARL.git@v2.2.1`); it is never uploaded to
a package index, because UCLA's licence does not allow the core to be transferred to others.

```python
from opencda_marl.envs.parallel_env import OpenCDAMARLParallelEnv

env = OpenCDAMARLParallelEnv.from_scenario(
    "sumo-marl/intersection/route_speed_only_pcmappo",
    overrides={"MARL": {"external": {"features": {
        "rel_x": 1, "rel_y": 1, "heading_angle": 1, "dist_to_intersection": 1}}}})
observations, infos = env.reset()
while not env.episode_over:          # env.agents is empty before the first spawn
    actions = {agent: my_policy(observations[agent]) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

Contract notes:

- Actions are **target speeds in km/h** (the planner and controller stay
  fixed — that is the matched protocol that makes comparisons fair). An
  agent you stop sending actions for is handed back to SUMO's own
  car-following model (under the scenario's `world.sumo_rl_runtime_mode`);
  an agent you have not commanded yet keeps the departure speed it spawned
  with. To control every agent, send an action for every agent in
  `env.agents`.
- Loop on `env.episode_over`, not on `env.agents`: vehicles spawn over
  time, so the live set is empty at the start of a replay and between
  waves of traffic.
- Agents join and leave mid-episode as vehicles spawn and terminate; a
  terminating agent appears one final time in the step returns
  (`infos[agent]["event"]` is `success` / `collision` / `removed`).
- The scenario declares the observation vector (`MARL.external.features`,
  in the profile or through `overrides` as above) — the benchmark defines
  the observation space, not the algorithm. The observation width follows
  those features (plus the preference entries when the scenario conditions
  on preferences), not the profile's `MARL.state_dim`, which belongs to the
  profile's own learner. The reference resolves exactly as on the command
  line (a `configs/` root under the working directory,
  `OPENCDA_MARL_CONFIG_ROOTS`, then the bundled configs), and a profile's
  `include:` list is composed.
- Path intent stays framework-side for this port (the port's actions are
  speeds): `MARL.action_space.path.selector` picks rule-based /
  random-legal / keep-current / a learned head, and
  `MARL.action_space.path.provider` names a registered path action of
  your own ([Architecture](architecture.md) has the seam map). Path
  actions are not exposed over the external port yet.
- `pettingzoo` / `gymnasium` are soft dependencies (installed in the pixi
  environment; the wheel's `[ports]` extra adds them): without them the
  same API works by duck typing and `observation_space()` raises an
  actionable error; with them the class is a genuine
  `pettingzoo.ParallelEnv` with real Gymnasium spaces.
- `from_scenario` supports the SUMO backend today; CARLA lands with the
  parallel-training work.

## Level 2: the Policy port

The rest of this page covers level 2.

## The contract

Two protocols in [`opencda_marl/api/policy.py`](https://github.com/radar-lab/OpenCDA-MARL/blob/main/opencda_marl/api/policy.py):

- **`Policy`** — the speed controller. Required.
- **`IntentPolicy`** — optional one-shot route-intent capability, needed only
  for `action_space.mode: path_speed` with a learned route selector.

A `Policy` implements seven methods and exposes two attributes:

| Member | Meaning |
| --- | --- |
| `select_action(multi_agent_obs, ego_agent_id, training)` | Target speed in km/h for one agent, or `None` (warmup) to leave the vehicle to the backend's own driver for that step: the vanilla driver on CARLA, SUMO's car-following model on SUMO (a vehicle never commanded keeps its departure speed) |
| `store_transition(obs, ego_id, action, reward, next_obs, done)` | Record one transition |
| `update()` | One learning step; returns a metrics dict |
| `reset_episode()` | Clear per-episode state |
| `get_training_info()` | Diagnostics for the monitor and metric sinks |
| `save(path)` / `load(path)` | Checkpoint round-trip |
| `max_action` | Upper speed clamp, km/h |
| `supports_batched_actions` | `True` to receive one batched `select_actions` call per step instead of N per-agent calls |

Observations arrive as `{agent_id: state_vector}` — every live agent, so a
centralized critic can see the whole scene. The ego agent's own vector is
`multi_agent_obs[ego_agent_id]`.

## A minimal algorithm

```python
# my_project/constant_policy.py
from opencda_marl.api import register_algorithm


class ConstantSpeedPolicy:
    supports_batched_actions = False

    def __init__(self, config, state_dim, action_dim):
        self.max_action = float(config.get("max_action", 45.0))
        self.speed = float(config.get("constant_speed", 25.0))

    def select_action(self, multi_agent_obs, ego_agent_id, training=True):
        return self.speed

    def store_transition(self, multi_agent_obs, ego_agent_id, action, reward,
                         next_multi_agent_obs, done):
        pass

    def update(self):
        return {}

    def reset_episode(self):
        pass

    def get_training_info(self):
        return {"algorithm_type": "constant"}

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass


@register_algorithm("constant", label="Constant")
def build_constant(config, *, state_dim, action_dim):
    return ConstantSpeedPolicy(config, state_dim, action_dim)
```

## Making it discoverable

The registration runs when the module is imported. Four ways to get it
imported -- none of them touch `opencda_marl/`:

| route | when to use |
|---|---|
| your own package with an entry point | a research repository that `pip install`s the framework — the standard workflow |
| drop the module into `opencda_marl/builtin/algorithms/contrib/` | you cloned this repository; contrib modules are auto-imported, zero configuration |
| `MARL.plugins: [my_project.constant_policy]` in the scenario YAML | pin the plugin to the experiment; loaded at initialize for every run, before either backend resolves a component, so it works with the matrix launcher, the monitor and evaluation |
| `OPENCDA_MARL_PLUGINS=my_project.constant_policy` (or a `.py` path; `os.pathsep`-separated) | ad-hoc, no config edit |

```toml
[project.entry-points."opencda_marl.plugins"]
constant = "my_project.constant_policy"
```

Declare the contract version your package targets as a module attribute —
``OPENCDA_MARL_POLICY_API = 1`` — next to your registrations. The engine
(`opencda_marl.plugins.POLICY_API_VERSION`) rolls back and reports a
plugin built against a different contract version instead of letting it
misbehave at run time; omitting the attribute skips the check.

The built-in learners are themselves a plugin (`opencda_marl/builtin/algorithms/`,
loaded through the entry point the wheel declares on itself), so a plugin
is a peer, not a special case — the engine imports no learners at all. `python -m opencda_marl.plugins` lists everything that
is registered and where it came from; a plugin that fails to import is
reported there and in the error you get when its name is requested.

Your package can ship its own scenario profile under its own config root and
inherit the framework's family default and scenario base:

```bash
pixi run python opencda.py -t sumo-marl/intersection/my_profile --marl \
    --marl-config-root path/to/my_project/configs
```

The smallest complete learner -- one file, one profile -- is
`opencda_marl/builtin/algorithms/contrib/random_speed.py` with
`opencda_marl/_bundled/configs/sumo-marl/intersection/random_speed.yaml`; run it with
`pixi run python opencda.py -t sumo-marl/intersection/random_speed --marl`.

### What the framework does for a plain policy

A policy that does not subclass `BaseAlgorithm` still gets: checkpoints
(`save`/`load` are called with the per-episode / best / latest paths), world
reloads on CARLA (the snapshot reads only what the policy has), and a
metrics store for the monitor -- the manager records every `update()`
result (every `outputs.metrics_store.log_frequency` calls) and every episode
summary under `metrics/<algorithm>/<run>/`, where `<algorithm>` is the key
you registered (a `BaseAlgorithm` subclass's store lands there too).

### What a checkpoint should hold

A checkpoint should not carry the run's configuration. The config your
factory receives is your `MARL.<section>` block plus `training_mode` (and
`reward_dim` when `vector_rewards` injects it) and the `tensorboard` and
`metrics_store` blocks the manager merges in from `outputs.*` (directories
under `outputs.root_dir`, the run name). Its top-level values arrive
resolved, but its list and dict values are OmegaConf nodes, each of which
keeps a reference to the whole run config: pickled as they are, they ship
every section. The built-in learners store `self.checkpoint_config()`,
which every `BaseAlgorithm` inherits from `TrainingLogger`, instead of
`self.config`. It is a plain copy without those two blocks, with any value
that is an absolute filesystem path replaced by `<absolute path removed>`;
every hyperparameter is kept. Interpolations inside nested blocks are kept
as written. A top-level value that resolved to an absolute path is
replaced, but a path inside a longer string (`run@${outputs.root_dir}`) is
saved as it resolved, so keep such strings out of your section's top
level. A `BaseAlgorithm` subclass that writes its config in its own `save`
should store the same. Any list or dict it reads from its config and saves
separately should go in as `portable_config(...)` from
`opencda_marl.builtin.algorithms.training_logger`, the function behind
`checkpoint_config()`. It takes an OmegaConf node or a plain dict or list,
returns plain dicts and lists, and replaces absolute paths the same way.
`OmegaConf.to_container(...)` raises on a plain dict, for example in a
learner built from plain dicts, and keeps absolute paths. A list of plain
numbers can also go in as `list(...)`. A policy that does not subclass
`BaseAlgorithm` can store `portable_config(config)`, which also leaves
those two blocks out.

## Configuring it

The shared extractor's `position_x` and `position_y` features are absolute
map coordinates in the common CARLA frame. They read the explicit normalized
position fields; `location` is a fallback because legacy CARLA observations
use that dictionary for junction-relative coordinates. Use `rel_x` and
`rel_y` for the relative position. The SUMO position features have always
read this way.

For replay vehicles with assigned junction geometry, `rel_x/rel_y` are the
vehicle's map position minus that junction's center. `dist_to_intersection`
and the approach decision-zone radius use the planar distance to the same
center on both backends. This replaces CARLA's former nearest-waypoint vector
and entry-distance trigger. Legacy planners without junction geometry retain
their adapter-provided distances; the legacy `location` alias is unchanged.

The config section is the algorithm name by default. Features declared here
drive the shared observation extractor — this is how your algorithm gets a
state vector without writing any extraction code:

```yaml
MARL:
  algorithm: constant
  state_dim: 3
  action_space:
    mode: speed_only
  constant:
    constant_speed: 27.5
    max_action: 45.0
    features:
      rel_x: 1
      rel_y: 1
      speed: 1
```

`state_dim` must equal the summed feature widths plus the preference
dimension when preference conditioning is on. A mismatch is logged as a
warning at construction — the configured value still wins, because it is
baked into saved checkpoints.

## Declaring capabilities

`register_algorithm` accepts:

| Argument | Default | Effect |
| --- | --- | --- |
| `label` | uppercased key | Display name in tables, logs and the monitor |
| `config_section` | the key | Which `MARL.<section>` block reaches your factory |
| `multi_agent` | `True` | `False` routes single-agent observations instead: `select_action(state, training)` and `store_transition(state, action, reward, next_state, done)` on the agent's own state |
| `reward_aggregation` | `"per_agent"` | `"team_mean"` gives every agent the team-mean reward |
| `supports_intent` | `False` | `True` declares the `IntentPolicy` capability |
| `vector_rewards` | `False` | `True` injects `reward_dim` for objective-vector runs |
| `override` | `False` | Required to replace an existing registration |

The framework dispatches on these declarations. There is no `isinstance`
check and no name list to join.

A single-agent policy (`multi_agent=False`) returns the target speed from
`select_action` too, unless its class declares `returns_action_index =
True`: then `select_action` returns an index into the policy's
`speed_actions` list, the framework drives that list entry, and the index
is what `store_transition` receives back as `action`. The declaration is
read, never inferred — owning a `speed_actions` table does not make a
policy index-returning (the bundled DQN returns speeds; tabular Q-learning
returns indices).

## Adding route-intent support

To support learned route selection, implement `IntentPolicy` — or inherit
the standard implementation from `RouteHeadMixin`, which is what the
built-in families do:

```python
from opencda_marl.builtin.algorithms.route_head import RouteHeadMixin


class MyIntentPolicy(RouteHeadMixin, ConstantSpeedPolicy):
    route_head_cls = "HybridRouteSACHead"   # or HybridRoutePPOHead
    route_head_shares_optimizer = False     # True for PPO-family heads


@register_algorithm("my_intent", supports_intent=True)
def build(config, *, state_dim, action_dim):
    return MyIntentPolicy(config, state_dim, action_dim)
```

The mixin requires `self.device` (and `self.optimizer` when
`route_head_shares_optimizer = True`).

Route intent is **one-shot**: the policy chooses once per vehicle at spawn
from the legal set the framework supplies, and the reward arrives at that
vehicle's terminal event — hence `store_pending` / `finalize_terminal`
rather than a per-step transition. Illegal actions are masked before
sampling; never select a masked-out index.

Registering with `supports_intent=True` without implementing the methods
fails at manager construction with a message naming what is missing, rather
than misbehaving at run time.

## Beyond algorithms: components and profiles

Research rarely stops at the learner. The same plugin door (entry point,
`MARL.plugins`, `OPENCDA_MARL_PLUGINS`, `contrib/`) carries every other
registration your package makes at import time — planners, agent types,
intent providers, observation extractors, reward calculators, the profile
tables, and axes of your own. [Replace a Component](replace_component.md)
is the mechanism, with one worked page per kind; overriding an existing
key — built-ins included — always requires `override=True`, so a typo
cannot shadow one silently. The per-vehicle *local* planner still lives
inside the vendored OpenCDA behavior stack and is not yet an axis.

## Verifying your algorithm

```bash
pixi run pytest test/route_intent/test_policy_registry.py -q
```

That file registers a policy defined entirely inside the test and drives it
through `MARLManager` — copy its shape to smoke-test your own before
launching a training run.

## Known rough edges

Honest notes, so you do not discover these the hard way:

- Actions are **km/h**, not a normalized action space. Conversion belongs in
  an env-side action space; until that lands it is part of the contract.
- `None` doubles as the "leave this vehicle to the backend's own driver"
  signal during warmup.
- Preference vectors are appended to the end of the state vector; policies
  that need them slice the last `pref_dim` entries.
