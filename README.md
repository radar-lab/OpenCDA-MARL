# OpenCDA-MARL

[![PyPI](https://img.shields.io/pypi/v/opencda-marl)](https://pypi.org/project/opencda-marl/)
[![Python](https://img.shields.io/pypi/pyversions/opencda-marl)](https://pypi.org/project/opencda-marl/)
[![Release](https://img.shields.io/github/v/release/radar-lab/OpenCDA-MARL)](https://github.com/radar-lab/OpenCDA-MARL/releases)
[![Docs](https://img.shields.io/badge/docs-radar--lab.github.io-blue)](https://radar-lab.github.io/OpenCDA-MARL/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%2B%20UCLA%20academic-lightgrey)](https://github.com/radar-lab/OpenCDA-MARL/blob/main/THIRD_PARTY_NOTICES.md)

OpenCDA-MARL extends the [OpenCDA](https://opencda-documentation.readthedocs.io/en/latest/) cooperative driving
automation framework (see [SAE J3216](https://www.sae.org/standards/content/j3216_202005/)) with **multi-agent
reinforcement learning** for driving in complex and mixed traffic: learners, a verified training lifecycle on CARLA and
SUMO, benchmark scenarios and the tools to watch and compare runs.

* **Learners out of the box**: MAPPO, MASAC, MATD3, DQN and tabular Q-learning, plus PCMAPPO and PCMASAC, which build on
  [PCMA](https://arxiv.org/abs/2606.14693) (without its Dirichlet preference planner) for multi-objective training,
  and the preference-conditioned PCMATD3.
* **Two backends, one loop**: CARLA 0.9.15 / 0.9.16 and SUMO (libsumo), plus CARLA–SUMO co-simulation. The same
  scenarios and learners run on both.
* **Replaceable parts**: the lifecycle is fixed, and every part an experiment might vary (algorithm, agent, planner,
  observation, reward, intent provider) is a plugin chosen by configuration.
* **Standard interfaces**: a PettingZoo `ParallelEnv` port, so external RL stacks can drive the same scenarios.
* **Mixed autonomy**: human-driven vehicles, rule-based AVs and learning agents in the same traffic.
* **Tools for experiments**: a matrix launcher for parallel training cells across GPUs, and a web monitor with a live
  traffic view.

## Quick start

**For research on the SUMO benchmark** (the library, the plugin system, the baselines; no simulator to install):

```bash
pip install opencda-marl
opencda-marl run -t sumo-marl/intersection/route_speed_only_pcmappo
```

That trains PCMAPPO on a signal-free intersection on SUMO, headless. The run starts a web monitor at
`http://127.0.0.1:8765` with losses, episode summaries and a live traffic view. The wheel from PyPI is the *public*
build: it leaves out the vendored OpenCDA core, whose UCLA licence does not allow it to travel in a package. `pip`
installs the CPU build of PyTorch; for a GPU, install the matching CUDA build from [pytorch.org](https://pytorch.org)
first. To build your own study on it, start from the
[research template](https://github.com/radar-lab/opencda-marl-research-template).

**For CARLA, co-simulation and the OpenCDA scenarios**, clone the repository (the clone brings the core):

```bash
git clone https://github.com/radar-lab/OpenCDA-MARL.git && cd OpenCDA-MARL
pixi install
pixi run python opencda.py -t sumo-marl/intersection/route_speed_only_pcmappo --marl
```

[Installation](https://radar-lab.github.io/OpenCDA-MARL/installation/) covers CARLA, and
[Quick Start](https://radar-lab.github.io/OpenCDA-MARL/quick-start/) goes on from here.

![the monitor's live traffic view during a SUMO training run](https://radar-lab.github.io/OpenCDA-MARL/images/monitor-traffic-view.gif)

## Architecture

![OpenCDA-MARL architecture: entry points, the sealed kernel, the api contracts and plugin system, the builtin plugins, telemetry and the simulation backends](https://radar-lab.github.io/OpenCDA-MARL/images/opencda-marl-architecture.svg)

One sealed lifecycle, many replaceable joints. `_kernel/` runs the episode loop and knows only the contracts in
`api/`; `builtin/` registers the framework's own learners, agents and planners through the same plugin system an
external package uses, and a test keeps the kernel from importing them. See
[Architecture](https://radar-lab.github.io/OpenCDA-MARL/marl/architecture/) for the joints, and
[Add an Algorithm](https://radar-lab.github.io/OpenCDA-MARL/marl/custom_algorithms/) for the largest worked example.

## What's New in OpenCDA-MARL

### v2.2.0

* **On PyPI**: `pip install opencda-marl` installs a public wheel without any UCLA-licensed path (the vendored OpenCDA
  core, the copied OpenCDA configs and the files with adapted OpenCDA code), licensed `Apache-2.0 AND MIT`. The SUMO
  benchmark, the library, the plugins and the baselines run on it; CARLA needs a `git clone`
  ([changelog](https://radar-lab.github.io/OpenCDA-MARL/marl/changelog/v2.2.0/)).

### v2.1.0

The first 2.x release ([changelog](https://radar-lab.github.io/OpenCDA-MARL/marl/changelog/v2.1.0/), including its
known issues).

* **A library, not just a repo**: `pixi run wheel` builds a wheel with the lifecycle, the SUMO benchmark and the
  baseline learners, and an `opencda-marl run` CLI. Every algorithm arrives through the `opencda_marl.plugins` entry
  point, so a research repository imports the library and registers its own components without touching it.
* **Verified lifecycle**: one episode loop across both backends, CARLA world reloads that carry the full learner
  state, and resumable runs.
* **SUMO backend**: the same scenarios and learners without CARLA, orders of magnitude faster per step.
* **Experiment infrastructure**: the matrix launcher with parallel cells, per-slot CARLA servers, GPU spreading and
  `--resume`, and opt-in throughput features.
* **Metrics store and monitor**: a per-run SQLite store and a dependency-free web monitor that overlays runs, lists
  episode summaries and draws a live traffic view.
* **Bundled baselines**: the framework paper's DQN, MAPPO and SAC checkpoints and most of its TD3 ones ship with their
  profiles and the deterministic traffic recordings they were trained against.

## Research template

Building a study on top of OpenCDA-MARL? The
[opencda-marl-research-template](https://github.com/radar-lab/opencda-marl-research-template) is the starting point: a
research repository that installs this library and registers its own components through the plugin entry point, with a
demo, a test and the framework pinned to a release. It is an initial version and is still being completed.

## Documentation

* **Users**: [Installation](https://radar-lab.github.io/OpenCDA-MARL/installation/),
  [Quick Start](https://radar-lab.github.io/OpenCDA-MARL/quick-start/),
  [CARLA Version Support](https://radar-lab.github.io/OpenCDA-MARL/carla-version-support/),
  [Run Experiments](https://radar-lab.github.io/OpenCDA-MARL/run-experiments/),
  [PCMA Learners](https://radar-lab.github.io/OpenCDA-MARL/marl/pcma/)
* **Developers**: [Add an Algorithm](https://radar-lab.github.io/OpenCDA-MARL/marl/custom_algorithms/),
  [Agent Workflow](https://radar-lab.github.io/OpenCDA-MARL/agents/workflow/) and
  [Code Index](https://radar-lab.github.io/OpenCDA-MARL/agents/code-index/) (what coding agents and people read
  before changing code), [OpenCDA Patch Ledger](https://radar-lab.github.io/OpenCDA-MARL/opencda/marl_patches/)

The framework is mainly tested on our customized maps; we do not guarantee the same robustness on other maps.
Contributions by people and by coding agents are welcome: see the [Contributing Guide](https://github.com/radar-lab/OpenCDA-MARL/blob/main/docs/contributing.md), file
[issues](https://github.com/radar-lab/OpenCDA-MARL/issues), and open [pull requests](https://github.com/radar-lab/OpenCDA-MARL/pulls)
with the [PR template](https://github.com/radar-lab/OpenCDA-MARL/blob/main/.github/PR_TEMPLATE.md).

## Citation

If you use OpenCDA-MARL in your research, please cite both the MARL extension and the original OpenCDA paper:

```bibtex
@ARTICLE{11396022,
  author={Guo, Lihao and Liu, Louis and Tang, Jiahao and Liu, Bo and Cao, Siyang},
  journal={IEEE Robotics and Automation Letters}, 
  title={OpenCDA-MARL: A Unified Benchmarking Framework for Cooperative Autonomous Intersection Management With Multi-Agent Reinforcement Learning}, 
  year={2026},
  volume={11},
  number={4},
  pages={4291-4298},
  keywords={Vehicle dynamics;Training;Robot sensing systems;Heuristic algorithms;Vehicle-to-everything;Benchmark testing;Real-time systems;Monitoring;Laser radar;Graphical user interfaces;Autonomous driving;benchmark testing;connected vehicles;cooperative systems;deep reinforcement learning;multi-agent systems;reinforcement learning;simulation},
  doi={10.1109/LRA.2026.3664656}
}


@misc{xu2021opencdaanopencooperativedriving,
      title={OpenCDA:An Open Cooperative Driving Automation Framework Integrated with Co-Simulation}, 
      author={Runsheng Xu and Yi Guo and Xu Han and Xin Xia and Hao Xiang and Jiaqi Ma},
      year={2021},
      eprint={2107.06260},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2107.06260}, 
}
```

The preference-conditioned learners (PCMASAC / PCMAPPO / PCMATD3) build on
*Preference Coordinated Multi-agent Policy Optimization* (PCMA), without its
Dirichlet preference planner. If you use them, please also cite:

```bibtex
@misc{wang2026pcma,
  title={Learning Coordinated Preference for Multi-Objective Multi-Agent Reinforcement Learning},
  author={Pengxin Wang and Lihao Guo and Yi Xie and Bo Liu and Siyang Cao and Jingdi Chen},
  year={2026},
  eprint={2606.14693},
  archivePrefix={arXiv},
  primaryClass={cs.MA},
  url={https://arxiv.org/abs/2606.14693},
}
```

What the implementation does, layer by layer, is documented in
[PCMA Learners](https://radar-lab.github.io/OpenCDA-MARL/marl/pcma/).

## License

The code written for OpenCDA-MARL (the `opencda_marl` package, scripts, tests, configuration profiles and
documentation) is licensed under the [Apache License 2.0](https://github.com/radar-lab/OpenCDA-MARL/blob/main/LICENSE); see also [NOTICE](https://github.com/radar-lab/OpenCDA-MARL/blob/main/NOTICE).

The vendored OpenCDA core (`opencda/`, `opencda.py` and the files derived from it) remains under the UCLA Mobility
Lab's Academic Software License: academic and nonprofit research use only, and no further transfer of the software
or its derivatives; commercial users should contact the UCLA Mobility Lab as that licence says. A running
installation includes that core, so use of the complete system is limited by UCLA's licence to academic and
nonprofit research; this does not narrow what the Apache License allows for our own files. CARLA code and map
data and the other third-party material keep their own terms. [THIRD_PARTY_NOTICES.md](https://github.com/radar-lab/OpenCDA-MARL/blob/main/THIRD_PARTY_NOTICES.md)
lists each part, the paths it occupies and its terms, and reproduces UCLA's licence text.

## Contributors

* **OpenCDA-MARL** (UA Radar Lab): Project Lead Lihao Guo; core team Louis Liu and Jiahao Tang; advisors Dr. Liu Bo and
  Dr. Siyang Cao.
* **OpenCDA** ([UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/)): Principal Investigator Dr. Jiaqi Ma
  ([UCLA Samueli](https://samueli.ucla.edu/people/jiaqi-ma/)); Project Lead Runsheng Xu
  ([github](https://github.com/DerrickXuNu)); core team Xu Han, Hao Xiang, Zhaoliang Zheng, Zonglin Meng and
  Dr. Xin Xia.

### Acknowledgements

* [UA Radar lab](https://github.com/radar-lab) and the [UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/)
* UC Davis Professor [Junshan Zhang's](https://faculty.engineering.ucdavis.edu/jzhang/) group for openScenario integration
* @GoodarzMehr for Docker support
* All contributors to the OpenCDA and OpenCDA-MARL projects
