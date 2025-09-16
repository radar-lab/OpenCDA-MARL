# OpenCDA-MARL

OpenCDA-MARL extends the original OpenCDA framework with **Multi-Agent Reinforcement Learning (MARL)** capabilities for cooperative driving automation research. Building upon OpenCDA's co-simulation-based framework integrated with prototype cooperative driving automation (CDA; see [SAE J3216](https://www.sae.org/standards/content/j3216_202005/)) pipelines, OpenCDA-MARL adds advanced MARL algorithms and training infrastructure for developing intelligent multi-agent driving policies in complex traffic scenarios.

OpenCDA-MARL combines the robust simulation capabilities of the original OpenCDA with state-of-the-art MARL algorithms to enable researchers to train and evaluate cooperative multi-agent driving policies. This extension supports various MARL paradigms including centralized training with decentralized execution (CTDE), communication-based cooperation, and emergent behaviors in mixed autonomy traffic.

The key features of OpenCDA-MARL are:

* <strong>MARL Integration</strong>: Native support for popular MARL algorithms (PPO, SAC, QMIX, MADDPG) with distributed training capabilities.
* <strong>Research Pipeline</strong>: Rich research pipelines including both rule-based CDA modules and learning-based MARL agents for platooning, cooperative perception, and traffic coordination.
* <strong>Integration</strong>: Seamless integration with CARLA and SUMO, plus compatibility with RLlib and other RL frameworks.
* <strong>Full-stack Simulation</strong>: Complete automated and cooperative driving platform in Python with perception, localization, planning, control, V2X communication, and MARL decision-making modules.
* <strong>Modularity</strong>: Highly modularized architecture allowing easy swapping between rule-based and learning-based components.
* <strong>Benchmark</strong>: MARL-specific benchmarks, training scenarios, and evaluation metrics for multi-agent driving tasks.
* <strong>Scalability</strong>: Distributed training infrastructure supporting large-scale multi-agent scenarios with hundreds of vehicles.
* <strong>Mixed Autonomy</strong>: Support for mixed traffic with human-driven vehicles, rule-based AVs, and learning-based agents.

Users can refer to our [documentation](#) for detailed guides on MARL integration, training procedures, and API references. For the original OpenCDA documentation, visit [OpenCDA documentation](https://opencda-documentation.readthedocs.io/en/latest/).

## What's New in OpenCDA-MARL

### August 2025

* **MARL Framework Integration**: Full integration of Multi-Agent Reinforcement Learning capabilities with support for PPO, SAC, QMIX, and MADDPG algorithms.
* **Distributed Training**: Scalable training infrastructure using Ray/RLlib for large-scale multi-agent scenarios.
* **Mixed Autonomy Support**: Seamless integration of learning-based agents with rule-based vehicles and human-driven traffic.

### Key Updates from Original OpenCDA

* **Environment Changes**: Changed Conda environment to Pixi for easy installation.
* **Enhanced Configuration System**: Clean YAML-based configuration with `default.yaml` template
* **Docker Support**: Easy deployment and reproducibility
* **Windows Compatibility**: Full support for Windows with Python 3.10.x and CUDA 12.8
* **HD Map Manager**: Real-time rasterization maps for RL planning
* **CARLA 0.9.15**: Latest CARLA version support with improved stability

## Major Components

![teaser](docs/images/OpenCDA_MARL_architecture.png)

OpenCDA-MARL extends the original four components with MARL-specific modules:

* <strong>MARL Training Framework</strong>: Distributed training infrastructure with multiple RL algorithms
* <strong>Cooperative Driving System</strong>: Enhanced with learning-based decision making
* <strong>Co-Simulation Tools</strong>: CARLA + SUMO integration with RL environment wrapper
* <strong>Data Manager and Repository</strong>: Training data collection and replay buffer management
* <strong>Scenario Manager</strong>: MARL-specific training and evaluation scenarios

Check our [documentation](#) for detailed architecture and MARL integration.

## Get Started

 ![teaser](docs/images/MARL-rule-based-simulation.gif)

### Users Guide

* [Overview](https://radar-lab.github.io/OpenCDA-MARL/)
* [Installation](https://radar-lab.github.io/OpenCDA-MARL/installation/)
* [Quick Start](https://radar-lab.github.io/OpenCDA-MARL/quick-start/)

Note: We continuously improve the performance of OpenCDA-MARL. Currently, it is mainly tested in our customized maps. However, we <strong>DO NOT </strong> guarantee the same level of robustness in other maps. We will update the documentation and the maps in the future.

### Developer Guide

* [Class Design](https://radar-lab.github.io/OpenCDA-MARL/architecture/)
* [Customize Your Algorithms](#)
* [API Reference](https://radar-lab.github.io/OpenCDA-MARL/api/opencda-marl/overview/) <br>

### Contributing

We welcome contributions to OpenCDA-MARL! Please see our [Contributing Guide](docs/contributing.md) for detailed instructions on:

* How to fork and clone the repository
* Creating feature branches
* Submitting pull requests
* Coding standards and guidelines

For quick reference:

* Report bugs and improvements by submitting [issues](https://github.com/radar-lab/OpenCDA-MARL/issues)
* Submit contributions via [pull requests](https://github.com/radar-lab/OpenCDA-MARL/pulls) using our [PR template](.github/PR_TEMPLATE.md)

## Citation

If you are using OpenCDA-MARL for your research, please cite both the MARL extension and the original OpenCDA paper:

```bibtex
@software{opencda-marl2025,
  title={OpenCDA-MARL: A Unified Benchmarking Framework for Cooperative Autonomous Intersection Management with Multi-Agent Reinforcement Learning},
  author={Lihao Guo, Louis Liu, Jiahao Tang, Liu Bo, Siyang Cao},
  year={2025},
  url={https://github.com/radar-lab/OpenCDA-MARL}
}

@inproceedings{xu2021opencda,
  title={OpenCDA: an open cooperative driving automation framework integrated with co-simulation},
  author={Xu, Runsheng and Guo, Yi and Han, Xu and Xia, Xin and Xiang, Hao and Ma, Jiaqi},
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
  pages={1155--1162},
  year={2021},
  organization={IEEE}
}
```

The arxiv link to the paper:  <https://arxiv.org/abs/2107.06260>

Also, under this LICENSE, OpenCDA is for non-commercial research only. Researchers can modify the source code for their own research only. Contracted work that generates corporate revenues and other general commercial use are prohibited under this LICENSE. See the LICENSE file for details and possible opportunities for commercial use.

## Contributors

### OpenCDA-MARL Team

The MARL extension is developed and maintained by researchers focusing on multi-agent reinforcement learning for autonomous driving.

* **Project Lead**: Lihao Guo
* **Core Team**: Louis Liu, Jiahao Tang
* **Advisors**: Dr. Liu Bo, Dr. Siyang Cao

### OpenCDA Team

OpenCDA is originally developed by the [UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/):

* **Principal Investigator**: Dr. Jiaqi Ma ([UCLA Samueli](https://samueli.ucla.edu/people/jiaqi-ma/))
* **Project Lead**: Runsheng Xu ([github](https://github.com/DerrickXuNu))
* **Core Team**: Xu Han, Hao Xiang, Zhaoliang Zheng, Zonglin Meng, Dr. Xin Xia

### Acknowledgements

* [UA Radar lab](https://github.com/radar-lab)
* [UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/)
* UC Davis Professor [Junshan Zhang's](https://faculty.engineering.ucdavis.edu/jzhang/) group for openScenario integration
* @GoodarzMehr for Docker support
* All contributors to the OpenCDA and OpenCDA-MARL projects
