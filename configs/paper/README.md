# Paper Experiment Configurations

This folder contains configurations for reproducible paper experiments comparing baseline agents with MARL agents.

## Folder Structure

```
configs/paper/
├── default.yaml          # Common settings (traffic, rewards, simulation)
├── agents/
│   ├── behavior.yaml     # Simple reactive agent
│   ├── vanilla.yaml      # Basic collision avoidance
│   ├── rule_based.yaml   # 3-stage hierarchical control
│   ├── td3.yaml          # TD3 MARL agent
│   ├── dqn.yaml          # DQN MARL agent
│   └── mappo.yaml        # MAPPO (future extension)
└── README.md
```

## Usage

Run experiments using the paper configs:

```bash
# Baseline agents (no training)
python run_marl.py --config configs/paper/agents/behavior.yaml
python run_marl.py --config configs/paper/agents/vanilla.yaml
python run_marl.py --config configs/paper/agents/rule_based.yaml

# MARL agents (training)
python run_marl.py --config configs/paper/agents/td3.yaml
python run_marl.py --config configs/paper/agents/dqn.yaml
```

## Experiment Design

### Traffic Scenario
- **12-lane intersection** with dense traffic (300 vph per direction)
- **Replay mode** for reproducibility
- **2-minute episodes** (2400 steps at 20 FPS)

### Reward Configuration (Safety-First)
All agents share the same reward structure for fair comparison:

| Reward | Value | Rationale |
|--------|-------|-----------|
| Collision | -500 | Strong deterrent |
| Success | +300 | Goal achievement |
| Step penalty | -0.5 | Reduced rush pressure |
| TTC critical | -15 | Strong near-miss penalty |
| Progress | +1.0 | Goal-seeking bonus |

### Agents Comparison

| Agent | Type | Description |
|-------|------|-------------|
| Behavior | Baseline | Simple stop/slow reactive control |
| Vanilla | Baseline | Basic collision avoidance |
| Rule-Based | Baseline | 3-stage hierarchical + TTC + FCFS |
| TD3 | MARL | Continuous action, no warmup |
| DQN | MARL | Discrete actions |
| MAPPO | MARL | Multi-agent PPO (future) |

## Reproducibility

For paper experiments:
1. Use fixed seed (`world.seed: 42`)
2. Run 3 seeds per agent (modify seed in config)
3. Report mean ± std across seeds
4. Use same traffic recording for all runs
