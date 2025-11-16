# SUMO MARL Training Guide

## Overview

This guide explains how to use SUMO for accelerated MARL training with transfer learning to CARLA.

### Training Pipeline

```
Phase 1: SUMO Pre-training (Fast)
    ↓
  1000 episodes @ 10-80x speed
    ↓
  Save checkpoint
    ↓
Phase 2: CARLA Fine-tuning (Accurate)
    ↓
  200 episodes with physics
    ↓
  Final policy
```

### Performance Benefits

| Metric | CARLA-only | SUMO → CARLA Transfer |
|--------|------------|----------------------|
| **Training Time** | ~5-7 days | ~1.5 days total |
| **Episodes (1000)** | 168 hours | 12 hours (SUMO) + 24 hours (CARLA) |
| **Agent Scalability** | 10 agents max | 50+ agents in SUMO |
| **GPU Usage** | High | Low (CPU-only SUMO phase) |

---

## Quick Start

### 1. SUMO Pre-training

Train a policy in SUMO (10-80x faster than CARLA):

```bash
# Standard training
pixi run python opencda.py -t intersection_sumo --marl

# With debug output
pixi run python opencda.py -t intersection_sumo --marl --debug

# With SUMO GUI (visual debugging)
# Edit configs/marl/intersection_sumo.yaml: set sumo_gui: true
pixi run python opencda.py -t intersection_sumo --marl
```

**Expected Output:**
```
[INFO] Initializing SUMO-only MARL environment
[INFO] Starting SUMO with config: opencda/assets/intersection_sumo/intersection.sumocfg
[INFO] SUMO started successfully
[INFO] Episode 1/1000 started...
...
[INFO] Checkpoint saved at episode 10
```

**Training Progress:**
- Episodes 1-100: Exploration phase (high collision rate)
- Episodes 100-500: Learning phase (collision rate decreasing)
- Episodes 500-1000: Convergence phase (stable policy)

**Checkpoint Location:**
- `checkpoints/sumo_td3/latest_checkpoint.pth`
- `checkpoints/sumo_td3/episode_100_checkpoint.pth`
- `checkpoints/sumo_td3/episode_500_checkpoint.pth`
- etc.

### 2. CARLA Fine-tuning

Transfer the SUMO policy to CARLA for physics-accurate training:

```bash
# Fine-tune from latest SUMO checkpoint
pixi run python opencda.py -t intersection_finetune --marl

# Fine-tune from specific checkpoint
# Edit configs/marl/intersection_finetune.yaml:
#   load_checkpoint: "checkpoints/sumo_td3/episode_1000_checkpoint.pth"
pixi run python opencda.py -t intersection_finetune --marl
```

**Expected Output:**
```
[INFO] Initializing CARLA MARL environment
[INFO] Loading checkpoint from: checkpoints/sumo_td3/latest_checkpoint.pth
[INFO] Resumed from episode 1000, step 2400000
[INFO] Episode 1/200 started... (fine-tuning)
```

### 3. Evaluation

Compare SUMO-trained vs CARLA-finetuned policies:

```bash
# Evaluate SUMO policy in CARLA (without training)
# Edit configs/marl/intersection_finetune.yaml: set training_mode: false
pixi run python opencda.py -t intersection_finetune --marl

# Check metrics
# Results saved in: data/evaluation/intersection_finetune/
```

---

## Configuration Files

### SUMO Training Config

File: `configs/marl/intersection_sumo.yaml`

**Key Parameters:**
```yaml
meta:
  simulator: "sumo"  # Use SUMO instead of CARLA
  sumo_cfg: "opencda/assets/intersection_sumo/intersection.sumocfg"

scenario:
  simulation:
    max_episodes: 1000  # More episodes (SUMO is fast)

MARL:
  training:
    checkpoint_dir: "checkpoints/sumo_td3/"
    load_checkpoint: null  # Start from scratch
```

### CARLA Fine-tuning Config

File: `configs/marl/intersection_finetune.yaml`

**Key Parameters:**
```yaml
meta:
  simulator: "carla"  # Back to CARLA

scenario:
  simulation:
    max_episodes: 200  # Fewer episodes (CARLA is slow)

MARL:
  td3:
    learning_rate_actor: 5e-4  # REDUCED for fine-tuning
    exploration_noise: 0.2  # REDUCED from 0.5

  training:
    checkpoint_dir: "checkpoints/carla_finetune_td3/"
    load_checkpoint: "checkpoints/sumo_td3/latest_checkpoint.pth"  # Load SUMO
```

---

## Advanced Usage

### Scaling Agent Count

SUMO can handle 50+ agents simultaneously:

```yaml
# configs/marl/intersection_sumo.yaml
agents:
  count: 50  # Increase from 10
```

### Custom SUMO Networks

1. Create custom XODR file
2. Convert to SUMO:
   ```bash
   python scripts/convert_xodr_to_sumo.py
   ```
3. Update config:
   ```yaml
   meta:
     sumo_cfg: "opencda/assets/custom_intersection/custom.sumocfg"
   ```

### Monitoring Training

View real-time metrics:
```bash
# Enable SUMO GUI for visualization
# configs/marl/intersection_sumo.yaml
world:
  sumo_gui: true
```

Check checkpoint quality:
```python
import torch

# Load checkpoint
ckpt = torch.load("checkpoints/sumo_td3/episode_500_checkpoint.pth")

# Check metrics
print(f"Episode: {ckpt['episode']}")
print(f"Collision rate: {ckpt['metrics']['collision_rate']}")
print(f"Success rate: {ckpt['metrics']['success_rate']}")
```

---

## Troubleshooting

### Issue: SUMO connection error

**Error:**
```
traci.exceptions.TraCIException: Could not connect to TraCI server at localhost:8873
```

**Solution:**
1. Check SUMO_HOME is set:
   ```bash
   echo $SUMO_HOME  # Should point to SUMO installation
   ```
2. Check port availability:
   ```bash
   netstat -an | grep 8873
   ```
3. Change port in config if needed:
   ```yaml
   world:
     sumo_port: 8874
   ```

### Issue: Transfer learning gap

**Problem:** Policy trained in SUMO performs poorly in CARLA

**Solutions:**
1. **Increase fine-tuning episodes:**
   ```yaml
   scenario:
     simulation:
       max_episodes: 500  # Instead of 200
   ```

2. **Reduce fine-tuning learning rate further:**
   ```yaml
   td3:
     learning_rate_actor: 1e-4  # Even lower
   ```

3. **Add domain randomization in SUMO:**
   ```yaml
   scenario:
     traffic:
       speed_variation: 0.3  # Higher variation
   ```

### Issue: Out of memory during CARLA fine-tuning

**Solution:** Reduce agent count:
```yaml
agents:
  count: 5  # Reduce from 10
```

---

## Performance Benchmarks

### Training Time (1000 episodes, 10 agents)

| Setup | Time | Speedup |
|-------|------|---------|
| CARLA-only (RTX 5090) | ~5-7 days | 1x |
| SUMO-only | ~12 hours | **10-14x** |
| SUMO (900) + CARLA (100) | ~1.5 days | **3-5x** |

### Memory Usage

| Setup | GPU VRAM | System RAM |
|-------|----------|-----------|
| CARLA (10 agents) | ~8-12 GB | ~4 GB |
| SUMO (50 agents) | 0 GB | ~2 GB |

---

## Best Practices

### 1. Observation Space Consistency

**Critical:** SUMO and CARLA must use identical observation features.

Current features (7D):
```python
{
    'relative_position_x',  # To intersection
    'relative_position_y',
    'speed',
    'heading_angle',
    'distance_to_intersection',
    'distance_to_front_vehicle',
    'lane_position'
}
```

**Do NOT modify** feature extraction in SUMO without updating CARLA config.

### 2. Reward Structure

Keep rewards identical between SUMO and CARLA:
```yaml
rewards:
  collision: -500.0  # Same everywhere
  success: 400.0
  step_penalty: -1.5
  speed_bonus: 0.5
```

### 3. Hyperparameter Tuning

Only tune these during fine-tuning:
- `learning_rate_actor`
- `learning_rate_critic`
- `exploration_noise`
- `warmup_steps`

Keep these **fixed** (match SUMO):
- `state_dim`
- `action_dim`
- Network architecture
- `discount`, `tau`, etc.

### 4. Checkpoint Management

Save checkpoints frequently in SUMO (cheap):
```yaml
training:
  save_freq: 10  # Every 10 episodes
```

Save less frequently in CARLA (expensive):
```yaml
training:
  save_freq: 5
```

---

## Example Workflow

### Complete Training Pipeline

```bash
# Step 1: SUMO Pre-training (12 hours)
pixi run python opencda.py -t intersection_sumo --marl

# Step 2: Evaluate SUMO policy
# Edit intersection_finetune.yaml: set training_mode: false
pixi run python opencda.py -t intersection_finetune --marl

# Step 3: CARLA Fine-tuning (24 hours)
# Edit intersection_finetune.yaml: set training_mode: true
pixi run python opencda.py -t intersection_finetune --marl

# Step 4: Final Evaluation
# Edit intersection_finetune.yaml: set training_mode: false
pixi run python opencda.py -t intersection_finetune --marl
```

### Comparing Baselines

```bash
# Baseline 1: CARLA-only (no SUMO)
pixi run python opencda.py -t intersection --marl

# Baseline 2: SUMO-only (no fine-tuning)
pixi run python opencda.py -t intersection_sumo --marl

# Baseline 3: Transfer learning (SUMO → CARLA)
pixi run python opencda.py -t intersection_sumo --marl
pixi run python opencda.py -t intersection_finetune --marl
```

---

## Next Steps

1. **Scale up:** Try 50+ agents in SUMO
2. **Custom scenarios:** Create highway or merge scenarios
3. **Distributed training:** Run multiple SUMO instances in parallel
4. **Real-world transfer:** Test on CARLA Town01-Town10

---

## References

- SUMO Documentation: https://sumo.dlr.de/docs/
- TraCI API: https://sumo.dlr.de/docs/TraCI.html
- OpenCDA Documentation: [Add link]
- Transfer Learning in RL: [Add research papers]
