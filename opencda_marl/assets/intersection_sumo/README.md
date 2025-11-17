# SUMO Intersection Assets (MARL)

This directory contains SUMO network files converted from OpenDRIVE format specifically for MARL training.

**Location:** `opencda_marl/assets/intersection_sumo/`

This is separate from OpenCDA's co-simulation assets because it's optimized for pure SUMO MARL training without CARLA.

## Files

- `intersection.net.xml` - SUMO network file (converted from XODR)
- `intersection.rou.xml` - Route definitions for intersection scenario
- `intersection.sumocfg` - SUMO configuration file

## Usage

These files are automatically loaded by `SumoMARLEnv` when using SUMO-only MARL training.

Configuration in `configs/marl/intersection_sumo.yaml`:
```yaml
meta:
  scenario_type: "intersection_sumo"
  sumo_cfg: "opencda_marl/assets/intersection_sumo/intersection.sumocfg"
```

## Manual Testing

To test the SUMO network manually:
```bash
sumo-gui -c intersection.sumocfg
```

## Regeneration

To regenerate these files from the XODR source:
```bash
python scripts/convert_xodr_to_sumo.py
```
