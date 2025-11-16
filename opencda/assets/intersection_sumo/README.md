# SUMO Intersection Assets

This directory contains SUMO network files converted from OpenDRIVE format for MARL training.

## Files

- `intersection.net.xml` - SUMO network file (converted from XODR)
- `intersection.rou.xml` - Route definitions (managed by MARL traffic manager)
- `intersection.sumocfg` - SUMO configuration file

## Usage

These files are automatically loaded by `SumoMARLEnv` when using the SUMO-only training mode.

Configuration in `configs/marl/intersection_sumo.yaml`:
```yaml
meta:
  scenario_type: "intersection_sumo"
  sumo_cfg: "opencda/assets/intersection_sumo/intersection.sumocfg"
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
