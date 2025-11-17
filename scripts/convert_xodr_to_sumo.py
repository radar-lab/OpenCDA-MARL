#!/usr/bin/env python
"""
Convert OpenDRIVE (.xodr) files to SUMO network files.

Usage:
    python scripts/convert_xodr_to_sumo.py
"""
import os
import sys
import subprocess
from pathlib import Path

# Check SUMO_HOME
if 'SUMO_HOME' not in os.environ:
    sys.exit("ERROR: Please set SUMO_HOME environment variable")

SUMO_TOOLS = os.path.join(os.environ['SUMO_HOME'], 'tools')
NETCONVERT = os.path.join(os.environ['SUMO_HOME'], 'bin', 'netconvert.exe')

def convert_xodr_to_sumo(xodr_path: str, output_dir: str):
    """
    Convert XODR file to SUMO network.

    Args:
        xodr_path: Path to input .xodr file
        output_dir: Directory to save SUMO files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output network file
    net_file = os.path.join(output_dir, 'intersection.net.xml')

    print(f"Converting {xodr_path} to SUMO network...")
    print(f"Output: {net_file}")

    # Run netconvert
    cmd = [
        NETCONVERT,
        '--opendrive', xodr_path,
        '--output-file', net_file,
        '--opendrive.curve-resolution', '1.0',  # Finer resolution
        '--junctions.corner-detail', '5',
        '--tls.discard-simple',  # Remove simple traffic lights
        '--tls.guess', 'true',  # Guess traffic light positions
        '--ramps.guess', 'true',
        '--junctions.join', 'true',
        '--geometry.remove', 'true',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Successfully converted to {net_file}")
            return net_file
        else:
            print(f"[ERROR] Conversion failed:")
            print(result.stderr)
            return None
    except FileNotFoundError:
        print(f"[ERROR] netconvert not found at {NETCONVERT}")
        print("  Make sure SUMO is installed and SUMO_HOME is set correctly")
        return None

def create_route_file(output_dir: str, net_file: str):
    """
    Create a basic route file for the intersection.

    Args:
        output_dir: Directory to save route file
        net_file: Path to network file
    """
    route_file = os.path.join(output_dir, 'intersection.rou.xml')

    # Simple route file template
    route_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

    <!-- Vehicle type definitions -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="70.0" guiShape="passenger"/>

    <!-- Routes will be defined by MARL traffic manager -->
    <!-- This is a placeholder for manual testing -->
    <!-- Valid edges: -0, -1, -2, -3, 0, 1, 2, 3 -->

    <route id="north_to_south" edges="-0 2"/>
    <route id="south_to_north" edges="-2 0"/>
    <route id="east_to_west" edges="-1 3"/>
    <route id="west_to_east" edges="-3 1"/>

    <!-- Example flows (can be commented out when using MARL) -->
    <!--
    <flow id="flow_north" type="car" route="north_to_south" begin="0" end="3600" vehsPerHour="200"/>
    <flow id="flow_south" type="car" route="south_to_north" begin="0" end="3600" vehsPerHour="200"/>
    <flow id="flow_east" type="car" route="east_to_west" begin="0" end="3600" vehsPerHour="200"/>
    <flow id="flow_west" type="car" route="west_to_east" begin="0" end="3600" vehsPerHour="200"/>
    -->

</routes>
'''

    with open(route_file, 'w') as f:
        f.write(route_xml)

    print(f"[OK] Created route file: {route_file}")
    return route_file

def create_sumo_config(output_dir: str, net_file: str, route_file: str):
    """
    Create SUMO configuration file.

    Args:
        output_dir: Directory to save config file
        net_file: Path to network file
        route_file: Path to route file
    """
    config_file = os.path.join(output_dir, 'intersection.sumocfg')

    # Get relative paths
    net_rel = os.path.basename(net_file)
    route_rel = os.path.basename(route_file)

    config_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{net_rel}"/>
        <route-files value="{route_rel}"/>
    </input>

    <time>
        <begin value="0"/>
        <step-length value="0.05"/>
    </time>

    <processing>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>

</configuration>
'''

    with open(config_file, 'w') as f:
        f.write(config_xml)

    print(f"[OK] Created SUMO config: {config_file}")
    return config_file

def create_readme(output_dir: str):
    """Create README for SUMO assets."""
    readme_file = os.path.join(output_dir, 'README.md')

    readme_content = '''# SUMO Intersection Assets

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
'''

    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"[OK] Created README: {readme_file}")

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    xodr_file = project_root / 'opencda_marl' / 'assets' / 'maps' / 'intersection.xodr'
    output_dir = project_root / 'opencda_marl' / 'assets' / 'intersection_sumo'  # MARL-specific location

    print("=" * 60)
    print("XODR to SUMO Conversion Script (for MARL)")
    print("=" * 60)
    print(f"Input XODR: {xodr_file}")
    print(f"Output directory: {output_dir}")
    print()

    # Check input file exists
    if not xodr_file.exists():
        print(f"[ERROR] XODR file not found: {xodr_file}")
        sys.exit(1)

    # Convert XODR to SUMO network
    net_file = convert_xodr_to_sumo(str(xodr_file), str(output_dir))
    if not net_file:
        sys.exit(1)

    # Create route file
    route_file = create_route_file(str(output_dir), net_file)

    # Create SUMO config
    config_file = create_sumo_config(str(output_dir), net_file, route_file)

    # Create README
    create_readme(str(output_dir))

    print()
    print("=" * 60)
    print("[OK] Conversion complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Test the network: sumo-gui -c", config_file)
    print("2. Update configs/marl/intersection_sumo.yaml")
    print("3. Run SUMO MARL training: pixi run start -t intersection_sumo --marl")

if __name__ == '__main__':
    main()
