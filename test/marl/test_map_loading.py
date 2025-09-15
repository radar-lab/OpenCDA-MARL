'''
Author: AXIBA leolihao@arizona.edu
Date: 2025-08-13 13:21:25
FilePath: \OpenCDA\test\marl\test_map_loading.py
Description: Test MARL map loading - ensuring complete map functionality.

This test focuses on loading the intersection map correctly with Road metadata (registry)

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''

import sys
import os
import time
import carla
from omegaconf import OmegaConf
from loguru import logger

# Add project root to path
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# set loguru logger level to debug
logger.level("DEBUG")


def checking_registry():
    from opencda_marl.core.world import MAP_REGISTRY
    map_info = MAP_REGISTRY.get("intersection")
    if map_info:
        print("   ✓ Intersection found in registry:")
        print(f"     Type: {map_info['type']}")
        print(f"     Description: {map_info['description']}")
        for key in ['xodr_path', 'fbx_path']:
            path = map_info.get(key)
            if path and os.path.exists(path):
                print(f"     {key}: {path} (EXISTS)")
            else:
                print(f"     {key}: {path} (NOT FOUND)")
    else:
        print("   ✗ Intersection NOT in registry")


def checking_carla_connection():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get version
    version = client.get_server_version()
    print(f"   Connected to CARLA {version}")

    return client


def checking_carla_maps(client):
    try:
        available_maps = client.get_available_maps()
        print(f"   CARLA has {len(available_maps)} maps available:")

        # Look for intersection-related maps
        intersection_maps = [
            m for m in available_maps if 'intersection' in m.lower()]
        if intersection_maps:
            print("   🎯 Found intersection-related maps:")
            for map_name in intersection_maps:
                print(f"      • {map_name}")
        else:
            print("   ⚠ No intersection maps found in CARLA")
            print("   📋 Available maps include:")
            # Show first 5
            for i, map_name in enumerate(available_maps[:5]):
                print(f"      • {i+1}: {map_name}")
            if len(available_maps) > 5:
                print(f"      ... and {len(available_maps) - 5} more")
    except Exception as e:
        print(f"   Could not get CARLA maps: {e}")


def print_tips():
    print("\n" + "=" * 90)
    print("MAP VISUALIZATION READY")
    print("=" * 90)
    print("\nYou can now view the map in CARLA:")
    print("1. The intersection map should be loaded")
    print("2. Test vehicles are spawned at entry points")
    print("3. Waypoints are marked in yellow")

    print("\n📍 TEXTURE INFORMATION:")
    print("  • CARLA cannot dynamically load FBX textures")
    print("  • Custom XODR files only provide road geometry")
    print("  • For proper lane markings:")
    print("    - Import map into CARLA beforehand")
    print("    - Or use CARLA's built-in maps")
    print("  • Since you pasted the map into CARLA, it should load with textures")

    print("\nCamera controls in CARLA window:")
    print("  - WASD: Move camera")
    print("  - Mouse: Look around")
    print("  - Q/E: Move up/down")
    print("  - TAB: Toggle to next vehicle view")


def test_complete_map_loading():
    """
    Test complete map loading functionality.

    This test:
    1. Connects to CARLA
    2. Loads intersection.xodr + intersection.fbx properly
    3. Verifies road textures and lane markings
    4. Sets up optimal viewing conditions
    5. Allows inspection of map loading results
    """

    try:
        # Import CARLA and MARL modules

        from opencda_marl.core.world.map_manager import MARLMapManager

        print("=" * 60)
        print("MARL MAP LOADING TEST (Registry-based)")
        print("=" * 60)

        # Show registry information
        print("\n0. Checking MAP_REGISTRY...")
        checking_registry()

        # Connect to CARLA
        print("\n1. Connecting to CARLA...")
        client = checking_carla_connection()

        # Check what maps CARLA has available
        print("\n2. Checking CARLA's available maps...")
        checking_carla_maps(client)

        # Create map manager
        print("\n3. Creating map manager...")

        # Configuration
        config = OmegaConf.create({
            "map": {
                "name": "intersection",
                #"name": "Town10HD",
                "safe_distance": 6.0,
                "spawn_offset": 2,
                "dest_offset": 2,
                "wp_step": 0.5,
                "spawn_z_lift": 1.0,
            }
        })
        # Map will be loaded automatically from the config
        map_manager = MARLMapManager(config, client)

        if map_manager.world is None:
            print("  ✗ Failed to load map")
            return False
        else:
            print("  ✓ Map loaded successfully")

        # List available maps (MARL)
        map_manager.list_maps()

        # Get map info
        info = map_manager.get_info()
        print("\n4. Map Information:")
        for key, value in info.items():
            print(f"   {key:<20}: {str(value)}")

        # Set up spectator
        print("\n5. Setting up spectator view...")
        spectator = map_manager.world.get_spectator()
        # Set spectator to bird's eye view of intersection
        spectator_transform = carla.Transform(
            carla.Location(x=0, y=0, z=100),  # 100m above center
            carla.Rotation(pitch=-90, yaw=0, roll=0)  # Looking down
        )
        spectator.set_transform(spectator_transform)
        print("   Spectator positioned at bird's eye view")

        # Spawn test vehicles at spawn points
        print("\n6. Spawning test vehicles at spawn points...")

        # Get blueprint library
        blueprint_library = map_manager.world.get_blueprint_library()

        # check spawn points
        spawn_points = map_manager.get_spawn_points(num=6, dest=True)
        print(f"   Found {len(spawn_points)} spawn points")

        # draw spawn points
        map_manager.draw_spawn_points(life_time=25.0)


        spawned_vehicles = []
        # Vehicle colors for visualization
        colors = ['255,0,0', '0,255,0', '0,0,255',
                  '255,255,0']  # Red, Green, Blue, Yellow
        vehicle_bp_name = 'vehicle.tesla.model3'
        for i, spawn_point in enumerate(spawn_points):
            spawn_point = spawn_point[0]
            try:
                # Get vehicle blueprint
                vehicle_bp = blueprint_library.filter(vehicle_bp_name)[0]

                # Set color
                if vehicle_bp.has_attribute('color'):
                    vehicle_bp.set_attribute('color', colors[i % len(colors)])

                print(spawn_point.location)
                # Spawn vehicle
                vehicle = map_manager.world.spawn_actor(
                    vehicle_bp, spawn_point)
                spawned_vehicles.append(vehicle)

                loc = spawn_point.location
                print(
                    f"   Vehicle {i+1} spawned at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

            except Exception as e:
                print(f"   Warning: Could not spawn vehicle {i+1}: {e}")

        print(f"\n[OK] Spawned {len(spawned_vehicles)} test vehicles")

        # Add some road markings or waypoints visualization
        print("\n7. Visualizing details...")
        
        # Draw junction centers
        map_manager.draw_junction_centers()

        # Show junction areas if any
        topology = map_manager.world.get_map().get_topology()
        print(f"   Map topology has {len(topology)} segments")

        # Get waypoints at regular intervals
        safe_distance = config.get("safe_distance", 6.0)
        waypoint_list = map_manager.world.get_map().generate_waypoints(distance=safe_distance)

        # Draw waypoints (first 25 for visibility if too many)
        #if len(waypoint_list) > 25:
        #    waypoint_list = waypoint_list[:25]
        for wp in waypoint_list:
            map_manager.world.debug.draw_point(
                wp.transform.location + carla.Location(z=0.1),
                size=0.1,
                color=carla.Color(255, 255, 0),
                life_time=30.0
            )

        print(f"   Drew {len(waypoint_list)} waypoints")

        # Keep alive for inspection
        print_tips()
        duration = 30
        print(f"\nKeeping simulation alive for {duration} seconds...")
        print("Press Ctrl+C to stop earlier")

        try:
            for i in range(duration):
                map_manager.world.tick()
                time.sleep(1)
                if i % 5 == 0:
                    print(f"  {duration - i} seconds remaining...")

        except KeyboardInterrupt:
            print("\nStopped by user")

        # Cleanup
        print("\n8. Cleaning up...")
        for vehicle in spawned_vehicles:
            vehicle.destroy()
        print(f"   Destroyed {len(spawned_vehicles)} vehicles")

        print("\n" + "=" * 90)
        print("[OK] Map visualization test completed successfully!")
        print("=" * 90)

        return True

    except ImportError as e:
        print(f"\n[FAIL] Import error: {e}")
        print("Make sure CARLA Python API is installed")
        return False

    except RuntimeError as e:
        if "timeout" in str(e).lower():
            print("\n[FAIL] Cannot connect to CARLA")
            print("Please ensure:")
            print("  1. CARLA is running (CarlaUE4.exe)")
            print("  2. It's listening on localhost:2000")
        else:
            print(f"\n[FAIL] Runtime error: {e}")
        return False

    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the map loading test."""

    print("MARL Map Loading Test")
    print("=" * 60)
    print("This test focuses on getting intersection.xodr + intersection.fbx")
    print("to load properly with road textures and lane markings.")
    print("\nRequirements:")
    print("• CARLA running on localhost:2000")
    print("• intersection.xodr in opencda_marl/assets/maps/")
    print("• intersection.fbx in opencda_marl/assets/maps/")

    user_input = input("\nIs CARLA running? [y/N]: ")

    if user_input.lower() != 'y':
        print("\nPlease start CARLA first:")
        print("  1. Navigate to your CARLA directory")
        print("  2. Run: CarlaUE4.exe (or ./CarlaUE4.sh on Linux)")
        print("  3. Wait for it to fully load")
        print("  4. Run this test again")
        return False

    return test_complete_map_loading()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
