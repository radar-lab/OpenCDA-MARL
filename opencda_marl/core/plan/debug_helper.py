'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-01 11:35:29
FilePath     : /OpenCDA-MARL/opencda_marl/core/plan/debug_helper.py
Description  : MARL specific debug helper functions for trajectory visualization.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import carla


def draw_trajetory_points(world, waypoints, z=0.25,
                         color=carla.Color(255, 0, 0),
                         lt=5, size=0.1, arrow_size=0.1):
    """
    Draw a list of trajectory points

    Parameters
    ----------
    size : float
        Time step between updating visualized waypoint.

    lt : int
        Number of waypoints being visualized.

    color : carla.Color
        The trajectory color.

    world : carla.world
        The simulation world.

    waypoints : list
        List of waypoints to draw.

    z : float
        Z-offset for drawing points.

    arrow_size : float
        Size of direction arrows.
    """
    if not waypoints:
        return
        
    for waypoint in waypoints:
        try:
            # Handle different waypoint types
            if hasattr(waypoint, 'transform'):
                # carla.Waypoint
                location = waypoint.transform.location + carla.Location(z=z)
                rotation = waypoint.transform.rotation
            elif hasattr(waypoint, 'location'):
                # carla.Transform
                location = waypoint.location + carla.Location(z=z)
                rotation = waypoint.rotation
            elif isinstance(waypoint, tuple) and len(waypoint) >= 1:
                # Tuple (waypoint, road_option) or (transform, speed)
                wp_or_transform = waypoint[0]
                if hasattr(wp_or_transform, 'transform'):
                    location = wp_or_transform.transform.location + carla.Location(z=z)
                    rotation = wp_or_transform.transform.rotation
                elif hasattr(wp_or_transform, 'location'):
                    location = wp_or_transform.location + carla.Location(z=z)
                    rotation = wp_or_transform.rotation
                else:
                    continue
            else:
                continue

            # Draw point
            world.debug.draw_point(
                location,
                size=size,
                color=color,
                life_time=lt
            )
            
            # Draw arrow to show direction
            if arrow_size > 0:
                forward_vec = rotation.get_forward_vector()
                end_location = location + carla.Location(
                    x=forward_vec.x * arrow_size,
                    y=forward_vec.y * arrow_size,
                    z=forward_vec.z * arrow_size
                )
                world.debug.draw_arrow(
                    location,
                    end_location,
                    thickness=0.02,
                    arrow_size=arrow_size * 0.5,
                    color=color,
                    life_time=lt
                )
        except Exception as e:
            # Skip problematic waypoints
            print(f"Warning: Could not draw waypoint: {e}")
            continue