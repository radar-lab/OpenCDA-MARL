'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 21:53:17
FilePath     : /OpenCDA-MARL/opencda_marl/core/traffic/utils.py
Description  : Utility functions for traffic planning.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import carla
from typing import List, Union


def with_z(t: Union[carla.Transform, 'Transform'], z: float) -> Union[carla.Transform, 'Transform']:
    """
    Add z-offset to a transform.
    Works with both CARLA Transform and SUMO mock Transform.
    """
    # Check if this is a SUMO mock Transform (has module 'sumo_adapter')
    if hasattr(t, '__class__') and 'sumo_adapter' in t.__class__.__module__:
        # SUMO mock object - use mock classes
        from opencda_marl.core.traffic.sumo_adapter import Transform, Location
        new_location = Location(
            x=t.location.x,
            y=t.location.y,
            z=t.location.z + float(z)
        )
        return Transform(location=new_location, rotation=t.rotation)
    else:
        # Real CARLA object
        return carla.Transform(t.location + carla.Location(z=float(z)), t.rotation)


def choose_same_lane(cands: List[carla.Waypoint], cur: carla.Waypoint) -> carla.Waypoint:
    return (
        [w for w in cands if (w.road_id, w.section_id, w.lane_id) ==
         (cur.road_id, cur.section_id, cur.lane_id)] or cands
    )[0]


def move_along_lane(start_wp: carla.Waypoint, distance: float, step: float = 1.0) -> carla.Waypoint:
    """
    Follow the lane center in small steps, keeping road/section/lane IDs.
    distance > 0 -> forward (next), distance < 0 -> backward (previous).
    Returns the last valid waypoint reached.
    """
    remaining = abs(float(distance))
    cur = start_wp
    step = float(step)

    while remaining > 1e-6:
        d = min(step, remaining)
        cands = cur.next(d) if distance >= 0 else cur.previous(d)
        if not cands:
            break
        nxt = choose_same_lane(cands, cur)
        # If we ever cross into a junction unintentionally on spawn side, stop a bit earlier
        cur = nxt
        remaining -= d
    return cur


def wp_to_transform(wp: carla.Waypoint, z_lift: float = 0.0) -> carla.Transform:
    return with_z(wp.transform, z_lift)


def wp_direction(wp: carla.Waypoint, junction_center: carla.Location) -> str:
    """
    Classify waypoint direction relative to junction center.
    
    Returns direction string: 'north', 'south', 'east', 'west'
    """
    dx = wp.transform.location.x - junction_center.x
    dy = wp.transform.location.y - junction_center.y
    
    if abs(dx) > abs(dy):
        return 'east' if dx > 0 else 'west'
    else:
        return 'north' if dy > 0 else 'south'