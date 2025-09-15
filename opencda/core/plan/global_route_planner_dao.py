'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-27 17:32:06
FilePath     : /OpenCDA-MARL/opencda/core/plan/global_route_planner_dao.py
Description  : This module provides implementation for GlobalRoutePlannerDAO

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
Copyright (c) 2018-2020 CVC.
'''
import numpy as np
import carla


class GlobalRoutePlannerDAO(object):
    """
    This class is the data access layer for fetching data from the carla
    server instance for GlobalRoutePlanner.

    Parameters
    -wmap : carla.world
        The current carla simulation world.
    -sampling_resolution : float
        sampling distance between waypoints.

    """

    def __init__(self, wmap, sampling_resolution):

        self._sampling_resolution = sampling_resolution
        self._wmap = wmap

    def get_topology(self):
        """
        Accessor for topology.
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects.

            :return topology:
                entry   -   waypoint of entry point of road segment
                entryxyz-   (x,y,z) of entry point of road segment
                exit    -   waypoint of exit point of road segment
                exitxyz -   (x,y,z) of exit point of road segment
                path    -   list of waypoints separated by 1m from entry
                            to exit
        """
        topology = []

        # Get base topology from Carla
        base_segments = self._wmap.get_topology()

        # print(f"Base topology segments: {len(base_segments)}")

        # Process base segments
        for segment in base_segments:
            wp1, wp2 = segment[0], segment[1]
            seg_dict = self._create_segment_dict(wp1, wp2)
            if seg_dict:
                topology.append(seg_dict)
                # print(f"Added base segment: road_id={wp1.road_id}, lane_id={wp1.lane_id}")

        # <OpenCDA-MARL> - discover missing road segments
        missing_segments = self._discover_missing_segments(topology)
        # print(f"Found {len(missing_segments)} missing segments")

        for seg_dict in missing_segments:
            topology.append(seg_dict)

        # print(f"Total topology segments: {len(topology)}")
        # </OpenCDA-MARL>
        return topology

    def _create_segment_dict(self, wp1, wp2):
        """
        Create a segment dictionary from two waypoints.
        """
        try:
            l1, l2 = wp1.transform.location, wp2.transform.location
            # Rounding off to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round(
                [l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (
                x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(
                    endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(
                        endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    w = w.next(self._sampling_resolution)[0]
            else:
                seg_dict['path'].append(wp1.next(self._sampling_resolution)[0])
            return seg_dict
        except Exception as e:
            print(f"Failed to create segment: {e}")
            return None

    def get_waypoint(self, location):
        """
        The method returns waypoint at given location.

        Args:
            -location (carla.lcoation): Vehicle location.
        Returns:
            -waypoint (carla.waypoint): Newly generated waypoint close
            to location.
        """
        waypoint = self._wmap.get_waypoint(location)
        return waypoint

    def get_resolution(self):
        """ Return the sampling resolution."""
        return self._sampling_resolution

    # --------------------------------------------------------------------- #
    # <OpenCDA-MARL> - discover missing road segments
    # --------------------------------------------------------------------- #
    def _discover_missing_segments(self, existing_topology):
        """
        Discover missing road segments by sampling waypoints across the map.
        This is crucial for intersection scenarios where some road segments
        are not included in the base topology.
        """
        missing_segments = []

        # Create a set of existing segments for quick lookup
        existing_segments = set()
        for seg in existing_topology:
            key = (seg['entry'].road_id,
                   seg['entry'].section_id, seg['entry'].lane_id)
            existing_segments.add(key)

        # Sample waypoints across the map to find missing segments
        # print("Sampling waypoints to find missing road segments...")
        sample_waypoints = self._wmap.generate_waypoints(
            5.0)  # Sample every 5 meters

        # Group waypoints by road/section/lane
        lane_waypoints = {}
        for wp in sample_waypoints:
            if wp.lane_type == carla.LaneType.Driving:  # Only driving lanes
                key = (wp.road_id, wp.section_id, wp.lane_id)
                if key not in lane_waypoints:
                    lane_waypoints[key] = []
                lane_waypoints[key].append(wp)

        # print(f"Found {len(lane_waypoints)} unique lane segments in map")

        # Create segments for lanes not in existing topology
        for (road_id, section_id, lane_id), waypoints in lane_waypoints.items():
            key = (road_id, section_id, lane_id)

            if key not in existing_segments:
                # print(f"Missing segment found: road_id={road_id},"
                #      f"section_id={section_id}, lane_id={lane_id}")

                # Sort waypoints by distance along the lane
                # s is the distance along the road
                waypoints.sort(key=lambda wp: wp.s)

                if len(waypoints) >= 2:
                    # Create segment from first to last waypoint in this lane
                    entry_wp = waypoints[0]
                    exit_wp = waypoints[-1]

                    # Make sure they're far enough apart to be a valid segment
                    if entry_wp.transform.location.distance(exit_wp.transform.location) > 10.0:
                        seg_dict = self._create_segment_dict(entry_wp, exit_wp)
                        if seg_dict:
                            missing_segments.append(seg_dict)
                            # print(f"Added missing segment: road_id={road_id}, lane_id={lane_id}, "
                            #      f"length={entry_wp.transform.location.distance(exit_wp.transform.location):.1f}m")

        return missing_segments
