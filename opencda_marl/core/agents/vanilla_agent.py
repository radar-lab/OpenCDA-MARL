'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-31 17:05:26
FilePath     : /OpenCDA-MARL/opencda_marl/core/agents/vanilla_agent.py
Description  : Simple MARL Agent

This agent provides simple route following without complex autonomous behaviors:
- Follow scheduled waypoints from start to destination  
- Simple collision avoidance (slow down, stop, yield only)
- No lane changes, overtaking, or route replanning
- Full perception capabilities via update_information interface
- Compatible with OpenCDA's VehicleManager architecture

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from opencda_marl.core.agents.basic_agent import BasicAgent
from opencda.core.plan.collision_check import CollisionChecker
from opencda.core.plan.planer_debug_helper import PlanDebugHelper


class VanillaAgent(BasicAgent):
    def __init__(self, vehicle, carla_map, config_yaml):
        opt_dict = {
            'ignore_traffic_light': config_yaml['ignore_traffic_light'],
            'local_planner': config_yaml.get('local_planner', {}),
            'debug': config_yaml['debug'],
        }
        super().__init__(vehicle=vehicle, map_inst=carla_map,
                         opt_dict=opt_dict)
        self._ego_pos = None
        self._ego_speed = 0.0

        # speed related, check yaml file to see the meaning
        self.max_speed = config_yaml['max_speed']

        # safety related
        self.emergency_param = config_yaml['emergency_param']
        self.break_distance = 0
        self.ttc = 1000

        # collision checker
        self._collision_check = CollisionChecker(
            time_ahead=config_yaml['collision_time_ahead'])

        # used to indicate whether a vehicle is on the planned path
        self.hazard_flag = False

        # white list of vehicle managers that the cav does not consider as
        # obstacles
        self.white_list = []
        self.obstacle_vehicles = []
        self.objects = {}

        # route planner related
        self.start_waypoint = None
        self.end_waypoint = None

        # debug helper
        self.debug_helper = PlanDebugHelper(self._vehicle.id, self._world)

    def set_destination(self, start_location, end_location,
                        clean=False, end_reset=True, clean_history=False):
        try:
            # Store waypoints for destination checking
            self.end_waypoint = self._map.get_waypoint(end_location)
            self.start_waypoint = self._map.get_waypoint(start_location)

            if not self.end_waypoint or not self.start_waypoint:
                raise RuntimeError(
                    f"Failed to get waypoints: start_wp={self.start_waypoint}, end_wp={self.end_waypoint}")
            
            # Use BasicAgent's set_destination method directly with explicit clean_queue
            super().set_destination(end_location=end_location, start_location=start_location, clean_queue=clean)

        except Exception as e:
            print(f"VanillaAgent: Error in set_destination: {e}")
            raise e

    def white_list_match(self, obstacles):
        """
        Match the detected obstacles with the white list.
        Remove the obstacles that are in white list.
        The white list contains all position of target platoon
        member for joining.

        Parameters
        ----------
        obstacles : list
            A list of carla.Vehicle or ObstacleVehicle

        Returns
        -------
        new_obstacle_list : list
            The new list of obstacles.
        """
        new_obstacle_list = []

        for o in obstacles:
            flag = False
            o_x = o.get_location().x
            o_y = o.get_location().y

            o_waypoint = self._map.get_waypoint(o.get_location())
            o_lane_id = o_waypoint.lane_id

            for vm in self.white_list:
                pos = vm.v2x_manager.get_ego_pos()
                vm_x = pos.location.x
                vm_y = pos.location.y

                w_waypoint = self._map.get_waypoint(pos.location)
                w_lane_id = w_waypoint.lane_id

                # if the id is different, then not matched for sure
                if o_lane_id != w_lane_id:
                    continue

                if abs(vm_x - o_x) <= 3.0 and abs(vm_y - o_y) <= 3.0:
                    flag = True
                    break
            if not flag:
                new_obstacle_list.append(o)

        return new_obstacle_list

    def update_information(self, ego_pos, ego_speed, objects):
        """
        Update the perception and localization information
        to the behavior agent.
        """
        # update localization information
        self._ego_speed = ego_speed
        self._ego_pos = ego_pos
        self.break_distance = self._ego_speed / 3.6 * self.emergency_param

        # update the localization info to trajectory planner
        self.get_local_planner().update_information(ego_pos, ego_speed)

        self.objects = objects
        # current version only consider about vehicles
        obstacle_vehicles = objects['vehicles']
        self.obstacle_vehicles = self.white_list_match(obstacle_vehicles)

        # update the debug helper
        self.debug_helper.update(ego_speed, self.ttc)

        # if self.ignore_traffic_light:
        #    self.light_state = "Green"
        # else:
        #    # This method also includes stop signs and intersections.
        #    self.light_state = str(self.vehicle.get_traffic_light_state())

    def is_close_to_destination(self):
        """
        Check if the current ego vehicle's position is close to destination
        """
        # Check if required components are initialized
        if not self._ego_pos or not self.end_waypoint:
            return False

        try:
            flag = abs(self._ego_pos.location.x - self.end_waypoint.transform.location.x) <= 10 and \
                abs(self._ego_pos.location.y -
                    self.end_waypoint.transform.location.y) <= 10
            return flag
        except AttributeError as e:
            print(f"VanillaAgent: Error accessing waypoint data: {e}")
            return False

    def run_step(self, target_speed=None):
        """
        Execute one step of navigation.
        Returns OpenCDA-compatible format: (target_speed, target_location)
        """
        # Check if agent is properly initialized
        if not self._ego_pos:
            print("VanillaAgent: Warning - ego position not set, returning zero speed")
            return 0.0, None

        # 0. Simulation ends condition
        if self.is_close_to_destination():
            # Signal completion without forceful exit - let the scenario manager handle cleanup
            raise StopIteration("Destination reached - simulation complete")

        try:
            # Get control from BasicAgent's CARLA implementation
            control = super().run_step()

            # Extract target speed from control (convert from throttle/brake to speed)
            if control.throttle > 0:
                # Use configured max_speed when accelerating
                calculated_target_speed = min(
                    self.max_speed, target_speed if target_speed else self.max_speed)
            elif control.brake > 0:
                # Reduce speed when braking
                calculated_target_speed = max(
                    0, self._ego_speed - 10)  # Gradual brake
            else:
                # Maintain current speed
                calculated_target_speed = self._ego_speed

        except Exception as e:
            print(f"VanillaAgent: Error in BasicAgent.run_step(): {e}")
            # Fallback to conservative behavior
            calculated_target_speed = min(
                5.0, self._ego_speed)  # Slow down safely

        # Get target location from local planner
        local_planner = self.get_local_planner()
        if local_planner and hasattr(local_planner, 'target_waypoint') and local_planner.target_waypoint:
            target_location = local_planner.target_waypoint.transform.location
        else:
            # Fallback: use current position if no waypoint available
            target_location = self._ego_pos.location if self._ego_pos else None

        return calculated_target_speed, target_location

    # --------------------------------------------------------------------- #
    # Public getter methods for state information
    # --------------------------------------------------------------------- #
    def get_speed(self) -> float:
        """Get current ego vehicle speed in m/s."""
        return self._ego_speed
    
    def get_position(self) -> tuple:
        """Get current ego vehicle position as (x, y) coordinates."""
        if self._ego_pos and hasattr(self._ego_pos, 'location'):
            return (self._ego_pos.location.x, self._ego_pos.location.y)
        return (0.0, 0.0)
    
    def get_lane_id(self) -> int:
        """Get current lane ID from ego vehicle position."""
        if self._ego_pos and hasattr(self._ego_pos, 'location'):
            try:
                current_waypoint = self._map.get_waypoint(self._ego_pos.location)
                if current_waypoint:
                    return current_waypoint.lane_id
            except Exception as e:
                print(f"VanillaAgent: Error getting lane ID: {e}")
        return 0
