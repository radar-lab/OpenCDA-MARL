# -*- coding: utf-8 -*-
"""
Scenario testing: merging vehicle joining a platoon in the
customized 2-lane freeway simplified map sorely with carla
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import carla

import opencda.scenario_testing.utils.sim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import \
    add_current_time


def run_scenario(opt, scenario_params):
    try:
        # first define the path of the yaml file and 2lanefreemap file
        scenario_params = add_current_time(scenario_params)
        current_path = os.path.dirname(os.path.realpath(__file__))
        xodr_path = os.path.join(
            current_path,
            '../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr')

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   xodr_path=xodr_path)
        if opt.record:
            scenario_manager.client. \
                start_recorder("platoon_joining_2lanefree_carla.log", True)

        # create platoon members
        platoon_list = \
            scenario_manager.create_platoon_manager(
                map_helper=map_api.spawn_helper_2lanefree,
                data_dump=False)

        # create single cavs
        single_cav_list = \
            scenario_manager.create_vehicle_manager(['platooning'],
                                                    map_api.
                                                    spawn_helper_2lanefree)

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='platoon_joining_2lanefree_carla',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()
        spectator_vehicle = platoon_list[0].vehicle_manager_list[1].vehicle

        # Keep track of all vehicles for cleanup
        all_platoons = list(platoon_list)
        all_single_cavs = list(single_cav_list)

        # run steps
        while True:
            scenario_manager.tick()
            transform = spectator_vehicle.get_transform()
            spectator.set_transform(
                carla.Transform(
                    transform.location +
                    carla.Location(
                        z=80),
                    carla.Rotation(
                        pitch=-
                        90)))
            # Process platoons in reverse to safely remove completed ones
            for i in range(len(platoon_list) - 1, -1, -1):
                platoon = platoon_list[i]
                try:
                    platoon.update_information()
                    platoon.run_step()
                except StopIteration:
                    # Platoon reached destination, remove from list
                    print(f"Platoon {i} reached destination, removing from simulation")
                    platoon_list.pop(i)

            # Process single CAVs in reverse to safely remove completed ones
            for i in range(len(single_cav_list) - 1, -1, -1):
                single_cav = single_cav_list[i]
                # this function should be added in wrapper
                if single_cav.v2x_manager.in_platoon():
                    single_cav_list.pop(i)
                else:
                    single_cav.update_info()
                    try:
                        control = single_cav.run_step()
                        single_cav.vehicle.apply_control(control)
                    except StopIteration:
                        # Vehicle reached destination, remove from list and continue
                        print(f"Vehicle {i} reached destination, removing from simulation")
                        single_cav_list.pop(i)
            
            # Check if all single vehicles and platoons are done
            if not single_cav_list and not platoon_list:
                print("All vehicles completed their routes. Ending simulation.")
                break

    finally:
        eval_manager.evaluate()

        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.close()

        # Clean up all vehicles, including completed ones
        for platoon in all_platoons:
            try:
                platoon.destroy()
            except Exception as e:
                print(f"Warning: Failed to destroy platoon: {e}")
                
        for cav in all_single_cavs:
            try:
                cav.destroy()
            except Exception as e:
                print(f"Warning: Failed to destroy single CAV: {e}")
        for v in bg_veh_list:
            v.destroy()
