# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time


def run_scenario(opt, scenario_params):
    # Initialize variables to None to handle exceptions properly
    scenario_manager = None
    eval_manager = None
    single_cav_list = []
    bg_veh_list = []
    
    try:
        scenario_params = add_current_time(scenario_params)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   town='Town06',
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("single_town06_carla.log", True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='single_2lanefree_carla',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()
        # run steps
        while True:
            scenario_manager.tick()
            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location +
                carla.Location(
                    z=50),
                carla.Rotation(
                    pitch=-
                    90)))

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                try:
                    control = single_cav.run_step()
                    single_cav.vehicle.apply_control(control)
                except StopIteration as e:
                    print(f"Simulation completed: {e}")
                    break
            else:
                continue
            break

    finally:
        # Only evaluate if eval_manager was successfully created
        if eval_manager is not None:
            eval_manager.evaluate()

        # Clean up scenario manager if it was created
        if scenario_manager is not None:
            if opt.record:
                scenario_manager.client.stop_recorder()
            scenario_manager.close()

        # Clean up vehicles if they were created
        for v in single_cav_list:
            v.destroy()
        for v in bg_veh_list:
            v.destroy()

