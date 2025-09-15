'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 17:05:42
FilePath     : /OpenCDA-MARL/test/marl/test_intersection_build.py
Description  : 
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# set log level to INFO
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_intersection_build():
    from opencda_marl.scenarios import ScenarioBuilder
    from opencda.core.common.cav_world import CavWorld
    
    scenario_builder = ScenarioBuilder()
    cfg = scenario_builder.create_intersection_scenario(
        weather='clear',
        town='intersection'
    )
    
    # add visualization to cfg
    if 'scenario' not in cfg:
        cfg['scenario'] = {}
    if 'traffic' not in cfg['scenario']:
        cfg['scenario']['traffic'] = {}
    if 'planner' not in cfg['scenario']['traffic']:
        cfg['scenario']['traffic']['planner'] = {}
    cfg['scenario']['traffic']['planner']['visualize'] = {
        'route_line': False,
        'junction_center': True,
        'text': True,
        'life_time': 1000000
    }
    
    cav_world = CavWorld(apply_ml=cfg['opt']['apply_ml'])
    scenario_manager = scenario_builder.build_from_config(
        config=cfg,
        apply_ml=cfg['opt']['apply_ml'],
        cav_world=cav_world
    )
    return scenario_manager

if __name__ == "__main__":
    from opencda_marl.envs import CarlaSpectator
    
    scenario_manager = test_intersection_build()
    
    world = scenario_manager.world
    spectator = CarlaSpectator(world=world, config={
        'preset': 'intersection_bird_eye'
    })
    
    traffic_manager = scenario_manager.traffic_manager
    planner = traffic_manager.planner
    
    junctions = planner.get_junctions()
    for junction_id, junction_data in junctions.items():
        # show how many routes in each junction
        print(f"Junction {junction_id} has {len(junction_data['routes'])} routes")
        
        # check first route
        route = junction_data['routes'][0]
        print(route)