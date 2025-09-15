'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-03 10:54:59
FilePath: \OpenCDA\opencda.py
Description  : Script to run different scenarios.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
# -*- coding: utf-8 -*-

# Author: Lihao Guo <leolihao@arizona.edu>
#         Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import importlib
import os
import sys
import traceback
from omegaconf import OmegaConf
from loguru import logger

from opencda import __version__


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # add arguments to the parser
    parser.add_argument('-t', "--test_scenario", required=True, type=str,
                        help='Define the name of the scenario you want to test. The given name must'
                             'match one of the testing scripts(e.g. single_2lanefree_carla) in '
                             'opencda/scenario_testing/ folder'
                             ' as well as the corresponding yaml file in configs/opencda/ (standard mode)'
                             ' or configs/marl/ (when using --marl flag).')
    parser.add_argument("--record", action='store_true',
                        help='whether to record and save the simulation process to .log file')
    parser.add_argument("--apply_ml",
                        action='store_true',
                        help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
                             'Set it to true only when you have installed the pytorch/sklearn package.')

    # MARL-specific arguments
    parser.add_argument("--marl", action='store_true',
                        help='Enable MARL (Multi-Agent Reinforcement Learning) mode. '
                             'Uses MARLCoordinator instead of standard OpenCDA execution.')
    parser.add_argument("--gui", action='store_true',
                        help='Launch GUI debug mode for step-by-step control. '
                             'Only valid with --marl flag.')
    parser.add_argument("-d", "--debug", action='store_true',
                        help='Enable debug mode. This will print more detailed information.')

    # parse the arguments and return the result
    opt = parser.parse_args()
    return opt


def load_configuration(opt):
    """Load and merge configuration based on mode."""
    base_dir = os.path.dirname(os.path.realpath(__file__))

    if opt.marl:
        # MARL mode: use configs/marl/ directory
        config_dir = 'configs/marl'
        default_yaml = os.path.join(base_dir, f'{config_dir}/default.yaml')
        config_yaml = os.path.join(
            base_dir, f'{config_dir}/{opt.test_scenario}.yaml')
        error_msg = f"{config_dir}/{opt.test_scenario}.yaml not found!"
    else:
        # Standard OpenCDA mode: use configs/opencda/ directory
        config_dir = 'configs/opencda'
        default_yaml = os.path.join(base_dir, f'{config_dir}/default.yaml')
        config_yaml = os.path.join(
            base_dir, f'{config_dir}/{opt.test_scenario}.yaml')
        error_msg = f"{config_dir}/{opt.test_scenario}.yaml not found!"

    # check if the yaml file for the specific testing scenario exists
    if not os.path.isfile(config_yaml):
        sys.exit(error_msg)

    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = OmegaConf.load(default_yaml)
    scene_dict = OmegaConf.load(config_yaml)

    # merge the dictionaries - use smart merge for MARL to extend lists
    if opt.marl:
        from opencda_marl.utils import smart_merge
        merged_config = smart_merge(default_dict, scene_dict)
    else:
        # Standard OpenCDA - keep original behavior
        merged_config = OmegaConf.merge(default_dict, scene_dict)

    return merged_config


def run_standard_opencda(opt, config):
    """Run standard OpenCDA scenario."""
    print("Running standard OpenCDA scenario...")

    # import the testing script
    testing_scenario = importlib.import_module(
        "opencda.scenario_testing.%s" % opt.test_scenario)

    # get the function for running the scenario from the testing script
    scenario_runner = getattr(testing_scenario, 'run_scenario')
    # run the scenario testing
    scenario_runner(opt, config)


def run_marl_scenario(opt, config):
    """Run MARL-enhanced scenario."""
    # remove the logger
    logger.remove()

    if opt.debug:
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.add(sys.stdout, level="INFO")

    logger.info("Running MARL-enhanced scenario...")
    coordinator = None
    try:
        # Import MARL components
        from opencda_marl import MARLCoordinator

        logger.info(f"Scenario: {opt.test_scenario}")

        # Determine execution mode
        if opt.gui:
            logger.info("Enabling GUI mode...")

        # append all arguments from opt to config
        opt_dict = vars(opt)
        config['opt'] = opt_dict

        # Create and initialize coordinator
        coordinator = MARLCoordinator(
            config=config,
        )

        # Initialize components
        coordinator.initialize()

        # Execute based on mode
        if opt.gui:
            coordinator.run_gui_mode()
        else:
            coordinator.run()

    except ImportError as e:
        print(f"Error: MARL components not available: {e}")
        print("Please ensure opencda_marl package is properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error in MARL execution: {e}")
        print(traceback.format_exc())
        sys.exit(1)
    finally:
        # <OpenCDA-MARL> Always clean up coordinator
        if coordinator and hasattr(coordinator, 'close'):
            coordinator.close()


def main():
    # parse the arguments
    opt = arg_parse()

    # print the version of OpenCDA
    print("OpenCDA Version: %s" % __version__)

    # load configuration
    config = load_configuration(opt)

    # determine execution path
    if opt.marl:
        # MARL-enhanced execution
        run_marl_scenario(opt, config)
    else:
        # Standard OpenCDA execution
        run_standard_opencda(opt, config)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
