'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-03 10:54:59
FilePath: /OpenCDA/opencda.py
Description  : Script to run different scenarios.
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

# On Windows `import torch` fails with WinError 127 (shm.dll) once cv2, open3d
# or matplotlib.pyplot are loaded, so torch goes first when it is installed.
try:
    import torch  # noqa: F401
except ImportError:
    pass

from omegaconf import OmegaConf

os.environ.setdefault("LOGURU_LEVEL", "INFO")
# The Qt behind OpenCV's windows logs "qt.qpa.screen: Unable to open monitor
# interface" for each display it cannot query on multi-monitor setups; other
# Qt categories and an explicit user setting stay untouched.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.screen=false")

from loguru import logger

from opencda import __version__
from opencda_marl.utils import resolve_marl_config_paths


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # add arguments to the parser
    parser.add_argument('-t', "--test_scenario", required=True, type=str,
                        help='Define the name of the scenario you want to test. The given name must '
                             'match one of the testing scripts (e.g. single_2lanefree_carla) in '
                             'opencda/scenario_testing/ folder'
                             ' as well as the corresponding yaml file in the bundled opencda family, opencda_marl/_bundled/configs/opencda/ (standard mode)'
                             ' or one of the MARL config families under configs/ or an added'
                             ' custom config root (when using --marl flag).'
                             ' Use refs like pcma/intersection/smoke,'
                             ' momarl/intersection/learn_path_pcmappo,'
                             ' or sumo-marl/intersection/sac.')
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
    parser.add_argument("--marl-config-root", action='append', default=None,
                        help='Additional MARL config root containing family directories '
                             'such as marl/, pcma/, momarl/, or sumo-marl/. Can be repeated.')
    parser.add_argument("-d", "--debug", action='store_true',
                        help='Enable debug mode. This will print more detailed information.')

    # parse the arguments and return the result
    opt = parser.parse_args()
    return opt


def load_configuration(opt):
    """Load and merge configuration based on mode."""
    base_dir = os.path.dirname(os.path.realpath(__file__))

    if opt.marl:
        resolved = resolve_marl_config_paths(
            base_dir,
            opt.test_scenario,
            custom_config_roots=getattr(opt, "marl_config_root", None),
        )
        default_yaml = str(resolved['default_yaml'])
        scenario_base_yaml = str(resolved['scenario_base_yaml'])
        config_yaml = str(resolved['scenario_yaml'])
    else:
        # Standard OpenCDA mode: the classic scenario family ships inside the
        # package; a checkout-level configs/opencda/ overrides it when present.
        config_dir = os.path.join(base_dir, 'configs', 'opencda')
        if not os.path.isdir(config_dir):
            from opencda_marl._bundled import BUNDLED_CONFIG_ROOT
            config_dir = str(BUNDLED_CONFIG_ROOT / 'opencda')
        default_yaml = os.path.join(config_dir, 'default.yaml')
        config_yaml = os.path.join(config_dir, f'{opt.test_scenario}.yaml')

    # check if the yaml file for the specific testing scenario exists
    if not os.path.isfile(config_yaml):
        sys.exit(f"{config_yaml} not found!")

    # load the default yaml file and the scenario yaml file as dictionaries
    from opencda_marl.utils.config_locator import load_profile_with_includes

    default_dict = OmegaConf.load(default_yaml)
    scene_dict = load_profile_with_includes(config_yaml)

    # merge the dictionaries - use smart merge for MARL to extend lists
    if opt.marl:
        from opencda_marl.utils import smart_merge
        scenario_base_dict = OmegaConf.load(scenario_base_yaml)
        merged_config = smart_merge(default_dict, scenario_base_dict)
        merged_config = smart_merge(merged_config, scene_dict)
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
        try:
            import torch  # Preload torch to stabilize the local Windows DLL load order.
            logger.debug(f"torch_preload: {torch.__version__}")
        except OSError as exc:
            raise RuntimeError(
                "Failed to import the MARL runtime because the local Torch installation "
                "could not load correctly. The current Windows experimental workaround "
                "is KMP_DUPLICATE_LIB_OK=TRUE; if this still fails then the pixi Torch "
                f"runtime itself is inconsistent. Low-level error: {exc}"
            ) from exc

        # Import MARL components
        from opencda_marl import MARLCoordinator

        logger.info(f"Scenario: {opt.test_scenario}")

        # append all arguments from opt to config
        opt_dict = vars(opt)
        config['opt'] = opt_dict

        # Create and initialize coordinator
        coordinator = MARLCoordinator(
            config=config,
        )

        # Initialize components
        coordinator.initialize()

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
