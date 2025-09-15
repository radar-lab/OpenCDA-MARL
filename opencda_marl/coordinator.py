'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-20 10:54:59
FilePath: \OpenCDA\opencda_marl\core\coordinator.py
Description  : High-level orchestration for MARL training and simulation.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from loguru import logger
from typing import Dict, List, Any, Callable

import traceback

from opencda.core.common.cav_world import CavWorld
from opencda_marl.envs import CarlaMonitor, CarlaSpectator, MARLEnv, EvaluationManager
from opencda_marl.scenarios import ScenarioBuilder


class MARLCoordinator:
    """
    High-level coordinator for MARL experiments.

    Orchestrates interaction between:
    - Scenario management (CARLA simulation)
    - User interfaces (GUI/CLI)
    """

    def __init__(
        self,
        config: Dict
    ):
        self.config = config

        # Core components (initialized later)
        self.scenario_manager = None
        self.marl_env = None
        self.cav_world = None
        self.evaluation_metrics = None

        # CARLA components
        self.carla_client = None
        self.carla_world = None
        self.carla_monitor = None

        # Callbacks for external control (GUI, etc.)
        self.pre_step_callbacks: List[Callable] = []
        self.post_step_callbacks: List[Callable] = []
        self.episode_callbacks: List[Callable] = []

    def initialize(self):
        """
        Initialize all required components:
        1. CAV world: A customized world object to save all CDA vehicle
                      information and shared ML models.

        2. Monitor the CARLA simulation?

        """
        # 1. Create CAV world
        self.cav_world = CavWorld(apply_ml=self.config.opt.apply_ml)

        # 2. Create Scenario Manager
        self.scenario_manager = ScenarioBuilder.build_from_config(
            config=self.config,
            cav_world=self.cav_world,
        )
        self.states = self.scenario_manager.states
        self.carla_client = self.scenario_manager.client
        self.carla_world = self.scenario_manager.world

        # 3. Create Spectator for GUI
        spectator_config = self.config.get('spectator', {})
        self.carla_spectator = CarlaSpectator(
            self.carla_world, spectator_config)

        # 4. Create MARL environment
        Marl_cfg = self.config.get('MARL', {})
        self.marl_env = MARLEnv(self.scenario_manager,
                                config=Marl_cfg)

        # 5. Create CARLA monitor
        marl_tm = self.scenario_manager.traffic_manager
        self.carla_monitor = CarlaMonitor(self.carla_world,
                                          marl_tm=marl_tm)

        # 6. Create Evaluation manager
        scenario_type = self.config.get("meta", {}).get("scenario_type", None)
        agent_name = self.config.get("agents", {}).get("agent_type", None)
        eval_cfg = self.config.get('evaluation', {})
        self.evaluation_manager = EvaluationManager(config=eval_cfg,
                                                    scenario_name=scenario_type,
                                                    agent_name=agent_name)

    # --------------------------------------------------------------------- #
    # main thread
    # --------------------------------------------------------------------- #
    def get_metrics(self):
        metrics = self.states.copy()
        if self.marl_env:
            metrics.update(self.marl_env.get_episode_metrics())
        return metrics

    def step(self):
        # Call pre-step callbacks
        for callback in self.pre_step_callbacks:
            callback()

        # Use MARL environment to step
        if self.marl_env:
            self.marl_env.step()
        else:
            self.scenario_manager.step()

        # Call post-step callbacks
        for callback in self.post_step_callbacks:
            callback()

        metrics = self.get_metrics()
        # Get current step rewards if MARL environment is available
        rewards = None
        if self.marl_env:
            rewards = self.marl_env.get_current_step_rewards()
        self.evaluation_manager.update_step(metrics, rewards)

        #print(f"states: {self.states}, rewards: {rewards}, metrics: {metrics}")
        
    def run(self):
        """
        Run the MARL scenario.
        """
        max_steps = self.states['max_steps']
        for episode in range(self.states['max_episodes']):
            logger.info(
                f"Starting episode {episode + 1}/{self.states['max_episodes']}")

            for _ in range(max_steps):
                self.step()
                # clean the event history
                

            # Reset for next episode
            self.reset_episode()

    def reset_episode(self):
        """
        Reset environment and scenario for new episode.
        """
        try:
            # Reset MARL environment first
            if self.marl_env:
                episode_metrics = self.marl_env.reset_episode()
                logger.info(f"Episode metrics: {episode_metrics}")

            # Reset scenario manager
            self.scenario_manager.reset_episode()

            # Call episode callbacks
            for callback in self.episode_callbacks:
                callback()

        except Exception as e:
            logger.error(f"Error during episode reset: {e}")
            import traceback
            traceback.print_exc()

        import gc
        gc.collect()

    def run_gui_mode(self):
        """
        Run in GUI mode with step-by-step control.
        Uses the Dashboard class for clean separation of GUI logic.
        """
        try:
            # Try to import GUI dependencies
            import sys
            from PySide6.QtWidgets import QApplication
            logger.info("GUI dependencies found, initializing GUI mode...")

            # Create QApplication FIRST - required before any Qt widgets
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            logger.info("QApplication created/obtained successfully")

            # Now import and create dashboard (which inherits from QMainWindow)
            from opencda_marl.gui.dashboard import Dashboard
            dashboard = Dashboard(self, app)
            return dashboard.run()

        except Exception as e:
            logger.error(f"Error initializing GUI: {e}")
            logger.error(traceback.format_exc())
            return 1

    # --------------------------------------------------------------------- #
    # hooks
    # --------------------------------------------------------------------- #
    def register_pre_step_callback(self, callback: Callable):
        """Register pre-step callback."""
        self.pre_step_callbacks.append(callback)

    def register_post_step_callback(self, callback: Callable):
        """Register post-step callback."""
        self.post_step_callbacks.append(callback)

    def register_episode_callback(self, callback: Callable):
        """Register episode callback."""
        self.episode_callbacks.append(callback)

    # --------------------------------------------------------------------- #
    # CARLA monitoring
    # --------------------------------------------------------------------- #
    def get_monitor_data(self) -> Dict[str, Any]:
        """
        Get current CARLA monitor data for GUI display.

        Returns
        -------
        dict
            Monitor data or empty dict if monitor not available
        """
        if not self.carla_monitor:
            raise RuntimeError("CARLA monitor not initialized")
        data = self.carla_monitor.get_monitor_data()

        scenario_data = self.scenario_manager.get_traffic_info()
        data['traffic']['queue_count'] = scenario_data['queue_count']

        data['system']['max_steps'] = self.states['max_steps']
        data['system']['max_episodes'] = self.states['max_episodes']
        data['system']['agent_type'] = scenario_data['agent_type']

        # The spawned vehicles need to minus the failure events
        data['traffic']['spawned_vehicles'] -= scenario_data['queue_count']

        data['reward'] = self.marl_env.get_reward_params()

        # Add training information including DQN-specific metrics
        training_info = self.marl_env.get_training_info()
        data['reward'].update({
            'episode_count': training_info.get('episode_count', 0),
            'algorithm': training_info.get('algorithm_type', 'unknown'),
            'epsilon': training_info.get('epsilon', 'N/A'),
            'training_mode': training_info.get('training_mode', True),
            # Loss metrics (for TD3, DQN, etc.)
            'actor_loss': training_info.get('actor_loss', 0.0),
            'critic_loss': training_info.get('critic_loss', 0.0),
            # DQN-specific metrics
            'memory_size': training_info.get('memory_size', 0),
            'memory_capacity': training_info.get('memory_capacity', 0),
            'target_updates': training_info.get('target_updates', 0),
            'training_step': training_info.get('training_step', 0),
            'device': training_info.get('device', 'N/A')
        })
        return data

    def get_observations(self) -> Dict[str, Any]:
        """
        Get agent observations for GUI display.
        """
        obs = {}
        if self.marl_env:
            obs.update(self.marl_env.get_observations())
        else:
            obs.update(self.scenario_manager.get_observations())

        # print(obs)
        return obs
    # --------------------------------------------------------------------- #
    # cleanup
    # --------------------------------------------------------------------- #

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'scenario_manager') and self.scenario_manager:
            self.scenario_manager.close()
            logger.info("Scenario manager cleaned up")

        if self.carla_monitor:
            self.carla_monitor.restore_original_settings()
            logger.info("CARLA settings restored to original")
        logger.info("Coordinator cleanup completed")
