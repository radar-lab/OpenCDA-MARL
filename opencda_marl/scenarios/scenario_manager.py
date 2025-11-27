'''
Author: AXIBA leolihao@arizona.edu
Date: 2025-08-21 17:47:21
FilePath: \OpenCDA\opencda_marl\scenarios\scenario_manager.py
Description: MARL Scenario Manager that extends OpenCDA's ScenarioManager.

Provides step-based control and MARL-specific features.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from typing import Dict, Any, Optional

from loguru import logger

# OpenCDA
from opencda.scenario_testing.utils.sim_api import ScenarioManager as BaseScenarioManager
from opencda.core.common.cav_world import CavWorld

# MARL
from opencda_marl.core.traffic import MARLTrafficManager
from opencda_marl.core import MARLAgentManager
from opencda_marl.core.agents.agent_factory import AgentFactory


class MARLScenarioManager(BaseScenarioManager):
    def __init__(
        self, scenario_params: Dict, apply_ml,
        xodr_path: Optional[str] = None, town: Optional[str] = None,
        cav_world: Optional[CavWorld] = None
    ):
        """
        Initialize MARL Scenario Manager.

        Parameters
        ----------
        scenario_params : dict
            OpenCDA scenario configuration
        apply_ml : bool
            Whether to apply ML models
        states : dict
            States of the scenario
        xodr_path : str, optional
            Path to custom map
        town : str, optional
            Town name
        cav_world : CavWorld, optional
            CAV world instance
        """
        try:
            super().__init__(
                scenario_params=scenario_params,
                apply_ml=apply_ml,
                town=town,
                xodr_path=xodr_path,
                cav_world=cav_world
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize OpenCDA's scenario manager: {e}")

        # MARL specific settings
        self.params = scenario_params.get('scenario', {})
        self.states = self.default_states()

        # Store fixed delta time for traffic manager
        self.states['fixed_dt'] = self.fixed_dt

        # Simulation settings
        sim_cfg = self.params.get("simulation", {})
        if 'max_steps' in sim_cfg:
            self.states['max_steps'] = sim_cfg['max_steps']
        if 'max_episodes' in sim_cfg:
            self.states['max_episodes'] = sim_cfg['max_episodes']

        # Traffic settings
        traffic_cfg = self.params.get("traffic", {})
        self.traffic_manager = MARLTrafficManager(self.world,
                                                  traffic_cfg,
                                                  self.states,
                                                  self.fixed_dt)

        agents_cfg = scenario_params.get("agents", {})
        self.agent_manager = MARLAgentManager(agents_cfg, self.states,
                                              self.world, self.cav_world)

        # if agent is baseline agent, max episode is 1
        if self.agent_manager.agent_type in AgentFactory.get_baseline_types():
            self.states['max_episodes'] = 1
            logger.info("Baseline agent, setting max episode to 1.")

        logger.success(
            "MARLScenarioManager initialized (CARLA-only, all CAV).")

    # --------------------------------------------------------------------- #
    # Public control API
    # --------------------------------------------------------------------- #
    def get_traffic_info(self) -> Dict[str, Any]:
        return {
            'queue_count': self.agent_manager.get_queue_count(),
            'agent_type': self.agent_manager.agent_type,
            'pending_spawns': self.traffic_manager.total_events
        }

    def get_observations(self) -> Dict[str, Any]:
        return self.agent_manager.get_all_observations()
    
    # --------------------------------------------------------------------- #
    # Single step
    # --------------------------------------------------------------------- #

    def step(self, target_speed: Dict[int, float] = {}) -> Dict[str, Any]:

        current_step = self.states['step']
        events = self.traffic_manager.update(current_step)
        self.agent_manager.step(events, target_speed)
        events = self.agent_manager.get_event_logs()

        # update states
        for event in events:
            if event.event_type == "collision":
                self.states['collision'] += 1
            elif event.event_type == "success":
                self.states['success'] += 1
        self.states['step'] += 1
        self.states['active_agents'] = len(self.agents)

        self.world.tick()
        return {
            "event": events,
        }

    # --------------------------------------------------------------------- #
    # Public properties
    # --------------------------------------------------------------------- #
    @property
    def agents(self) -> Dict[Any, Any]:
        return self.agent_manager._vehicle_adapters
    # --------------------------------------------------------------------- #
    # Cleanup
    # --------------------------------------------------------------------- #

    def default_states(self):
        return {
            'max_steps': 500,
            'max_episodes': 1,
            'step': 0,
            'episode': 0,
            'collision': 0,
            'success': 0,
            'active_agents': 0
        }

    def reset_episode(self):
        """Reset only episode-related metrics."""
        self.states['step'] = 0
        self.states['episode'] += 1
        self.states['collision'] = 0
        self.states['success'] = 0
        self.states['active_agents'] = 0
        
        # Reset agent manager
        self.agent_manager.reset()
        # Reset traffic manager
        self.traffic_manager.reset()

    def reset(self):
        """Reset scenario manager for new episode.

        """
        # Preserve episode config
        max_episodes = self.states['max_episodes']
        max_steps = self.states['max_steps']
        self.states = self.default_states()
        self.states['max_episodes'] = max_episodes
        self.states['max_steps'] = max_steps

        # Reset agent manager
        self.agent_manager.reset()
        # Reset traffic manager
        self.traffic_manager.reset()

        logger.warning("MARLScenarioManager reset completed")

    def close(self):
        try:
            self.agent_manager.cleanup()
            self.traffic_manager.cleanup()
            super().close()
        except Exception as e:
            logger.warning(f"Base close error: {e}")
        logger.info("MARLScenarioManager closed.")
