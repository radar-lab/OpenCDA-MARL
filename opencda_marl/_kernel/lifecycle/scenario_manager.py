'''
Author: AXIBA leolihao@arizona.edu
Date: 2025-08-21 17:47:21
FilePath: opencda_marl/core/lifecycle/scenario_manager.py
Description: MARL Scenario Manager that extends OpenCDA's ScenarioManager.

Provides step-based control and MARL-specific features.

Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from typing import Dict, Any, Optional
import copy

from loguru import logger

# OpenCDA
from opencda.scenario_testing.utils.sim_api import (
    ScenarioManager as BaseScenarioManager, is_client_timeout)
from opencda.core.common.cav_world import CavWorld

# MARL
from opencda_marl._kernel.traffic.traffic_manager import MARLTrafficManager
from opencda_marl._kernel.agent_manager import MARLAgentManager
from opencda_marl.api.components import BASELINE_AGENT_TYPES
from opencda_marl.api.intent import build_provider_from_config
from opencda_marl._kernel.policy.action_space import derive_route_intent_config


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
            # The world build already ran under a raised client time-out
            # (sim_api.build_world_with_retry); say what to change.
            hint = (
                "; the CARLA server did not answer in time: check that it is "
                "running, or raise world.timeout on a loaded machine"
                if is_client_timeout(e) else "")
            raise RuntimeError(
                "Failed to initialize OpenCDA's scenario manager "
                f"(town={town!r}, xodr_path={xodr_path!r}): {e}{hint}"
            ) from e

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
        agents_cfg = copy.deepcopy(scenario_params.get("agents", {}))
        agents_cfg["world"] = scenario_params.get("world", {})
        agents_cfg["scenario"] = scenario_params.get("scenario", {})
        intent_cfg = derive_route_intent_config(
            agents_cfg, scenario_params.get("MARL", {})
        )
        agents_cfg["route_intent"] = intent_cfg
        self.traffic_manager = MARLTrafficManager(self.world,
                                                  traffic_cfg,
                                                  self.states,
                                                  self.fixed_dt,
                                                  intent_cfg=intent_cfg,
                                                  population_cfg=agents_cfg.get("population"),
                                                  outputs_cfg=scenario_params.get("outputs"))
        # Eval-mode awareness: training=false means the intent provider must
        # use greedy argmax instead of stochastic sampling, otherwise an
        # evaluation run picks up intent-policy noise that would otherwise
        # be deterministic at deployment.
        _training_mode_for_intent = scenario_params.get('MARL', {}).get(
            'training', {}).get('training_mode', True)
        intent_provider = build_provider_from_config(
            intent_cfg, self.traffic_manager.planner,
            marl_manager=getattr(self, "marl_manager", None),
            training=_training_mode_for_intent,
        )
        self.agent_manager = MARLAgentManager(agents_cfg, self.states,
                                              self.world, self.cav_world,
                                              client=self.client,
                                              planner=self.traffic_manager.planner,
                                              intent_provider=intent_provider,
                                              marl_manager=getattr(self, "marl_manager", None))

        # If marl_manager is attached later (e.g., training loop sets
        # scenario.marl_manager = manager), the intent provider may need
        # to be re-bound once the manager exists.
        if (getattr(self, "marl_manager", None) is not None
                and intent_provider is None
                and intent_cfg
                and intent_cfg.get("mode") == "marl"):
            self.agent_manager.intent_provider = self.marl_manager.get_intent_provider(
                training=_training_mode_for_intent
            )
            self.agent_manager.marl_manager = self.marl_manager

        # A baseline agent runs one episode unless the config asks for more:
        # scenario.simulation.max_episodes is honoured as on SUMO.
        if (self.agent_manager.agent_type in BASELINE_AGENT_TYPES
                and 'max_episodes' not in sim_cfg):
            self.states['max_episodes'] = 1
            logger.info("Baseline agent without simulation.max_episodes: "
                        "running 1 episode.")

        # In evaluation mode, default to one episode only when the config does
        # not request a specific evaluation horizon. Batch eval launchers patch
        # scenario.simulation.max_episodes explicitly, so honoring it keeps the
        # run receipt ("N eval episodes") aligned with the data on disk.
        marl_cfg = scenario_params.get('MARL', {})
        training_mode = marl_cfg.get('training', {}).get('training_mode', True)
        if not training_mode and self.agent_manager.agent_type == 'marl':
            self.states['max_episodes'] = max(1, int(self.states.get('max_episodes', 1)))
            logger.info(
                "Evaluation mode: using max episode "
                f"{self.states['max_episodes']} (training_mode: false)"
            )

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
        self.world.tick()
        events = self.agent_manager.get_event_logs()

        # update states
        for event in events:
            if event.event_type == "collision":
                self.states['collision'] += 1
            elif event.event_type == "planner_failure":
                self.states['planner_failure'] += 1
            elif event.event_type == "success":
                self.states['success'] += 1
        self.states['step'] += 1
        self.states['active_agents'] = len(self.agents)
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

    def rebuild_after_world_reload(self, builder) -> None:
        """Rebuild traffic + agent managers after a CARLA world reload."""
        from opencda_marl._kernel.traffic.traffic_manager import MARLTrafficManager
        from opencda_marl._kernel.agent_manager import MARLAgentManager
        from opencda_marl._kernel.policy.action_space import derive_route_intent_config

        traffic_cfg = self.params.get("traffic", {})
        agents_cfg = copy.deepcopy(builder.config.get("agents", {}))
        agents_cfg["world"] = builder.config.get("world", {})
        agents_cfg["scenario"] = builder.config.get("scenario", {})
        intent_cfg = derive_route_intent_config(
            agents_cfg, builder.config.get("MARL", {})
        )
        agents_cfg["route_intent"] = intent_cfg

        previous_provider = getattr(
            getattr(self, "agent_manager", None), "intent_provider", None
        )
        self.traffic_manager = MARLTrafficManager(
            self.world, traffic_cfg, self.states, self.fixed_dt,
            intent_cfg=intent_cfg,
            # Carried across the reload like the intent provider below: a
            # rebuild without them turned a mixed-autonomy run homogeneous
            # (and its recording destination back to the default root).
            population_cfg=agents_cfg.get("population"),
            outputs_cfg=builder.config.get("outputs"),
        )
        # The built-in providers (tao_fixed / naive_random / rule_based_path)
        # close over the planner, which was just rebuilt with the world, so
        # they are rebuilt here exactly as at first construction. Leaving
        # this None (as before) made every rule/random path cell silently
        # keep-current after the first periodic reload; only the learned
        # (marl) provider is owned by the MARL manager and re-wired by
        # builder.wire_intent_provider afterwards.
        training_mode = bool(
            builder.config.get("MARL", {}).get("training", {}).get("training_mode", True)
        )
        intent_provider = build_provider_from_config(
            intent_cfg, self.traffic_manager.planner,
            marl_manager=getattr(self, "marl_manager", None),
            training=training_mode,
            previous=previous_provider,
        )
        self.agent_manager = MARLAgentManager(
            agents_cfg, self.states, self.world, self.cav_world,
            client=getattr(self, "client", None),
            planner=self.traffic_manager.planner,
            intent_provider=intent_provider,
            marl_manager=getattr(self, "marl_manager", None),
        )

    def default_states(self):
        return {
            'max_steps': 500,
            'max_episodes': 1,
            'step': 0,
            'episode': 0,
            'collision': 0,
            'planner_failure': 0,
            'success': 0,
            'active_agents': 0
        }

    def reset_episode(self):
        """Reset only episode-related metrics."""
        self.states['step'] = 0
        self.states['episode'] += 1
        self.states['collision'] = 0
        self.states['planner_failure'] = 0
        self.states['success'] = 0
        self.states['active_agents'] = 0

        # Reset agent manager
        self.agent_manager.reset()
        # Reset traffic manager
        self.traffic_manager.reset()

        # Tick world to ensure CARLA processes cleanup and releases GPU memory
        try:
            self.world.tick()
        except Exception:
            pass

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

        # Tick world to ensure CARLA processes cleanup and releases GPU memory
        try:
            self.world.tick()
        except Exception:
            pass

        logger.warning("MARLScenarioManager reset completed")

    def close(self):
        # Each stage is isolated: a cleanup failure must never skip the
        # world-settings restoration in super().close(), which returns the
        # CARLA server to asynchronous mode.
        try:
            self.agent_manager.cleanup()
        except Exception as e:
            logger.warning(f"Agent cleanup during close failed: {e}")
        try:
            self.traffic_manager.cleanup()
        except Exception as e:
            logger.warning(f"Traffic cleanup during close failed: {e}")
        try:
            super().close()
        except Exception as e:
            logger.warning(f"Base close error: {e}")
        logger.info("MARLScenarioManager closed.")
