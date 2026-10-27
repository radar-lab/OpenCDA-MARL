"""PipelineBuilder centralizes construction of MARL runtime components.

Used by MARLCoordinator at first init and by WorldManager during world reload
so both paths share one source of truth.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Dict

from loguru import logger

if TYPE_CHECKING:
    # A type annotation only: the SUMO path never needs the vendored OpenCDA
    # core, and the public wheel does not ship it.
    from opencda.core.common.cav_world import CavWorld


#: Route-intent modes that never need the MARL manager: the disabled states
#: and the built-in providers that decide without a learner.
_MODES_WITHOUT_LEARNER = frozenset({
    "disabled", "off", "", "tao_fixed", "naive_random", "rule_based_path", "rule_path", "rule_based"})


def scenario_name_from_config(config: Mapping) -> Optional[str]:
    """The scenario reference (``-t <family>/<scenario>/<profile>``) that
    labels the run in the metrics store, or None when the run has none.

    The CLI options live under ``config['opt']``. Both backends hand this to
    the MARL manager as ``scenario_name``.
    """
    opt = config.get('opt') or {}
    scenario_ref = opt.get('test_scenario') if hasattr(opt, 'get') else None
    return str(scenario_ref) if scenario_ref else None


class PipelineBuilder:
    """Single source of truth for runtime component construction."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build_output_section_config(self, section_name: str) -> Dict[str, Any]:
        outputs_cfg = self.config.get('outputs', {})
        section_cfg = outputs_cfg.get(section_name, {})
        if not isinstance(section_cfg, Mapping):
            return {}
        resolved = dict(section_cfg)
        root_dir = outputs_cfg.get('root_dir')
        section_dir = resolved.pop('dir', None)
        if section_dir:
            resolved['root_dir'] = (
                os.path.join(root_dir, section_dir) if root_dir else section_dir
            )
        elif root_dir and 'root_dir' not in resolved:
            resolved['root_dir'] = root_dir
        if root_dir and 'metrics_root' not in resolved:
            resolved['metrics_root'] = os.path.join(root_dir, 'metrics')
        return resolved

    def build_scenario(self, cav_world: CavWorld):
        """Construct the MARLScenarioManager from config.

        Delegates to ``ScenarioBuilder.build_from_config`` which extracts
        ``scenario.map.town`` and ``scenario.map.xodr_path`` from the config
        and feeds them to ``MARLScenarioManager``. Constructing
        ``MARLScenarioManager`` directly with only ``scenario_params=...``
        silently drops those two map parameters and leaves CARLA on whatever
        map happened to be loaded last, so always go through
        ``ScenarioBuilder``.
        """
        from opencda_marl._kernel.scenarios import ScenarioBuilder
        return ScenarioBuilder.build_from_config(
            config=self.config,
            cav_world=cav_world,
        )

    def build_marl_manager(self, scenario) -> Any:
        from opencda_marl._kernel.policy import MARLManager
        marl_cfg = self.config.get('MARL', {})
        if hasattr(marl_cfg, 'copy'):
            marl_cfg = marl_cfg.copy()
        else:
            marl_cfg = dict(marl_cfg)
        if 'outputs' not in marl_cfg and 'outputs' in self.config:
            marl_cfg['outputs'] = self.config.get('outputs', {})
        scenario_ref = scenario_name_from_config(self.config)
        if scenario_ref and 'scenario_name' not in marl_cfg:
            marl_cfg['scenario_name'] = scenario_ref
        algorithm = marl_cfg.get('algorithm')
        manager = MARLManager(marl_cfg, algorithm)
        manager.agent_manager = scenario.agent_manager
        scenario.marl_manager = manager
        scenario.agent_manager.marl_manager = manager
        return manager

    def build_env(self, scenario, marl_manager) -> Any:
        from opencda_marl.envs import MARLEnv
        marl_cfg = self.config.get('MARL', {})
        if hasattr(marl_cfg, 'copy'):
            marl_cfg = marl_cfg.copy()
        else:
            marl_cfg = dict(marl_cfg)
        if 'outputs' not in marl_cfg and 'outputs' in self.config:
            marl_cfg['outputs'] = self.config.get('outputs', {})
        return MARLEnv(scenario, config=marl_cfg, marl_manager=marl_manager)

    def build_world_manager(self, coordinator):
        reset_cfg = self.config.get('runtime', {}).get('reset', {})
        enabled = reset_cfg.get('enabled', True) and (
            reset_cfg.get('reset_frequency', 0) > 0 or
            reset_cfg.get('auto_reset', {}).get('enabled', False)
        )
        if not enabled:
            return None
        from opencda_marl._kernel.lifecycle.world_manager import WorldManager
        wm = WorldManager(reset_cfg, coordinator)
        logger.info("WorldManager initialized for CARLA memory management")
        return wm

    def build_trace_writer(self, coordinator):
        trace_cfg = self.build_output_section_config('trace')
        if not trace_cfg.get('enabled', False):
            return None
        from opencda_marl.telemetry.trace import RouteActionTraceWriter
        writer = RouteActionTraceWriter(trace_cfg, coordinator)
        coordinator.register_post_step_callback(
            lambda: writer.record_step(coordinator)
        )
        coordinator.register_episode_callback(lambda: writer.flush())
        logger.info(f"Route-action trace enabled: {writer.root_dir}")
        return writer

    def wire_intent_provider(self, scenario, marl_manager) -> None:
        """Bind the MARL manager's cached intent_provider to the agent manager.

        Also restores the bidirectional back-reference
        ``marl_manager.agent_manager = scenario.agent_manager``.

        Without this, post-world-reset rebuilds end up with the marl_manager
        still pointing at the destroyed old agent_manager. The intent
        provider's cached closure (``_get_active_adapters_for_intent``)
        then returns ``[]`` permanently, and -- more importantly -- the
        traffic/agent-manager pair the marl_manager thinks it owns no longer
        matches what the scenario manager actually drives. Symptom: episodes
        2+ after a periodic ``WorldManager`` reset spawn 0 vehicles.

        Which provider the agent manager ends up with follows what the
        profile configured. The framework's own learned action (``mode:
        marl``) is the manager's cached bridge. A *registered* path action
        is rebuilt here through the registry, now that the manager exists to
        hand it the learner: the scenario manager built its first copy before
        the manager was created, so that copy has no learner and could only
        draw at random -- and installing the bridge over it, as this method
        once did unconditionally, fed the bridge's own observation into a
        head sized for the plugin's. The non-learned built-ins (rule /
        random / fixed) need no manager and keep the scenario manager's copy.
        """
        from opencda_marl._kernel.policy.action_space import derive_route_intent_config
        from opencda_marl.api.intent import build_provider_from_config

        # The back-reference first: a registered provider may read the
        # planner and the live adapters through the manager.
        marl_manager.agent_manager = scenario.agent_manager

        intent_cfg = derive_route_intent_config(
            dict(self.config.get("agents", {}) or {}), self.config.get("MARL", {}) or {})
        mode = str(intent_cfg.get("mode", "disabled") or "disabled").lower()
        if mode == "marl":
            provider = marl_manager.intent_provider  # cached property; built once
        elif mode in _MODES_WITHOUT_LEARNER:
            provider = None
        else:
            training = bool((self.config.get("MARL", {}) or {}).get("training", {}).get("training_mode", True))
            provider = build_provider_from_config(
                intent_cfg, getattr(scenario.agent_manager, "planner", None),
                marl_manager=marl_manager, training=training,
                previous=getattr(scenario.agent_manager, "intent_provider", None))
        if provider is not None:
            scenario.agent_manager.intent_provider = provider
