'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-20 10:54:59
FilePath: /OpenCDA-MARL/opencda_marl/core/lifecycle/coordinator.py
Description  : High-level orchestration for MARL training and simulation.
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import os
from collections.abc import Mapping
from loguru import logger
from typing import Dict, List, Any, Callable

import traceback

from opencda_marl.utils.config_merger import normalize_run_config


def require_opencda_core() -> None:
    """Raise a pointed ImportError when the vendored OpenCDA core is absent.

    The public wheel ships without it (UCLA's licence does not allow the core to
    travel in a package), so a CARLA or co-simulation run needs a repository
    checkout, which brings the core along.
    """
    try:
        import opencda  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The CARLA backend needs the OpenCDA core, which the public opencda-marl wheel "
            "does not ship. Run from a repository checkout "
            "(git clone https://github.com/radar-lab/OpenCDA-MARL, then "
            "`python opencda.py -t <scenario> --marl`), or install the full build with "
            "`pip install git+https://github.com/radar-lab/OpenCDA-MARL.git`."
        ) from exc

# Backend imports are lazy: the SUMO path must run from a pip-installed
# wheel with neither the carla client nor the vendored `opencda` core (nor
# matplotlib) present; each branch imports what it actually drives.


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
        # The run's config is finalised here for every coordinator-driven
        # entry: every component below reads the top-level sections as
        # mappings, and a section left empty in YAML (None) would crash the
        # first `.get` on it.
        self.config = normalize_run_config(config)

        # Core components (initialized later)
        self.scenario_manager = None
        self.marl_env = None
        self.cav_world = None
        self.evaluation_metrics = None

        # CARLA components
        self.carla_client = None
        self.carla_world = None
        self.carla_monitor = None
        self.world_manager = None
        self.trace_writer = None

        # Callbacks for external control (GUI, etc.)
        self.pre_step_callbacks: List[Callable] = []
        self.post_step_callbacks: List[Callable] = []
        self.episode_callbacks: List[Callable] = []

    def _build_output_section_config(self, section_name: str) -> Dict[str, Any]:
        outputs_cfg = self.config.get('outputs', {})
        section_cfg = outputs_cfg.get(section_name, {})
        if not isinstance(section_cfg, Mapping):
            return {}

        resolved = dict(section_cfg)
        root_dir = outputs_cfg.get('root_dir')
        section_dir = resolved.pop('dir', None)
        if section_dir:
            resolved['root_dir'] = os.path.join(root_dir, section_dir) if root_dir else section_dir
        elif root_dir and 'root_dir' not in resolved:
            resolved['root_dir'] = root_dir
        if root_dir and 'metrics_root' not in resolved:
            resolved['metrics_root'] = os.path.join(root_dir, 'metrics')
        return resolved

    # --------------------------------------------------------------------- #
    # Schematic traffic view (telemetry/traffic_view.py)
    # --------------------------------------------------------------------- #

    def _service_section_config(self, section_name: str) -> Dict[str, Any]:
        """``outputs.<section>`` as a dict; a bare bool is ``{enabled: <bool>}``.

        ``monitor: false`` used to read as an empty dict (``False or {}``) and
        so start the service, and ``monitor: true`` crashed initialize() on
        ``True.get``. The sibling sections accept the shorthand too.
        """
        section = (self.config.get('outputs', {}) or {}).get(section_name, {})
        if isinstance(section, bool):
            return {'enabled': section}
        return dict(section) if isinstance(section, Mapping) else {}

    def _resolve_view_run_dir(self):
        """The run's metrics-store directory: learner store, else the
        external-policy store, else the evaluation mirror."""
        manager = getattr(self, 'marl_manager', None) or \
            getattr(self.marl_env, 'marl_manager', None)
        algorithm = getattr(manager, 'algorithm', None) if manager is not None else None
        for store in (getattr(algorithm, 'metrics_store', None),
                      getattr(manager, '_external_store', None) if manager is not None else None):
            path = getattr(store, 'path', None)
            if path is not None:
                return path.parent
        evaluation = getattr(self, 'evaluation_manager', None)
        if evaluation is not None and evaluation.is_enabled():
            return evaluation.store_dir
        return None

    def _attach_evaluation_to_run_store(self):
        """Hand the run's metrics store, the registered learner key and the
        run's training mode to the evaluation manager, so a learner run is
        one monitor entry and a separate evaluation run (no store of its
        own) is labelled with the learner and the right mode."""
        evaluation = getattr(self, 'evaluation_manager', None)
        if evaluation is None or not hasattr(evaluation, 'attach_run_store'):
            return
        manager = getattr(self, 'marl_manager', None) or \
            getattr(getattr(self, 'marl_env', None), 'marl_manager', None)
        store = getattr(manager, 'run_store', None) if manager is not None else None
        training = ((self.config.get('MARL', {}) or {}).get('training', {}) or {}).get('training_mode', True)
        evaluation.attach_run_store(store, algorithm_name=getattr(manager, 'algorithm_name', None),
                                    training=bool(training))

    def _setup_monitor_service(self):
        """Starting a simulation starts (or reuses) the web monitor.

        ``outputs.monitor`` controls it: {enabled: true, host, port,
        allow_hosts}, or a bare bool. One monitor per artifacts root covers every run under
        that root (another root gets its own, on the next free port); the
        run just prints where to look.
        """
        monitor_cfg = self._service_section_config('monitor')
        if not monitor_cfg.get('enabled', True):
            return
        root = self.config.get('outputs', {}).get('root_dir', 'artifacts')
        try:
            from opencda_marl.telemetry.monitor import ensure_service

            url, started = ensure_service(
                root,
                host=str(monitor_cfg.get('host', '127.0.0.1')),
                port=int(monitor_cfg.get('port', 8765)),
                allow_hosts=monitor_cfg.get('allow_hosts') or (),
            )
            if url is None:
                logger.warning(f"monitor not started: every port from "
                               f"{monitor_cfg.get('port', 8765)} upward serves another root")
            else:
                logger.info(f"Monitor {'started' if started else 'already running'}: "
                            f"{url}  (metrics + traffic view for runs under {root})")
        except Exception as exc:  # noqa: BLE001 - the monitor must never kill a run
            logger.warning(f"monitor service unavailable: {exc}")

    def _setup_traffic_view(self):
        """Publish schematic vehicle poses for the monitor's traffic panel.

        The publisher is created lazily on the first due tick, because the
        run's metrics store (whose directory the view files share) may not
        exist until the algorithm or the evaluation manager first writes.
        ``outputs.traffic_view`` is {enabled, interval} or a bare bool.
        """
        view_cfg = self._service_section_config('traffic_view')
        if not view_cfg.get('enabled', True):
            return
        if self.marl_env is None or not hasattr(self.marl_env, 'collect_traffic_view'):
            return
        interval = float(view_cfg.get('interval', 0.2))
        self.traffic_view = None

        def _publish():
            publisher = self.traffic_view
            if publisher is None:
                run_dir = self._resolve_view_run_dir()
                if run_dir is None:
                    return
                from opencda_marl.telemetry.traffic_view import TrafficViewPublisher
                publisher = TrafficViewPublisher(run_dir, interval=interval)
                self.traffic_view = publisher
                logger.info(f"Traffic view enabled -> {run_dir}")
            if not publisher.due():
                return
            # Geometry before the first state, so a page that finds a state
            # also finds the lanes to draw it on (retried on every due tick
            # until a collection succeeds; a failure never holds back the
            # vehicle poses).
            if not publisher.geometry_written:
                try:
                    publisher.publish_geometry(self.marl_env.collect_traffic_geometry())
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"traffic view geometry skipped: {exc}")
            try:
                vehicles, episode, step = self.marl_env.collect_traffic_view()
                publisher.publish_state(vehicles, episode=episode, step=step)
            except Exception as exc:  # noqa: BLE001 - the view must never kill a run
                logger.debug(f"traffic view publish skipped: {exc}")

        self.register_post_step_callback(_publish)

    def initialize(self):
        """
        Initialize all required components:
        1. CAV world: A customized world object to save all CDA vehicle
                      information and shared ML models.

        2. Monitor the CARLA simulation?

        """
        # Check if using SUMO-only mode
        runtime_cfg = self.config.get('runtime', {})
        simulator = runtime_cfg.get('backend', 'carla')

        # MARL.plugins first, before either backend resolves anything. The
        # CARLA pipeline resolves the planner, the intent provider and the
        # agent types while it builds the scenario, before the MARL manager
        # (whose own load used to be the only one) exists, and a run without
        # a learner never loaded them at all: a component registered only
        # through MARL.plugins was an "unknown component" there.
        from opencda_marl.plugins.loader import load_plugins
        load_plugins((self.config.get('MARL', {}) or {}).get('plugins'))

        # Seed every RNG (random / numpy / torch) from world.seed BEFORE any
        # network is built, on both backends. Previously only numpy/random
        # were seeded, and only on the CARLA path, so the learner itself
        # was never reproducible from the recorded seed.
        from opencda_marl.utils.seeding import seed_everything
        seed_everything(
            self.config.get('world', {}).get('seed'),
            deterministic_torch=bool(runtime_cfg.get('deterministic_torch', False)),
        )

        if simulator == 'sumo':
            # SUMO-only mode (no CARLA, no CAV world needed)
            logger.info("Initializing SUMO-only MARL environment")

            # Create SUMO MARL environment directly
            from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv

            self.marl_env = SumoMARLEnv(config=self.config)

            # Set placeholders for CARLA components (not used in SUMO mode)
            self.cav_world = None
            self.scenario_manager = None
            self.carla_client = None
            self.carla_world = None
            self.carla_spectator = None
            self.carla_monitor = None

            # Populate states from config for SUMO mode
            simulation_cfg = self.config.get('scenario', {}).get('simulation', {})
            self.states = {
                'max_steps': simulation_cfg.get('max_steps', 2400),
                'max_episodes': simulation_cfg.get('max_episodes', 1000)
            }

            # Evaluation mode honours the configured horizon, exactly as the
            # CARLA backend does (scenario_manager.build_scenario). Forcing 1
            # here meant a SUMO eval config asking for N episodes silently ran
            # one, and its receipt disagreed with the data on disk.
            marl_cfg = self.config.get('MARL', {})
            training_mode = marl_cfg.get('training', {}).get('training_mode', True)
            if not training_mode:
                self.states['max_episodes'] = max(
                    1, int(self.states.get('max_episodes', 1)))
                logger.info(
                    "Evaluation mode: using max episode "
                    f"{self.states['max_episodes']} (training_mode: false)")

            # Create Evaluation manager
            scenario_type = self.config.get("scenario", {}).get("type", None)
            agent_name = self.config.get("agents", {}).get("agent_type", None)
            eval_cfg = self._build_output_section_config('evaluation')
            from opencda_marl.evaluation.manager import EvaluationManager

            self.evaluation_manager = EvaluationManager(config=eval_cfg,
                                                        scenario_name=scenario_type,
                                                        agent_name=agent_name)
            # The step trace is a backend-agnostic consumer of observations
            # and events; the SUMO mirror needs it for the same offline
            # metrics as CARLA.
            self.trace_writer = self._build_trace_writer()
        elif simulator == 'carla':
            # CARLA mode (standard)
            logger.info("Initializing CARLA MARL environment")

            require_opencda_core()
            from opencda.core.common.cav_world import CavWorld

            from opencda_marl._kernel.sensing.carla_monitor import CarlaMonitor
            from opencda_marl._kernel.sensing.carla_spectator import CarlaSpectator
            from opencda_marl.evaluation.manager import EvaluationManager
            from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder

            builder = PipelineBuilder(self.config)
            self.cav_world = CavWorld(apply_ml=self.config.opt.apply_ml)
            self.scenario_manager = builder.build_scenario(self.cav_world)
            self.states = self.scenario_manager.states
            self.carla_client = self.scenario_manager.client
            self.carla_world = self.scenario_manager.world

            scenario_cfg = self.config.get('scenario', {})
            spectator_config = scenario_cfg.get('spectator', self.config.get('spectator', {}))
            self.carla_spectator = CarlaSpectator(self.carla_world, spectator_config)

            self.marl_manager = builder.build_marl_manager(self.scenario_manager)
            self.marl_env = builder.build_env(self.scenario_manager, self.marl_manager)
            builder.wire_intent_provider(self.scenario_manager, self.marl_manager)

            self.carla_monitor = CarlaMonitor(
                self.carla_world, marl_tm=self.scenario_manager.traffic_manager
            )

            scenario_type = self.config.get('scenario', {}).get('type', None)
            agent_name = self.config.get('agents', {}).get('agent_type', None)
            eval_cfg = builder.build_output_section_config('evaluation')
            self.evaluation_manager = EvaluationManager(
                config=eval_cfg, scenario_name=scenario_type, agent_name=agent_name,
            )

            self.world_manager = builder.build_world_manager(self)
            self.trace_writer = builder.build_trace_writer(self)
        else:
            raise ValueError(
                f"Unsupported MARL runtime backend '{simulator}'. "
                "Supported backends are 'carla' and 'sumo'."
            )

        self._attach_evaluation_to_run_store()
        self._setup_traffic_view()
        self._setup_monitor_service()

        # The run's behaviour set is now fixed: every plugin (algorithms,
        # components, profiles) is loaded and every part constructed. Late
        # registrations would make results irreproducible — reject them.
        from opencda_marl import plugins as _plugins
        _plugins.freeze()

        # Mid-cell resume offset: when the launcher injects
        # MARL.training.start_episode (because a prior cell run was
        # interrupted with N pkl files already on disk), shift the episode
        # loop bound so we run episodes N..max-1 instead of 0..max-1. This
        # is a no-op (offset=0) for fresh runs and for evaluation mode.
        self.start_episode = int(
            self.config.get('MARL', {}).get('training', {}).get('start_episode', 0)
        )
        if self.start_episode > 0:
            logger.info(
                f"Mid-cell resume offset: start_episode={self.start_episode}, "
                f"will run episodes {self.start_episode + 1}..{self.states.get('max_episodes', '?')}"
            )

    # --------------------------------------------------------------------- #
    # main thread
    # --------------------------------------------------------------------- #
    def get_metrics(self):
        simulator = self.config.get('runtime', {}).get('backend', 'carla')

        if simulator == 'sumo':
            # SUMO mode: the same progress keys CARLA gets from the scenario
            # states (step / episode / outcome counts), then the environment's
            # own episode metrics on top. Without the progress keys every
            # consumer that keys on them -- the evaluation manager, the GUI
            # status line -- saw an empty episode.
            states = getattr(self, 'states', None) or {}
            metrics = {
                'max_steps': states.get('max_steps', 0),
                'max_episodes': states.get('max_episodes', 0),
                'fixed_dt': states.get('fixed_dt'),
            }
            env = self.marl_env
            if env is not None:
                episode = getattr(env, 'episode', None)
                departed = set(getattr(episode, 'departed_vehicles', ()) or ())
                collided = set(getattr(episode, 'collided_vehicles', ()) or ())
                # ``active_vehicles`` records every vehicle observed this
                # episode and is never pruned; a finished one is a success or
                # a collision, not still active. Counting it here too made
                # the evaluation read every success as a timeout as well, and
                # its all-finished step could never be reached.
                seen = getattr(episode, 'active_vehicles', ()) or ()
                active = {veh for veh in seen if veh not in departed and veh not in collided}
                # A vehicle still in SUMO's insertion queue was spawned but
                # never entered the network: an unfinished route, counted
                # with the active ones as CARLA counts its never-entered
                # spawns. Reported as a pending spawn it fell out of the
                # evaluation's total, so an episode that ended with a queue
                # overstated every rate.
                active |= self._sumo_queued_insertions(env)
                metrics.update({
                    'step': int(getattr(env, 'current_step', 0)),
                    'episode': int(getattr(env, 'current_episode', 0)),
                    'success': len(departed),
                    'collision': len(collided),
                    'active_agents': len(active),
                    'pending_spawns': self._sumo_pending_spawns(env),
                })
                metrics.update(env.get_episode_metrics())
            return metrics
        if simulator == 'carla':
            # CARLA mode: combine scenario states and environment metrics
            metrics = self.states.copy()
            if self.marl_env:
                metrics.update(self.marl_env.get_episode_metrics())
            # Add pending spawns from traffic manager for accurate throughput calculation
            if self.scenario_manager:
                traffic_info = self.scenario_manager.get_traffic_info()
                metrics['pending_spawns'] = traffic_info.get('pending_spawns', 0)
            return metrics
        raise ValueError(
            f"Unsupported MARL runtime backend '{simulator}'. "
            "Supported backends are 'carla' and 'sumo'."
        )

    @staticmethod
    def _sumo_pending_spawns(env) -> int:
        """Spawn events of the SUMO episode not yet due (vehicles already
        spawned into SUMO's insertion queue are active, see
        ``_sumo_queued_insertions``); 0 when the env cannot say."""
        counter = getattr(env, 'pending_spawn_count', None)
        if not callable(counter):
            return 0
        try:
            return int(counter())
        except Exception as exc:  # noqa: BLE001 - metrics must never kill a run
            logger.debug(f"SUMO pending spawn count unavailable: {exc}")
            return 0

    @staticmethod
    def _sumo_queued_insertions(env) -> set:
        """Vehicles the SUMO episode spawned that SUMO still holds in its
        insertion queue; empty when the env cannot say."""
        getter = getattr(env, 'queued_insertions', None)
        if not callable(getter):
            return set()
        try:
            return set(getter())
        except Exception as exc:  # noqa: BLE001 - metrics must never kill a run
            logger.debug(f"SUMO insertion queue unavailable: {exc}")
            return set()

    def step(self) -> bool:
        """Advance one step. Returns True when the environment reports the
        episode is done (envs without a done signal always return False)."""
        import time
        step_start = time.perf_counter()

        # Call pre-step callbacks
        for callback in self.pre_step_callbacks:
            callback()

        # Use MARL environment to step
        if self.marl_env:
            result = self.marl_env.step()
        else:
            result = self.scenario_manager.step()

        # Call post-step callbacks
        for callback in self.post_step_callbacks:
            callback()

        # The evaluation manager is the only per-step consumer of the full
        # metrics snapshot; assembling it (episode summaries, traffic
        # statistics, competition tables) is ~4 ms per CARLA step and is
        # thrown away when evaluation is disabled — the common training
        # configuration.
        evaluation_enabled = getattr(self.evaluation_manager, "is_enabled", lambda: True)
        if evaluation_enabled():
            metrics = self.get_metrics()
            # Get current step rewards if MARL environment is available
            rewards = None
            if self.marl_env:
                rewards = self.marl_env.get_current_step_rewards()
            self.evaluation_manager.update_step(metrics, rewards)

        # Record step time for world reset manager performance monitoring
        step_duration = time.perf_counter() - step_start
        if self.world_manager:
            self.world_manager.record_step_time(step_duration)

        return bool(result.get('done', False)) if isinstance(result, dict) else False

    def run(self):
        """
        Run the MARL scenario.
        """
        max_steps = self.states['max_steps']
        try:
            for episode in range(self.start_episode, self.states['max_episodes']):
                logger.info(
                    f"Starting episode {episode + 1}/{self.states['max_episodes']}")

                for step_index in range(max_steps):
                    if self.step():
                        # The env declared the episode over (e.g. SUMO traffic
                        # drained) — burning the remaining steps would tick an
                        # empty world and skew episode-length metrics.
                        logger.info(
                            f"Episode ended early by environment at step "
                            f"{step_index + 1}/{max_steps}")
                        break

                # Reset for next episode
                self.reset_episode()
        finally:
            if self.trace_writer is not None:
                self.trace_writer.close()

    def _end_evaluation_episode(self):
        """Tell the evaluation manager the episode is over (see
        EvaluationManager.end_episode for why it must not guess)."""
        manager = getattr(self, 'evaluation_manager', None)
        if manager is None or not getattr(manager, 'is_enabled', lambda: False)():
            return
        try:
            max_episodes = int(self.states.get('max_episodes', 0) or 0)
            episode = int(self.states.get('episode', 0) or 0)
            if self.marl_env is not None and hasattr(self.marl_env, 'current_episode'):
                episode = int(self.marl_env.current_episode)
            manager.end_episode(self.get_metrics(),
                                self.marl_env.get_current_step_rewards() if self.marl_env else None,
                                final=(episode >= max_episodes - 1))
        except Exception as exc:  # noqa: BLE001 - evaluation must not break training
            logger.warning(f"Evaluation episode close failed: {exc}")

    def reset_episode(self):
        """
        Reset environment and scenario for new episode.
        """
        self._end_evaluation_episode()
        # Each stage is isolated: one failing stage must not skip the others,
        # or leaked actors and stale registries accumulate episode over
        # episode (the exact pattern the world-reset watchdog papers over).
        try:
            # Reset MARL environment first
            if self.marl_env:
                episode_metrics = self.marl_env.reset_episode()
                # Log a curated subset; full metrics are archived under
                # outputs.metrics_archive.dir for post-hoc analysis.
                summary_keys = (
                    "episode", "step_length",
                    "total_reward", "avg_reward",
                    "success_rate", "collision_rate", "timeout_rate",
                    "successes", "collisions", "timeouts",
                )
                summary = {k: episode_metrics.get(k) for k in summary_keys
                           if k in episode_metrics}
                logger.info(f"Episode summary: {summary}")
        except Exception as e:
            logger.error(f"Error during environment episode reset: {e}")
            import traceback
            traceback.print_exc()

        try:
            # Reset scenario manager (only in CARLA mode)
            if self.scenario_manager is not None:
                self.scenario_manager.reset_episode()
        except Exception as e:
            logger.error(f"Error during scenario episode reset: {e}")
            import traceback
            traceback.print_exc()

        for callback in self.episode_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Episode callback {callback!r} failed: {e}")

        import gc
        gc.collect()

        # Check for world reset AFTER episode completes (safe timing)
        # This prevents CARLA server-side memory accumulation slowdown
        if self.world_manager:
            self.world_manager.on_episode_end()

    # --------------------------------------------------------------------- #
    # hooks
    # --------------------------------------------------------------------- #
    def register_pre_step_callback(self, callback: Callable):
        """Register pre-step callback."""
        self.pre_step_callbacks.append(callback)

    def _build_trace_writer(self):
        """The route-action step trace (``outputs.trace``), on any backend."""
        trace_cfg = self._build_output_section_config('trace')
        if not trace_cfg.get('enabled', False):
            return None
        from opencda_marl.telemetry.trace import RouteActionTraceWriter

        writer = RouteActionTraceWriter(trace_cfg, self)
        self.register_post_step_callback(lambda: writer.record_step(self))
        self.register_episode_callback(lambda: writer.flush())
        logger.info(f"Route-action trace enabled: {writer.root_dir}")
        return writer

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
        """Tear down writers, the environment, and simulator state.

        The scenario manager runs last because its close() restores the
        pre-run (asynchronous) world settings. CarlaMonitor must not touch
        settings here: its snapshot is taken after sync mode was applied, so
        "restoring" it used to leave the CARLA server frozen in synchronous
        mode with no client ticking.
        """
        if self.trace_writer is not None:
            try:
                self.trace_writer.close()
            except Exception as exc:
                logger.warning(f"Trace writer close failed: {exc}")

        # Flush the metrics archive before the environment goes away. The
        # exporter only writes every `export_interval` episodes, so with an
        # episode budget that is not a multiple of it -- or any interrupted
        # run -- the trailing episodes lived only in memory and were lost,
        # while the launcher's resume counted them as never run.
        metrics = getattr(self.marl_env, 'metrics', None) if self.marl_env else None
        if metrics is not None and getattr(metrics, 'export_interval', 0) > 0:
            try:
                metrics.export_history()
            except Exception as exc:
                logger.warning(f"Metrics history flush failed: {exc}")

        if self.marl_env is not None and hasattr(self.marl_env, 'close'):
            try:
                self.marl_env.close()
                logger.info("MARL environment closed")
            except Exception as exc:
                logger.warning(f"MARL environment close failed: {exc}")

        # Flush and close the TensorBoard writer — previously nothing called
        # this, leaking a writer thread per run and losing tail events.
        if getattr(self, 'marl_manager', None) is not None:
            try:
                self.marl_manager.close()
            except Exception as exc:
                logger.warning(f"MARL manager close failed: {exc}")

        if getattr(self, 'scenario_manager', None):
            try:
                self.scenario_manager.close()
                logger.info("Scenario manager cleaned up")
            except Exception as exc:
                logger.warning(f"Scenario manager close failed: {exc}")

        logger.info("Coordinator cleanup completed")
