'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-11-25
FilePath     : /OpenCDA-MARL/opencda_marl/core/lifecycle/world_manager.py
Description  : CARLA World Reset Manager for preventing training slowdown.

Manages world reload to prevent server-side memory accumulation while
preserving all Python-side training state.

Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import copy
import gc
import time
import statistics

import torch
from collections import deque
from typing import Dict, Any, Optional, TYPE_CHECKING
from loguru import logger

import carla

if TYPE_CHECKING:
    from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator
    from opencda_marl._kernel.envs.marl_env import MARLEnv


#: Floor for the client time-out while ``reload_world`` runs. On a windowed
#: server a reload takes 10 to 35 s, longer than the usual ``world.timeout``.
RELOAD_TIMEOUT_S = 120.0


class WorldResetError(RuntimeError):
    """The CARLA world could not be reloaded; the run cannot continue."""


class TrainingStateSnapshot:
    """
    Captures all training state that must survive CARLA world reset.

    This includes replay buffers, model weights, optimizers, training
    counters, and history deques - everything needed to continue
    training seamlessly after world reload.
    """

    def __init__(self):
        # Replay buffer (reference, not copy - avoid memory duplication)
        self.replay_buffer = None

        # Model weights
        self.model_state_dicts = {}
        self.optimizer_state_dict = None

        # Training progress
        self.epsilon = None
        self.training_step = 0
        self.episode_count = 0

        # History deques
        self.reward_history = None
        self.episode_length_history = None
        self.success_rate_history = None
        self.collision_rate_history = None

        # Convergence state
        self.is_converged = False
        self.convergence_episode = None

        # TensorBoard writer (keep reference for continuous logging)
        self.tensorboard_writer = None
        self.tensorboard_dir = None
        self.tensorboard_root_dir = None
        self.tb_enabled = False
        self.tb_mode = 'full'
        self.tb_log_frequency = 1
        self.tb_metrics = {}
        self.tb_category_frequencies = {}
        self.tb_histograms_enabled = False
        self.tb_max_episodes_per_run = 0
        self.tb_keep_last_runs = 0
        self.tb_run_index = 0
        self.tb_logged_episodes_in_run = 0
        self.tb_run_name = None

        # On-policy rollout state (references): the partial rollout, its
        # open trajectories and the progress counters survive the reload
        # instead of being thrown away with the old learner object.
        self.rollout_state = {}

        # Environment state
        self.metrics = None
        self.checkpoint_manager = None
        self.training_config = None

    # Progress counters every learner may carry (captured/restored by name).
    _COUNTER_ATTRS = ('steps_collected', 'updates_performed', '_steps_since_update')
    # Rollout objects handed over by reference (incl. the running return
    # statistics of value-normalising critics).
    _ROLLOUT_ATTRS = ('rollout_buffer', '_open_next_obs', 'value_normalizer', 'vector_value_normalizer')

    @classmethod
    def capture(cls, marl_env: 'MARLEnv') -> 'TrainingStateSnapshot':
        """
        Capture current training state before world reset.

        Args:
            marl_env: The MARL environment containing training state

        Returns:
            TrainingStateSnapshot with all state captured
        """
        snapshot = cls()

        # Get algorithm from MARL manager
        algorithm = marl_env.marl_manager.algorithm

        if algorithm is not None:
            # Capture replay buffer reference (not a copy!)
            # Works for DQN, TD3, and other buffer-based algorithms
            if hasattr(algorithm, 'memory'):
                snapshot.replay_buffer = algorithm.memory
                logger.debug(f"Captured replay buffer with {len(snapshot.replay_buffer)} experiences")

            # Capture model weights (DQN has q_network/target_network)
            if hasattr(algorithm, 'q_network'):
                snapshot.model_state_dicts['q_network'] = \
                    copy.deepcopy(algorithm.q_network.state_dict())
                snapshot.model_state_dicts['target_network'] = \
                    copy.deepcopy(algorithm.target_network.state_dict())
                logger.debug("Captured DQN network weights")

            # Capture TD3 networks if present (has actor_target)
            if hasattr(algorithm, 'actor') and hasattr(algorithm, 'actor_target'):
                snapshot.model_state_dicts['actor'] = copy.deepcopy(algorithm.actor.state_dict())
                snapshot.model_state_dicts['actor_target'] = copy.deepcopy(algorithm.actor_target.state_dict())
                snapshot.model_state_dicts['critic'] = copy.deepcopy(algorithm.critic.state_dict())
                snapshot.model_state_dicts['critic_target'] = copy.deepcopy(algorithm.critic_target.state_dict())
                logger.debug("Captured TD3 network weights")

            # Capture SAC networks (twin critic target + entropy temperature).
            # SAC has actor+critic but no actor_target, so it previously fell
            # into the MAPPO branch and silently lost critic_target,
            # log_alpha, the alpha optimizer, and the LSTM encoder.
            elif hasattr(algorithm, 'log_alpha') and hasattr(algorithm, 'actor'):
                snapshot.model_state_dicts['actor'] = copy.deepcopy(algorithm.actor.state_dict())
                snapshot.model_state_dicts['critic'] = copy.deepcopy(algorithm.critic.state_dict())
                if hasattr(algorithm, 'critic_target'):
                    snapshot.model_state_dicts['critic_target'] = \
                        copy.deepcopy(algorithm.critic_target.state_dict())
                snapshot.model_state_dicts['log_alpha'] = \
                    algorithm.log_alpha.detach().clone()
                if getattr(algorithm, 'alpha_optimizer', None) is not None:
                    snapshot.model_state_dicts['alpha_optimizer'] = \
                        copy.deepcopy(algorithm.alpha_optimizer.state_dict())
                if hasattr(algorithm, 'lstm_encoder'):
                    snapshot.model_state_dicts['lstm_encoder'] = \
                        copy.deepcopy(algorithm.lstm_encoder.state_dict())
                if hasattr(algorithm, 'actor_optimizer'):
                    snapshot.model_state_dicts['actor_optimizer'] = \
                        copy.deepcopy(algorithm.actor_optimizer.state_dict())
                if hasattr(algorithm, 'critic_optimizer'):
                    snapshot.model_state_dicts['critic_optimizer'] = \
                        copy.deepcopy(algorithm.critic_optimizer.state_dict())
                logger.debug("Captured SAC network weights "
                             "(incl. critic_target/log_alpha)")

            # Capture MAPPO networks if present (no target networks)
            elif hasattr(algorithm, 'actor') and hasattr(algorithm, 'critic'):
                snapshot.model_state_dicts['actor'] = copy.deepcopy(algorithm.actor.state_dict())
                snapshot.model_state_dicts['critic'] = copy.deepcopy(algorithm.critic.state_dict())
                # MAPPO may have separate optimizers
                if hasattr(algorithm, 'actor_optimizer'):
                    snapshot.model_state_dicts['actor_optimizer'] = copy.deepcopy(algorithm.actor_optimizer.state_dict())
                if hasattr(algorithm, 'critic_optimizer'):
                    snapshot.model_state_dicts['critic_optimizer'] = copy.deepcopy(algorithm.critic_optimizer.state_dict())
                # Capture rollout buffer if present (MAPPO on-policy buffer)
                if hasattr(algorithm, 'rollout_buffer'):
                    snapshot.model_state_dicts['has_rollout_buffer'] = True
                logger.debug("Captured MAPPO network weights")

            # Capture optimizer state
            if hasattr(algorithm, 'optimizer'):
                snapshot.optimizer_state_dict = copy.deepcopy(algorithm.optimizer.state_dict())

            # Family-independent pieces the branches above did not all
            # cover: the LSTM context encoder (TD3 and the PPO family lost
            # it on every reload — a re-randomised encoder under a trained
            # critic) and the per-network optimizers (TD3 lost its moments).
            if hasattr(algorithm, 'lstm_encoder') and 'lstm_encoder' not in snapshot.model_state_dicts:
                snapshot.model_state_dicts['lstm_encoder'] = \
                    copy.deepcopy(algorithm.lstm_encoder.state_dict())
            for name in ('actor_optimizer', 'critic_optimizer'):
                if hasattr(algorithm, name) and name not in snapshot.model_state_dicts:
                    snapshot.model_state_dicts[name] = copy.deepcopy(getattr(algorithm, name).state_dict())

            # On-policy rollout state and progress counters (by reference /
            # by value) so a partial rollout is not discarded by the reload.
            for name in cls._ROLLOUT_ATTRS + cls._COUNTER_ATTRS:
                if hasattr(algorithm, name):
                    snapshot.rollout_state[name] = getattr(algorithm, name)

            # Capture training progress (from BaseAlgorithm). External
            # policies need not subclass BaseAlgorithm: read what exists.
            snapshot.epsilon = getattr(algorithm, 'epsilon', None)
            snapshot.training_step = getattr(algorithm, 'training_step', 0)
            snapshot.episode_count = getattr(algorithm, 'episode_count', 0)

            # Capture history deques (convert to lists for safe copying)
            if hasattr(algorithm, 'reward_history'):
                snapshot.reward_history = list(algorithm.reward_history)
            if hasattr(algorithm, 'episode_length_history'):
                snapshot.episode_length_history = list(algorithm.episode_length_history)
            if hasattr(algorithm, 'success_rate_history'):
                snapshot.success_rate_history = list(algorithm.success_rate_history)
            if hasattr(algorithm, 'collision_rate_history'):
                snapshot.collision_rate_history = list(algorithm.collision_rate_history)

            # Capture convergence state
            snapshot.is_converged = getattr(algorithm, 'is_converged', False)
            snapshot.convergence_episode = getattr(algorithm, 'convergence_episode', None)

            # Keep TensorBoard writer reference (file-based, survives reload)
            snapshot.tensorboard_writer = getattr(algorithm, 'writer', None)
            snapshot.tensorboard_dir = getattr(algorithm, 'tensorboard_dir', None)
            snapshot.tensorboard_root_dir = getattr(algorithm, 'tensorboard_root_dir', None)
            snapshot.tb_enabled = getattr(algorithm, 'tb_enabled', False)
            snapshot.tb_mode = getattr(algorithm, 'tb_mode', 'full')
            snapshot.tb_log_frequency = getattr(algorithm, 'tb_log_frequency', 1)
            snapshot.tb_metrics = getattr(algorithm, 'tb_metrics', {})
            snapshot.tb_category_frequencies = getattr(algorithm, 'tb_category_frequencies', {})
            snapshot.tb_histograms_enabled = getattr(algorithm, 'tb_histograms_enabled', False)
            snapshot.tb_max_episodes_per_run = getattr(algorithm, 'tb_max_episodes_per_run', 0)
            snapshot.tb_keep_last_runs = getattr(algorithm, 'tb_keep_last_runs', 0)
            snapshot.tb_run_index = getattr(algorithm, 'tb_run_index', 0)
            snapshot.tb_logged_episodes_in_run = getattr(algorithm, 'tb_logged_episodes_in_run', 0)
            snapshot.tb_run_name = getattr(algorithm, 'tb_run_name', None)

        # Capture environment state
        snapshot.metrics = marl_env.metrics
        snapshot.checkpoint_manager = marl_env.checkpoint_manager
        snapshot.training_config = marl_env.training_config

        logger.info(f"Training state captured: episode={snapshot.episode_count}, "
                   f"training_step={snapshot.training_step}, "
                   f"epsilon={snapshot.epsilon}, "
                   f"replay_buffer_size={len(snapshot.replay_buffer) if snapshot.replay_buffer else 0}")

        return snapshot

    def restore(self, marl_env: 'MARLEnv'):
        """
        Restore training state after world reset.

        Args:
            marl_env: The MARL environment to restore state to
        """
        algorithm = marl_env.marl_manager.algorithm

        if algorithm is not None:
            # Restore replay buffer reference
            if self.replay_buffer is not None and hasattr(algorithm, 'memory'):
                algorithm.memory = self.replay_buffer
                logger.debug(f"Restored replay buffer with {len(self.replay_buffer)} experiences")

            # Restore DQN model weights
            if 'q_network' in self.model_state_dicts:
                algorithm.q_network.load_state_dict(
                    self.model_state_dicts['q_network'])
                algorithm.target_network.load_state_dict(
                    self.model_state_dicts['target_network'])
                logger.debug("Restored DQN network weights")

            # Restore TD3 model weights (has target networks)
            if 'actor_target' in self.model_state_dicts:
                algorithm.actor.load_state_dict(self.model_state_dicts['actor'])
                algorithm.actor_target.load_state_dict(self.model_state_dicts['actor_target'])
                algorithm.critic.load_state_dict(self.model_state_dicts['critic'])
                algorithm.critic_target.load_state_dict(self.model_state_dicts['critic_target'])
                logger.debug("Restored TD3 network weights")

            # Restore SAC weights (keyed by the captured entropy temperature)
            elif 'log_alpha' in self.model_state_dicts:
                algorithm.actor.load_state_dict(self.model_state_dicts['actor'])
                algorithm.critic.load_state_dict(self.model_state_dicts['critic'])
                if 'critic_target' in self.model_state_dicts and \
                        hasattr(algorithm, 'critic_target'):
                    algorithm.critic_target.load_state_dict(
                        self.model_state_dicts['critic_target'])
                # copy_ into the existing tensor: the alpha optimizer holds a
                # reference to it, so rebinding would silently detach them.
                with torch.no_grad():
                    algorithm.log_alpha.data.copy_(
                        self.model_state_dicts['log_alpha'])
                if 'alpha_optimizer' in self.model_state_dicts and \
                        getattr(algorithm, 'alpha_optimizer', None) is not None:
                    algorithm.alpha_optimizer.load_state_dict(
                        self.model_state_dicts['alpha_optimizer'])
                if 'lstm_encoder' in self.model_state_dicts and \
                        hasattr(algorithm, 'lstm_encoder'):
                    algorithm.lstm_encoder.load_state_dict(
                        self.model_state_dicts['lstm_encoder'])
                if 'actor_optimizer' in self.model_state_dicts and \
                        hasattr(algorithm, 'actor_optimizer'):
                    algorithm.actor_optimizer.load_state_dict(
                        self.model_state_dicts['actor_optimizer'])
                if 'critic_optimizer' in self.model_state_dicts and \
                        hasattr(algorithm, 'critic_optimizer'):
                    algorithm.critic_optimizer.load_state_dict(
                        self.model_state_dicts['critic_optimizer'])
                logger.debug("Restored SAC network weights "
                             "(incl. critic_target/log_alpha)")

            # Restore MAPPO model weights (no target networks)
            elif 'actor' in self.model_state_dicts and 'actor_target' not in self.model_state_dicts:
                algorithm.actor.load_state_dict(self.model_state_dicts['actor'])
                algorithm.critic.load_state_dict(self.model_state_dicts['critic'])
                # Restore separate optimizers if present
                if 'actor_optimizer' in self.model_state_dicts and hasattr(algorithm, 'actor_optimizer'):
                    algorithm.actor_optimizer.load_state_dict(self.model_state_dicts['actor_optimizer'])
                if 'critic_optimizer' in self.model_state_dicts and hasattr(algorithm, 'critic_optimizer'):
                    algorithm.critic_optimizer.load_state_dict(self.model_state_dicts['critic_optimizer'])
                logger.debug("Restored MAPPO network weights")

            # Restore optimizer state
            if self.optimizer_state_dict and hasattr(algorithm, 'optimizer'):
                algorithm.optimizer.load_state_dict(self.optimizer_state_dict)

            # Family-independent pieces (see capture): encoder, per-network
            # optimizers (best-effort — the parameter set can legitimately
            # differ, e.g. train_context_encoder), rollout state, counters.
            if 'lstm_encoder' in self.model_state_dicts and hasattr(algorithm, 'lstm_encoder'):
                algorithm.lstm_encoder.load_state_dict(self.model_state_dicts['lstm_encoder'])
            for name in ('actor_optimizer', 'critic_optimizer'):
                if name in self.model_state_dicts and hasattr(algorithm, name):
                    try:
                        getattr(algorithm, name).load_state_dict(self.model_state_dicts[name])
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(f"World reload: {name} state not applied ({exc}); fresh moments")
            for name, value in self.rollout_state.items():
                if hasattr(algorithm, name):
                    setattr(algorithm, name, value)
            # Optimizer state was reloaded above: a learner that replays a
            # captured update graph must re-capture against the new tensors.
            invalidate = getattr(algorithm, '_invalidate_update_graph', None)
            if callable(invalidate):
                invalidate()

            # Restore training progress
            if self.epsilon is not None:
                algorithm.epsilon = self.epsilon
            # Only hand back what the learner tracks itself: an external
            # policy without these counters must not grow them on restore.
            for name in ('training_step', 'episode_count'):
                if hasattr(algorithm, name):
                    setattr(algorithm, name, getattr(self, name))

            # Restore history deques (recreate with maxlen)
            if self.reward_history is not None:
                algorithm.reward_history = deque(self.reward_history, maxlen=100)
            if self.episode_length_history is not None:
                algorithm.episode_length_history = deque(
                    self.episode_length_history, maxlen=100)
            if self.success_rate_history is not None:
                algorithm.success_rate_history = deque(
                    self.success_rate_history, maxlen=100)
            if self.collision_rate_history is not None:
                algorithm.collision_rate_history = deque(
                    self.collision_rate_history, maxlen=100)

            # Restore convergence state
            if hasattr(algorithm, 'is_converged'):
                algorithm.is_converged = self.is_converged
                algorithm.convergence_episode = self.convergence_episode

            # Restore TensorBoard writer
            if self.tensorboard_writer:
                algorithm.writer = self.tensorboard_writer
                algorithm.tensorboard_dir = self.tensorboard_dir
                algorithm.tensorboard_root_dir = self.tensorboard_root_dir
                algorithm.tb_enabled = self.tb_enabled
                algorithm.tb_mode = self.tb_mode
                algorithm.tb_log_frequency = self.tb_log_frequency
                algorithm.tb_metrics = self.tb_metrics
                algorithm.tb_category_frequencies = self.tb_category_frequencies
                algorithm.tb_histograms_enabled = self.tb_histograms_enabled
                algorithm.tb_max_episodes_per_run = self.tb_max_episodes_per_run
                algorithm.tb_keep_last_runs = self.tb_keep_last_runs
                algorithm.tb_run_index = self.tb_run_index
                algorithm.tb_logged_episodes_in_run = self.tb_logged_episodes_in_run
                algorithm.tb_run_name = self.tb_run_name

        # Restore environment state
        marl_env.metrics = self.metrics
        marl_env.checkpoint_manager = self.checkpoint_manager
        marl_env.training_config = self.training_config

        logger.info(f"Training state restored: episode={self.episode_count}, "
                   f"training_step={self.training_step}, "
                   f"epsilon={self.epsilon}")


class WorldManager:
    """
    Manages CARLA world reset/reload to prevent server-side memory accumulation.

    CARLA's server accumulates internal state (physics buffers, destroyed actor
    references, collision detection caches) over time, causing simulation slowdown.
    Python's gc.collect() cannot clean this - only client.reload_world() can.

    This manager:
    1. Tracks epoch count and triggers reset at configurable frequency
    2. Monitors step duration for automatic slowdown detection
    3. Orchestrates safe 7-phase world reload sequence
    4. Coordinates with MARLEnv to preserve all training state

    Key responsibilities:
    - Performance monitoring (step time tracking, baseline establishment)
    - Reset decision making (periodic + auto-reset on slowdown)
    - Safe reload sequence execution
    - Training state preservation across reloads
    """

    def __init__(self, config: Dict[str, Any], coordinator: 'MARLCoordinator'):
        """
        Initialize WorldManager.

        Args:
            config: world_reset configuration section
            coordinator: Reference to MARLCoordinator for accessing components
        """
        self.config = config
        self.coordinator = coordinator

        # Reset frequency configuration
        self.reset_frequency = config.get('reset_frequency', 15)

        # Auto-reset configuration
        auto_cfg = config.get('auto_reset', {})
        self.auto_reset_enabled = auto_cfg.get('enabled', True)
        self.slowdown_threshold = auto_cfg.get('slowdown_threshold', 1.5)
        self.monitoring_window = auto_cfg.get('monitoring_window', 100)
        self.min_epochs_before_auto = auto_cfg.get('min_epochs_before_auto', 3)

        # Performance monitoring state
        self.baseline_step_time: Optional[float] = None
        self.step_times: deque = deque(maxlen=200)
        self.epochs_since_reset = 0

        # Reset tracking
        self.total_resets = 0
        self.last_reset_epoch = 0

        # Logging configuration
        self.log_reset_events = config.get('log_reset_events', True)
        self.log_step_times = config.get('log_step_times', False)

        logger.info(f"WorldManager initialized: "
                   f"reset_frequency={self.reset_frequency}, "
                   f"auto_reset={self.auto_reset_enabled}, "
                   f"slowdown_threshold={self.slowdown_threshold}x")

    def record_step_time(self, step_duration: float):
        """
        Record step duration for performance monitoring.

        Args:
            step_duration: Time in seconds for the last step
        """
        self.step_times.append(step_duration)

        if self.log_step_times:
            logger.debug(f"Step time: {step_duration:.4f}s")

        # Establish baseline after collecting enough stable data
        # Wait for 50 steps to avoid including initialization overhead
        if self.baseline_step_time is None and len(self.step_times) >= 50:
            self.baseline_step_time = statistics.median(self.step_times)
            logger.info(f"Step time baseline established: {self.baseline_step_time:.4f}s")

    def should_auto_reset(self) -> bool:
        """
        Check if auto-reset should be triggered based on performance degradation.

        Returns:
            True if slowdown detected and auto-reset should trigger
        """
        if not self.auto_reset_enabled:
            return False

        if self.baseline_step_time is None:
            return False

        if len(self.step_times) < self.monitoring_window:
            return False

        if self.epochs_since_reset < self.min_epochs_before_auto:
            return False

        # Calculate recent median step time
        recent_times = list(self.step_times)[-self.monitoring_window:]
        recent_median = statistics.median(recent_times)
        slowdown_ratio = recent_median / self.baseline_step_time

        if slowdown_ratio > self.slowdown_threshold:
            logger.warning(f"Performance degradation detected: "
                          f"{slowdown_ratio:.2f}x slower "
                          f"(baseline: {self.baseline_step_time:.4f}s, "
                          f"current: {recent_median:.4f}s)")
            return True

        return False

    def on_episode_end(self):
        """
        Called after each episode ends. Check if world reset is needed.

        This is the ONLY safe time to reset - never mid-episode.

        Raises:
            WorldResetError: a triggered reset failed and the world is unusable
        """
        self.epochs_since_reset += 1

        should_reset = False
        reset_reason = ""

        # Check periodic reset
        if (self.reset_frequency > 0 and
            self.epochs_since_reset >= self.reset_frequency):
            should_reset = True
            reset_reason = f"Periodic reset (every {self.reset_frequency} epochs)"

        # Check auto-reset (only if periodic didn't trigger)
        elif self.should_auto_reset():
            should_reset = True
            reset_reason = "Auto-reset due to performance degradation"

        if should_reset:
            if self.log_reset_events:
                logger.warning(f"Triggering world reset: {reset_reason}")

            success = self.reload_world()

            if success:
                logger.success(f"World reset #{self.total_resets} completed. "
                              f"Reason: {reset_reason}")
            else:
                # A failed reload leaves a dead world: the next tick would
                # time out or abort natively, so stop the run here.
                raise WorldResetError(
                    f"World reset #{self.total_resets + 1} failed ({reset_reason}); "
                    "the CARLA world is no longer usable, stopping the run. "
                    "See the errors above for the cause.")

    def reload_world(self) -> bool:
        """
        Safely reload CARLA world while preserving all training state.

        Executes a 7-phase reload sequence:
        1. Capture training state
        2. Cleanup CARLA resources
        3. Reload world (client.reload_world())
        4. Reapply world settings
        5. Rebuild scenario components
        6. Restore training state
        7. Finalize and reset baseline

        MUST be called AFTER episode completes, NEVER mid-episode.

        Returns:
            True if reload successful, False otherwise
        """
        logger.warning(f"{'='*20} CARLA World Reset #{self.total_resets + 1} {'='*20}")

        try:
            # Phase 1: Capture training state
            logger.info("Phase 1/7: Capturing training state...")
            state_snapshot = TrainingStateSnapshot.capture(
                self.coordinator.marl_env)

            # Phase 2: Cleanup existing CARLA resources
            logger.info("Phase 2/7: Cleaning up CARLA resources...")
            self._cleanup_carla_resources()

            # Phase 3: Reload world
            logger.info("Phase 3/7: Reloading CARLA world...")
            success = self._reload_carla_world()
            if not success:
                logger.error("World reload failed!")
                return False

            # Phase 4: Reapply world settings
            logger.info("Phase 4/7: Reapplying world settings...")
            self._apply_world_settings()

            # Phase 5: Rebuild scenario components
            logger.info("Phase 5/7: Rebuilding scenario components...")
            self._rebuild_scenario_components()

            # Phase 6: Restore training state
            logger.info("Phase 6/7: Restoring training state...")
            state_snapshot.restore(self.coordinator.marl_env)

            # Phase 7: Finalize reset
            logger.info("Phase 7/7: Finalizing reset...")
            self.total_resets += 1
            self.epochs_since_reset = 0
            self.last_reset_epoch = self.coordinator.states.get('episode', 0)

            # Reset performance baseline (will re-establish after 50 steps)
            self.baseline_step_time = None
            self.step_times.clear()

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
            except ImportError:
                pass

            logger.success(f"World reset completed successfully. Total resets: {self.total_resets}")
            return True

        except Exception as e:
            logger.error(f"World reset failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _cleanup_carla_resources(self):
        """
        Cleanup existing CARLA resources before reload.

        Each manager handles its own sensor cleanup - no pre-stopping needed.
        IMPORTANT: world.tick() is called after each cleanup phase to ensure
        CARLA server processes destroy commands and releases GPU memory.
        """
        sm = self.coordinator.scenario_manager
        world = self.coordinator.carla_world

        # Cleanup agent manager (destroys all vehicles and sensors)
        if sm.agent_manager:
            sm.agent_manager.cleanup()
            logger.debug("Agent manager cleaned up")

        # Cleanup traffic manager
        if sm.traffic_manager:
            sm.traffic_manager.cleanup()
            logger.debug("Traffic manager cleaned up")

        # Tick to process manager cleanups
        try:
            if world:
                world.tick()
        except Exception:
            pass

        # Destroy any remaining actors in the world
        # NOTE: We must re-fetch actors list after manager cleanups since
        # the old list contains stale references to already-destroyed actors
        try:
            if world:
                # Re-fetch fresh actor list after manager cleanups
                actors = world.get_actors()

                # Destroy remaining sensors first (they depend on vehicles)
                sensor_count = 0
                sensor_errors = 0
                for actor in actors.filter('sensor.*'):
                    try:
                        if actor.is_alive:
                            # Stop the data stream before destroying, or the
                            # registered callback is dropped on the floor.
                            try:
                                actor.stop()
                            except Exception:
                                pass
                            actor.destroy()
                            sensor_count += 1
                    except RuntimeError:
                        # Actor already destroyed, ignore
                        sensor_errors += 1
                if sensor_count > 0:
                    logger.debug(f"Destroyed {sensor_count} remaining sensor actors")
                if sensor_errors > 0:
                    logger.debug(f"Skipped {sensor_errors} already-destroyed sensors")

                # Tick after sensor cleanup
                world.tick()

                # Re-fetch actors list again for vehicles
                actors = world.get_actors()

                # Destroy remaining vehicles
                vehicle_count = 0
                vehicle_errors = 0
                for actor in actors.filter('vehicle.*'):
                    try:
                        if actor.is_alive:
                            actor.destroy()
                            vehicle_count += 1
                    except RuntimeError:
                        # Actor already destroyed, ignore
                        vehicle_errors += 1
                if vehicle_count > 0:
                    logger.debug(f"Destroyed {vehicle_count} remaining vehicle actors")
                if vehicle_errors > 0:
                    logger.debug(f"Skipped {vehicle_errors} already-destroyed vehicles")

                # Final tick to process all destroys and release GPU memory
                world.tick()

        except Exception as e:
            logger.warning(f"Error during actor cleanup: {e}")

    def _reload_carla_world(self) -> bool:
        """
        Reload world using client.reload_world().

        This clears CARLA server-side memory and reloads the current map.

        Returns:
            True if successful, False otherwise
        """
        client = self.coordinator.carla_client
        configured = float(self.coordinator.config.get('world', {}).get('timeout', 20.0))
        client.set_timeout(max(configured, RELOAD_TIMEOUT_S))
        try:
            # reload_world() clears server memory and reloads current map
            # This is the key call that fixes the slowdown!
            client.reload_world()

            # Wait for world to be ready and re-get reference
            for attempt in range(5):
                try:
                    time.sleep(1.0)
                    self.coordinator.carla_world = client.get_world()
                    if self.coordinator.carla_world:
                        logger.info("World reference obtained successfully")
                        return True
                except Exception as e:
                    logger.warning(f"Retry {attempt + 1}/5 getting world: {e}")

            return False

        except Exception as e:
            logger.error(f"reload_world() failed: {e}")
            return False
        finally:
            client.set_timeout(configured)

    def _apply_world_settings(self):
        """Reapply world settings after reload."""
        world = self.coordinator.carla_world
        sm = self.coordinator.scenario_manager

        # Apply synchronous mode settings from scenario manager
        settings = world.get_settings()
        settings.synchronous_mode = True  # Always sync mode
        settings.fixed_delta_seconds = sm.fixed_dt
        world.apply_settings(settings)

        logger.debug(f"World settings reapplied: sync_mode=True, fixed_dt={sm.fixed_dt}")

    def _rebuild_scenario_components(self):
        """Rebuild scenario components by calling the same builder used at init."""
        from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder
        from opencda_marl._kernel.sensing import CarlaMonitor, CarlaSpectator

        builder = PipelineBuilder(self.coordinator.config)
        sm = self.coordinator.scenario_manager
        sm.world = self.coordinator.carla_world
        sm.carla_map = sm.world.get_map()

        # Rebuild traffic + agent managers. The MARL manager is unaffected; its
        # state is preserved by TrainingStateSnapshot in the reload sequence.
        sm.rebuild_after_world_reload(builder)
        builder.wire_intent_provider(sm, self.coordinator.marl_manager)

        self.coordinator.marl_env.world = sm.world
        self.coordinator.marl_env.sm = sm
        self.coordinator.carla_monitor = CarlaMonitor(
            sm.world, marl_tm=sm.traffic_manager,
        )

        scenario_cfg = self.coordinator.config.get('scenario', {})
        spectator_config = scenario_cfg.get(
            'spectator', self.coordinator.config.get('spectator', {}),
        )
        self.coordinator.carla_spectator = CarlaSpectator(sm.world, spectator_config)
        logger.info("All scenario components rebuilt successfully")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the world reset manager.

        Returns:
            Dictionary with status information
        """
        status = {
            'total_resets': self.total_resets,
            'epochs_since_reset': self.epochs_since_reset,
            'reset_frequency': self.reset_frequency,
            'auto_reset_enabled': self.auto_reset_enabled,
            'baseline_step_time': self.baseline_step_time,
        }

        if self.baseline_step_time and len(self.step_times) >= self.monitoring_window:
            recent_times = list(self.step_times)[-self.monitoring_window:]
            recent_median = statistics.median(recent_times)
            status['current_step_time'] = recent_median
            status['slowdown_ratio'] = recent_median / self.baseline_step_time

        return status
