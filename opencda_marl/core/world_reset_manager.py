'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-11-25
FilePath     : /OpenCDA-MARL/opencda_marl/core/world_reset_manager.py
Description  : CARLA World Reset Manager for preventing training slowdown.

Manages world reload to prevent server-side memory accumulation while
preserving all Python-side training state.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import gc
import time
import statistics
from collections import deque
from typing import Dict, Any, Optional, TYPE_CHECKING
from loguru import logger

import carla

if TYPE_CHECKING:
    from opencda_marl.coordinator import MARLCoordinator
    from opencda_marl.envs.marl_env import MARLEnv


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
        self.tb_enabled = False
        self.tb_log_frequency = 1
        self.tb_metrics = {}

        # Environment state
        self.metrics = None
        self.checkpoint_manager = None
        self.training_config = None

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
                    algorithm.q_network.state_dict()
                snapshot.model_state_dicts['target_network'] = \
                    algorithm.target_network.state_dict()
                logger.debug("Captured DQN network weights")

            # Capture TD3 networks if present
            if hasattr(algorithm, 'actor'):
                snapshot.model_state_dicts['actor'] = algorithm.actor.state_dict()
                snapshot.model_state_dicts['actor_target'] = algorithm.actor_target.state_dict()
                snapshot.model_state_dicts['critic'] = algorithm.critic.state_dict()
                snapshot.model_state_dicts['critic_target'] = algorithm.critic_target.state_dict()
                logger.debug("Captured TD3 network weights")

            # Capture optimizer state
            if hasattr(algorithm, 'optimizer'):
                snapshot.optimizer_state_dict = algorithm.optimizer.state_dict()

            # Capture training progress (from BaseAlgorithm)
            snapshot.epsilon = getattr(algorithm, 'epsilon', None)
            snapshot.training_step = algorithm.training_step
            snapshot.episode_count = algorithm.episode_count

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
            snapshot.is_converged = algorithm.is_converged
            snapshot.convergence_episode = algorithm.convergence_episode

            # Keep TensorBoard writer reference (file-based, survives reload)
            snapshot.tensorboard_writer = algorithm.writer
            snapshot.tensorboard_dir = getattr(algorithm, 'tensorboard_dir', None)
            snapshot.tb_enabled = algorithm.tb_enabled
            snapshot.tb_log_frequency = getattr(algorithm, 'tb_log_frequency', 1)
            snapshot.tb_metrics = getattr(algorithm, 'tb_metrics', {})

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

            # Restore TD3 model weights
            if 'actor' in self.model_state_dicts:
                algorithm.actor.load_state_dict(self.model_state_dicts['actor'])
                algorithm.actor_target.load_state_dict(self.model_state_dicts['actor_target'])
                algorithm.critic.load_state_dict(self.model_state_dicts['critic'])
                algorithm.critic_target.load_state_dict(self.model_state_dicts['critic_target'])
                logger.debug("Restored TD3 network weights")

            # Restore optimizer state
            if self.optimizer_state_dict and hasattr(algorithm, 'optimizer'):
                algorithm.optimizer.load_state_dict(self.optimizer_state_dict)

            # Restore training progress
            if self.epsilon is not None:
                algorithm.epsilon = self.epsilon
            algorithm.training_step = self.training_step
            algorithm.episode_count = self.episode_count

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
            algorithm.is_converged = self.is_converged
            algorithm.convergence_episode = self.convergence_episode

            # Restore TensorBoard writer
            if self.tensorboard_writer:
                algorithm.writer = self.tensorboard_writer
                algorithm.tensorboard_dir = self.tensorboard_dir
                algorithm.tb_enabled = self.tb_enabled
                algorithm.tb_log_frequency = self.tb_log_frequency
                algorithm.tb_metrics = self.tb_metrics

        # Restore environment state
        marl_env.metrics = self.metrics
        marl_env.checkpoint_manager = self.checkpoint_manager
        marl_env.training_config = self.training_config

        logger.info(f"Training state restored: episode={self.episode_count}, "
                   f"training_step={self.training_step}, "
                   f"epsilon={self.epsilon}")


class WorldResetManager:
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
        Initialize WorldResetManager.

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

        logger.info(f"WorldResetManager initialized: "
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
                logger.error(f"World reset FAILED. Training may continue to slow down.")

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
        """Cleanup existing CARLA resources before reload."""
        sm = self.coordinator.scenario_manager

        # Cleanup agent manager (destroys all vehicles and sensors)
        if sm.agent_manager:
            sm.agent_manager.cleanup()
            logger.debug("Agent manager cleaned up")

        # Cleanup traffic manager
        if sm.traffic_manager:
            sm.traffic_manager.cleanup()
            logger.debug("Traffic manager cleaned up")

        # Destroy any remaining actors in the world
        try:
            world = self.coordinator.carla_world
            if world:
                actors = world.get_actors()

                # First destroy sensors (they are attached to vehicles)
                sensor_count = 0
                for actor in actors.filter('sensor.*'):
                    if actor.is_alive:
                        try:
                            actor.stop()  # Stop listening first
                            actor.destroy()
                            sensor_count += 1
                        except Exception as e:
                            logger.debug(f"Error destroying sensor {actor.id}: {e}")
                if sensor_count > 0:
                    logger.debug(f"Destroyed {sensor_count} remaining sensor actors")

                # Then destroy vehicles
                vehicle_count = 0
                for actor in actors.filter('vehicle.*'):
                    if actor.is_alive:
                        actor.destroy()
                        vehicle_count += 1
                if vehicle_count > 0:
                    logger.debug(f"Destroyed {vehicle_count} remaining vehicle actors")
        except Exception as e:
            logger.warning(f"Error during actor cleanup: {e}")

    def _reload_carla_world(self) -> bool:
        """
        Reload world using client.reload_world().

        This clears CARLA server-side memory and reloads the current map.

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self.coordinator.carla_client

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
        """Rebuild scenario manager components after world reload."""
        from opencda_marl.core.traffic import MARLTrafficManager
        from opencda_marl.core import MARLAgentManager
        from opencda_marl.envs import CarlaMonitor, CarlaSpectator

        sm = self.coordinator.scenario_manager

        # Update world reference in scenario manager
        sm.world = self.coordinator.carla_world
        sm.carla_map = sm.world.get_map()

        # Get config sections
        traffic_cfg = sm.params.get("traffic", {})
        agents_cfg = self.coordinator.config.get("agents", {})

        # Rebuild traffic manager with fresh world reference
        sm.traffic_manager = MARLTrafficManager(
            sm.world,
            traffic_cfg,
            sm.states,
            sm.fixed_dt
        )
        logger.debug("Traffic manager rebuilt")

        # Rebuild agent manager with fresh world reference
        sm.agent_manager = MARLAgentManager(
            agents_cfg,
            sm.states,
            sm.world,
            sm.cav_world
        )
        logger.debug("Agent manager rebuilt")

        # Update MARL environment references
        self.coordinator.marl_env.world = sm.world
        self.coordinator.marl_env.sm = sm

        # Rebuild CARLA monitor
        self.coordinator.carla_monitor = CarlaMonitor(
            sm.world,
            marl_tm=sm.traffic_manager
        )
        logger.debug("CARLA monitor rebuilt")

        # Update spectator
        spectator_config = self.coordinator.config.get('spectator', {})
        self.coordinator.carla_spectator = CarlaSpectator(
            sm.world,
            spectator_config
        )
        logger.debug("Spectator rebuilt")

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
