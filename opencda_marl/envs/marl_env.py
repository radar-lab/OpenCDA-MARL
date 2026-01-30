'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 12:49:47
FilePath     : /OpenCDA-MARL/opencda_marl/envs/marl_env.py
Description  : MARL environment with modular evaluation system
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import math
from typing import Dict, List, Any
from loguru import logger

from opencda_marl.scenarios.scenario_manager import MARLScenarioManager
from opencda_marl.core.events import StepEvent
from opencda_marl.core.marl import MARLManager
from opencda_marl.core.marl.metrics import TrainingMetrics as Metrics
from opencda_marl.core.marl.checkpoint import CheckpointManager


class MARLEnv:

    def __init__(self, scenario_manager: MARLScenarioManager, config: Dict = {}):
        """
        Initialize MARL environment with reward calculation capabilities.
        """
        self.sm = scenario_manager
        self.config = config
        self.world = scenario_manager.world

        # Extract MARL specific config
        algorithm = self.config.get('algorithm', 'q_learning')
        
        # Check agent type first
        from opencda_marl.core.agents.agent_factory import AgentFactory
        agent_type = self.sm.agent_manager.agent_type
        
        # Use 'none' algorithm for baseline agents to skip MARL processing
        if agent_type in AgentFactory.get_baseline_types():
            algorithm = 'none'

        # MARL Manager for speed control
        self.marl_manager = MARLManager(self.config, algorithm)

        # RL-specific tracking
        self.events = None
        self.episode_events = []  # Track events for reward calculation
        self.previous_observations = {}  # Store for algorithm updates
        self.terminal_agents = set()  # Track agents that received terminal rewards
        self.current_step_rewards = {}  # Store current step rewards for evaluation

        # Near-miss tracking (TTC < threshold without collision)
        # Shows safety learning: decreasing near-misses = learning to avoid danger
        self.near_miss_count = 0  # Count per episode
        self.near_miss_agents = set()  # Track agents with near-miss this step (prevent double counting)

        # TTC violation tracking (TTC < safe_threshold)
        # Tracks % of agent-steps with dangerous TTC for paper metrics
        self.ttc_violation_count = 0  # Count of TTC violations per episode
        self.ttc_check_count = 0  # Total TTC checks per episode

        # Reward parameters
        self.reward_params = self._default_reward_params()
        self.reward_params.update(self.config.get('rewards', {}))

        # Training configuration
        self.training_config = self.config.get('training', {})

        # Training metrics (export history every N episodes)
        metrics_export_interval = self.training_config.get('metrics_export_interval', 10)
        metrics_export_dir = self.training_config.get('metrics_export_dir', 'metrics_history')
        self.metrics = Metrics(export_interval=metrics_export_interval, export_dir=metrics_export_dir)
        self.is_training_mode = self.training_config.get('training_mode', True)
        
        # Override training mode for baseline agents
        if agent_type in AgentFactory.get_baseline_types():
            self.is_training_mode = False
            logger.info(f"Baseline agent '{agent_type}' detected - disabling training mode")

        # Initialize checkpoint manager
        # Note: We need checkpoint manager even in evaluation mode to load trained weights
        checkpoint_dir = self.training_config.get(
            'checkpoint_dir', f'checkpoints/{algorithm}')
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir, algorithm) if self.is_training_mode else None

        # Load checkpoint if specified - works in BOTH training and evaluation modes
        # In evaluation mode (training_mode: false), this loads trained weights to test performance
        load_checkpoint = self.training_config.get('load_checkpoint')
        if load_checkpoint:
            self._load_checkpoint_from_config(load_checkpoint)
            if not self.is_training_mode:
                logger.info(f"Evaluation mode: Loaded checkpoint '{load_checkpoint}' for testing")

    def step(self):
        # Get current observations from scenario manager
        observations = self.sm.get_observations()

        # MARL computes target speeds based on observations (pass training mode)
        target_speeds = self.marl_manager.compute_actions(
            observations, training=self.is_training_mode)

        # Execute scenario manager step with target speeds
        out = self.sm.step(target_speeds)

        # Get new observations for reward calculation (includes min_ttc, distance_to_destination)
        next_observations = self.sm.get_observations()

        # Calculate rewards from events AND observations (Phase 3: includes TTC and progress rewards)
        self.events = out.get('event', [])
        rewards = self._calculate_rewards(self.events, observations=next_observations)

        # Update MARL algorithm with CURRENT step's transition
        # Transition: (observations, action, reward, next_observations)
        # - observations: state before action (S_t)
        # - action: stored in last_actions during compute_actions
        # - reward: calculated for taking action from observations
        # - next_observations: state after action (S_t+1)
        if self.is_training_mode:
            self.marl_manager.update(
                rewards, observations, next_observations)  # Use current observations!

        # Store current observations as previous for next step's progress calculation
        self.previous_observations = next_observations.copy()

        # Store current step rewards for evaluation
        self.current_step_rewards = rewards.copy()

        self.episode_events.extend(self.events)

        # Count successes this step for accurate episode_length tracking
        step_successes = sum(1 for e in self.events if e.event_type == "success")

        # Update training metrics with observations and RL-commanded target speeds
        # Pass target_speeds directly so we track what RL commanded (not adapter cached values)
        self.metrics.update_step(rewards, next_observations, target_speeds, step_successes)

    # --------------------------------------------------------------------- #
    # Public Methods
    # --------------------------------------------------------------------- #
    def get_current_events(self):
        return self.events.copy()

    def get_episode_metrics(self):
        return self.metrics.get_current_metrics()

    def get_reward_params(self):
        return self.reward_params

    def get_current_step_rewards(self):
        """Get rewards from the current step for evaluation."""
        return self.current_step_rewards.copy()

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information from MARL algorithm."""
        info = self.marl_manager.get_training_metrics()
        current_episode = self.sm.states.get('episode', 0)
        max_episodes = self.sm.states.get('max_episodes', 1)
        info.update({
            'episode_count': current_episode,
            'max_episodes': max_episodes,
            'current_metrics': self.metrics.get_current_metrics(),
            'training_mode': self.is_training_mode
        })
        return info

    # --------------------------------------------------------------------- #
    # Training control methods (centralized here)
    # --------------------------------------------------------------------- #
    def get_observations(self):
        """
        Get observations from scenario manager with RL enhancements.
        """
        try:
            base_observations = self.sm.get_observations()

            # Enhance observations with RL-specific data
            observations = {}

            for agent_id, obs in base_observations.items():
                enhanced_obs = obs.copy()

                # Add reward information
                enhanced_obs['current_reward'] = self.metrics.get_agent_reward(
                    agent_id)

                observations[agent_id] = enhanced_obs

            return observations

        except Exception as e:
            logger.error(f"Error getting enhanced observations: {e}")
            return self.sm.get_observations()  # Fallback to base observations

    # --------------------------------------------------------------------- #
    # Checkpoint methods
    # --------------------------------------------------------------------- #
    def _handle_training_episode_end(self, episode_metrics: Dict[str, Any], current_episode: int):
        """Handle end-of-episode training logic."""
        if not self.checkpoint_manager:
            return

        save_freq = self.training_config.get('save_freq', 10)

        # Always save checkpoint at episode end for training progress
        self._save_training_checkpoint(episode_metrics, current_episode)

        # Save best checkpoint if this episode has best reward
        episode_reward = episode_metrics.get('total_reward', 0.0)
        if hasattr(self.checkpoint_manager, 'best_reward'):
            if episode_reward > self.checkpoint_manager.best_reward:
                self.checkpoint_manager.best_reward = episode_reward
                self.checkpoint_manager.best_episode = current_episode
                self._save_best_checkpoint(episode_metrics, current_episode)

        # Cleanup old checkpoints periodically
        if current_episode % save_freq == 0:
            keep_checkpoints = self.training_config.get('keep_checkpoints', 5)
            self.checkpoint_manager.cleanup_old_checkpoints(keep_checkpoints)

    def _save_training_checkpoint(self, episode_metrics: Dict[str, Any], current_episode: int):
        """Save training checkpoint."""
        try:
            self.checkpoint_manager.save_checkpoint(
                self.marl_manager.algorithm,
                current_episode,
                episode_metrics
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _save_best_checkpoint(self, episode_metrics: Dict[str, Any], current_episode: int):
        """Save best checkpoint."""
        try:
            self.checkpoint_manager.save_checkpoint(
                self.marl_manager.algorithm,
                current_episode,
                episode_metrics,
                checkpoint_type="best"
            )
            logger.info(
                f"New best checkpoint saved at episode {current_episode} with reward {episode_metrics.get('total_reward', 0.0)}")
        except Exception as e:
            logger.error(f"Failed to save best checkpoint: {e}")

    def _load_checkpoint_from_config(self, checkpoint_path: str):
        """Load checkpoint specified in configuration."""
        try:
            if checkpoint_path:
                logger.debug(
                    f"Loading checkpoint from config: {checkpoint_path}")
                self.marl_manager.load_checkpoint(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to load checkpoint from config: {e}")

    # --------------------------------------------------------------------- #
    # Private Methods
    # --------------------------------------------------------------------- #

    def _calculate_rewards(self, events: List[StepEvent], observations: Dict = None) -> Dict[str, float]:
        """
        Calculate rewards for all agents based on events and current state.

        Includes Phase 3 enhancements:
        - TTC-based safety reward (proactive penalty for dangerous proximity)
        - Progress-toward-goal reward (dense feedback for movement toward destination)

        Args:
            events: List of event strings from scenario manager
            observations: Current observations dict (optional, for TTC/progress rewards)

        Returns:
            Dict mapping agent_id to reward value
        """
        rewards = {}

        step_penalty = self.reward_params.get("step_penalty", 0.0)
        collision_reward = self.reward_params.get("collision", 0.0)
        success_reward = self.reward_params.get("success", 0.0)

        # Get speed bonus parameters
        speed_bonus = self.reward_params.get("speed_bonus", 0.0)
        speed_threshold = self.reward_params.get("speed_threshold", 40.0)  # km/h

        # Get stop penalty parameters (Phase 3.1)
        stop_threshold = self.reward_params.get("stop_threshold", 5.0)  # km/h
        stop_penalty = self.reward_params.get("stop_penalty", -3.0)

        try:
            # Get current agents from scenario manager
            agents = self.sm.agents

            # Initialize rewards for all agents with step penalty + speed bonus
            for agent in agents:
                agent_id = agent.actor_id
                base_reward = step_penalty

                # Get agent's current speed in km/h
                speed_kmh = 0.0
                try:
                    velocity = agent.vehicle.get_velocity()
                    speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

                    # Add speed bonus if agent is going fast enough
                    if speed_bonus > 0 and speed_kmh > speed_threshold:
                        base_reward += speed_bonus
                        logger.debug(f"Speed bonus applied to agent {agent_id}: {speed_kmh:.1f} km/h > {speed_threshold} km/h")

                    # Add stop penalty if agent has nearly stopped (Phase 3.1)
                    # Encourages gradual deceleration over hard stops
                    if speed_kmh < stop_threshold:
                        base_reward += stop_penalty
                        logger.debug(f"Stop penalty applied to agent {agent_id}: {speed_kmh:.1f} km/h < {stop_threshold} km/h")

                except Exception as e:
                    logger.debug(f"Could not get speed for agent {agent_id}: {e}")

                # Phase 3: TTC-based safety reward - DISABLED
                # TTC penalty was causing overly conservative behavior
                # Collision penalty (-500) is sufficient safety signal
                # Keep TTC tracking for metrics only
                if observations and agent_id in observations:
                    obs = observations[agent_id]
                    min_ttc = obs.get('min_ttc', float('inf'))

                    # Track TTC metrics (for paper/analysis) without affecting reward
                    self._track_ttc_metrics(min_ttc, agent_id)

                    # Clearance speed bonus: reward faster speeds when path is clear
                    # This teaches: "go fast when safe, collision penalty teaches when to slow"
                    clearance_speed_bonus = self.reward_params.get('clearance_speed_bonus', 0.3)
                    clearance_threshold = self.reward_params.get('clearance_threshold', 30.0)
                    dist_to_front = obs.get('distance_to_front_vehicle', float('inf'))

                    if dist_to_front > clearance_threshold or dist_to_front == float('inf'):
                        if speed_kmh > 20.0:  # Above minimum useful speed
                            speed_ratio = min(speed_kmh / 65.0, 1.0)
                            clearance_bonus = clearance_speed_bonus * speed_ratio
                            base_reward += clearance_bonus
                            logger.debug(f"Clearance bonus to agent {agent_id}: speed={speed_kmh:.1f}, dist_front={dist_to_front:.1f}m, bonus={clearance_bonus:.3f}")

                    # Yielding bonus: reward for slowing down when nearby vehicle has low TTC
                    # This encourages cooperative behavior - yield to let others pass safely
                    yielding_bonus = self.reward_params.get('yielding_bonus', 0.0)
                    if yielding_bonus > 0 and agent_id in self.previous_observations:
                        yielding_ttc_threshold = self.reward_params.get('yielding_ttc_threshold', 3.0)
                        yielding_speed_drop = self.reward_params.get('yielding_speed_drop', 5.0)

                        # Check if there's a nearby vehicle with low TTC
                        if min_ttc < yielding_ttc_threshold and min_ttc != float('inf'):
                            # Check if ego vehicle is slowing down (yielding behavior)
                            prev_speed = self.previous_observations[agent_id].get('speed', 0.0)
                            if prev_speed - speed_kmh >= yielding_speed_drop:
                                base_reward += yielding_bonus
                                logger.debug(f"Yielding bonus to agent {agent_id}: slowed {prev_speed:.1f} -> {speed_kmh:.1f} km/h, TTC={min_ttc:.2f}s")

                    # Phase 3: Add progress reward (requires previous observations)
                    if agent_id in self.previous_observations:
                        prev_obs = self.previous_observations[agent_id]
                        progress_reward = self._calculate_progress_reward(
                            current_dist_to_dest=obs.get('distance_to_destination', 999.0),
                            prev_dist_to_dest=prev_obs.get('distance_to_destination', 999.0),
                            current_dist_to_int=obs.get('distance_to_intersection', 100.0),
                            prev_dist_to_int=prev_obs.get('distance_to_intersection', 100.0)
                        )
                        base_reward += progress_reward

                rewards[agent_id] = base_reward

            # Process events for rewards (prevent duplicate terminal rewards)
            for event in events:
                agent_id = event.vehicle_id

                if agent_id not in rewards:
                    # Apply same logic for agents not yet in rewards
                    base_reward = step_penalty
                    if speed_bonus > 0:
                        try:
                            # Find the agent by ID
                            agent = next((a for a in agents if a.actor_id == agent_id), None)
                            if agent:
                                velocity = agent.vehicle.get_velocity()
                                speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                                if speed_kmh > speed_threshold:
                                    base_reward += speed_bonus
                        except Exception:
                            pass  # Use base reward if speed calculation fails
                    rewards[agent_id] = base_reward

                if event.event_type == "collision" and agent_id not in self.terminal_agents:
                    rewards[agent_id] = collision_reward
                    self.terminal_agents.add(agent_id)
                    logger.warning(
                        f"Collision reward applied to agent {agent_id}")
                elif event.event_type == "success" and agent_id not in self.terminal_agents:
                    rewards[agent_id] = success_reward
                    self.terminal_agents.add(agent_id)
                    logger.info(
                        f"Success reward applied to agent {agent_id}")

        except Exception as e:
            logger.error(f"Error calculating rewards: {e}")
            raise e

        return rewards

    def _default_reward_params(self):
        return {
            "collision": -120.0,
            "success": 120.0,
            "step_penalty": -0.5,
            "speed_bonus": 0.0,      # Bonus for maintaining speed above threshold
            "speed_threshold": 40.0,  # km/h threshold for speed bonus
            # TTC-based safety reward parameters (Phase 3)
            "ttc_safe_threshold": 3.0,      # seconds - no penalty above this
            "ttc_caution_threshold": 2.0,   # seconds - caution zone
            "ttc_danger_threshold": 1.0,    # seconds - danger zone
            "ttc_caution_penalty": -0.5,    # penalty in caution zone
            "ttc_danger_penalty": -2.0,     # penalty in danger zone
            "ttc_critical_penalty": -5.0,   # penalty in critical zone (< 1s)
            # Progress reward parameters (Phase 3)
            "progress_scale": 0.5,          # scale factor for progress reward
            "junction_threshold": 5.0,      # meters - threshold for "in junction"
            # Stop penalty parameters (Phase 3.1)
            "stop_threshold": 5.0,          # km/h - below this speed is considered "stopped"
            "stop_penalty": -3.0            # penalty for stopping (encourages gradual deceleration)
        }

    # --------------------------------------------------------------------- #
    # TTC and Progress Reward Methods (Phase 3 Enhancement)
    # --------------------------------------------------------------------- #

    def _calculate_ttc_reward(self, min_ttc: float, agent_id: int = None) -> float:
        """
        Calculate TTC-based safety reward using smooth exponential penalty.

        Uses smooth exponential function instead of hard thresholds for better
        gradient-based learning. Penalty increases smoothly as TTC decreases.

        Formula: penalty = -max_penalty * exp(-decay_rate * ttc)

        Args:
            min_ttc: Minimum time-to-collision across all nearby vehicles (seconds)
            agent_id: Agent ID for near-miss tracking (optional)

        Returns:
            float: TTC reward (0.0 for safe, negative for dangerous situations)
        """
        safe_threshold = self.reward_params.get('ttc_safe_threshold', 4.0)
        max_penalty = self.reward_params.get('ttc_max_penalty', 10.0)
        decay_rate = self.reward_params.get('ttc_decay_rate', 1.5)
        near_miss_threshold = self.reward_params.get('ttc_near_miss_threshold', 2.0)

        # Track TTC checks for violation rate calculation
        self.ttc_check_count += 1

        # Safe - no penalty (above threshold or no nearby vehicles)
        if min_ttc > safe_threshold or min_ttc == float('inf'):
            return 0.0

        # Track any TTC below safe threshold as a violation (for paper metrics)
        self.ttc_violation_count += 1

        # Smooth exponential penalty: increases as TTC decreases
        # At TTC=4.0s: penalty ≈ -0.02 (near zero)
        # At TTC=2.0s: penalty ≈ -0.5
        # At TTC=1.0s: penalty ≈ -2.2
        # At TTC=0.5s: penalty ≈ -4.7
        # At TTC=0.0s: penalty = -10.0
        penalty = -max_penalty * math.exp(-decay_rate * min_ttc)

        # Near-miss tracking (for metrics) - count when TTC drops below threshold
        if min_ttc < near_miss_threshold and agent_id is not None:
            if agent_id not in self.near_miss_agents:
                self.near_miss_count += 1
                self.near_miss_agents.add(agent_id)
                logger.debug(f"Near-miss: agent {agent_id}, TTC={min_ttc:.2f}s, penalty={penalty:.2f}")

        return penalty

    def _track_ttc_metrics(self, min_ttc: float, agent_id: int = None) -> None:
        """
        Track TTC metrics without affecting reward (for paper/analysis).

        This is called after TTC reward was disabled to maintain metric tracking
        for violation rates and near-miss counts.

        Args:
            min_ttc: Minimum time-to-collision across all nearby vehicles (seconds)
            agent_id: Agent ID for near-miss tracking (optional)
        """
        safe_threshold = self.reward_params.get('ttc_safe_threshold', 4.0)
        near_miss_threshold = self.reward_params.get('ttc_near_miss_threshold', 2.0)

        # Track TTC checks for violation rate calculation
        self.ttc_check_count += 1

        # Track any TTC below safe threshold as a violation (for paper metrics)
        if min_ttc < safe_threshold and min_ttc != float('inf'):
            self.ttc_violation_count += 1

        # Near-miss tracking (for metrics)
        if min_ttc < near_miss_threshold and agent_id is not None:
            if agent_id not in self.near_miss_agents:
                self.near_miss_count += 1
                self.near_miss_agents.add(agent_id)
                logger.debug(f"Near-miss tracked: agent {agent_id}, TTC={min_ttc:.2f}s")

    def _calculate_progress_reward(self, current_dist_to_dest: float, prev_dist_to_dest: float,
                                    current_dist_to_int: float, prev_dist_to_int: float) -> float:
        """
        Calculate progress-toward-goal reward.

        Two-phase approach:
        1. Before intersection: reward reducing distance to intersection
        2. In/past junction: reward reducing distance to destination

        Args:
            current_dist_to_dest: Current distance to destination (meters)
            prev_dist_to_dest: Previous distance to destination (meters)
            current_dist_to_int: Current distance to intersection (meters)
            prev_dist_to_int: Previous distance to intersection (meters)

        Returns:
            float: Progress reward (positive for progress, negative for regression)
        """
        junction_threshold = self.reward_params.get('junction_threshold', 5.0)
        in_junction = current_dist_to_int < junction_threshold

        if not in_junction:
            # Approaching intersection phase
            progress = prev_dist_to_int - current_dist_to_int
        else:
            # In/past junction: use destination distance
            progress = prev_dist_to_dest - current_dist_to_dest

        # Normalize: typical step progress ~2-4m at 40-80 km/h
        # Cap at ±5m of progress per step
        import numpy as np
        progress_reward = np.clip(progress / 5.0, -1.0, 1.0)
        scale = self.reward_params.get('progress_scale', 0.5)

        return progress_reward * scale

    # --------------------------------------------------------------------- #
    # Clean Up
    # --------------------------------------------------------------------- #
    def reset_episode(self):
        """
        Reset environment for new episode and handle training logic.
        """
        # Create episode-specific state snapshot BEFORE any resets
        # This captures the actual current episode stats
        current_episode = self.sm.states.get('episode', 0)

        # Calculate TTC violation rate (% of TTC checks that were violations)
        ttc_violation_rate = 0.0
        if self.ttc_check_count > 0:
            ttc_violation_rate = (self.ttc_violation_count / self.ttc_check_count) * 100.0

        episode_states = {
            'max_steps': self.sm.states['max_steps'],
            'max_episodes': self.sm.states['max_episodes'],
            'step': self.sm.states['step'],
            'episode': current_episode,  # Current episode, not incremented yet
            'collision': self.sm.states['collision'],  # Current episode stats
            'success': self.sm.states['success'],  # Current episode stats
            'active_agents': self.sm.states['active_agents'],
            'fixed_dt': self.sm.states.get('fixed_dt', 0.05),
            # Near-miss tracking for learning analysis
            'near_miss_count': self.near_miss_count,
            # TTC violation rate for paper metrics (% of checks with TTC < safe threshold)
            'ttc_violation_rate': ttc_violation_rate
        }

        # Finish current episode metrics with clean episode-specific snapshot
        episode_metrics = self.metrics.finish_episode(episode_states)

        # Handle training-specific logic with correct episode number
        if self.is_training_mode:
            self._handle_training_episode_end(episode_metrics, current_episode)

        # Reset MARL algorithm and log episode metrics to TensorBoard
        self.marl_manager.reset_episode(episode_metrics=episode_metrics)

        # Reset environment state
        self.events = None
        self.episode_events.clear()
        self.previous_observations.clear()
        self.terminal_agents = set()
        self.current_step_rewards.clear()

        # Reset near-miss tracking for new episode
        self.near_miss_count = 0
        self.near_miss_agents.clear()

        # Reset TTC violation tracking for new episode
        self.ttc_violation_count = 0
        self.ttc_check_count = 0

        logger.info(f"Episode {current_episode} reset completed")
        return episode_metrics
