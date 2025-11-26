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

        # Initialize checkpoint manager if training
        if self.is_training_mode:
            checkpoint_dir = self.training_config.get(
                'checkpoint_dir', f'checkpoints/{algorithm}')
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir, algorithm)

            # Load checkpoint if specified
            load_checkpoint = self.training_config.get('load_checkpoint')
            if load_checkpoint:
                self._load_checkpoint_from_config(load_checkpoint)
        else:
            self.checkpoint_manager = None

    def step(self):
        # Get current observations from scenario manager
        observations = self.sm.get_observations()

        # MARL computes target speeds based on observations (pass training mode)
        target_speeds = self.marl_manager.compute_actions(
            observations, training=self.is_training_mode)

        # Execute scenario manager step with target speeds
        out = self.sm.step(target_speeds)

        # Calculate rewards from events
        self.events = out.get('event', [])
        rewards = self._calculate_rewards(self.events)

        # Get new observations for MARL learning
        next_observations = self.sm.get_observations()

        # Update MARL algorithm with CURRENT step's transition
        # Transition: (observations, action, reward, next_observations)
        # - observations: state before action (S_t)
        # - action: stored in last_actions during compute_actions
        # - reward: calculated for taking action from observations
        # - next_observations: state after action (S_t+1)
        if self.is_training_mode:
            self.marl_manager.update(
                rewards, observations, next_observations)  # Use current observations!

        # Store for metrics/debugging only
        self.previous_observations = observations.copy()

        # Store current step rewards for evaluation
        self.current_step_rewards = rewards.copy()

        self.episode_events.extend(self.events)

        # Update training metrics with observations for traffic performance tracking
        self.metrics.update_step(rewards, observations)

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

    def _calculate_rewards(self, events: List[StepEvent]) -> Dict[str, float]:
        """
        Calculate rewards for all agents based on events and current state.

        Args:
            events: List of event strings from scenario manager

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

        try:
            # Get current agents from scenario manager
            agents = self.sm.agents

            # Initialize rewards for all agents with step penalty + speed bonus
            for agent in agents:
                agent_id = agent.actor_id
                base_reward = step_penalty
                
                # Add speed bonus if agent is going fast enough
                if speed_bonus > 0:
                    try:
                        # Get agent's current speed in km/h
                        velocity = agent.vehicle.get_velocity()
                        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                        
                        if speed_kmh > speed_threshold:
                            base_reward += speed_bonus
                            logger.debug(f"Speed bonus applied to agent {agent_id}: {speed_kmh:.1f} km/h > {speed_threshold} km/h")
                            
                    except Exception as e:
                        logger.debug(f"Could not get speed for agent {agent_id}: {e}")
                
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
            "speed_threshold": 40.0   # km/h threshold for speed bonus
        }

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
        episode_states = {
            'max_steps': self.sm.states['max_steps'],
            'max_episodes': self.sm.states['max_episodes'],
            'step': self.sm.states['step'],
            'episode': current_episode,  # Current episode, not incremented yet
            'collision': self.sm.states['collision'],  # Current episode stats
            'success': self.sm.states['success'],  # Current episode stats
            'active_agents': self.sm.states['active_agents'],
            'fixed_dt': self.sm.states.get('fixed_dt', 0.05)
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

        logger.info(f"Episode {current_episode} reset completed")
        return episode_metrics
