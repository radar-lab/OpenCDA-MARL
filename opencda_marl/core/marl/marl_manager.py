'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 17:58:12
FilePath     : /OpenCDA-MARL/opencda_marl/core/marl/marl_manager.py
Description  : MARL Manager for speed control of autonomous vehicles
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from typing import Dict, Any
from loguru import logger

from .extractor import ObservationExtractor
from .algorithms import QLearningAlgorithm, DQNAlgorithm, TD3Algorithm


class MARLManager:
    """Manages RL algorithms and speed control decisions"""

    def __init__(self, config: Dict, algorithm: str):
        self.config = config
        self.algorithm_name = algorithm
        self.algorithm = self.build_algorithm(algorithm)
        self.observation_extractor = ObservationExtractor(config, algorithm)
        
        # Track last actions for Q-learning updates
        self.last_actions = {}  # Dict[agent_id, action_index]
        
        logger.success(f"MARLManager initialized with {algorithm}")

    def build_algorithm(self, algorithm: str):
        """Build algorithm instance based on configuration"""

        if algorithm == 'q_learning':
            # Get Q-learning specific config
            q_config = self.config.get('q_learning', {})

            # Calculate action_dim from speed actions (algorithm-specific)
            speed_actions = q_config.get('speed_actions', [0, 15, 30, 45, 60])
            action_dim = len(speed_actions)

            # Calculate state dimension from features
            state_features = q_config.get('state_features', {})
            state_dim = self._calculate_state_dim(state_features)

            return QLearningAlgorithm(q_config, state_dim, action_dim)

        elif algorithm == 'dqn':
            # Get DQN specific config
            dqn_config = self.config.get('dqn', {})
            
            # Calculate action_dim from speed actions (algorithm-specific)
            speed_actions = dqn_config.get('speed_actions', [0, 15, 30, 45, 60])
            action_dim = len(speed_actions)

            # DQN uses continuous states - get from config
            state_dim = self.config.get('state_dim', 7)  # Use configured state dimension

            return DQNAlgorithm(dqn_config, state_dim, action_dim)

        elif algorithm == 'td3':
            # Get TD3 specific config
            td3_config = self.config.get('td3', {})
            
            # TD3 uses continuous states
            state_dim = self.config.get('state_dim', 7)  # Use configured state dimension
            action_dim = self.config.get('action_dim', 1)
            return TD3Algorithm(td3_config, state_dim, action_dim)

        elif algorithm == 'ppo':
            # Placeholder - implement PPO later
            raise NotImplementedError("PPO not implemented yet")
        elif algorithm == 'sac':
            # Placeholder - implement SAC later
            raise NotImplementedError("SAC not implemented yet")
        elif algorithm == 'none':
            # No MARL algorithm - for baseline agents
            logger.info("No MARL algorithm initialized (baseline agent mode)")
            return None
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")

    # --------------------------------------------------------------------- #
    # Core MARL methods
    # --------------------------------------------------------------------- #
    def compute_actions(self, observations: Dict, training: bool = True) -> Dict[int, float]:
        """
        Extract features and compute target speeds for agents.

        Args:
            observations: Dict from scenario_manager.get_observations()
            training: Whether in training mode (affects exploration)

        Returns:
            Dict[agent_id, target_speed] for speed control
        """
        try:
            # Handle baseline agents (no MARL algorithm)
            if self.algorithm is None:
                # Return empty dict - baseline agents control their own speed
                return {}
            
            # Extract MARL-relevant features
            states = self.observation_extractor.extract(observations)

            # Compute actions (speeds) for each agent
            target_speeds = {}
            
            # Handle TD3 differently due to multi-agent structure
            if isinstance(self.algorithm, TD3Algorithm):
                for agent_id_str, multi_agent_data in states.items():
                    # Keep agent_id as-is (can be string in SUMO or int in CARLA)
                    agent_id = agent_id_str

                    # Extract all agent states for TD3
                    all_agent_states = multi_agent_data['all_states']
                    action = self.algorithm.select_action(all_agent_states, agent_id, training=training)
                    speed = self._compute_td3_action(action, agent_id)
                    
                    # If speed is None (warmup phase), skip this agent to use vanilla agent
                    if speed is None:
                        logger.debug(f"TD3: Agent {agent_id} in warmup phase, using vanilla agent")
                        # Track vanilla agent's current speed for memory storage
                        if agent_id in observations:
                            vanilla_speed = observations[agent_id].get('speed', 45.0)
                            self.last_actions[agent_id] = vanilla_speed  # Track for TD3 learning
                        continue  # Don't add to target_speeds, let vanilla agent handle it

                    # Clamp speed to TD3 action bounds
                    max_action = self.algorithm.max_action
                    clamped_speed = max(0.0, min(max_action, speed))
                    target_speeds[agent_id] = clamped_speed

                    # Log the target speed assignment
                    logger.debug(f"TD3: Agent {agent_id} target speed set to {clamped_speed:.2f} km/h")
            else:
                # Handle single-agent algorithms (Q-learning, DQN)
                for agent_id, state in states.items():
                    action = self.algorithm.select_action(state, training=training)

                    # Convert action to speed based on algorithm type
                    if isinstance(self.algorithm, QLearningAlgorithm):
                        speed = self._compute_q_learning_action(action, agent_id)
                    elif isinstance(self.algorithm, DQNAlgorithm):
                        speed = self._compute_dqn_action(action, agent_id)
                    else:
                        # Default fallback
                        speed = float(action) if isinstance(
                            action, (int, float)) else 8.0

                    # Clamp speed to maximum allowed
                    speed_actions = self._get_speed_actions()
                    max_speed = max(speed_actions) if speed_actions else 15.0
                    target_speeds[agent_id] = max(0.0, min(max_speed, speed))

            return target_speeds

        except Exception as e:
            logger.error(f"Error computing MARL actions: {e}")
            # Return empty dict to use default agent speeds
            return {}

    def update(self, rewards: Dict, observations: Dict, next_observations: Dict):
        """
        Update RL algorithm with experience.

        Args:
            rewards: Dict[agent_id, reward]
            observations: Current observations
            next_observations: Next step observations
        """
        try:
            # Skip updates for baseline agents (no algorithm)
            if self.algorithm is None:
                return
                
            if isinstance(self.algorithm, QLearningAlgorithm):
                self._update_q_learning(rewards, observations, next_observations)
            elif isinstance(self.algorithm, DQNAlgorithm):
                self._update_dqn(rewards, observations, next_observations)
            elif isinstance(self.algorithm, TD3Algorithm):
                self._update_td3(rewards, observations, next_observations)
            
        except Exception as e:
            logger.error(f"Error updating MARL algorithm: {e}")
            import traceback
            traceback.print_exc()
    
    # --------------------------------------------------------------------- #
    # Q-Learning specific methods
    # --------------------------------------------------------------------- #
    def _compute_q_learning_action(self, action, agent_id: int) -> float:
        """Compute speed from Q-learning action index."""
        speed_actions = self.config.get('q_learning', {}).get(
            'speed_actions', [0, 5, 10, 15])
        
        # Convert numpy.int64 to Python int for ListConfig indexing
        action_idx = int(action)
        speed = speed_actions[min(action_idx, len(speed_actions)-1)]
        
        # Store the action for Q-learning updates
        self.last_actions[agent_id] = action_idx
        
        return float(speed)

    def _update_q_learning(self, rewards: Dict, observations: Dict, next_observations: Dict):
        """Q-learning specific update logic."""
        states = self.observation_extractor.extract(observations)
        next_states = self.observation_extractor.extract(next_observations)
        
        # Store transitions for each agent
        for agent_id in states.keys():
            if (agent_id in rewards and 
                agent_id in self.last_actions):
                
                state = states[agent_id]
                action = self.last_actions[agent_id]
                reward = rewards[agent_id]
                
                # Check if agent is still active (not done)
                if agent_id in next_states:
                    next_state = next_states[agent_id]
                    done = False
                else:
                    # Agent removed (collision/success), use current state as next_state
                    next_state = state
                    done = True
                
                self.algorithm.store_transition(state, action, reward, next_state, done)
        
        # Call update without arguments for Q-learning
        self.algorithm.update()

    def _calculate_state_dim(self, state_features: Dict) -> int:
        """Calculate total discrete state space size from feature bins"""
        if not state_features:
            return 1  # Fallback to single state

        state_dim = 1
        for feature, settings in state_features.items():
            bins = settings.get('bins', [])
            # Number of bins is len(bins) + 1 (for values beyond last bin)
            num_bins = len(bins) + 1 if bins else 2
            state_dim *= num_bins

        logger.info(
            f"Calculated state_dim: {state_dim} from features: {list(state_features.keys())}")
        return state_dim

    # --------------------------------------------------------------------- #
    # DQN specific methods
    # --------------------------------------------------------------------- #
    def _compute_dqn_action(self, action, agent_id: int) -> float:
        """Compute speed from DQN action (direct speed value)."""
        # DQN returns target speed directly
        speed = float(action)
        
        # Store the action for DQN updates
        self.last_actions[agent_id] = speed
        
        return speed

    def _update_dqn(self, rewards: Dict, observations: Dict, next_observations: Dict):
        """DQN specific update logic."""
        states = self.observation_extractor.extract(observations)
        next_states = self.observation_extractor.extract(next_observations)
        
        for agent_id in states.keys():
            if (agent_id in rewards and 
                agent_id in next_states and 
                agent_id in self.last_actions):
                
                state = states[agent_id]
                action = self.last_actions[agent_id]  # For DQN this is the speed value
                reward = rewards[agent_id]
                next_state = next_states[agent_id]
                done = agent_id not in next_states
                
                self.algorithm.store_transition(state, action, reward, next_state, done)
        
        self.algorithm.update()

    # --------------------------------------------------------------------- #
    # TD3 specific methods
    # --------------------------------------------------------------------- #
    def _compute_td3_action(self, action, agent_id: int):
        """Compute speed from TD3 action (direct continuous speed value)."""
        # During warmup, TD3 returns None to use vanilla agent
        if action is None:
            return None  # Signal to use vanilla agent speed
            
        # TD3 returns target speed directly (continuous action)
        speed = float(action)
        
        # Store the action for TD3 updates
        self.last_actions[agent_id] = speed
        
        return speed

    def _update_td3(self, rewards: Dict, observations: Dict, next_observations: Dict):
        """TD3 specific update logic."""
        states = self.observation_extractor.extract(observations)
        next_states = self.observation_extractor.extract(next_observations)
        
        # Store transitions for each agent
        for agent_id_str in states.keys():
            # Use agent_id_str directly (can be string in SUMO or int in CARLA)
            agent_id = agent_id_str

            if (agent_id in rewards and
                agent_id in next_states and
                agent_id in self.last_actions):

                # Get multi-agent observations for TD3
                multi_agent_obs = states[agent_id]['all_states']
                next_multi_agent_obs = next_states[agent_id]['all_states']

                action = self.last_actions[agent_id]
                reward = rewards[agent_id]
                done = agent_id not in next_states

                # Store multi-agent transition
                self.algorithm.store_transition(
                    multi_agent_obs, agent_id, action, reward,
                    next_multi_agent_obs, done
                )
        
        self.algorithm.update()

    # --------------------------------------------------------------------- #
    # Algorithm interface methods (strict implementation required)
    # --------------------------------------------------------------------- #
    def reset_episode(self):
        """Reset algorithm for new episode."""
        if self.algorithm is None:
            logger.info("Episode reset skipped for baseline agent")
            return
        self.algorithm.reset_episode()
        logger.info(f"Episode reset for {self.algorithm_name}")

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics from algorithm."""
        if self.algorithm is None:
            return {"algorithm_type": "none", "epsilon": "N/A", "training_mode": False}
        return self.algorithm.get_training_info()

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        if self.algorithm is None:
            logger.warning("Cannot save checkpoint for baseline agent")
            return
        self.algorithm.save(filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        if self.algorithm is None:
            logger.warning("Cannot load checkpoint for baseline agent")
            return
        self.algorithm.load(filepath)

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _get_speed_actions(self) -> list:
        """Get speed actions based on algorithm type."""
        if self.algorithm_name == 'q_learning':
            return self.config.get('q_learning', {}).get(
                'speed_actions', [0, 5, 10, 15])
        elif self.algorithm_name == 'dqn':
            return self.config.get('dqn', {}).get(
                'speed_actions', [0, 5, 10, 15])
        else:
            return [0, 5, 10, 15]

    def _action_to_speed(self, action) -> float:
        """Convert algorithm action to target speed"""
        # For now, assume action is already a speed value
        return float(action)
