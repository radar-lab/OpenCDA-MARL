import numpy as np
from typing import Dict
from loguru import logger


class ObservationExtractor:
    """Extract MARL-relevant features from simulation observations"""

    def __init__(self, config: Dict, algorithm: str):
        self.config = config
        self.algorithm = algorithm
        
        if algorithm == 'q_learning':
            self.q_config = config.get('q_learning', {})
            self.state_features = self.q_config.get('state_features', {})
            self.state_dim = self._calculate_state_dim()
        elif algorithm == 'td3':
            self.td3_config = config.get('td3', {})
            self.features_config = self.td3_config.get('features', {})
            self.custom_features = self.features_config  # Direct access to features
            self.state_dim = self._calculate_custom_feature_dim()

    def extract(self, observations: Dict) -> Dict[int, np.ndarray]:
        """
        Extract state features for MARL from simulation observations.

        Args:
            observations: Dict from scenario_manager.get_observations()

        Returns:
            Dict[agent_id, state_vector] for MARL decision making
        """
        if self.algorithm == 'q_learning':
            return self._extract_discrete(observations)
        elif self.algorithm == 'td3':
            return self._extract_multi_agent(observations)
        else:
            return self._extract_continuous(observations)

    def _extract_continuous(self, observations: Dict) -> Dict[int, np.ndarray]:
        """Extract continuous state features for DQN and other deep RL algorithms (7D enhanced feature set)"""
        states = {}

        for agent_id, obs in observations.items():
            # New 7D feature set for enhanced intersection navigation
            state_features = []

            # 1-2. Relative position to intersection (x, y)
            if 'relative_position_to_intersection' in obs:
                state_features.extend([
                    obs['relative_position_to_intersection']['x'], 
                    obs['relative_position_to_intersection']['y']
                ])
            else:
                # Fallback to legacy location if available
                if 'location' in obs:
                    state_features.extend([obs['location']['x'], obs['location']['y']])
                else:
                    state_features.extend([0.0, 0.0])

            # 3. Speed (magnitude of velocity)
            if 'speed' in obs:
                state_features.append(obs['speed'])
            else:
                state_features.append(0.0)

            # 4. Heading angle (vehicle orientation)
            if 'heading_angle' in obs:
                state_features.append(obs['heading_angle'])
            else:
                state_features.append(0.0)

            # 5. Distance to intersection
            if 'distance_to_intersection' in obs:
                state_features.append(obs['distance_to_intersection'])
            else:
                state_features.append(100.0)  # Default large distance

            # 6. Distance to front vehicle
            if 'distance_to_front_vehicle' in obs:
                state_features.append(obs['distance_to_front_vehicle'])
            else:
                state_features.append(999.0)  # Default large distance

            # 7. Lane position (discrete but normalized to continuous)
            if 'lane_position' in obs:
                # Normalize lane position from [-3, 3] to [-1, 1]
                lane_pos = float(obs['lane_position']) / 3.0
                state_features.append(lane_pos)
            else:
                state_features.append(0.0)  # Unknown lane

            # Convert to numpy array (should be 7 features)
            states[agent_id] = np.array(state_features, dtype=np.float32)

            # Debug: Log feature dimensions occasionally
            if len(state_features) != 7:
                logger.warning(f"Agent {agent_id}: Expected 7 features, got {len(state_features)}: {state_features}")

        return states

    def _extract_multi_agent(self, observations: Dict) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract multi-agent observations for TD3 algorithm.
        
        Returns a structured format where each agent has access to all agents' states.
        
        Args:
            observations: Dict from scenario_manager.get_observations()
            
        Returns:
            Dict mapping agent_id to dict containing:
                - 'ego_state': Own custom feature vector
                - 'all_states': Dict of all agents' custom feature vectors
        """
        # Extract individual states using custom feature extractor
        individual_states = self._extract_custom_features(observations)
        
        # Restructure for multi-agent access
        multi_agent_obs = {}
        
        for ego_id, ego_state in individual_states.items():
            multi_agent_obs[str(ego_id)] = {
                'ego_state': ego_state,
                'all_states': {str(agent_id): state for agent_id, state in individual_states.items()}
            }
        
        return multi_agent_obs

    def _extract_custom_features(self, observations: Dict) -> Dict[int, np.ndarray]:
        """Extract custom configurable features for TD3 algorithm"""
        states = {}

        for agent_id, obs in observations.items():
            state_features = []

            # Extract features based on custom configuration
            for feature_name, feature_dim in self.custom_features.items():
                if feature_name == 'rel_x':
                    if 'relative_position_to_intersection' in obs:
                        state_features.append(obs['relative_position_to_intersection']['x'])
                    else:
                        state_features.append(0.0)
                
                elif feature_name == 'rel_y':
                    if 'relative_position_to_intersection' in obs:
                        state_features.append(obs['relative_position_to_intersection']['y'])
                    else:
                        state_features.append(0.0)
                
                elif feature_name == 'position_x':
                    if 'location' in obs:
                        state_features.append(obs['location']['x'])
                    else:
                        state_features.append(0.0)
                
                elif feature_name == 'position_y':
                    if 'location' in obs:
                        state_features.append(obs['location']['y'])
                    else:
                        state_features.append(0.0)
                
                elif feature_name == 'lane_position':
                    if 'lane_position' in obs:
                        # Use raw lane_position values (0,1,2,3)
                        # 0=at intersection, 1=left, 2=middle, 3=right
                        lane_pos = obs['lane_position']
                        state_features.append(float(lane_pos))
                    else:
                        state_features.append(2.0)  # Default to middle lane
                
                elif feature_name == 'lane_position_onehot':
                    # One-hot encoding of lane_position (4D)
                    if 'lane_position' in obs:
                        lane_pos = int(obs['lane_position'])
                        # Create one-hot: [at_intersection, left, middle, right]
                        onehot = [0.0, 0.0, 0.0, 0.0]
                        if 0 <= lane_pos <= 3:
                            onehot[lane_pos] = 1.0
                        state_features.extend(onehot)
                    else:
                        # Default to middle lane (index 2)
                        state_features.extend([0.0, 0.0, 1.0, 0.0])
                
                elif feature_name == 'heading_angle':
                    if 'heading_angle' in obs:
                        state_features.append(obs['heading_angle'])
                    else:
                        state_features.append(0.0)
                
                elif feature_name == 'dist_to_intersection':
                    if 'distance_to_intersection' in obs:
                        state_features.append(obs['distance_to_intersection'])
                    else:
                        state_features.append(100.0)  # Default large distance
                
                elif feature_name == 'dist_to_front_vehicle':
                    if 'distance_to_front_vehicle' in obs:
                        state_features.append(obs['distance_to_front_vehicle'])
                    else:
                        state_features.append(999.0)  # Default large distance
                
                elif feature_name == 'waypoint_buffer':
                    if 'waypoint_buffer_size' in obs:
                        # Use raw waypoint count (0-50 range)
                        waypoint_count = float(obs['waypoint_buffer_size'])
                        state_features.append(waypoint_count)
                    else:
                        # Fallback: estimate from distance to intersection
                        dist_to_int = obs.get('distance_to_intersection', 100.0)
                        # Rough estimate: ~2m per waypoint
                        estimated_waypoints = min(50.0, max(0.0, dist_to_int / 2.0))
                        state_features.append(estimated_waypoints)

                elif feature_name == 'nearby_vehicles':
                    # Nearby vehicle features: 35D = 5 vehicles × 7 features each
                    # Features per vehicle: rel_x, rel_y, rel_vx, rel_vy, heading_diff, distance, ttc
                    if 'nearby_vehicles' in obs:
                        nearby_features = obs['nearby_vehicles']
                        state_features.extend(nearby_features)
                    else:
                        # Default: 5 slots × 7 features with safe values
                        # Empty slots have zeros except distance=1.0, ttc=1.0 (safe/far away)
                        for _ in range(5):
                            state_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

                else:
                    logger.warning(f"Unknown custom feature: {feature_name}")
                    state_features.extend([0.0] * feature_dim)

            # Convert to numpy array
            states[agent_id] = np.array(state_features, dtype=np.float32)

            # Debug: Log feature dimensions occasionally  
            expected_dim = sum(self.custom_features.values())
            if len(state_features) != expected_dim:
                logger.warning(f"Agent {agent_id}: Expected {expected_dim} features, got {len(state_features)}: {state_features}")

        return states

    def _extract_discrete(self, observations: Dict) -> Dict[int, int]:
        """Extract and discretize states for Q-learning based on config"""
        states = {}

        for agent_id, obs in observations.items():
            state_indices = []

            # Process each configured feature
            for feature_name, settings in self.state_features.items():
                if feature_name in obs:
                    value = obs[feature_name]
                    bins = settings.get('bins', [])
                    
                    if bins:
                        # Use numpy.digitize to bin the value
                        bin_idx = np.digitize(value, bins)
                        state_indices.append(bin_idx)
                    else:
                        # No bins specified, treat as binary (0 or 1)
                        state_indices.append(1 if value > 0 else 0)
                else:
                    logger.warning(f"Feature '{feature_name}' not found in observations for agent {agent_id}")
                    logger.debug(f"Available features: {list(obs.keys())}")
                    # Default to 0 if feature not available
                    state_indices.append(0)

            # Convert multi-dimensional state indices to single state index
            single_state_idx = self._to_single_index(state_indices)
            states[agent_id] = single_state_idx

        return states

    def _to_single_index(self, state_indices: list) -> int:
        """Convert multi-dimensional state indices to single state index"""
        if not state_indices:
            return 0
            
        # Calculate single index using positional encoding
        single_idx = 0
        multiplier = 1
        
        for feature_name, settings in self.state_features.items():
            if state_indices:
                idx = state_indices.pop(0)
                bins = settings.get('bins', [])
                num_bins = len(bins) + 1 if bins else 2
                
                single_idx += idx * multiplier
                multiplier *= num_bins
        
        return single_idx

    def _calculate_state_dim(self) -> int:
        """Calculate total discrete state space size from feature bins"""
        if not self.state_features:
            return 1  # Fallback to single state
            
        state_dim = 1
        for feature, settings in self.state_features.items():
            bins = settings.get('bins', [])
            # Number of bins is len(bins) + 1 (for values beyond last bin)
            num_bins = len(bins) + 1 if bins else 2  # Default to 2 bins if none specified
            state_dim *= num_bins
            
        return state_dim

    def _calculate_custom_feature_dim(self) -> int:
        """Calculate total custom feature dimension from configuration"""
        if not self.custom_features:
            logger.warning("No custom features defined, falling back to default 7D features")
            return 7  # Fallback to continuous features
            
        total_dim = sum(self.custom_features.values())
        logger.info(f"Custom feature dimension: {total_dim}D from features: {list(self.custom_features.keys())}")
        return total_dim
