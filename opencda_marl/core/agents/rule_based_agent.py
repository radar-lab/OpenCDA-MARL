'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-03
FilePath     : /OpenCDA-MARL/opencda_marl/core/agents/rule_based_agent.py
Description  : Rule-Based Agent implementing three-stage hierarchical control

This agent implements the three-stage rule-based control system enhanced with
Junction-Aware Traffic Management from JAIR paper:
"A rule-based cooperative merging strategy" + "Junction-aware traffic management"
Using trajectory prediction and time-to-collision for better intersection handling

Three-stage hierarchy:
1. Junction Management (Highest Priority) - FCFS conflict resolution
2. Car Following - Safe distance maintenance  
3. Cruising - Default free-flow behavior

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import math
import carla
import itertools
from typing import Dict, List, Optional, Tuple
from loguru import logger

from opencda_marl.core.agents.vanilla_agent import VanillaAgent


class RuleBasedAgent(VanillaAgent):
    """
    Rule-based agent implementing three-stage hierarchical control for intersection management.
    
    Based on the paper's three-stage algorithm:
    1. Junction Management: Detect conflicts and apply cautious behavior
    2. Car Following: Maintain safe following distance
    3. Cruising: Default maximum speed behavior
    """
    
    def __init__(self, vehicle, carla_map, config_yaml):
        """
        Initialize rule-based agent with three-stage control parameters.
        
        Args:
            vehicle: CARLA vehicle actor
            carla_map: CARLA HD map
            config_yaml: Configuration containing rule_based parameters
        """
        # Initialize VanillaAgent base
        super().__init__(vehicle, carla_map, config_yaml)
        
        # Stage 1: Junction Management parameters (JAIR enhanced)
        self.junction_approach_distance = config_yaml.get('junction_approach_distance', 70.0)  # meters
        self.junction_conflict_distance = config_yaml.get('junction_conflict_distance', 50.0)   # meters
        self.cautious_speed = config_yaml.get('cautious_speed', 35.0)  # km/h
        
        # Trajectory-based conflict detection (replaces heading difference)
        self.trajectory_lookahead_time = config_yaml.get('trajectory_lookahead_time', 3.0)  # seconds
        self.ttc_threshold = config_yaml.get('ttc_threshold', 2.0)  # seconds - min TTC difference for conflict
        self.trajectory_step_size = config_yaml.get('trajectory_step_size', 1.0)  # meters
        
        # Stage 2: Car Following parameters
        self.time_headway = config_yaml.get('time_headway', 2.0)  # seconds
        self.following_gain = config_yaml.get('following_gain', 0.5)  # control gain
        self.minimum_distance_buffer = config_yaml.get('minimum_distance_buffer', 5.0)  # meters
        self.same_lane_tolerance = math.radians(config_yaml.get('same_lane_tolerance_deg', 30))  # radians
        self.front_cone_angle = math.radians(config_yaml.get('front_cone_angle_deg', 45))  # radians
        
        # Speed control parameters - Dynamic adjustment factors instead of fixed speeds
        self.junction_speed_factor = config_yaml.get('junction_speed_factor', 0.6)  # Reduce to 60% at junctions
        self.junction_min_reduction = config_yaml.get('junction_min_reduction', 15.0)  # Minimum reduction (km/h)
        self.following_speed_buffer = config_yaml.get('following_speed_buffer', 5.0)  # Buffer below leader (km/h)
        
        # Overall speed control
        self.max_speed = config_yaml.get('max_speed', 65.0)  # Absolute speed limit - NEVER exceed
        self.cruising_speed_factor = config_yaml.get('cruising_speed_factor', 0.95)  # 95% of vanilla when cruising
        
        # Debug and logging
        self.debug_rules = config_yaml.get('debug_rules', False)
        self.current_stage = 0  # For debugging: 1=junction, 2=following, 3=cruising
        
        logger.debug(f"RuleBasedAgent initialized for vehicle {vehicle.id} with three-stage control")

    # --------------------------------------------------------------------- #
    # Main Step Function
    # --------------------------------------------------------------------- #    
    def run_step(self, target_speed=None):
        """
        Execute one step of rule-based navigation.
        Uses VanillaAgent for route following and applies three-stage rules for speed control only.
        
        Returns:
            Tuple[float, carla.Location]: (rule_based_speed, vanilla_location)
        """
        # Check if agent is properly initialized
        if not self._ego_pos:
            if self.debug_rules:
                logger.warning("RuleBasedAgent: ego position not set")
            return 0.0, None
        
        # Check destination
        if self.is_close_to_destination():
            raise StopIteration("Destination reached - simulation complete")
        
        try:
            # Let parent VanillaAgent handle route navigation
            vanilla_speed, vanilla_location = super().run_step(target_speed)
            
            # Apply three-stage rule-based speed control using vanilla_speed as base
            rule_based_speed = self._calculate_rule_based_speed(vanilla_speed)
            
            # Return rule-based speed with VanillaAgent's navigation
            if self.debug_rules:
                stage_name = {1: 'Junction', 2: 'Following', 3: 'Cruising'}.get(self.current_stage, 'Unknown')
                logger.info(f"Vehicle {self._vehicle.id}: {stage_name} - Speed: {rule_based_speed:.1f} km/h")
            
            return rule_based_speed, vanilla_location
            
        except Exception as e:
            logger.error(f"RuleBasedAgent error for vehicle {self._vehicle.id}: {e}")
            # Fallback to parent behavior
            try:
                return super().run_step(target_speed)
            except Exception:
                return self.cautious_speed, self._ego_pos.location if self._ego_pos else None
    

    # --------------------------------------------------------------------- #
    # Private Functions
    # --------------------------------------------------------------------- #

    def _calculate_rule_based_speed(self, vanilla_speed: float) -> float:
        """
        Calculate speed based on three-stage rule system using vanilla_speed as base.
        
        Args:
            vanilla_speed: Base speed from VanillaAgent (km/h)
        
        Returns:
            float: Target speed in km/h based on current stage
        """
        try:
            calculated_speed = 0.0
            
            # Stage 1: Junction Management (Highest Priority)
            if self._should_apply_junction_management():
                self.current_stage = 1
                # Dynamic adjustment: reduce speed by factor or minimum reduction, whichever is higher
                factor_reduction = vanilla_speed * self.junction_speed_factor
                fixed_reduction = max(0, vanilla_speed - self.junction_min_reduction)
                calculated_speed = max(factor_reduction, fixed_reduction)
            
            # Stage 2: Car Following (Medium Priority)
            elif (following_speed := self._calculate_car_following_speed(vanilla_speed)) is not None:
                self.current_stage = 2
                calculated_speed = following_speed
            
            # Stage 3: Cruising (Default - with adjustment)
            else:
                self.current_stage = 3
                # Apply cruising factor to vanilla_speed instead of using it directly
                calculated_speed = vanilla_speed * self.cruising_speed_factor
                
            # FINAL CHECK: Enforce max_speed limit on all stages
            final_speed = min(calculated_speed, self.max_speed)
            
            if self.debug_rules and calculated_speed > self.max_speed:
                stage_name = {1: 'Junction', 2: 'Following', 3: 'Cruising'}.get(self.current_stage, 'Unknown')
                logger.info(f"Vehicle {self._vehicle.id}: {stage_name} speed limited from "
                           f"{calculated_speed:.1f} to {self.max_speed:.1f} km/h")
            
            return final_speed
            
        except Exception as e:
            logger.error(f"Error calculating rule-based speed: {e}")
            # Fallback with max_speed enforcement
            fallback_speed = max(10.0, vanilla_speed * 0.7) if vanilla_speed > 0 else 30.0
            return min(fallback_speed, self.max_speed)

    def _calculate_car_following_speed(self, vanilla_speed: float) -> Optional[float]:
        """
        Calculate car following speed if there's a leading vehicle.
        
        Args:
            vanilla_speed: Base speed from VanillaAgent (km/h)
        
        Returns:
            Optional[float]: Following speed or None if no leading vehicle
        """
        # Find leading vehicle
        leading_vehicle, distance = self._get_leading_vehicle()
        
        if leading_vehicle is None:
            return None  # No leading vehicle
        
        # Calculate safe following speed
        safe_speed = self._calculate_safe_following_speed(leading_vehicle, distance, vanilla_speed)
        
        if self.debug_rules:
            logger.info(f"Vehicle {self._vehicle.id}: Following vehicle at {distance:.1f}m, "
                       f"target speed: {safe_speed:.1f} km/h")
        
        return safe_speed

    def _calculate_ttc_to_junction(self, trajectory: List[Tuple[float, float, float]], 
                                  junction_dist: float) -> float:
        """
        Calculate time-to-collision to junction entry.
        
        Args:
            trajectory: Vehicle trajectory
            junction_dist: Current distance to junction
            
        Returns:
            float: Time to reach junction in seconds
        """
        try:
            if not trajectory:
                return float('inf')
            
            # Simple approximation: use current speed and distance
            speed_ms = self._ego_speed / 3.6 if self._ego_speed > 0 else 1.0
            return junction_dist / speed_ms
            
        except Exception as e:
            logger.error(f"Error calculating TTC to junction: {e}")
            return float('inf')
    
    # --------------------------------------------------------------------- #
    # Junction Management
    # --------------------------------------------------------------------- #         
    def _should_apply_junction_management(self) -> bool:
        """
        Check if junction management should be applied.
        
        Returns:
            bool: True if should apply cautious speed for junction
        """
        # Get distance to nearest junction
        junction_dist = self._get_junction_distance()
        
        # Check if approaching junction
        if junction_dist > self.junction_approach_distance:
            return False  # Not approaching junction
        
        # Detect conflicts with other vehicles
        conflicts = self._detect_junction_conflicts(junction_dist)
        
        if conflicts:
            # FCFS Logic: Check if any other vehicle has priority (arrives first)
            has_priority_conflicts = [c for c in conflicts if c['has_priority']]
            
            if has_priority_conflicts:
                # Other vehicles arrive first - yield
                if self.debug_rules:
                    # Fix: Use carla_id attribute instead of id method
                    priority_vehicles = [str(getattr(c['vehicle'], 'carla_id', 'unknown')) for c in has_priority_conflicts]
                    logger.info(f"Vehicle {self._vehicle.id}: Yielding at junction, "
                               f"distance: {junction_dist:.1f}m, yielding to vehicles: {priority_vehicles}")
                return True
            else:
                # Ego has priority - proceed but monitor
                if self.debug_rules:
                    logger.info(f"Vehicle {self._vehicle.id}: Has priority at junction, "
                               f"proceeding with conflicts: {len(conflicts)}")
        
        return False
       
    def _get_junction_distance(self) -> float:
        """
        Calculate distance to nearest junction/intersection.
        
        Returns:
            float: Distance to nearest junction in meters
        """
        if not self._ego_pos:
            return float('inf')
        
        try:
            # Get current waypoint
            current_waypoint = self._map.get_waypoint(self._ego_pos.location)
            if not current_waypoint:
                return float('inf')
            
            # Look ahead for junction waypoints
            lookahead_distance = 0.0
            waypoint = current_waypoint
            step_size = 5.0  # meters
            
            while lookahead_distance < self.junction_approach_distance:
                # Check if current waypoint is in junction
                if waypoint.is_junction:
                    return lookahead_distance
                
                # Get next waypoint
                next_waypoints = waypoint.next(step_size)
                if not next_waypoints:
                    break
                    
                waypoint = next_waypoints[0]
                lookahead_distance += step_size
            
            return float('inf')  # No junction found within approach distance
            
        except Exception as e:
            logger.error(f"Error calculating junction distance: {e}")
            return float('inf')
    
    def _detect_junction_conflicts(self, junction_dist: float) -> List[Dict]:
        """
        Detect conflicting vehicles at junction using trajectory prediction and TTC.
        Enhanced JAIR approach for better intersection handling.
        
        Args:
            junction_dist: Distance to junction
            
        Returns:
            List[Dict]: List of conflicting vehicles with TTC information
        """
        conflicts = []
        
        if not self._ego_pos or not hasattr(self, 'obstacle_vehicles'):
            return conflicts
        
        try:
            # Get ego vehicle's predicted trajectory
            ego_trajectory = self._predict_vehicle_trajectory(
                self._ego_pos, self._ego_speed, is_ego=True
            )
            
            if not ego_trajectory:
                if self.debug_rules:
                    logger.warning(f"Vehicle {self._vehicle.id}: Could not predict ego trajectory")
                return conflicts
            
            # Calculate ego's TTC to junction
            ego_junction_ttc = self._calculate_ttc_to_junction(ego_trajectory, junction_dist)
            
            if self.debug_rules:
                logger.info(f"Vehicle {self._vehicle.id}: Ego TTC to junction: {ego_junction_ttc:.1f}s")
            
            for vehicle in self.obstacle_vehicles:
                # Get vehicle distance to junction
                vehicle_location = vehicle.get_location()
                vehicle_junction_dist = vehicle_location.distance(self._ego_pos.location)
                
                # Only consider vehicles near junction
                if vehicle_junction_dist > self.junction_conflict_distance:
                    continue
                
                # Get vehicle speed and transform
                vehicle_transform = vehicle.get_transform()
                vehicle_speed = self._get_vehicle_speed(vehicle)  # km/h
                
                # Predict other vehicle's trajectory
                other_trajectory = self._predict_vehicle_trajectory(
                    vehicle_transform, vehicle_speed, is_ego=False
                )
                
                if not other_trajectory:
                    continue
                
                # Find collision point and calculate TTC for both vehicles
                collision_data = self._find_collision_point(ego_trajectory, other_trajectory)
                
                if collision_data:
                    ego_ttc, other_ttc, collision_point = collision_data
                    
                    # Check if this is a potential conflict
                    # Conflict exists if both vehicles will reach collision point within threshold
                    ttc_diff = abs(ego_ttc - other_ttc)
                    
                    if ttc_diff < self.ttc_threshold:
                        # Use ego_junction_ttc for more accurate priority determination
                        # Compare collision TTC with junction arrival TTC
                        has_priority = other_ttc < min(ego_ttc, ego_junction_ttc)
                        
                        conflicts.append({
                            'vehicle': vehicle,
                            'distance': vehicle_junction_dist,
                            'ego_ttc': ego_ttc,
                            'ego_junction_ttc': ego_junction_ttc,  # Now using this value
                            'other_ttc': other_ttc,
                            'ttc_diff': ttc_diff,
                            'collision_point': collision_point,
                            'has_priority': has_priority  # Enhanced priority logic
                        })
                        
                        if self.debug_rules:
                            priority_str = "OTHER" if has_priority else "EGO"
                            # Fix: Use carla_id attribute instead of id method
                            vehicle_id = getattr(vehicle, 'carla_id', 'unknown')
                            logger.info(
                                f"Vehicle {self._vehicle.id}: Conflict detected with vehicle {vehicle_id}, "
                                f"TTC ego: {ego_ttc:.1f}s, other: {other_ttc:.1f}s, "
                                f"junction: {ego_junction_ttc:.1f}s, priority: {priority_str}"
                            )
                    
        except Exception as e:
            logger.error(f"Error detecting junction conflicts: {e}")
        
        return conflicts
    
    def _predict_vehicle_trajectory(self, vehicle_transform: 'carla.Transform', 
                                  vehicle_speed: float, is_ego: bool = False) -> List[Tuple[float, float, float]]:
        """
        Predict vehicle trajectory for lookahead time.
        
        Args:
            vehicle_transform: Vehicle's current transform
            vehicle_speed: Vehicle speed in km/h
            is_ego: Whether this is the ego vehicle
            
        Returns:
            List[Tuple[float, float, float]]: List of (x, y, time) points along predicted path
        """
        trajectory = []
        
        try:
            if is_ego:
                # Use ego vehicle's planned waypoints from local planner
                local_planner = self.get_local_planner()
                if local_planner and hasattr(local_planner, 'get_waypoint_buffer'):
                    waypoint_buffer = local_planner.get_waypoint_buffer()
                    if waypoint_buffer:
                        current_time = 0.0
                        speed_ms = vehicle_speed / 3.6  # Convert km/h to m/s
                        
                        for waypoint, _ in itertools.islice(waypoint_buffer, 20):  # Limit to avoid infinite loops
                            if current_time > self.trajectory_lookahead_time:
                                break
                            
                            trajectory.append((
                                waypoint.transform.location.x,
                                waypoint.transform.location.y,
                                current_time
                            ))
                            
                            # Estimate time to next waypoint (simplified)
                            current_time += self.trajectory_step_size / max(speed_ms, 1.0)
            else:
                # For other vehicles, predict based on current heading and speed
                speed_ms = vehicle_speed / 3.6
                if speed_ms < 0.1:  # Nearly stopped
                    return trajectory
                
                # Get current waypoint and follow road topology
                current_waypoint = self._map.get_waypoint(vehicle_transform.location)
                if not current_waypoint:
                    return trajectory
                
                current_time = 0.0
                distance_covered = 0.0
                waypoint = current_waypoint
                
                while (current_time < self.trajectory_lookahead_time and 
                       distance_covered < self.trajectory_lookahead_time * speed_ms):
                    
                    trajectory.append((
                        waypoint.transform.location.x,
                        waypoint.transform.location.y,
                        current_time
                    ))
                    
                    # Get next waypoint
                    next_waypoints = waypoint.next(self.trajectory_step_size)
                    if not next_waypoints:
                        break
                    
                    waypoint = next_waypoints[0]
                    distance_covered += self.trajectory_step_size
                    current_time += self.trajectory_step_size / speed_ms
                    
        except Exception as e:
            logger.error(f"Error predicting trajectory: {e}")
        
        return trajectory
    
    def _find_collision_point(self, ego_trajectory: List[Tuple[float, float, float]], 
                            other_trajectory: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float, Tuple[float, float]]]:
        """
        Find collision point between two trajectories.
        
        Args:
            ego_trajectory: Ego vehicle trajectory [(x, y, time), ...]
            other_trajectory: Other vehicle trajectory [(x, y, time), ...]
            
        Returns:
            Optional[Tuple[float, float, Tuple[float, float]]]: (ego_ttc, other_ttc, collision_point) or None
        """
        min_distance = float('inf')
        best_collision = None
        collision_threshold = 4.0  # meters - consider collision if closer than this
        
        try:
            for ego_point in ego_trajectory:
                ego_x, ego_y, ego_time = ego_point
                
                for other_point in other_trajectory:
                    other_x, other_y, other_time = other_point
                    
                    # Calculate distance between trajectory points
                    distance = math.sqrt((ego_x - other_x)**2 + (ego_y - other_y)**2)
                    
                    if distance < collision_threshold and distance < min_distance:
                        min_distance = distance
                        collision_point = ((ego_x + other_x) / 2, (ego_y + other_y) / 2)
                        best_collision = (ego_time, other_time, collision_point)
            
            return best_collision
            
        except Exception as e:
            logger.error(f"Error finding collision point: {e}")
            return None

    def _calculate_safe_following_speed(self, leading_vehicle: 'carla.Vehicle', distance: float, vanilla_speed: float) -> float:
        """
        Calculate safe following speed using time headway principle and leading vehicle speed.
        
        Args:
            leading_vehicle: The vehicle being followed
            distance: Distance to leading vehicle
            vanilla_speed: Base speed from VanillaAgent (km/h)
            
        Returns:
            float: Target speed for safe following (km/h)
        """
        try:
            # Get leading vehicle speed
            leading_speed = self._get_vehicle_speed(leading_vehicle)  # km/h
            
            # Calculate safe distance: minimum buffer + time-based distance
            time_based_distance = (self._ego_speed / 3.6) * self.time_headway  # convert km/h to m/s
            safe_distance = self.minimum_distance_buffer + time_based_distance
            
            # Calculate gap error
            gap_error = distance - safe_distance
            
            # Dynamic following speed calculation using vanilla_speed as base
            # 1. Don't exceed leading vehicle speed + buffer
            max_speed_from_leader = leading_speed + self.following_speed_buffer
            
            # 2. Apply proportional control for gap adjustment
            speed_adjustment = self.following_gain * gap_error
            gap_adjusted_speed = self._ego_speed + speed_adjustment
            
            # 3. Final speed: minimum of vanilla_speed, leader-based speed, and gap-adjusted speed
            target_speed = max(0, min(
                vanilla_speed,
                max_speed_from_leader, 
                gap_adjusted_speed
            ))
            
            return target_speed
            
        except Exception as e:
            logger.error(f"Error calculating safe following speed: {e}")
            return self.cautious_speed
        
    # --------------------------------------------------------------------- #
    # Private Functions
    # --------------------------------------------------------------------- #
    def _get_vehicle_speed(self, vehicle: 'carla.Vehicle') -> float:
        """
        Get vehicle speed in km/h.
        
        Args:
            vehicle: CARLA vehicle
            
        Returns:
            float: Speed in km/h
        """
        try:
            velocity = vehicle.get_velocity()
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            return speed_ms * 3.6  # Convert to km/h
        except Exception as e:
            logger.error(f"Error getting vehicle speed: {e}")
            return 0.0
    
    def _get_leading_vehicle(self) -> Tuple[Optional['carla.Vehicle'], float]:
        """
        Find the leading vehicle in the same lane.
        
        Returns:
            Tuple[Optional[carla.Vehicle], float]: (vehicle, distance) or (None, inf)
        """
        if not self._ego_pos or not hasattr(self, 'obstacle_vehicles'):
            return None, float('inf')
        
        ego_heading = math.radians(self._ego_pos.rotation.yaw)
        ego_location = self._ego_pos.location
        
        closest_vehicle = None
        min_distance = float('inf')
        
        try:
            for vehicle in self.obstacle_vehicles:
                vehicle_location = vehicle.get_location()
                vehicle_heading = math.radians(vehicle.get_transform().rotation.yaw)
                
                # Check heading similarity (same lane)
                heading_diff = abs(ego_heading - vehicle_heading)
                if heading_diff > math.pi:
                    heading_diff = 2 * math.pi - heading_diff
                
                if heading_diff > self.same_lane_tolerance:
                    continue  # Not in same lane
                
                # Calculate distance and direction
                distance = ego_location.distance(vehicle_location)
                
                # Check if vehicle is in front (within front cone)
                dx = vehicle_location.x - ego_location.x
                dy = vehicle_location.y - ego_location.y
                angle_to_vehicle = math.atan2(dy, dx)
                
                angle_diff = abs(ego_heading - angle_to_vehicle)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # Vehicle must be in front cone and closer than current minimum
                if angle_diff < self.front_cone_angle and distance < min_distance:
                    min_distance = distance
                    closest_vehicle = vehicle
                    
        except Exception as e:
            logger.error(f"Error finding leading vehicle: {e}")
        
        return closest_vehicle, min_distance
    
    # --------------------------------------------------------------------- #
    # Public Functions
    # --------------------------------------------------------------------- #
    def get_current_stage(self) -> int:
        """Get current active stage (for debugging/monitoring)."""
        return self.current_stage
    
    def get_debug_info(self) -> Dict:
        """Get debug information about current state."""
        return {
            'agent_type': 'rule_based_jair_enhanced',
            'current_stage': self.current_stage,
            'stage_name': {1: 'Junction', 2: 'Following', 3: 'Cruising'}.get(self.current_stage, 'Unknown'),
            'junction_distance': self._get_junction_distance(),
            'leading_vehicle_distance': self._get_leading_vehicle()[1],
            'ego_speed': self._ego_speed,
            'configuration': {
                'junction_approach_distance': self.junction_approach_distance,
                'cautious_speed': self.cautious_speed,
                'trajectory_lookahead_time': self.trajectory_lookahead_time,
                'ttc_threshold': self.ttc_threshold,
                'time_headway': self.time_headway,
                'max_speed': self.max_speed,
                'cruising_speed_factor': self.cruising_speed_factor
            }
        }