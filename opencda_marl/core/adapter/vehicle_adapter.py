'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 18:22:20
FilePath     : /OpenCDA-MARL/opencda_marl/core/adapter/vehicle_adapter.py
Description  : 
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from typing import Dict, Any
from loguru import logger
import carla
import numpy as np


from opencda.core.common.vehicle_manager import VehicleManager

from opencda_marl.core.adapter.vehicle_defaults import get_vehicle_manager_defaults
from opencda_marl.core.adapter.exception import CollisionException
from opencda_marl.core.safety.marl_safety_manager import MARLSafetyManager
from opencda_marl.core.agents import AgentFactory
from opencda_marl.core.agents.vanilla_agent import VanillaAgent

# Default nearby vehicle detection constants (can be overridden by config)
DEFAULT_NEARBY_DETECTION_RADIUS = 50.0  # meters - detection radius for nearby vehicles
DEFAULT_MAX_NEARBY_VEHICLES = 5  # Maximum number of nearby vehicles to track (K slots)
DEFAULT_FEATURES_PER_VEHICLE = 7  # Features per vehicle: rel_x, rel_y, rel_vx, rel_vy, heading_diff, distance, ttc
DEFAULT_MAX_RELATIVE_VELOCITY = 40.0  # m/s (~144 km/h) - for normalization
DEFAULT_MAX_TTC = 10.0  # seconds - TTC values clamped to this maximum


class MARLVehicleAdapter:

    def __init__(self, config: Dict[str, Any],
                 vehicle: carla.Actor, carla_map, cav_world,
                 dump_data: bool = False,
                 agent_type: str = "behavior"):
        self.config = config
        self.vm_cfg = self._merge_config(config.get("vehicle", {}))
        # Uncomment to print the vehicle config for debugging
        # self._print_vm_cfg()

        self.vehicle = vehicle
        self.actor_id = vehicle.id
        self.carla_map = carla_map
        self.cav_world = cav_world
        self.world = vehicle.get_world()
        self.dump_data = dump_data
        self.agent_type = agent_type

        # Store target speed for monitoring
        self.target_speed = 0.0

        # Load nearby vehicle config from MARL config (or use defaults)
        self._init_nearby_vehicle_config(config)

        self.vm = self.get_vm()
        if hasattr(self.vm, 'safety_manager'):
            self._use_marl_safety_manager()

    def _init_nearby_vehicle_config(self, config: Dict[str, Any]):
        """Initialize nearby vehicle detection parameters from config."""
        # Try to get config from MARL.td3.nearby_vehicle_config
        marl_config = config.get('MARL', {})
        td3_config = marl_config.get('td3', {})
        nearby_config = td3_config.get('nearby_vehicle_config', {})

        # Load values from config with defaults
        self.nearby_detection_radius = nearby_config.get('detection_radius', DEFAULT_NEARBY_DETECTION_RADIUS)
        self.max_nearby_vehicles = nearby_config.get('max_vehicles', DEFAULT_MAX_NEARBY_VEHICLES)
        self.features_per_vehicle = nearby_config.get('features_per_vehicle', DEFAULT_FEATURES_PER_VEHICLE)
        self.max_relative_velocity = nearby_config.get('max_relative_velocity', DEFAULT_MAX_RELATIVE_VELOCITY)
        self.max_ttc = nearby_config.get('max_ttc', DEFAULT_MAX_TTC)
    # --------------------------------------------------------------------- #
    # Public control API
    # --------------------------------------------------------------------- #

    def step(self, target_speed: float = None):
        # Store target speed for monitoring
        if target_speed is not None:
            self.target_speed = target_speed
        else:
            # During warmup (target_speed=None): use current vehicle speed as "target"
            # This reflects what vanilla agent is commanding, for TensorBoard tracking
            try:
                velocity = self.vm.vehicle.get_velocity()
                current_speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                self.target_speed = current_speed
            except Exception:
                pass  # Keep previous value if velocity unavailable

        self.vm.update_info()

        if self.check_collision():
            raise CollisionException(f"Vehicle {self.actor_id} collided")

        control = self.vm.run_step(target_speed)
        self.vm.vehicle.apply_control(control)

    def set_destination(self, start_location: carla.Location, end_location: carla.Location,
                        clean: bool = False, end_reset: bool = True):
        self.vm.set_destination(start_location, end_location,
                                clean, end_reset)

    def set_target_speed(self, target_speed: float):
        if isinstance(self.vm.agent, VanillaAgent):
            logger.debug(
                f"Setting target speed for vehicle {self.actor_id} to {target_speed}")
            self.vm.agent.set_target_speed(target_speed)
        # no need to set target speed for BehaviorAgent

    def check_collision(self) -> bool:
        """
        Simple collision check using MARL safety manager.
        """
        try:
            if self.vm.safety_manager:
                return self.vm.safety_manager.check_collision()
        except Exception as e:
            logger.error(
                f"Warning: Could not check collision for vehicle {self.actor_id}: {e}")
        return False

    def get_observation(self) -> Dict[str, Any]:
        """
        Get comprehensive observation data for GUI display.

        Returns:
            dict: Agent observation data
        """
        try:
            # Ensure vehicle manager info is up to date before getting observations
            if hasattr(self, 'vm') and hasattr(self.vm, 'update_info'):
                self.vm.update_info()

            obs = {'vehicle_id': self.actor_id}

            # Use agent getter methods if available
            if hasattr(self.vm, 'agent'):
                agent = self.vm.agent

                # Get speed (already in km/h from agent)
                if hasattr(agent, 'get_speed'):
                    obs['speed'] = round(agent.get_speed(), 2)
                elif hasattr(agent, '_ego_speed'):
                    obs['speed'] = round(agent._ego_speed, 2)
                else:
                    obs['speed'] = 0.0

                # Get position using getter method
                if hasattr(agent, 'get_position'):
                    pos_x, pos_y = agent.get_position()
                    obs['position_x'] = round(pos_x, 1)
                    obs['position_y'] = round(pos_y, 1)
                elif hasattr(agent, '_ego_pos') and agent._ego_pos:
                    obs['position_x'] = round(agent._ego_pos.location.x, 1)
                    obs['position_y'] = round(agent._ego_pos.location.y, 1)
                else:
                    obs['position_x'] = 0.0
                    obs['position_y'] = 0.0

                # Get lane ID using getter method
                # if hasattr(agent, 'get_lane_id'):
                #    obs['lane_id'] = agent.get_lane_id()
                # elif hasattr(agent, '_map') and hasattr(agent, '_ego_pos') and agent._ego_pos:
                #    waypoint = agent._map.get_waypoint(agent._ego_pos.location)
                #    obs['lane_id'] = waypoint.lane_id if waypoint else 'N/A'
                # else:
                #    obs['lane_id'] = 'N/A'

                # Get waypoint buffer size from local planner
                if hasattr(agent, '_local_planner'):
                    planner = agent._local_planner
                    buffer = planner.get_waypoint_buffer()
                    obs['waypoint_buffer_size'] = len(buffer) if buffer else 0
                else:
                    obs['waypoint_buffer_size'] = 0
            else:
                # Fallback values if no agent
                obs['speed'] = 0.0
                obs['position_x'] = 0.0
                obs['position_y'] = 0.0
                obs['lane_id'] = 'N/A'
                obs['waypoint_buffer_size'] = 0

            # Basic status - can be expanded based on agent state
            obs['status'] = self.vehicle.is_alive

            # Add MARL-specific features
            obs['distance_to_intersection'] = self._calculate_distance_to_intersection()
            obs['distance_to_front_vehicle'] = self._calculate_distance_to_front_vehicle()
            obs['lane_position'] = self._classify_lane_position()

            # Add nearby vehicle features for Phase 3 enhancement (35D)
            nearby_features, min_ttc = self._compute_nearby_vehicle_features()
            obs['nearby_vehicles'] = nearby_features  # 35D list (5 vehicles × 7 features)
            obs['min_ttc'] = min_ttc  # For TTC-based reward calculation
            obs['distance_to_destination'] = self._calculate_distance_to_destination()  # For progress reward

            # Add new enhanced DQN features (9D feature set)
            # 1. Relative position to intersection (replaces absolute position)
            rel_x, rel_y = self._calculate_relative_position_to_intersection()
            obs['relative_position_to_intersection'] = {
                'x': round(rel_x, 2),
                'y': round(rel_y, 2)
            }
            
            # 2. Vehicle heading angle 
            obs['heading_angle'] = round(self._get_vehicle_heading_angle(), 3)
            
            # 3. Raw waypoint buffer count (already available as waypoint_buffer_size)

            # Keep legacy DQN observations for backward compatibility
            # Location dictionary for DQN extractor (now uses relative position)
            obs['location'] = {
                'x': rel_x,
                'y': rel_y
            }
            
            # Keep velocity for backward compatibility but remove from new feature set
            try:
                velocity = self.vehicle.get_velocity()
                obs['velocity'] = {
                    'x': velocity.x,
                    'y': velocity.y
                }
            except Exception as e:
                logger.debug(f"Could not get velocity for vehicle {self.actor_id}: {e}")
                obs['velocity'] = {'x': 0.0, 'y': 0.0}

            # Add target speed for monitoring
            obs['target_speed'] = round(self.target_speed, 2)

            return obs
        except Exception as e:
            return {'vehicle_id': self.actor_id, 'error': str(e)}

    # --------------------------------------------------------------------- #
    # MARL observation helpers
    # --------------------------------------------------------------------- #

    def _calculate_distance_to_intersection(self) -> float:
        """Calculate distance to nearest intersection in meters"""
        try:
            if hasattr(self.vm, 'agent') and hasattr(self.vm.agent, '_ego_pos') and self.vm.agent._ego_pos:
                current_location = self.vm.agent._ego_pos.location

                # Get current waypoint
                current_waypoint = self.carla_map.get_waypoint(
                    current_location)
                if not current_waypoint:
                    return 100.0  # Default large distance if no waypoint

                # Look ahead for intersections
                search_distance = 100.0  # meters
                waypoint = current_waypoint
                distance_traveled = 0.0

                while distance_traveled < search_distance:
                    # Check if this waypoint is at an intersection
                    if waypoint.is_intersection:
                        return distance_traveled

                    # Move to next waypoint
                    next_waypoints = waypoint.next(2.0)  # 2 meter steps
                    if not next_waypoints:
                        break

                    waypoint = next_waypoints[0]
                    distance_traveled += 2.0

                # No intersection found within search distance
                return search_distance
            else:
                return 100.0  # Default if no position available

        except Exception as e:
            logger.debug(
                f"Error calculating distance to intersection for vehicle {self.actor_id}: {e}")
            return 100.0  # Safe default

    def _calculate_distance_to_front_vehicle(self) -> float:
        """Calculate distance to front vehicle in meters"""
        try:
            # Use direct vehicle location instead of agent position
            current_location = self.vehicle.get_location()

            # Get all vehicles in world
            all_vehicles = self.world.get_actors().filter('vehicle.*')

            min_distance = 100.0  # Increased detection range
            current_waypoint = self.carla_map.get_waypoint(current_location)

            if not current_waypoint:
                logger.debug(f"No waypoint found for vehicle {self.actor_id}")
                return 999.0  # Different value for "no waypoint"

            vehicles_checked = 0
            vehicles_ahead = 0

            for other_vehicle in all_vehicles:
                if other_vehicle.id == self.vehicle.id:
                    continue  # Skip self

                if not other_vehicle.is_alive:
                    continue

                vehicles_checked += 1
                other_location = other_vehicle.get_location()

                # Quick distance check before expensive waypoint calculations
                rough_distance = current_location.distance(other_location)
                if rough_distance > min_distance:
                    continue  # Too far, skip waypoint calculation

                other_waypoint = self.carla_map.get_waypoint(other_location)
                if not other_waypoint:
                    continue

                # Check if other vehicle is in the same lane and ahead
                if (other_waypoint.lane_id == current_waypoint.lane_id and
                        other_waypoint.road_id == current_waypoint.road_id):

                    # Check if vehicle is ahead using forward vector
                    vehicle_transform = self.vehicle.get_transform()
                    forward_vector = vehicle_transform.get_forward_vector()
                    to_other = other_location - current_location

                    # If dot product is positive, other vehicle is ahead
                    dot_product = forward_vector.x * to_other.x + forward_vector.y * to_other.y
                    if dot_product > 0:
                        vehicles_ahead += 1
                        if rough_distance < min_distance:
                            min_distance = rough_distance

            # Log debugging info
            if vehicles_checked == 0:
                logger.debug(
                    f"Vehicle {self.actor_id}: No other vehicles found in world")
            elif vehicles_ahead == 0:
                logger.debug(
                    f"Vehicle {self.actor_id}: No vehicles ahead in same lane (checked {vehicles_checked} vehicles)")
            else:
                logger.debug(
                    f"Vehicle {self.actor_id}: Found {vehicles_ahead} vehicles ahead, closest at {min_distance:.1f}m")

            # Return 999.0 if no vehicle detected ahead, otherwise return actual distance
            return min_distance if min_distance < 100.0 else 999.0

        except Exception as e:
            logger.warning(
                f"Error calculating distance to front vehicle for vehicle {self.actor_id}: {e}")
            return 999.0  # Different error value

    def _classify_lane_position(self) -> int:
        """
        Classify lane position for Q-learning:
        Entry lanes: 1 (left), 2 (center), 3 (right)  
        Exit lanes: -1 (left), -2 (center), -3 (right)
        Junction/unknown: 0
        """
        try:
            current_location = self.vehicle.get_location()
            current_waypoint = self.carla_map.get_waypoint(current_location)

            if not current_waypoint:
                return 0

            # Check if actually in junction
            if current_waypoint.is_junction or current_waypoint.get_junction() is not None:
                return 0

            # Get lane number using abs(lane_id) - negative just means opposite direction
            lane_id = current_waypoint.lane_id
            abs_lane = abs(lane_id)
            lane_number = min(abs_lane, 3)  # Cap at 3 for our discrete space

            # Determine if approaching (entry) or leaving (exit) intersection
            is_entry_lane = self._is_approaching_intersection(current_waypoint)
            is_exit_lane = self._is_leaving_intersection(current_waypoint)

            if is_entry_lane:
                return lane_number  # Positive for entry lanes
            elif is_exit_lane:
                return -lane_number  # Negative for exit lanes
            else:
                return 0  # Unknown/no intersection nearby

        except Exception as e:
            logger.debug(
                f"Error classifying lane position for vehicle {self.actor_id}: {e}")
            return 0

    def _is_approaching_intersection(self, current_waypoint, search_distance: float = 50.0) -> bool:
        """Check if approaching an intersection by looking ahead."""
        try:
            waypoint = current_waypoint
            distance = 0.0
            
            while distance < search_distance:
                next_waypoints = waypoint.next(2.0)  # 2m steps
                if not next_waypoints:
                    break
                    
                waypoint = next_waypoints[0]
                distance += 2.0
                
                # Found intersection ahead
                if waypoint.is_junction:
                    return True
                    
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if approaching intersection: {e}")
            return False

    def _is_leaving_intersection(self, current_waypoint, search_distance: float = 30.0) -> bool:
        """Check if leaving an intersection by looking behind."""
        try:
            waypoint = current_waypoint
            distance = 0.0
            
            while distance < search_distance:
                prev_waypoints = waypoint.previous(2.0)  # 2m steps backward
                if not prev_waypoints:
                    break
                    
                waypoint = prev_waypoints[0]
                distance += 2.0
                
                # Found intersection behind
                if waypoint.is_junction:
                    return True
                    
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if leaving intersection: {e}")
            return False

    def _calculate_relative_position_to_intersection(self) -> tuple:
        """Calculate relative position to nearest intersection (x, y in meters)"""
        try:
            current_location = self.vehicle.get_location()
            current_waypoint = self.carla_map.get_waypoint(current_location)
            
            if not current_waypoint:
                return (0.0, 0.0)  # Default if no waypoint
            
            # Search in all directions for nearest intersection
            search_distance = 100.0  # meters
            closest_intersection = None
            min_distance = float('inf')
            
            # Search forward
            waypoint = current_waypoint
            distance_traveled = 0.0
            while distance_traveled < search_distance:
                if waypoint.is_intersection:
                    dist = current_location.distance(waypoint.transform.location)
                    if dist < min_distance:
                        min_distance = dist
                        closest_intersection = waypoint.transform.location
                        
                next_waypoints = waypoint.next(2.0)
                if not next_waypoints:
                    break
                waypoint = next_waypoints[0]
                distance_traveled += 2.0
            
            # Search backward
            waypoint = current_waypoint
            distance_traveled = 0.0
            while distance_traveled < search_distance:
                if waypoint.is_intersection:
                    dist = current_location.distance(waypoint.transform.location)
                    if dist < min_distance:
                        min_distance = dist
                        closest_intersection = waypoint.transform.location
                        
                prev_waypoints = waypoint.previous(2.0)
                if not prev_waypoints:
                    break
                waypoint = prev_waypoints[0]
                distance_traveled += 2.0
            
            if closest_intersection:
                # Calculate relative position (intersection - current)
                rel_x = closest_intersection.x - current_location.x
                rel_y = closest_intersection.y - current_location.y
                return (rel_x, rel_y)
            else:
                return (100.0, 0.0)  # Default large distance if no intersection found
                
        except Exception as e:
            logger.debug(f"Error calculating relative position to intersection: {e}")
            return (0.0, 0.0)

    def _get_vehicle_heading_angle(self) -> float:
        """Get vehicle heading angle in radians (-π to π)"""
        try:
            vehicle_transform = self.vehicle.get_transform()
            # CARLA uses degrees, convert to radians and normalize to [-π, π]
            yaw_degrees = vehicle_transform.rotation.yaw
            yaw_radians = np.radians(yaw_degrees)

            # Normalize to [-π, π]
            while yaw_radians > np.pi:
                yaw_radians -= 2 * np.pi
            while yaw_radians < -np.pi:
                yaw_radians += 2 * np.pi

            return yaw_radians

        except Exception as e:
            logger.debug(f"Error getting vehicle heading angle: {e}")
            return 0.0

    # --------------------------------------------------------------------- #
    # Nearby vehicle detection for MARL
    # --------------------------------------------------------------------- #

    def _get_nearby_vehicles(self) -> list:
        """
        Get nearby vehicles within detection radius, sorted by distance.
        Uses bounding box pre-filter for efficiency.

        Returns:
            List of tuples: [(vehicle, distance), ...] sorted by distance, max K vehicles
        """
        try:
            current_location = self.vehicle.get_location()
            all_vehicles = self.world.get_actors().filter('vehicle.*')

            nearby = []
            for vehicle in all_vehicles:
                if vehicle.id == self.vehicle.id:
                    continue  # Skip self

                if not vehicle.is_alive:
                    continue

                other_loc = vehicle.get_location()

                # Quick bounding box check (faster than full distance calculation)
                if (abs(other_loc.x - current_location.x) > self.nearby_detection_radius or
                    abs(other_loc.y - current_location.y) > self.nearby_detection_radius):
                    continue

                distance = current_location.distance(other_loc)
                if distance <= self.nearby_detection_radius:
                    nearby.append((vehicle, distance))

            # Sort by distance and return top K
            nearby.sort(key=lambda x: x[1])
            return nearby[:self.max_nearby_vehicles]

        except Exception as e:
            logger.debug(f"Error getting nearby vehicles for {self.actor_id}: {e}")
            return []

    def _calculate_ttc_to_vehicle(self, other_vehicle: carla.Actor) -> float:
        """
        Calculate Time-to-Collision (TTC) to another vehicle using constant velocity assumption.

        Uses quadratic equation: |rel_pos + t*rel_vel|² = collision_radius²

        Args:
            other_vehicle: The other CARLA vehicle actor

        Returns:
            float: TTC in seconds (inf if no collision predicted within horizon)
        """
        try:
            ego_loc = self.vehicle.get_location()
            ego_vel = self.vehicle.get_velocity()
            other_loc = other_vehicle.get_location()
            other_vel = other_vehicle.get_velocity()

            # Relative position and velocity
            rel_x = other_loc.x - ego_loc.x
            rel_y = other_loc.y - ego_loc.y
            rel_vx = other_vel.x - ego_vel.x
            rel_vy = other_vel.y - ego_vel.y

            collision_radius = 4.0  # Combined vehicle radii (meters)

            # Quadratic equation coefficients: at² + bt + c = 0
            a = rel_vx**2 + rel_vy**2
            b = 2 * (rel_x * rel_vx + rel_y * rel_vy)
            c = rel_x**2 + rel_y**2 - collision_radius**2

            # No relative motion - check if already colliding
            if abs(a) < 1e-6:
                return float('inf')

            discriminant = b**2 - 4 * a * c

            # No collision trajectory
            if discriminant < 0:
                return float('inf')

            # Find smallest positive root
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)

            if t1 > 0:
                return t1
            elif t2 > 0:
                return t2
            else:
                return float('inf')  # Collision in past

        except Exception as e:
            logger.debug(f"Error calculating TTC: {e}")
            return float('inf')

    def _compute_nearby_vehicle_features(self) -> tuple:
        """
        Compute normalized features for nearby vehicles (35D total = 5 slots × 7 features).

        Per-vehicle features (7D):
        - rel_x: Relative X position (normalized to [-1, 1])
        - rel_y: Relative Y position (normalized to [-1, 1])
        - rel_vx: Relative velocity X (normalized to [-1, 1])
        - rel_vy: Relative velocity Y (normalized to [-1, 1])
        - heading_diff: Heading difference (normalized to [-1, 1])
        - distance: Euclidean distance (normalized to [0, 1])
        - ttc: Time-to-collision (normalized to [0, 1])

        Empty slots are filled with safe defaults (distance=1.0, ttc=1.0).

        Returns:
            Tuple[List[float], float]: (35D feature vector, minimum TTC across all vehicles)
        """
        try:
            nearby_vehicles = self._get_nearby_vehicles()

            ego_loc = self.vehicle.get_location()
            ego_vel = self.vehicle.get_velocity()
            ego_heading = np.radians(self.vehicle.get_transform().rotation.yaw)

            features = []
            min_ttc = float('inf')

            for i in range(self.max_nearby_vehicles):
                if i < len(nearby_vehicles):
                    vehicle, distance = nearby_vehicles[i]
                    other_loc = vehicle.get_location()
                    other_vel = vehicle.get_velocity()
                    other_heading = np.radians(vehicle.get_transform().rotation.yaw)

                    # Relative position (normalized by detection radius)
                    rel_x = (other_loc.x - ego_loc.x) / self.nearby_detection_radius
                    rel_y = (other_loc.y - ego_loc.y) / self.nearby_detection_radius

                    # Relative velocity (normalized by max velocity)
                    rel_vx = (other_vel.x - ego_vel.x) / self.max_relative_velocity
                    rel_vy = (other_vel.y - ego_vel.y) / self.max_relative_velocity

                    # Heading difference (normalized by π)
                    heading_diff = other_heading - ego_heading
                    # Wrap to [-π, π]
                    while heading_diff > np.pi:
                        heading_diff -= 2 * np.pi
                    while heading_diff < -np.pi:
                        heading_diff += 2 * np.pi
                    heading_diff_norm = heading_diff / np.pi

                    # Distance (normalized by detection radius)
                    distance_norm = distance / self.nearby_detection_radius

                    # TTC (normalized by max TTC, clamped)
                    ttc = self._calculate_ttc_to_vehicle(vehicle)
                    min_ttc = min(min_ttc, ttc)
                    ttc_norm = min(ttc, self.max_ttc) / self.max_ttc

                    features.extend([
                        np.clip(rel_x, -1.0, 1.0),
                        np.clip(rel_y, -1.0, 1.0),
                        np.clip(rel_vx, -1.0, 1.0),
                        np.clip(rel_vy, -1.0, 1.0),
                        np.clip(heading_diff_norm, -1.0, 1.0),
                        np.clip(distance_norm, 0.0, 1.0),
                        np.clip(ttc_norm, 0.0, 1.0)
                    ])
                else:
                    # Empty slot: zeros except distance and TTC = 1.0 (safe/far away)
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

            return features, min_ttc

        except Exception as e:
            logger.debug(f"Error computing nearby vehicle features: {e}")
            # Return safe defaults: K slots × 7 features with safe values
            default_features = []
            for _ in range(self.max_nearby_vehicles):
                default_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
            return default_features, float('inf')

    def _calculate_distance_to_destination(self) -> float:
        """
        Calculate distance to destination for progress reward calculation.

        Returns:
            float: Distance to destination in meters (999.0 if not available)
        """
        try:
            if hasattr(self.vm, 'agent') and hasattr(self.vm.agent, '_local_planner'):
                planner = self.vm.agent._local_planner
                if hasattr(planner, '_destination') and planner._destination:
                    current_loc = self.vehicle.get_location()
                    dest_loc = planner._destination
                    return current_loc.distance(dest_loc)
            return 999.0
        except Exception as e:
            logger.debug(f"Error calculating distance to destination: {e}")
            return 999.0

    # --------------------------------------------------------------------- #
    # Private functions
    # --------------------------------------------------------------------- #

    def _use_marl_safety_manager(self):
        """
        Replace the VehicleManager's SafetyManager with MARL version.

        IMPORTANT: Must properly destroy the old SafetyManager before creating
        the new one to prevent orphaned sensors that cause "sensor went out of
        scope" warnings.
        """
        try:
            params = self.vm_cfg.get('safety_manager', {
                'collision_sensor': {'history_size': 4000, 'col_thresh': 50},
                'stuck_dector': {'len_thresh': 300, 'speed_thresh': 0.05},
                'offroad_dector': {'speed_thresh': 5},
                'traffic_light_detector': {'light_dist_thresh': 20},
                'print_message': False,
                'queue_maxlen': 2000
            })

            # CRITICAL: Destroy the old safety manager's sensors BEFORE creating new one
            # This prevents orphaned sensors that cause "sensor went out of scope" warnings
            if self.vm.safety_manager:
                old_sm = self.vm.safety_manager
                self.vm.safety_manager = None  # Clear reference first
                old_sm.destroy()  # Then destroy

            # Now create the new MARL safety manager
            # MARLSafetyManager is standalone (doesn't inherit from SafetyManager)
            # so it won't create duplicate sensors
            self.vm.safety_manager = MARLSafetyManager(
                self.cav_world,
                self.vehicle,
                params
            )

        except Exception as e:
            logger.error(
                f"Warning: Failed to initialize MARL safety manager for vehicle {self.actor_id}: {e}")
            import traceback
            traceback.print_exc()

    # --------------------------------------------------------------------- #
    # Helper functions
    # --------------------------------------------------------------------- #
    def _merge_config(self, config: Dict[str, Any] = {}):
        """
        Deep merge configuration with defaults.
        """
        import copy
        from collections.abc import Mapping

        def deep_merge(dct, merge_dct):
            """Recursive dict merge. Inspired by dict.update(), but for nested dicts."""
            for k, v in merge_dct.items():
                if k in dct and isinstance(dct[k], Mapping) and isinstance(v, Mapping):
                    deep_merge(dct[k], v)
                else:
                    dct[k] = v

        # Start with a deep copy of defaults
        default_vm_cfg = get_vehicle_manager_defaults()
        merged_cfg = copy.deepcopy(default_vm_cfg)
        deep_merge(merged_cfg, config)
        return merged_cfg

    def _print_vm_cfg(self):
        import pprint
        print("="*100)
        print("Merged VM Config:")
        pprint.pprint(self.vm_cfg)
        print("="*100)
        print("Default VM Config:")
        pprint.pprint(get_vehicle_manager_defaults())
        print("="*100)

    def get_vm(self):
        try:
            agent = AgentFactory.get_agent(self.agent_type,
                                           self.vehicle, self.carla_map,
                                           self.config)
            vm = VehicleManager(
                vehicle=self.vehicle,
                config_yaml=self.vm_cfg,
                application=['single'],
                carla_map=self.carla_map,
                cav_world=self.cav_world,
                agent=agent
            )


            return vm
        except Exception as e:
            logger.error(
                f"Error getting agent for vehicle {self.actor_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_carla_id(self) -> int:
        """Get the CARLA actor ID of this vehicle.

        Returns:
            int: CARLA actor ID
        """
        return self.actor_id

    # --------------------------------------------------------------------- #
    # Cleanup
    # --------------------------------------------------------------------- #

    def destroy(self):
        """
        Destroy the vehicle manager and all its sensors.

        Note: The VehicleManager.destroy() now handles proper sensor cleanup
        before destroying the vehicle to prevent Signal 11 crashes.
        """
        try:
            self.cav_world.remove_vehicle_manager(self.vm)
            if self.vm:
                self.vm.destroy()
                self.vm = None
        except Exception as e:
            logger.debug(f"Error destroying vehicle manager: {e}")
        
