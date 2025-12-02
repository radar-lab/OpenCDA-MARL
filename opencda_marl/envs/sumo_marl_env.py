'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-11-16
FilePath     : /OpenCDA-MARL/opencda_marl/envs/sumo_marl_env.py
Description  : SUMO-only MARL environment for fast training
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import math
import os
import sys
from typing import Dict, List, Any, Tuple
from loguru import logger
import numpy as np

# Check SUMO_HOME
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci  # pylint: disable=import-error
import sumolib  # pylint: disable=import-error

from opencda_marl.core.events import StepEvent
from opencda_marl.core.marl import MARLManager
from opencda_marl.core.marl.metrics import TrainingMetrics as Metrics
from opencda_marl.core.marl.checkpoint import CheckpointManager
from opencda_marl.core.traffic.sumo_adapter import SumoWorld, SumoMARLPlanner
from opencda_marl.core.traffic.traffic_manager import MARLTrafficManager
from opencda_marl.core.traffic.sumo_spawner import SumoVehicleSpawner


class SumoMARLEnv:
    """
    SUMO-only MARL environment for fast training.

    Uses TraCI to control SUMO vehicles directly without CARLA rendering.
    Provides same observation/action space as CARLA-based MARLEnv for transfer learning.
    """

    def __init__(self, config: Dict):
        """
        Initialize SUMO MARL environment.

        Args:
            config: Full configuration dictionary including:
                - meta.sumo_cfg: Path to SUMO configuration file
                - world.fixed_delta_seconds: Simulation step length
                - scenario.simulation: Episode parameters
                - MARL: Algorithm configuration
                - rewards: Reward parameters
        """
        self.config = config

        # Extract configuration sections
        self.meta_config = config.get('meta', {})
        self.world_config = config.get('world', {})
        self.scenario_config = config.get('scenario', {})
        self.simulation_config = self.scenario_config.get('simulation', {})
        self.marl_config = config.get('MARL', {})
        self.training_config = self.marl_config.get('training', {})

        # SUMO connection parameters
        self.sumo_cfg = self.meta_config.get('sumo_cfg')
        if not self.sumo_cfg or not os.path.isfile(self.sumo_cfg):
            raise ValueError(f"SUMO configuration file not found: {self.sumo_cfg}")

        self.step_length = self.world_config.get('fixed_delta_seconds', 0.05)
        self.sumo_port = self.world_config.get('sumo_port', 8873)
        self.use_gui = self.world_config.get('sumo_gui', False)

        # Episode parameters
        self.max_steps = self.simulation_config.get('max_steps', 2400)
        self.max_episodes = self.simulation_config.get('max_episodes', 1000)

        # State tracking
        self.current_step = 0
        self.current_episode = 0
        self.total_steps = 0

        # Vehicle tracking
        self.active_vehicles = {}  # {veh_id: vehicle_info}
        self.departed_vehicles = set()  # Vehicles that have arrived
        self.collided_vehicles = set()  # Vehicles involved in collisions

        # Intersection center (will be calculated from network)
        self.intersection_center = None

        # Traffic Manager (will be initialized after SUMO starts)
        self.traffic_manager = None
        self.vehicle_spawner = None
        self.use_traffic_manager = self.scenario_config.get('traffic', {}).get('mode') in ['live', 'record', 'replay']

        # MARL Manager
        algorithm = self.marl_config.get('algorithm', 'td3')
        self.marl_manager = MARLManager(self.marl_config, algorithm)

        # Reward parameters
        self.reward_params = self._default_reward_params()
        self.reward_params.update(self.marl_config.get('rewards', {}))

        # Episode tracking
        self.episode_events = []
        self.previous_observations = {}
        self.terminal_agents = set()
        self.current_step_rewards = {}

        # Training metrics (export history every N episodes)
        metrics_export_interval = self.training_config.get('metrics_export_interval', 10)
        metrics_export_dir = self.training_config.get('metrics_export_dir', 'metrics_history')
        self.metrics = Metrics(export_interval=metrics_export_interval, export_dir=metrics_export_dir)

        # Training mode
        self.is_training_mode = self.training_config.get('training_mode', True)

        # Checkpoint manager
        if self.is_training_mode:
            checkpoint_dir = self.training_config.get(
                'checkpoint_dir', f'checkpoints/sumo_{algorithm}')
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir, algorithm)

            # Load checkpoint if specified
            load_checkpoint = self.training_config.get('load_checkpoint')
            if load_checkpoint:
                self._load_checkpoint_from_config(load_checkpoint)
        else:
            self.checkpoint_manager = None

        # Initialize SUMO
        self._start_sumo()

    def _start_sumo(self):
        """Start SUMO simulation via TraCI."""
        logger.info(f"Starting SUMO with config: {self.sumo_cfg}")

        # Choose SUMO binary
        if self.use_gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')

        # Start TraCI
        sumo_cmd = [
            sumo_binary,
            '--step-length', str(self.step_length),
            '--collision.action', 'teleport',  # Remove vehicles on collision (realistic for transfer learning)
            '--collision.check-junctions', 'true',
            '--collision.mingap-factor', '1.0',  # Minimum gap for collision detection
            '--no-warnings', 'false',
            '--quit-on-end', 'false',  # Don't auto-quit
        ]

        # If using traffic manager, only load network file (no routes)
        if self.use_traffic_manager:
            # Extract network file from sumocfg
            net_file = self._get_network_file_from_config()
            sumo_cmd.extend(['--net-file', net_file])
            logger.info("Using MARLTrafficManager - loading network only (no route file)")
        else:
            # Use full config (includes routes)
            sumo_cmd.extend(['--configuration-file', self.sumo_cfg])
            logger.info("Using static route file from sumocfg")

        # Add auto-start flag for GUI to prevent manual start button
        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd, port=self.sumo_port)

        # Calculate intersection center from network
        self._calculate_intersection_center()

        # Initialize traffic manager if using dynamic traffic
        if self.use_traffic_manager:
            self._initialize_traffic_manager()

        logger.info("SUMO started successfully")

    def _get_network_file_from_config(self) -> str:
        """Extract network file path from SUMO configuration file."""
        import xml.etree.ElementTree as ET
        from pathlib import Path

        # Parse sumocfg file
        tree = ET.parse(self.sumo_cfg)
        root = tree.getroot()

        # Find net-file element
        net_elem = root.find('.//net-file')
        if net_elem is None:
            raise ValueError(f"No net-file found in {self.sumo_cfg}")

        net_file = net_elem.get('value')

        # Resolve relative path
        cfg_dir = Path(self.sumo_cfg).parent
        net_path = cfg_dir / net_file

        return str(net_path)

    def _calculate_intersection_center(self):
        """Calculate the center of the intersection from SUMO network."""
        # Get all junctions
        junctions = traci.junction.getIDList()

        if not junctions:
            logger.warning("No junctions found in SUMO network")
            self.intersection_center = (0.0, 0.0)
            return

        # Find junction 4 (main intersection in our network)
        # This ensures coordinate consistency with CARLA
        main_junction = '4' if '4' in junctions else junctions[0]
        pos_sumo = traci.junction.getPosition(main_junction)

        # IMPORTANT: Convert SUMO coordinates back to CARLA coordinates
        # SUMO applies netOffset (99.80, 100.00) during conversion
        # To get CARLA coordinates: carla_coord = sumo_coord - offset
        net_offset = (99.80, 100.00)
        pos_carla = (pos_sumo[0] - net_offset[0], pos_sumo[1] - net_offset[1])

        # Store in CARLA coordinate system for consistency with state features
        self.intersection_center = pos_carla

        logger.info(f"Intersection center (CARLA coords): {self.intersection_center} from SUMO junction {main_junction} at {pos_sumo}")
        logger.debug(f"Available junctions: {junctions}")

    def _initialize_traffic_manager(self):
        """Initialize MARLTrafficManager with SUMO adapter."""
        logger.info("Initializing MARL Traffic Manager for SUMO")

        try:
            # Create SUMO world adapter
            sumo_world = SumoWorld()
            logger.debug("SUMO world adapter created")

            # Get traffic configuration
            traffic_config = self.scenario_config.get('traffic', {})
            logger.debug(f"Traffic config: {traffic_config}")

            # Create state dict for traffic manager
            state = {
                'step': 0,
                'episode': 0
            }

            # Initialize traffic manager with SUMO world
            logger.debug("Creating MARLTrafficManager...")
            self.traffic_manager = MARLTrafficManager(
                world=sumo_world,
                config=traffic_config,
                state=state,
                fix_dlt=self.step_length
            )
            logger.debug("MARLTrafficManager created")

            # Initialize vehicle spawner
            self.vehicle_spawner = SumoVehicleSpawner()
            logger.debug("Vehicle spawner created")

            logger.success(f"Traffic Manager initialized with {self.traffic_manager.total_events} spawn events")

        except Exception as e:
            logger.error(f"Failed to initialize traffic manager: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _default_reward_params(self) -> Dict:
        """Default reward parameters matching CARLA MARLEnv."""
        return {
            'collision': -500.0,
            'success': 400.0,
            'step_penalty': -1.5,
            'speed_bonus': 0.5,
            'speed_threshold': 45.0,  # km/h
        }

    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.

        Returns:
            Dict containing rewards and metrics
        """
        # Spawn vehicles from traffic manager
        if self.use_traffic_manager and self.traffic_manager:
            spawn_events = self.traffic_manager.update(self.current_step)
            if spawn_events:
                self.vehicle_spawner.spawn_vehicles(spawn_events)

        # Get current observations
        observations = self._get_observations()

        if not observations:
            # No active vehicles yet - step simulation
            traci.simulationStep()
            self.current_step += 1
            self.total_steps += 1
            return {'rewards': {}, 'done': self.current_step >= self.max_steps}

        # Compute actions from MARL manager
        target_speeds = self.marl_manager.compute_actions(
            observations, training=self.is_training_mode)

        # Apply actions to SUMO vehicles
        self._apply_actions(target_speeds)

        # Step SUMO simulation
        traci.simulationStep()

        # Check for events (collisions, arrivals)
        events = self._check_events()

        # Calculate rewards for the current transition (state -> action -> next_state)
        rewards = self._calculate_rewards(events, observations)

        # Get next observations for learning
        next_observations = self._get_observations()

        # Update MARL algorithm with CURRENT step's transition
        # Transition: (observations, action, reward, next_observations)
        # - observations: state before action (S_t)
        # - action: stored in last_actions during compute_actions
        # - reward: calculated for taking action from observations
        # - next_observations: state after action (S_t+1)
        if self.is_training_mode:
            self.marl_manager.update(
                rewards=rewards,
                observations=observations,  # Use current observations, not previous!
                next_observations=next_observations
            )

        # Store for metrics/debugging only (not used for learning anymore)
        self.previous_observations = observations.copy()
        self.current_step_rewards = rewards

        # Update step counters
        self.current_step += 1
        self.total_steps += 1

        # Update metrics with observations for traffic performance tracking
        self._update_metrics(rewards, events, observations)

        # Check if episode is done
        done = self.current_step >= self.max_steps or not next_observations

        return {
            'rewards': rewards,
            'events': events,
            'done': done,
            'observations': next_observations
        }

    def _get_observations(self) -> Dict[int, Dict]:
        """
        Extract observations from SUMO.

        Returns same format as CARLA MARLEnv for compatibility.
        """
        observations = {}

        # Get all vehicle IDs in simulation
        vehicle_ids = traci.vehicle.getIDList()

        for veh_id in vehicle_ids:
            # Skip if vehicle has terminated
            if veh_id in self.terminal_agents:
                continue

            try:
                # Get vehicle state from SUMO
                pos = traci.vehicle.getPosition(veh_id)
                speed_ms = traci.vehicle.getSpeed(veh_id)
                speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
                angle_deg = traci.vehicle.getAngle(veh_id)
                angle_rad = math.radians(angle_deg)

                # Get leader vehicle
                leader_info = traci.vehicle.getLeader(veh_id, 100.0)  # Look ahead 100m
                if leader_info is not None:
                    distance_to_front_vehicle = leader_info[1]
                else:
                    distance_to_front_vehicle = 999.0  # No vehicle ahead

                # Calculate relative position to intersection
                rel_x = pos[0] - self.intersection_center[0]
                rel_y = pos[1] - self.intersection_center[1]

                # Calculate distance to intersection
                distance_to_intersection = math.sqrt(rel_x**2 + rel_y**2)

                # Get lane position (simplified)
                # 0 = at intersection, 1 = left lane, 2 = middle, 3 = right lane
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)
                lane_position = lane_index if distance_to_intersection < 10 else lane_index + 1

                # Build observation dictionary (matching CARLA format)
                observations[veh_id] = {
                    'location': {'x': pos[0], 'y': pos[1], 'z': 0.0},
                    'speed': speed_kmh,
                    'heading_angle': angle_rad,
                    'relative_position_to_intersection': {'x': rel_x, 'y': rel_y},
                    'distance_to_intersection': distance_to_intersection,
                    'distance_to_front_vehicle': distance_to_front_vehicle,
                    'lane_position': lane_position,
                }

                # Track active vehicle
                if veh_id not in self.active_vehicles:
                    self.active_vehicles[veh_id] = {
                        'spawn_step': self.current_step,
                        'collisions': 0
                    }

            except traci.exceptions.TraCIException as e:
                logger.warning(f"Failed to get observation for vehicle {veh_id}: {e}")
                continue

        return observations

    def _apply_actions(self, target_speeds: Dict[str, float]):
        """
        Apply MARL actions to SUMO vehicles.

        Args:
            target_speeds: Dict[vehicle_id, speed_kmh]
        """
        for veh_id, speed_kmh in target_speeds.items():
            try:
                # Convert km/h to m/s
                speed_ms = speed_kmh / 3.6

                # Clip to reasonable range
                speed_ms = max(0.0, min(speed_ms, 30.0))  # Max ~108 km/h

                # Apply speed to SUMO vehicle
                traci.vehicle.setSpeed(veh_id, speed_ms)

            except traci.exceptions.TraCIException as e:
                logger.warning(f"Failed to set speed for vehicle {veh_id}: {e}")

    def _check_events(self) -> List[StepEvent]:
        """
        Check for events (collisions, arrivals).

        Returns:
            List of StepEvent objects
        """
        events = []

        # Check for arrived vehicles
        arrived_vehicles = traci.simulation.getArrivedIDList()
        for veh_id in arrived_vehicles:
            if veh_id not in self.departed_vehicles:
                self.departed_vehicles.add(veh_id)
                self.terminal_agents.add(veh_id)
                events.append(StepEvent(
                    step=self.current_step,
                    event_id=f'success_{veh_id}_{self.current_step}',
                    vehicle_id=veh_id,
                    event_type='success'
                ))

        # Check for collisions
        collisions = traci.simulation.getCollisions()
        for collision in collisions:
            # Get colliding vehicle IDs
            collider = collision.collider
            victim = collision.victim

            for veh_id in [collider, victim]:
                if veh_id and veh_id not in self.collided_vehicles:
                    self.collided_vehicles.add(veh_id)
                    self.terminal_agents.add(veh_id)
                    events.append(StepEvent(
                        step=self.current_step,
                        event_id=f'collision_{veh_id}_{self.current_step}',
                        vehicle_id=veh_id,
                        event_type='collision'
                    ))

        return events

    def _calculate_rewards(self, events: List[StepEvent], observations: Dict) -> Dict[str, float]:
        """
        Calculate rewards from events and observations.

        Matches CARLA MARLEnv reward structure for transfer learning.
        """
        rewards = {}

        # Process terminal events first
        for event in events:
            agent_id = event.vehicle_id

            if event.event_type == 'collision':
                rewards[agent_id] = self.reward_params['collision']
            elif event.event_type == 'success':
                rewards[agent_id] = self.reward_params['success']

        # Add step penalties and bonuses for non-terminal agents
        for agent_id, obs in observations.items():
            if agent_id in self.terminal_agents:
                continue  # Skip if already terminated

            # Step penalty (encourages faster completion)
            reward = self.reward_params['step_penalty']

            # Speed bonus (encourages maintaining speed)
            if obs['speed'] >= self.reward_params['speed_threshold']:
                reward += self.reward_params['speed_bonus']

            rewards[agent_id] = reward

        return rewards

    def _update_metrics(self, rewards: Dict, events: List[StepEvent], observations: Dict = None):
        """Update training metrics."""
        # Count successes this step for accurate episode_length tracking
        step_successes = sum(1 for e in events if e.event_type == 'success')

        # Update step metrics with all agent rewards and observations for traffic tracking
        self.metrics.update_step(rewards, observations, step_successes=step_successes)

        # Track events
        for event in events:
            if event.event_type == 'collision':
                self.metrics.collisions += 1
            elif event.event_type == 'success':
                self.metrics.successes += 1

    def reset(self):
        """Reset the environment for a new episode."""
        logger.info(f"Resetting episode {self.current_episode}")

        # Remove all vehicles
        for veh_id in list(traci.vehicle.getIDList()):
            try:
                traci.vehicle.remove(veh_id)
            except traci.exceptions.TraCIException:
                pass

        # Reset state
        self.current_step = 0
        self.active_vehicles.clear()
        self.departed_vehicles.clear()
        self.collided_vehicles.clear()
        self.terminal_agents.clear()
        self.previous_observations.clear()
        self.episode_events.clear()

        # Reset traffic manager for new episode
        if self.use_traffic_manager and self.traffic_manager:
            self.traffic_manager.reset()

        # Log episode metrics before reset (use finish_episode to compute traffic metrics before reset)
        if self.current_episode > 0:
            # Create episode states snapshot for finish_episode
            episode_states = {
                'step': self.current_step,
                'collision': self.collision_count,
                'success': self.success_count,
                'fixed_dt': self.step_length,  # For throughput calculation
            }
            # finish_episode computes traffic metrics (including max_speed) BEFORE resetting
            episode_metrics = self.metrics.finish_episode(episode_states)
            logger.info(f"Episode {self.current_episode} metrics: {episode_metrics}")

            # Reset MARL algorithm and log episode metrics to TensorBoard
            self.marl_manager.reset_episode(episode_metrics=episode_metrics)

            # Save checkpoint periodically
            if self.is_training_mode and self.checkpoint_manager:
                save_freq = self.training_config.get('save_freq', 10)
                if self.current_episode % save_freq == 0:
                    self._save_checkpoint()

        self.current_episode += 1

        # Step SUMO a few times to spawn initial vehicles
        for _ in range(10):
            traci.simulationStep()

    def close(self):
        """Close the SUMO connection."""
        logger.info("Closing SUMO environment")
        try:
            traci.close()
        except:
            pass

    def get_episode_metrics(self) -> Dict:
        """Get current episode metrics (includes traffic metrics like max_speed)."""
        metrics = self.metrics.get_current_metrics()
        # Add fixed_dt for evaluation manager
        metrics['fixed_dt'] = self.step_length
        return metrics

    def get_current_step_rewards(self) -> Dict[int, float]:
        """Get rewards from the current step for evaluation."""
        return self.current_step_rewards.copy()

    def reset_episode(self) -> Dict:
        """
        Reset episode and return metrics (alias for coordinator compatibility).
        Returns episode metrics before resetting.

        Note: This uses get_current_metrics() which returns metrics without
        resetting. The actual reset happens in self.reset().
        """
        metrics = self.get_episode_metrics()
        self.reset()
        return metrics

    def _save_checkpoint(self):
        """Save training checkpoint."""
        # Use MARLManager's built-in checkpoint saving
        checkpoint_dir = self.config.get('meta', {}).get('checkpoint_dir', 'checkpoints')
        scenario_type = self.config.get('meta', {}).get('scenario_type', 'intersection_sumo')
        checkpoint_path = f"{checkpoint_dir}/{scenario_type}/td3_episode_{self.current_episode}.pth"

        import os
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        self.marl_manager.save_checkpoint(checkpoint_path)
        logger.info(f"Checkpoint saved at episode {self.current_episode}: {checkpoint_path}")

    def _load_checkpoint_from_config(self, checkpoint_path: str):
        """Load checkpoint from file."""
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint_data = self.checkpoint_manager.load(checkpoint_path)

        if checkpoint_data:
            self.marl_manager.load_checkpoint_data(checkpoint_data)
            self.current_episode = checkpoint_data.get('episode', 0)
            self.total_steps = checkpoint_data.get('total_steps', 0)
            logger.info(f"Resumed from episode {self.current_episode}, step {self.total_steps}")
        else:
            logger.warning(f"Failed to load checkpoint: {checkpoint_path}")
