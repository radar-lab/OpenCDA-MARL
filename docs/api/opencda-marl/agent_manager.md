# Agent Manager API

The Agent Manager system is the core component responsible for multi-agent coordination in OpenCDA-MARL. It manages the lifecycle of all vehicles and their associated agents, bridging traffic generation with intelligent vehicle control.

!!! success "Implementation Status"
    The Agent Manager system is **fully implemented** and provides comprehensive multi-agent management capabilities including spawning, coordination, and lifecycle management.

```text
Agent Management System
├── MARLAgentManager      # Central multi-agent coordinator
├── Agent Factory         # Agent type creation and selection
├── Agent Types           # Baseline and MARL agent implementations
├── Vehicle Adapters      # Vehicle-agent interface bridges
└── Configuration System  # Agent behavior and type configuration
```

## MARLAgentManager

=== "Multi-Agent Coordination Flow"

    ```mermaid
    graph TB
        subgraph "Traffic Generation"
            TM[TrafficManager] --> SM[Spawn Events]
            SM --> AM[MARLAgentManager]
        end
        
        subgraph "Agent Management"
            AM --> AF[AgentFactory]
            AF --> AT{Agent Type}
            AT -->|behavior| BA[BehaviorAgent]
            AT -->|vanilla| VA[VanillaAgent] 
            AT -->|rule_based| RA[RuleBasedAgent]
            AT -->|marl| MA[MARLAgent]
        end
        
        subgraph "Vehicle Control"
            BA --> VAD[VehicleAdapter]
            VA --> VAD
            RA --> VAD
            MA --> VAD
            VAD --> VC[Vehicle Control]
        end
        
        subgraph "Simulation Step"
            VC --> US[Update State]
            US --> AC[Apply Control]
            AC --> CD[Collision Detection]
            CD --> LM[Lifecycle Management]
        end
        
        classDef traffic fill:#e3f2fd
        classDef management fill:#f3e5f5
        classDef control fill:#e8f5e8
        classDef simulation fill:#fff3e0
        
        class TM,SM traffic
        class AM,AF,AT,BA,VA,RA,MA management
        class VAD,VC control
        class US,AC,CD,LM simulation
    ```

=== "Core Functionality"

    The `MARLAgentManager` serves as the central coordinator for all vehicle agents in the simulation.

    ```python
    # opencda_marl/core/agent_manager.py
    class MARLAgentManager:
        """
        Manages all vehicles and their agents in MARL scenarios.
        
        Features:
        - Processes TrafficManager spawn events
        - Creates vehicles with configured agent types
        - Coordinates multi-agent simulation steps
        - Handles vehicle lifecycle and cleanup
        - Integrates with collision detection system
        """
        
        def __init__(self, config: Dict[str, Any], state: Dict[str, Any],
                     world: carla.World, cav_world):
            """
            Initialize Agent Manager.

            Parameters
            ----------
            config : dict
                Agent configuration including agent_type and behavior settings
            state : dict
                Shared simulation state (traffic events, map data)
            world : carla.World
                CARLA world instance
            cav_world : CavWorld
                CAV world instance for vehicle management
            """
    ```

## Key Methods

=== "Simulation Control"

    ```python
    def step_simulation(self, external_actions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute one simulation step for all agents.
        
        Process:
        1. Pull spawn events from traffic manager
        2. Spawn new vehicles with configured agents
        3. Update all vehicle states and run control steps
        4. Apply vehicle controls and handle collisions
        5. Clean up destroyed vehicles
        
        Parameters
        ----------
        external_actions : dict, optional
            External actions for agents (e.g., from MARL algorithms)
            Format: {agent_id: target_speed} or {agent_id: {'target_speed': value}}
            
        Returns
        -------
        dict
            Step statistics including:
            - step_count: Current simulation step
            - spawn: Spawn attempt statistics
            - control: Vehicle control statistics
            - active_agents: List of active agent IDs
            - total_actors: Total number of active vehicles
            
        Example
        -------
        # Autonomous mode
        stats = agent_manager.step_simulation()
        
        # With external control
        external_actions = {
            'agent_001': 25.0,  # Target speed in km/h
            'agent_002': {'target_speed': 30.0}
        }
        stats = agent_manager.step_simulation(external_actions)
        """
    ```

=== "Agent Information"

    ```python
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get comprehensive agent system information.
        
        Returns
        -------
        dict
            Agent system statistics:
            - total_vehicles: Number of spawned vehicles
            - total_agents: Number of active agents (equals total_vehicles in MARL)
            - vehicles_without_agents: Should be 0 in MARL mode
            - agent_coverage_ratio: Should be 1.0 in MARL mode
            - collision_enabled_vehicles: Vehicles with collision detection
            - collision_sensor_coverage: Collision detection coverage ratio
            
        Example
        -------
        info = agent_manager.get_agent_info()
        print(f"Active agents: {info['total_agents']}")
        print(f"Coverage: {info['agent_coverage_ratio']*100:.1f}%")
        """
        
    def get_collision_debug_info(self) -> Dict[str, Any]:
        """
        Get detailed collision system debug information.
        
        Returns
        -------
        dict
            Collision system debug info:
            - collision_marked_for_destruction: Vehicles marked for collision removal
            - vehicle_adapters: Per-vehicle adapter information
        """
    ```

=== "Lifecycle Management"

    ```python
    def reset_episode(self) -> None:
        """
        Reset agent manager for new episode.
        
        Resets step counter while preserving active vehicles.
        Used for multi-episode training scenarios.
        """
        
    def cleanup(self) -> None:
        """
        Clean up all vehicles and adapters.
        
        Called at simulation end to properly destroy all vehicles
        and clean up resources.
        """
    ```

## Agent Factory System

=== "Agent Type Creation"

    The `AgentFactory` provides centralized agent creation based on configuration:

    ```python
    # opencda_marl/core/agents/agent_factory.py
    class AgentFactory:
        """
        Factory class for creating different agent types.
        
        Supported Agent Types:
        - behavior: Standard OpenCDA BehaviorAgent (fallback)
        - vanilla: Enhanced collision avoidance agent
        - rule_based: 3-stage rule-based intersection agent
        - marl: MARL agent with RL speed control
        """
        
        @staticmethod
        def create_agent(agent_type: str, vehicle: carla.Vehicle, 
                        carla_map: carla.Map, behavior_config: Dict[str, Any],
                        **kwargs) -> BehaviorAgent:
            """Create agent based on specified type."""
            
        @staticmethod
        def get_available_types() -> List[str]:
            """Get all available agent types."""
            return ["vanilla", "rule_based", "marl", "behavior"]
            
        @staticmethod
        def get_baseline_types() -> List[str]:
            """Get baseline agent types for benchmarking."""
            return ["behavior", "vanilla", "rule_based"]
    ```

=== "Agent Type Selection"

    ```yaml
    # configs/marl/intersection.yaml - Agent type configuration
    agents:
      agent_type: "vanilla"              # Primary agent type selection
      
      # Agent-specific configurations
      agent_kwargs:
        algorithm: "PPO"                 # For MARL agents
        blend: 0.7                       # Action blending factor
        clip_kmh: [0.0, 70.0]           # Speed clipping range
    ```

=== "Runtime Agent Creation"

    ```python
    # Example of runtime agent creation
    def create_agent_for_vehicle(vehicle, config):
        agent_type = config.get("agent_type", "behavior")
        behavior_config = config.get("agent_behavior", {})
        
        # Create agent through factory
        agent = AgentFactory.create_agent(
            agent_type=agent_type,
            vehicle=vehicle,
            carla_map=carla_map,
            behavior_config=behavior_config,
            **config.get("agent_kwargs", {})
        )
        
        return agent
    ```

## Agent Types and Configuration

=== "BehaviorAgent (OpenCDA Standard)"

    Standard OpenCDA autonomous driving agent with full perception and planning pipeline.

    ```yaml
    # Agent configuration
    agents:
    agent_type: "behavior"
    
    agent_behavior:
        max_speed: 35                      # Maximum speed (km/h)
        safety_time: 3.0                   # TTC safety threshold
        emergency_param: 0.4               # Emergency braking aggressiveness
        collision_time_ahead: 1.5          # Collision prediction horizon
        min_proximity_threshold: 10        # Minimum collision check distance
        overtake_allowed: true             # Allow overtaking maneuvers
        ignore_traffic_light: false        # Respect traffic lights
        sample_resolution: 1.0             # Waypoint sampling resolution
        
        # Local planner settings
        local_planner:
        buffer_size: 12                  # Waypoint buffer size
        trajectory_update_freq: 15       # Trajectory update frequency
        waypoint_update_freq: 9          # Waypoint update frequency
        min_dist: 2                      # Waypoint pop distance
        trajectory_dt: 0.10              # Trajectory sampling time
    ```

    **Features:**
    
    - Complete OpenCDA autonomous driving stack
    - Perception, planning, and control integration
    - Production-ready performance
    - Proven baseline for comparison

=== "VanillaAgent (Enhanced Safety)"

    Enhanced BehaviorAgent with improved perception-based collision avoidance.

    ```yaml
    # VanillaAgent configuration
    agents:
    agent_type: "vanilla"
    
    vanilla:
        intersection_safety_multiplier: 2.0 # Safety boost at intersections
        intersection_detection_distance: 50.0 # Intersection detection range
        multi_vehicle_ttc: true            # Track multiple vehicles for TTC
        max_vehicles_to_track: 5           # Maximum vehicles to track
        lateral_check_distance: 15.0       # Lateral conflict detection range
        lateral_safety_margin: 3.0         # Lateral safety buffer
        min_safety_distance: 8.0           # Minimum safety distance
        prediction_horizon: 3.0            # Collision prediction time
        conflict_threshold_angle: 45.0     # Angle threshold for conflict detection
    ```

    **Features:**

    - Multi-vehicle Time-to-Collision (TTC) tracking
    - Intersection safety multipliers
    - Lateral conflict detection
    - Predictive collision avoidance
    - Enhanced perception-based safety

=== "RuleBasedAgent (3-Stage Rules)"

    Three-stage rule-based agent specifically designed for intersection management.

    ```yaml
    # RuleBasedAgent configuration  
    agents:
    agent_type: "rule_based"
    
    rule_based:
        # Stage 1: Junction Management
        junction_approach_distance: 70.0   # Junction detection distance
        junction_conflict_distance: 50.0   # Conflict check distance
        cautious_speed: 20.0               # Speed when conflicts detected
        min_heading_diff_deg: 60           # Minimum heading difference for conflict
        max_heading_diff_deg: 120          # Maximum heading difference for conflict
        
        # Stage 2: Car Following
        time_headway: 2.0                  # Following time headway
        following_gain: 0.5                # Car following controller gain
        minimum_distance_buffer: 5.0       # Minimum following distance
        same_lane_tolerance_deg: 30        # Same lane detection tolerance
        front_cone_angle_deg: 45           # Front detection cone angle
        
        # Stage 3: Cruising
        max_speed: 35                      # Maximum cruising speed
    ```

    **Three-Stage Logic:**

    1. **Junction Management**: Approach detection and conflict resolution
    2. **Car Following**: Time headway-based following control
    3. **Cruising**: Target speed maintenance

=== "MARLAgent (Reinforcement Learning)"

    MARL agent with RL-controlled speed. The local planner handles steering and waypoint following; the RL algorithm controls speed only.

    ```yaml
    # MARLAgent configuration
    agents:
      agent_type: "marl"

    MARL:
      algorithm: "td3"                     # td3, dqn, q_learning, mappo, sac
      state_dim: 8                         # Observation vector dimension
      action_dim: 1                        # Speed control only
      training: true
    ```

    **Features:**

    - Five RL algorithms: TD3, DQN, Q-Learning, MAPPO, SAC
    - Speed-only control (local planner handles steering)
    - Configurable observation features via ObservationExtractor
    - Training and evaluation modes

## Multi-Agent Coordination Examples

=== "Basic Multi-Agent Setup"

    ```python
    # Initialize agent manager with multiple agent types
    config = {
        "agent_type": "vanilla",
        "agent_behavior": {
            "max_speed": 35,
            "safety_time": 3.0,
            "emergency_param": 0.4
        },
        "vanilla": {
            "intersection_safety_multiplier": 2.0,
            "multi_vehicle_ttc": True,
            "max_vehicles_to_track": 5
        }
    }

    agent_manager = MARLAgentManager(
        traffic_manager=traffic_manager,
        config=config,
        cav_world=cav_world
    )
    ```

=== "Simulation Step with External Control"

    ```python
    # Multi-agent simulation with external actions
    def run_multi_agent_simulation():
        for step in range(1000):
            # Get external actions from RL algorithms
            external_actions = {}
            
            # Get current observations
            agent_info = agent_manager.get_agent_info()
            active_agents = agent_info.get('active_agents', [])
            
            # Generate actions for each agent
            for agent_id in active_agents:
                # Example: RL policy action
                observation = get_agent_observation(agent_id)
                action = rl_policy.get_action(observation)
                external_actions[agent_id] = action
            
            # Execute simulation step
            step_stats = agent_manager.step_simulation(external_actions)
            
            # Log statistics
            print(f"Step {step}: {step_stats['total_actors']} active vehicles")
            if step_stats['control']['collision_destroyed'] > 0:
                print(f"Collisions: {step_stats['control']['collision_destroyed']}")
            
            # World step
            world.tick()
    ```

=== "Agent Performance Monitoring"

    ```python
    # Monitor agent performance during simulation
    def monitor_agent_performance(agent_manager):
        info = agent_manager.get_agent_info()
        collision_info = agent_manager.get_collision_debug_info()
        
        print("=== Agent System Status ===")
        print(f"Active Vehicles: {info['total_vehicles']}")
        print(f"Active Agents: {info['total_agents']}")
        print(f"Agent Coverage: {info['agent_coverage_ratio']*100:.1f}%")
        print(f"Collision Detection Coverage: {info['collision_sensor_coverage']*100:.1f}%")
        
        print("\n=== Collision System ===")
        print(f"Vehicles Marked for Destruction: {collision_info['collision_marked_for_destruction']}")
        
        return info
    ```

=== "Multi-Episode Training Integration"

    ```python
    # Multi-episode training with agent manager
    def run_training_episodes(num_episodes=100):
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Reset for new episode
            agent_manager.reset_episode()
            
            episode_data = []
            for step in range(300):  # 5-minute episodes
                # Collect observations
                observations = collect_observations(agent_manager)
                
                # Get actions from RL algorithms
                actions = get_rl_actions(observations)
                
                # Execute step
                step_stats = agent_manager.step_simulation(actions)
                
                # Store experience
                episode_data.append({
                    'observations': observations,
                    'actions': actions,
                    'step_stats': step_stats
                })
                
                # Check episode termination
                if should_terminate_episode(step_stats):
                    break
            
            # Process episode data for training
            process_episode_data(episode_data)
            
            # Episode cleanup (vehicles auto-cleaned between episodes)
            print(f"Episode completed with {len(episode_data)} steps")
    ```

=== "General Configuration Structure"

    ```yaml
    # Complete agent configuration template
    agents:
    # Global agent settings
    agent_type: "vanilla"                # Primary agent type
    
    # OpenCDA behavior configuration (inherited by all agent types)
    agent_behavior:
        max_speed: 35                      # Maximum speed (km/h)
        tailgate_speed: 40                 # Tailgating speed limit
        ...

    # Agent-specific configurations
    vanilla:
        intersection_safety_multiplier: 2.0
        intersection_detection_distance: 50.0
        multi_vehicle_ttc: true
        max_vehicles_to_track: 5
        lateral_check_distance: 15.0
        lateral_safety_margin: 3.0
        min_safety_distance: 8.0
        prediction_horizon: 3.0
        conflict_threshold_angle: 45.0

    rule_based:
        # Junction Management
        junction_approach_distance: 70.0
        junction_conflict_distance: 50.0
        cautious_speed: 20.0
        min_heading_diff_deg: 60
        max_heading_diff_deg: 120
        # Car Following
        time_headway: 2.0
        following_gain: 0.5
        minimum_distance_buffer: 5.0
        same_lane_tolerance_deg: 30
        front_cone_angle_deg: 45
        # Cruising
        max_speed: 35

    # MARL algorithm configuration
    MARL:
      algorithm: "td3"
      state_dim: 8
      action_dim: 1
      training: true
    ```

## Integration Points

=== "With MARLEnvironment"

    ```python
    # Agent manager integration with MARL environment
    class MARLEnvironment:
        def __init__(self, scenario_manager):
            self.scenario_manager = scenario_manager
            self.agent_manager = scenario_manager.agent_manager
        
        def step(self, actions):
            # Pass actions to agent manager
            step_stats = self.agent_manager.step_simulation(actions)
            
            # Get observations from active agents
            observations = self._get_observations()
            rewards = self._calculate_rewards(step_stats)
            dones = self._check_termination(step_stats)
            
            return observations, rewards, dones, step_stats
    ```

=== "With Benchmark System"

    ```python
    # Benchmark integration for performance testing
    def run_agent_benchmark(agent_types, scenarios):
        results = {}
        
        for agent_type in agent_types:
            for scenario in scenarios:
                # Configure agent manager with specific type
                config = load_config(scenario)
                config['agent_type'] = agent_type
                
                # Run test
                agent_manager = MARLAgentManager(traffic_manager, config, cav_world)
                performance = run_test_scenario(agent_manager, scenario)
                
                results[f"{agent_type}_{scenario}"] = performance
        
        return results
    ```
