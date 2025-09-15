# MARL Environment API

!!! info "Implementation Status"
    The MARL Environment is **fully implemented** as a custom environment that provides direct integration with CARLA and OpenCDA.

The MARL Environment provides a clean, direct interface for multi-agent reinforcement learning that's perfectly suited for CARLA's dynamic vehicle nature.

```text
MARLEnvironment
├── Episode Management    # reset_episode(), step execution
├── Observation System    # Direct vehicle state access
├── Reward Calculation    # Multi-objective reward functions
├── Termination Logic     # Collision, completion, timeout
└── MARLTrainer          # Training and evaluation workflows
```

## Core Classes

=== "MARLEnvironment"

    Custom MARL environment with direct CARLA integration.

    ```python
    class MARLEnvironment:
        """
        Custom MARL Environment for OpenCDA.
        
        Provides clean, direct interface without Gym's complexity.
        Perfect fit for CARLA's dynamic vehicle nature.
        """
    ```

=== "Constructor"

    ```python
    def __init__(self, scenario_manager, config: Dict[str, Any]):
        """
        Initialize MARL Environment.
        
        Parameters
        ----------
        scenario_manager : MARLScenarioManager
            The scenario manager to wrap
        config : dict
            Environment configuration
        """
    ```

## Key Methods

=== "Episode Management"

    ```python
    def reset_episode(self) -> Dict[str, Any]:
        """
        Reset for new episode.
        
        Returns
        -------
        episode_info : dict
            Initial episode information with observations
            - episode: Episode count
            - step: Step count (0)
            - observations: Initial agent observations
            - agent_ids: List of active agent IDs
            - active_agents: Number of active agents
        """
    
    def step(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute environment step.
        
        Parameters
        ----------
        actions : dict
            Actions for agents (agent_id -> action)
        
        Returns
        -------
        step_info : dict
            Complete step information
            - step: Current step
            - episode: Current episode
            - observations: Agent observations
            - rewards: Agent rewards
            - dones: Agent termination states
            - scenario_info: Scenario manager results
            - agent_ids: Active agent IDs
            - active_agents: Number of active agents
            - done: Episode termination
            - truncated: Maximum steps reached
        """
    ```

=== "Observation System"

    ```python
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get observations for all active agents.
        
        Returns
        -------
        observations : dict
            Agent observations (agent_id -> observation vector)
            
        Observation Vector Format:
            [0-2]: Position (x, y, z)
            [3-5]: Velocity (x, y, z) 
            [6-8]: Rotation (pitch, yaw, roll)
            [9]: Speed magnitude
        """
    ```

=== "Reward Calculation"

    ```python
    def _calculate_rewards(self, scenario_result: Dict) -> Dict[str, float]:
        """
        Calculate rewards for all agents.
        
        Multi-objective reward function:
        1. Forward progress reward: min(speed / 10.0, 1.0)
        2. Safety reward: +1.0 if distance > 5.0m, -5.0 if < 2.0m
        3. Efficiency reward: +2.0 for ideal speed range (5-15 m/s)
        
        Returns
        -------
        rewards : dict
            Agent rewards (agent_id -> reward value)
        """
    ```

=== "Termination Logic"

    ```python
    def _check_termination(self) -> Dict[str, bool]:
        """
        Check termination conditions for each agent.
        
        Returns
        -------
        dones : dict
            Agent termination states (agent_id -> done boolean)
            
        Termination Conditions:
        - Collision (instant destruction enabled)
        - Destination reached (planned)
        - Simulation error
        """
    ```

## MARLTrainer Integration

=== "MARLTrainer Class"

    ```python
    class MARLTrainer:
        """
        Custom MARL Trainer for multi-agent scenarios.
        
        Provides training/evaluation loops without Gym dependency.
        """
        
        def __init__(self, environment: MARLEnvironment):
            """Initialize trainer with environment."""
    ```

=== "Training Methods"

    ```python
    def train_episode(self, agents: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train single episode.
        
        Parameters
        ----------
        agents : dict, optional
            Agent instances for training (agent_id -> agent)
        
        Returns
        -------
        episode_stats : dict
            Episode training statistics
            - episode: Episode number
            - steps: Number of steps
            - rewards: Per-agent rewards
            - total_reward: Sum of all rewards
            - avg_reward: Average reward per agent
            - active_agents: Number of agents
        """
    
    def train_multiple_episodes(self, num_episodes: int, agents: Dict[str, Any] = None) -> List[Dict]:
        """Train multiple episodes and return statistics."""
    
    def evaluate(self, num_episodes: int = 10, agents: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate agents over multiple episodes.
        
        Returns
        -------
        evaluation_summary : dict
            - episodes: Number of evaluation episodes
            - avg_reward: Average reward across episodes
            - avg_steps: Average steps per episode
            - success_rate: Rate of positive reward episodes
            - detailed_results: Full episode statistics
        """
    ```

## Usage Examples

=== "Basic Training Setup"

    ```python
    from opencda_marl.core.marl_environment import MARLEnvironment, MARLTrainer
    
    # Create environment with scenario manager
    environment = MARLEnvironment(scenario_manager, config)
    trainer = MARLTrainer(environment)
    
    # Single episode training
    episode_stats = trainer.train_episode()
    print(f"Episode reward: {episode_stats['total_reward']:.3f}")
    
    # Multi-episode training
    results = trainer.train_multiple_episodes(num_episodes=100)
    avg_reward = sum(r['avg_reward'] for r in results) / len(results)
    print(f"Training average reward: {avg_reward:.3f}")
    ```

=== "Agent Integration"

    ```python
    # Training with RL agents (planned)
    agents = {
        'agent_1': PPOAgent(observation_space, action_space),
        'agent_2': PPOAgent(observation_space, action_space)
    }
    
    # Training loop with agents
    for episode in range(1000):
        episode_stats = trainer.train_episode(agents)
        
        # Update agent policies based on episode results
        for agent_id, agent in agents.items():
            agent.update(episode_stats['rewards'][agent_id])
    ```

=== "Evaluation Workflow"

    ```python
    # Evaluate trained agents
    evaluation_results = trainer.evaluate(
        num_episodes=50,
        agents=trained_agents
    )
    
    print(f"Evaluation Results:")
    print(f"  Average Reward: {evaluation_results['avg_reward']:.3f}")
    print(f"  Success Rate: {evaluation_results['success_rate']:.3f}")
    print(f"  Average Steps: {evaluation_results['avg_steps']:.1f}")
    ```

=== "Custom Reward Function"

    ```python
    class CustomMARLEnvironment(MARLEnvironment):
        """Custom environment with specialized reward function."""
        
        def _calculate_rewards(self, scenario_result: Dict) -> Dict[str, float]:
            """Custom reward calculation."""
            rewards = {}
            
            for agent_id, adapter in self.scenario_manager.agents.items():
                reward = 0.0
                
                # Custom reward components
                reward += self._cooperation_reward(agent_id, adapter)
                reward += self._efficiency_reward(agent_id, adapter)
                reward += self._safety_reward(agent_id, adapter)
                
                rewards[agent_id] = reward
            
            return rewards
    ```

## Configuration Integration

=== "Environment Configuration"

    ```yaml
    # MARL environment settings
    environment:
      max_episode_steps: 1000        # Maximum steps per episode
      reward_function: "multi_objective"  # Reward function type
      observation_type: "vector"     # Observation format
      
    # Reward function parameters
    rewards:
      progress_weight: 1.0          # Forward progress reward weight
      safety_weight: 5.0            # Safety reward weight
      efficiency_weight: 2.0        # Efficiency reward weight
      cooperation_weight: 1.0       # Cooperation reward weight (planned)
    ```