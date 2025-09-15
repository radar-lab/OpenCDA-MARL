'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 20:36:17
FilePath     : /OpenCDA-MARL/opencda_marl/core/agents/agent_factory.py
Description  : 
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from typing import Dict, Any

import carla

from opencda_marl.core.agents.marl_behavior_agent import MARLBehaviorAgent
from opencda_marl.core.agents.vanilla_agent import VanillaAgent
from opencda_marl.core.agents.rule_based_agent import RuleBasedAgent
from opencda_marl.core.agents.marl_agent import MARLAgent


class AgentFactory:
    @staticmethod
    def get_available_types() -> list:
        """Get list of all available agent types."""
        return ["behavior", "vanilla", "rule_based", "marl"]

    @staticmethod
    def get_baseline_types() -> list:
        """Get list of baseline agent types for benchmarking."""
        return ["behavior", "vanilla", "rule_based"]
    
    @staticmethod
    def get_rl_types() -> list:
        """Get list of reinforcement learning agent types."""
        return ["marl"]

    @staticmethod
    def validate_type(agent_type: str) -> bool:
        """Check if agent type is valid."""
        return agent_type.lower() in AgentFactory.get_available_types()

    @staticmethod
    def get_agent(agent_type: str, vehicle: carla.Vehicle, carla_map: carla.Map,
                  config: Dict[str, Any]):
        agent_type = agent_type.lower()
        behavior_config = config.get(agent_type, {})
        if agent_type == "behavior":
            behavior_config = config.get('vehicle', {}).get(agent_type, {})
            return MARLBehaviorAgent(vehicle, carla_map, behavior_config)
        elif agent_type == "vanilla":
            return VanillaAgent(vehicle, carla_map, behavior_config)
        elif agent_type == "rule_based":
            return RuleBasedAgent(vehicle, carla_map, behavior_config)
        elif agent_type == "marl":
            return MARLAgent(vehicle, carla_map, behavior_config)
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")
