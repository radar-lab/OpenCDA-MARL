'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 15:42:17
FilePath     : /OpenCDA-MARL/opencda_marl/scenarios/scenario_builder.py
Description  : Scenario builder for MARL scenarios.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''

from typing import List, Dict, Any

from .templates import Intersection
from .scenario_manager import MARLScenarioManager


class ScenarioBuilder:
    __templates = {
        'intersection': Intersection(),
    }

    # --------------------------------------------------------------------- #
    # Primary API
    # --------------------------------------------------------------------- #
    @classmethod
    def build_from_config(cls, **kwargs) -> MARLScenarioManager:
        """
        Build scenario manager from configuration.

        Parameters
        ----------
        **kwargs : dict
            Configuration dictionary

        Returns
        -------
        scenario_manager : MARLScenarioManager
            Configured scenario manager
        """

        # Get scenario type from kwargs
        config = kwargs.get('config', {})
        scenario_type = config.get('meta', {}).get('scenario_type', None)
        if scenario_type is None:
            raise ValueError(
                "Missing required 'scenario_type' field in meta section of configuration file")

        if scenario_type not in cls.__templates:
            available = cls.get_available_scenarios()
            raise ValueError(
                f"Unknown scenario type: {scenario_type}. "
                f"Available: {available}."
            )

        template = cls.__templates[scenario_type]

        # Build scenario using template
        return template.build_scenario(**kwargs)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    @classmethod
    def get_available_scenarios(cls) -> List[str]:
        """Get list of all scenario types (including placeholders)."""
        return list(cls.__templates.keys())

    @classmethod
    def get_template_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all available templates."""
        return {
            name: template.get_parameter_info()
            for name, template in cls.__templates.items()
        }

    # --------------------------------------------------------------------- #
    # Programmatic API
    # --------------------------------------------------------------------- #

    @classmethod
    def create_intersection_scenario(cls, **kwargs) -> MARLScenarioManager:
        """Create intersection scenario configuration."""
        template = cls.__templates['intersection']
        return template.create_config(**kwargs)
