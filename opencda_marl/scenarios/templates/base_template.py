'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 15:44:21
FilePath     : /OpenCDA-MARL/opencda_marl/scenarios/templates/base_template.py
Description  : Base Scenario Template for MARL scenarios.

Defines the abstract interface that all scenario templates must implement.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from typing import Dict, Any
from abc import ABC, abstractmethod

from ..scenario_manager import MARLScenarioManager


class ScenarioTemplate(ABC):
    """
    Abstract base class for scenario templates.
    
    Scenario templates are responsible for:
    - Converting high-level parameters into detailed configurations
    - Building scenario managers with appropriate settings
    - Providing scenario-specific defaults and validation
    """
    
    @abstractmethod
    def build_scenario(self,**kwargs) -> MARLScenarioManager:
        """
        Build scenario manager from configuration.
        
        Returns
        -------
        scenario_manager : MARLScenarioManager
            Configured scenario manager instance
        """
        pass
    
    @abstractmethod
    def create_config(self, **kwargs) -> Dict[str, Any]:
        """
        Create scenario configuration from parameters.
        
        Parameters
        ----------
        **kwargs
            Scenario-specific parameters
            
        Returns
        -------
        config : dict
            Complete scenario configuration dictionary
        """
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for this scenario type.
        
        Returns
        -------
        defaults : dict
            Default parameter values
        """
        pass
    
    @abstractmethod
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available parameters.
        """
        pass
    
    @abstractmethod
    def validate_config(self, **kwargs):
        """
        Validate scenario configuration.
        """
        pass
