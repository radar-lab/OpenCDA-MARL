'''
Author: AXIBA leolihao@arizona.edu
Date: 2025-08-23 21:19:31
FilePath: OpenCDA/opencda_marl/utils/config_merger.py
Description: Smart Configuration Merger for MARL

Provides intelligent merging of configurations where specified fields
can be extended instead of replaced.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from typing import Dict, Union, List, Any
from omegaconf import OmegaConf, DictConfig, ListConfig
import copy


class SmartConfigMerger:
    """
    Smart configuration merger with selective extend behavior.

    Default Behavior:
    - Uses standard OmegaConf.merge (replace behavior) for all fields
    - Only fields listed in _extend_fields will be extended instead of replaced

    Usage in YAML:
    ```yaml
    _extend_fields:
      - "scenario.traffic.excluded_vehicle_types"  # Extend this list
      - "world.seed"                               # Add to base seed
    ```
    """

    @staticmethod
    def merge(base_config: Union[Dict, DictConfig],
              scenario_config: Union[Dict, DictConfig],
              extend_fields: List[str] = None) -> DictConfig:
        """
        Merge configurations with selective extend behavior.

        Parameters
        ----------
        base_config : Dict or DictConfig
            Base configuration (e.g., default.yaml)
        scenario_config : Dict or DictConfig  
            Scenario configuration (e.g., intersection.yaml)
        extend_fields : List[str], optional
            List of field paths that should extend instead of replace. 
            If None, will extract from scenario_config._extend_fields

        Returns
        -------
        DictConfig
            Merged configuration with selective extend behavior
        """
        # Convert to OmegaConf if needed
        if not isinstance(base_config, DictConfig):
            base_config = OmegaConf.create(base_config)
        if not isinstance(scenario_config, DictConfig):
            scenario_config = OmegaConf.create(scenario_config)

        # Extract extend fields from config if not provided
        if extend_fields is None:
            extend_fields = []
            if '_extend_fields' in scenario_config:
                extend_fields = list(scenario_config._extend_fields)

        # Remove _extend_fields from scenario config before merging
        scenario_clean = copy.deepcopy(scenario_config)
        if '_extend_fields' in scenario_clean:
            del scenario_clean['_extend_fields']

        # First, do standard OmegaConf merge (replace behavior)
        merged = OmegaConf.merge(base_config, scenario_clean)

        # Then, selectively extend specified fields
        for field_path in extend_fields:
            base_value = SmartConfigMerger._get_nested_value(base_config, field_path)
            scenario_value = SmartConfigMerger._get_nested_value(scenario_clean, field_path)
            
            if base_value is not None and scenario_value is not None:
                extended_value = SmartConfigMerger._extend_values(base_value, scenario_value)
                SmartConfigMerger._set_nested_value(merged, field_path, extended_value)

        return merged

    @staticmethod
    def _get_nested_value(config: DictConfig, path: str) -> Any:
        """Get a value from nested config using dot notation path."""
        try:
            keys = path.split('.')
            value = config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, AttributeError):
            return None

    @staticmethod
    def _set_nested_value(config: DictConfig, path: str, value: Any):
        """Set a value in nested config using dot notation path."""
        keys = path.split('.')
        target = config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    @staticmethod
    def _extend_values(base_value: Any, scenario_value: Any) -> Any:
        """
        Extend values based on their types.
        
        - Lists: Concatenate and remove duplicates
        - Numbers: Add together
        - Strings: Concatenate
        - Other: Return scenario value (fallback to replace)
        """
        if isinstance(base_value, (list, ListConfig)) and isinstance(scenario_value, (list, ListConfig)):
            # Extend lists, removing duplicates while preserving order
            extended = list(base_value)
            for item in scenario_value:
                if item not in extended:
                    extended.append(item)
            return extended
        
        elif isinstance(base_value, (int, float)) and isinstance(scenario_value, (int, float)):
            # Add numeric values
            return base_value + scenario_value
        
        elif isinstance(base_value, str) and isinstance(scenario_value, str):
            # Concatenate strings
            return base_value + scenario_value
        
        else:
            # For other types or mismatched types, just replace
            return scenario_value


# Convenience function for direct usage
def smart_merge(base_config: Union[Dict, DictConfig],
                scenario_config: Union[Dict, DictConfig],
                extend_fields: List[str] = None) -> DictConfig:
    """
    Convenience function for smart config merging.

    Usage:
        from opencda_marl.utils.config_merger import smart_merge

        base = OmegaConf.load('default.yaml')
        scenario = OmegaConf.load('intersection.yaml') 
        config = smart_merge(base, scenario)
        
        # Or with explicit extend fields:
        extend_fields = [
            "scenario.traffic.excluded_vehicle_types",
            "world.seed"
        ]
        config = smart_merge(base, scenario, extend_fields)
    """
    return SmartConfigMerger.merge(base_config, scenario_config, extend_fields)