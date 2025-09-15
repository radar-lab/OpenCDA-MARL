from typing import Dict, Any
from .base_template import ScenarioTemplate
from ..scenario_manager import MARLScenarioManager
from opencda.core.common.cav_world import CavWorld


class Intersection(ScenarioTemplate):
    """
    Template for intersection negotiation scenarios.
    """
    # ============================= Scenario building =============================

    def build_scenario(self, **kwargs) -> MARLScenarioManager:
        """Build intersection scenario manager."""

        scenario_params = kwargs.get('config', {})
        meta = scenario_params.get('meta', {})
        opt = scenario_params.get('opt', {})
        cav_world = kwargs.get('cav_world', None)
        self.validate_config(meta, opt, cav_world)

        # Create scenario manager
        scenario_manager = MARLScenarioManager(
            scenario_params=scenario_params,
            apply_ml=opt.get('apply_ml'),
            town=meta.get('town'),
            xodr_path=meta.get('xodr_path'),
            cav_world=cav_world
        )

        return scenario_manager

    def validate_config(self, meta: Dict[str, Any],
                        opt: Dict[str, Any],
                        cav_world: CavWorld):
        """Validate intersection-specific configuration."""

        if 'apply_ml' not in opt:
            raise ValueError(
                "Missing required 'apply_ml' field in options/[opt.apply_ml] in config file")

        if 'town' not in meta:
            raise ValueError(
                "Missing required 'town' field in configuration")

        if cav_world is None:
            raise ValueError(
                "Missing required 'cav_world' field in options")

    # --------------------------------------------------------------------- #
    # Scenario creation
    # --------------------------------------------------------------------- #

    def create_config(self, **kwargs) -> Dict[str, Any]:
        """Create intersection scenario configuration."""
        weather_conditions = kwargs.get('weather', 'clear')
        town = kwargs.get('town', 'intersection')
        default_config = self.get_default_parameters()

        # update default config
        default_config['meta']['town'] = town
        default_config['world']['weather'] = self._get_weather_config(
            weather_conditions)

        return default_config

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about intersection scenario parameters."""
        return {
            'weather': {
                'description': 'Quickly set weather conditions for the scenario '
                'or use custom weather configuration in world.weather',
                'type': 'str',
                'choices': ['clear', 'cloudy', 'rainy'],
                'default': 'clear'
            },
            'town': {
                'description': 'Name of the town to use for the scenario',
                'type': 'str',
                'default': 'intersection'
            }
        }
    # --------------------------------------------------------------------- #
    # Default parameters
    # --------------------------------------------------------------------- #

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for intersection scenarios."""
        return {
            'meta': {
                'scenario_type': 'intersection',
                'town': 'intersection',
            },
            'world': {
                'sync_mode': True,
                'host': '127.0.0.1',
                'client_port': 2000,
                'timeout': 20.0,
                'fixed_delta_seconds': 0.05,
                'seed': 11,
                'weather': self._get_weather_config('clear')
            },
            'opt': {
                'apply_ml': True
            },
            'agents': {
                'agent_behavior': {
                    'marl_mode': True,
                    'collision_tolerance_threshold': 5  
                }
            }
        }

    def _get_weather_config(self, weather_conditions: str) -> Dict[str, Any]:
        """Get weather configuration for CARLA."""
        weather_configs = {
            'clear': {
                'sun_altitude_angle': 15,
                'cloudiness': 0,
                'precipitation': 0,
                'precipitation_deposits': 0,
                'wind_intensity': 0,
                'fog_density': 0,
                'fog_distance': 0,
                'fog_falloff': 0,
                'wetness': 0
            },
            'cloudy': {
                'sun_altitude_angle': 15,
                'cloudiness': 80,
                'precipitation': 0,
                'precipitation_deposits': 0,
                'wind_intensity': 10,
                'fog_density': 0,
                'fog_distance': 0,
                'fog_falloff': 0,
                'wetness': 0
            },
            'rainy': {
                'sun_altitude_angle': 15,
                'cloudiness': 100,
                'precipitation': 80,
                'precipitation_deposits': 0,
                'wind_intensity': 50,
                'fog_density': 10,
                'fog_distance': 0,
                'fog_falloff': 0,
                'wetness': 0
            }
        }

        return weather_configs.get(weather_conditions, weather_configs['clear'])
