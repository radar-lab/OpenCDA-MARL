"""
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 12:49:18
FilePath     : /OpenCDA-MARL/opencda_marl/envs/carla_monitor.py
Description  : CARLA Monitor for OpenCDA-MARL.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
"""

import time
import carla
from loguru import logger
from typing import Dict, Any, Optional

from opencda_marl.core.traffic.traffic_manager import MARLTrafficManager


class CarlaMonitor:
    """
    Monitor and manage CARLA world settings for MARL environment.
    """

    def __init__(
        self,
        world: carla.World,
        tm: Optional[carla.TrafficManager] = None,
        marl_tm: Optional[MARLTrafficManager] = None,
    ):
        """
        Initialize CARLA monitor.

        Parameters
        ----------
        world : carla.World
            CARLA world instance
        traffic_manager : carla.TrafficManager, optional
            Traffic manager instance
        """
        self.world = world
        self.tm = tm
        self.marl_tm = marl_tm

        self._cached_map_info = None

        # Store original settings for restoration
        self.original_settings = world.get_settings()
        logger.info("CarlaMonitor initialized")

    # --------------------------------------------------------------------- #
    # Information API
    # --------------------------------------------------------------------- #
    def get_current_settings(self) -> Dict[str, Any]:
        """
        Get current CARLA world settings.

        Returns
        -------
        dict
            Current world settings
        """
        settings = self.world.get_settings()
        return {
            "map_name": self.world.get_map().name,
            "sync_mode": settings.synchronous_mode,
            "fixed_delta_seconds": settings.fixed_delta_seconds,
            "max_substeps": settings.max_substeps,
            "max_substep_delta_time": settings.max_substep_delta_time,
            "substepping": settings.substepping,
            "spectator_as_ego": settings.spectator_as_ego,
            "no_rendering_mode": settings.no_rendering_mode,
        }

    def get_weather_info(self) -> Dict[str, Any]:
        """
        Get current weather information.

        Returns
        -------
        dict
            Weather information
        """
        weather = self.world.get_weather()
        return {
            "cloudiness": weather.cloudiness,
            "precipitation": weather.precipitation,
            "precipitation_deposits": weather.precipitation_deposits,
            "wind_intensity": weather.wind_intensity,
            "sun_azimuth_angle": weather.sun_azimuth_angle,
            "sun_altitude_angle": weather.sun_altitude_angle,
            "fog_density": weather.fog_density,
            "fog_distance": weather.fog_distance,
            "fog_falloff": weather.fog_falloff,
            "wetness": weather.wetness,
            "scattering_intensity": weather.scattering_intensity,
            "mie_scattering_scale": weather.mie_scattering_scale,
            "rayleigh_scattering_scale": weather.rayleigh_scattering_scale,
        }

    def get_map_info(self) -> Dict[str, Any]:
        """
        Get current map information.

        Returns
        -------
        dict
            Map information
        """
        if self._cached_map_info is None:
            world_map = self.world.get_map()
            self._cached_map_info = {
                "map_name": world_map.name.split("/")[-1],
                "map_layers": len(world_map.get_topology()),
                "waypoints_count": len(world_map.generate_waypoints(2.0)),
            }
        return self._cached_map_info

    def get_traffic_manager_info(self) -> Dict[str, Any]:
        """
        Get traffic manager information.

        Returns
        -------
        dict
            Traffic manager information
        """
        data = {}

        if self.marl_tm is not None:
            # get the number of flows
            data["n_flows"] = len(self.marl_tm.flows)
            data["n_events"] = len(self.marl_tm.events)
            data["spawned_vehicles"] = len(self.marl_tm._event_traces)

        if self.tm is not None:
            data["tm_port"] = self.tm.get_port()
            data["tm_sync_mode"] = self.tm.get_sync_mode()
            data["global_distance"] = self.tm.get_global_distance()
        else:
            data["carla_tm"] = False

        return data

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_monitor_data(self) -> Dict[str, Any]:
        """
        Get all monitoring data for GUI display.

        Returns
        -------
        dict
            Complete monitoring data
        """
        data = {
            "system": self.get_current_settings(),
            "weather": self.get_weather_info(),
            "traffic": self.get_traffic_manager_info(),
        }

        data["traffic"].update(self.get_map_info())

        return data

    def apply_settings(self, settings_dict: Dict[str, Any]):
        """
        Apply new settings to the world.

        Parameters
        ----------
        settings_dict : dict
            Settings to apply
        """
        current_settings = self.world.get_settings()

        # Update settings from dictionary
        if "sync_mode" in settings_dict:
            current_settings.synchronous_mode = settings_dict["sync_mode"]

        if "fixed_delta_seconds" in settings_dict:
            current_settings.fixed_delta_seconds = settings_dict["fixed_delta_seconds"]

        if "max_substeps" in settings_dict:
            current_settings.max_substeps = settings_dict["max_substeps"]

        if "no_rendering_mode" in settings_dict:
            current_settings.no_rendering_mode = settings_dict["no_rendering_mode"]

        # Apply the updated settings
        self.world.apply_settings(current_settings)
        logger.info(f"Applied new world settings: {settings_dict}")

    def restore_original_settings(self):
        """Restore original world settings."""
        self.world.apply_settings(self.original_settings)
        logger.info("Restored original world settings")
