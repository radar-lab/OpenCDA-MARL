'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 20:22:37
FilePath     : /OpenCDA-MARL/opencda_marl/envs/carla_spectator.py
Description  : 
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import carla

from loguru import logger
from typing import Dict, Any, Optional


class CarlaSpectator:
    """
    Manager for CARLA spectator camera control.
    """

    def __init__(self, world: carla.World, config: Optional[Dict[str, Any]] = None):
        self.world = world
        self.spectator = world.get_spectator()
        self.config = config or {}

        # Current view state
        self.target_agent_id = None
        self.follow_offset = carla.Vector3D(0, 0, 50)  # Default follow offset
        
        self._view_presets = {
            'intersection_bird_eye': {
                'position': carla.Location(x=0, y=0, z=100),
                'rotation': carla.Rotation(pitch=-90, yaw=0, roll=0),
                'description': 'Bird\'s eye view of intersection center'
            }
        }
        
        preset = self.config.get('preset')
        if preset and preset in self._view_presets:
            self.use_preset(preset)
            self.world.tick()
        
    def use_preset(self, preset_name: str) -> bool:
        """
        Apply predefined view preset.
        """
        if preset_name not in self._view_presets:
            available = list(self._view_presets.keys())
            logger.error(f"Unknown preset '{preset_name}'. Available: {available}")
            return False
        
        preset = self._view_presets[preset_name]
        success = self.set_fixed_view(preset['position'], preset['rotation'])
        
        if success:
            logger.debug(f"Applied preset '{preset_name}': {preset['description']}")
        
        return success
    
    # --------------------------------------------------------------------- #
    # Set view
    # --------------------------------------------------------------------- #    
    def set_fixed_view(self, position: carla.Location, 
                      rotation: carla.Rotation) -> bool:
        """
        Set fixed camera position and rotation.
        """
        try:
            transform = carla.Transform(position, rotation)
            self.spectator.set_transform(transform)
            
            logger.debug(f"Set fixed view at {position}, rotation {rotation}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set fixed view: {e}")
            return False
    
    def get_current_view(self) -> Dict[str, Any]:
        """
        Get current camera view information.
        
        Returns
        -------
        view_info : dict
            Current view mode, position, and rotation
        """
        current_transform = self.spectator.get_transform()
        
        return {
            'position': current_transform.location,
            'rotation': current_transform.rotation,
        }