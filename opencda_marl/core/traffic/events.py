'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 09:39:40
FilePath     : /OpenCDA-MARL/opencda_marl/core/traffic/events.py
Description  : Event classes for traffic manager.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import carla
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid

@dataclass
class SpawnEvent:
    """Event representing a vehicle spawn with all necessary parameters."""
    
    # Identifiers
    event_id: str
    vehicle_id: str
    flow_name: str
    
    # Timing
    spawn_step: int
    
    # Spatial information
    junction_id: int
    route_id: int
    lane_id: int
    transform: carla.Transform
    destination: carla.Transform
    
    # Vehicle configuration
    blueprint: carla.ActorBlueprint
    target_speed: float  # km/h
    
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, 
               vehicle_id: str,
               flow_name: str,
               spawn_step: int,
               junction_id: int,
               route_id: int,
               lane_id: int,
               transform: carla.Transform,
               destination: carla.Transform,
               blueprint: carla.ActorBlueprint,
               target_speed: float,
               metadata: Optional[Dict[str, Any]] = None) -> 'SpawnEvent':
        """Create a new SpawnEvent with auto-generated event ID."""
        return cls(
            event_id=str(uuid.uuid4())[:8],  # Short UUID for readability
            vehicle_id=vehicle_id,
            flow_name=flow_name,
            spawn_step=spawn_step,
            junction_id=junction_id,
            route_id=route_id,
            lane_id=lane_id,
            transform=transform,
            destination=destination,
            blueprint=blueprint,
            target_speed=target_speed,
            metadata=metadata or {}
        )
    
    def __str__(self) -> str:
        return (f"SpawnEvent({self.event_id}: {self.vehicle_id} from {self.flow_name} "
                f"at step {self.spawn_step}, route {self.route_id})")

@dataclass
class TrafficFlow:
    """Configuration for a traffic flow."""
    name: str
    lanes: list[int]  # which lanes to use (0, 1, 2, etc.)
    direction: str    # entry direction (north, south, east, west)
    speed_variation: float = 0.0  # ±variation as fraction (0.2 = ±20%)
    
    # Step-based parameters
    rate_vph: Optional[float] = None     # vehicles per hour per lane
    start_step: Optional[int] = None     # start at step N
    end_step: Optional[int] = None       # end at step M
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        
        # Check that we have step-based config
        has_step_config = all(x is not None for x in [self.rate_vph, self.start_step, self.end_step])
        
        if not has_step_config:
            raise ValueError("Must specify step-based config (rate_vph, start_step, end_step)")
        
        # Validation
        if self.rate_vph <= 0:
            raise ValueError("Flow rate must be greater than 0")
        if self.start_step >= self.end_step:
            raise ValueError("Flow start step must be before end step")
        if not self.lanes:
            raise ValueError("At least one lane must be specified")
        if self.direction not in ['north', 'south', 'east', 'west']:
            raise ValueError("Direction must be one of: north, south, east, west")
        if not (0.0 <= self.speed_variation <= 1.0):
            raise ValueError("Speed variation must be between 0.0 and 1.0")
    
    @property
    def duration_steps(self) -> int:
        """Get flow duration in steps."""
        return self.end_step - self.start_step