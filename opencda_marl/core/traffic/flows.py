'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 13:51:40
FilePath     : /OpenCDA-MARL/opencda_marl/core/traffic/flows.py
Description  : Traffic flow configuration
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import carla
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


from .events import SpawnEvent
from .planner import MARLPlanner


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

    # Simple middle peak configuration
    middle_peak: Optional[dict] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check that we have step-based config
        has_step_config = all(x is not None for x in [
                              self.rate_vph, self.start_step, self.end_step])

        if not has_step_config:
            raise ValueError(
                "Must specify step-based config (rate_vph, start_step, end_step)")

        # Validation
        if self.rate_vph <= 0:
            raise ValueError("Flow rate must be greater than 0")
        if self.start_step >= self.end_step:
            raise ValueError("Flow start step must be before end step")
        if not self.lanes:
            raise ValueError("At least one lane must be specified")
        if self.direction not in ['north', 'south', 'east', 'west']:
            raise ValueError(
                "Direction must be one of: north, south, east, west")
        if not (0.0 <= self.speed_variation <= 1.0):
            raise ValueError("Speed variation must be between 0.0 and 1.0")

        # Set default middle peak if not provided
        if self.middle_peak is None:
            self.middle_peak = self.default_middle_peak
        else:
            self.middle_peak = {**self.default_middle_peak, **self.middle_peak}

    # --------------------------------------------------------------------- #   
    # Properties
    # --------------------------------------------------------------------- #
    @property
    def duration_steps(self) -> int:
        """Get flow duration in steps."""
        return self.end_step - self.start_step

    @property
    def default_middle_peak(self) -> dict:
        """Get default middle peak configuration."""
        return {
            'intensity': 0.3,
            'position': 0.5,
            'width': 0.3
        }
    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def calc_expected_vehicles(self, fix_dlt=0.05) -> int:
        """Calculate expected vehicles"""
        # convert rate_vph to rate_vps  
        rate_vps = self.rate_vph / 3600 / (1 / fix_dlt)
        
        # calculate exact expected vehicles
        expected_vehicles = rate_vps * self.duration_steps * len(self.lanes)
        
        # Round to nearest integer
        return int(round(expected_vehicles))

    # --------------------------------------------------------------------- #
    # Private API
    # --------------------------------------------------------------------- #
    def _generate_events(self, vbps: List[carla.ActorBlueprint],
                         planner: MARLPlanner,
                         junction_id: int,
                         fix_dlt: float = 0.05,
                         base_speed: float = 30.0) -> List[SpawnEvent]:
        """Generate traffic events with middle peak distribution."""
        events = []
        vehicle_count = 0
        
        # Calculate target vehicle count - EXACT matching
        target_vehicles = self.calc_expected_vehicles(fix_dlt)
        
        if target_vehicles == 0:
            return events
        
        # Calculate timing
        steps_per_second = 1 / fix_dlt
        seconds_total = self.duration_steps / steps_per_second
        
        # Distribute vehicles evenly across lanes
        total_lanes = len(self.lanes)
        base_vehicles_per_lane = target_vehicles // total_lanes
        extra_vehicles = target_vehicles % total_lanes
        
        # Get peak configuration
        peak_intensity = self.middle_peak.get('intensity')
        peak_position = self.middle_peak.get('position')
        peak_width = self.middle_peak.get('width')
        
        # Generate events for each lane
        for lane_i, lane_idx in enumerate(self.lanes):
            vehicles_for_lane = base_vehicles_per_lane
            if lane_i < extra_vehicles:
                vehicles_for_lane += 1
                
            if vehicles_for_lane == 0:
                continue
            
            # Generate spawn times with middle peak distribution
            spawn_times = self._generate_peak_times(
                vehicles_for_lane, 
                seconds_total, 
                peak_intensity, 
                peak_position, 
                peak_width
            )
            
            # Create events for this lane
            for spawn_time in spawn_times:
                # Convert to steps
                spawn_step = self.start_step + int(spawn_time * steps_per_second)
                spawn_step = max(self.start_step, min(self.end_step - 1, spawn_step))
                
                # Get route and blueprint
                route = planner.get_random_route_by_direction(junction_id, self.direction)
                if not route:
                    raise ValueError(f"No route found for direction: {self.direction}")
                    
                blueprint = random.choice(vbps)
                
                # Speed variation based on traffic density
                normalized_time = spawn_time / seconds_total
                density_factor = self._get_peak_density(normalized_time, peak_intensity, peak_position, peak_width)
                
                # Higher density = slightly slower speeds
                density_speed_factor = 1.0 - (density_factor - 1.0) * 0.1  # Small effect
                speed_var = random.uniform(-self.speed_variation, self.speed_variation)
                target_speed = base_speed * (1.0 + speed_var) * density_speed_factor
                
                direction = {"north": 'N', "south": 'S', "east": 'E', "west": 'W'}[self.direction]
                vehicle_id = f"{self.name}_{direction}_{lane_idx}_{vehicle_count}"

                # Create spawn event
                event = SpawnEvent.create(
                    vehicle_id=vehicle_id,
                    flow_name=self.name,
                    spawn_step=spawn_step,
                    junction_id=junction_id,
                    route_id=route['route_id'],
                    lane_id=lane_idx,
                    transform=route['spawn_tf'],
                    destination=route['dest_tf'],
                    blueprint=blueprint,
                    target_speed=target_speed,
                    metadata={
                        'route_type': route.get('route_type', 'unknown'),
                        'entry_direction': route.get('entry_direction', ''),
                        'exit_direction': route.get('exit_direction', ''),
                        'lane_id': lane_idx,
                        'peak_density': density_factor
                    }
                )
                events.append(event)
                vehicle_count += 1

        # Sort by spawn time
        events.sort(key=lambda x: x.spawn_step)
        
        # Verify exact count
        if len(events) != target_vehicles:
            print(f"ERROR: {self.name} generated {len(events)} but expected {target_vehicles}")
        
        return events

    def _generate_peak_times(self, num_vehicles: int, total_time: float, 
                           peak_intensity: float, peak_position: float, peak_width: float) -> List[float]:
        """Generate spawn times with a middle peak distribution."""
        if num_vehicles <= 0:
            return []
        
        # Use inverse transform sampling for peak distribution
        spawn_times = []
        
        for i in range(num_vehicles):
            # Generate a random value and transform it to follow peak distribution
            u = np.random.random()
            
            # Transform uniform random to peak-weighted distribution
            # Use a simple mixture of uniform + gaussian around peak
            if u < 0.3:  # 30% follow peak distribution
                # Gaussian around peak position
                time_normalized = np.random.normal(peak_position, peak_width * 0.3)
                time_normalized = max(0.05, min(0.95, time_normalized))  # Clamp
            else:  # 70% uniform distribution
                time_normalized = 0.1 + 0.8 * np.random.random()  # Avoid edges
            
            spawn_time = time_normalized * total_time
            spawn_times.append(spawn_time)
        
        return sorted(spawn_times)

    def _get_peak_density(self, normalized_time: float, peak_intensity: float, 
                         peak_position: float, peak_width: float) -> float:
        """Get traffic density factor at a given time (for speed adjustment)."""
        # Gaussian peak centered at peak_position
        distance_from_peak = abs(normalized_time - peak_position)
        if distance_from_peak < peak_width:
            # Inside peak area
            gaussian_factor = np.exp(-((distance_from_peak / (peak_width * 0.5)) ** 2))
            density = 1.0 + peak_intensity * gaussian_factor
        else:
            # Outside peak area
            density = 1.0
        
        return density

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Traffic flow: {self.name},\n" \
               f"rate_vph: {self.rate_vph},\n" \
               f"lanes: {self.lanes},\n" \
               f"direction: {self.direction},\n" \
               f"speed_variation: {self.speed_variation},\n" \
               f"middle_peak: {self.middle_peak},\n" \
               f"start_step: {self.start_step},\n" \
               f"end_step: {self.end_step}"
               
