#!/usr/bin/env python3
"""
Script to discover all available CARLA vehicle types and categorize them.
This helps select appropriate vehicles for traffic simulation.

Usage: python list_vehicle_types.py
"""

import carla
import sys
from collections import defaultdict


def categorize_vehicle(bp_id: str) -> str:
    """Categorize vehicle based on its blueprint ID."""
    bp_lower = bp_id.lower()
    
    # Trucks and large vehicles
    if any(word in bp_lower for word in ['truck', 'semi', 'trailer', 'cargo']):
        return 'truck'
    
    # Buses
    if any(word in bp_lower for word in ['bus']):
        return 'bus'
    
    # Emergency vehicles
    if any(word in bp_lower for word in ['firetruck', 'ambulance', 'police']):
        return 'emergency'
    
    # Motorcycles and bikes
    if any(word in bp_lower for word in ['motorcycle', 'bike', 'yamaha', 'kawasaki', 'harley']):
        return 'motorcycle'
    
    # Special/weird vehicles
    if any(word in bp_lower for word in ['isetta', 'carlacola', 'cybertruck', 't2']):
        return 'special'
    
    # Vans and SUVs
    if any(word in bp_lower for word in ['van', 'suv', 'jeep', 'range_rover']):
        return 'suv'
    
    # Luxury/sports cars
    if any(word in bp_lower for word in ['tesla', 'mustang', 'crown', 'charger', 'challenger']):
        return 'luxury'
    
    # Regular sedans/hatchbacks
    return 'sedan'


def retrieve_vehicle_types():
    try:
        # Connect to CARLA
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Get world and blueprint library
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        
        # Get all vehicle blueprints
        vehicle_bps = bp_lib.filter('vehicle.*')
        
        print(f"Found {len(vehicle_bps)} total vehicle blueprints\\n")
        
        # Categorize vehicles
        categories = defaultdict(list)
        
        for bp in vehicle_bps:
            category = categorize_vehicle(bp.id)
            categories[category].append(bp.id)
        
        # Print categorized results
        print("=" * 80)
        print("CARLA VEHICLE TYPES BY CATEGORY")
        print("=" * 80)
        
        category_order = ['sedan', 'luxury', 'suv', 'truck', 'bus', 'motorcycle', 'emergency', 'special']
        
        for category in category_order:
            if category in categories:
                vehicles = sorted(categories[category])
                print(f"\\n{category.upper()} ({len(vehicles)} vehicles):")
                print("-" * 40)
                for vehicle in vehicles:
                    print(f"  {vehicle}")
        
        print("\\n" + "=" * 80)
        print("RECOMMENDED CONFIGURATION FOR DEFAULT.YAML")
        print("=" * 80)
        
        print("""
# Recommended vehicle configuration for consistent simulation:
scenario:
  traffic:
    # Use inclusion-based approach - specify exactly which vehicles to use
    included_vehicle_types:
      # Regular sedans (good collision detection, consistent size)
      - "vehicle.audi.a2"           # Compact sedan
      - "vehicle.bmw.grandtourer"   # Family sedan  
      - "vehicle.citroen.c3"        # Compact car
      - "vehicle.dodge.charger_2020" # Modern sedan
      - "vehicle.lincoln.mkz_2017"  # Luxury sedan
      - "vehicle.nissan.patrol_2021" # SUV
      - "vehicle.seat.leon"         # Compact sedan
      - "vehicle.tesla.model3"      # Electric sedan
      - "vehicle.toyota.prius"      # Hybrid sedan
      - "vehicle.volkswagen.t2_2021" # Modern van
      
    # Exclude all other vehicle types to avoid collision issues
    excluded_vehicle_types:
      # Large vehicles (collision sensor issues)
      - "truck"
      - "bus" 
      - "firetruck"
      - "ambulance"
      
      # Motorcycles (different physics)
      - "motorcycle"
      - "bike"
      - "yamaha"
      - "kawasaki"
      - "harley"
      
      # Special/problematic vehicles
      - "isetta"        # Too small
      - "carlacola"     # Non-standard
      - "cybertruck"    # Unusual shape
      - "t2"            # Old van with issues
      
      # Static objects
      - "walker"
      - "static"
      - "sensor"
      - "controller"
        """)
        
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        print("Make sure CARLA server is running on localhost:2000")
        sys.exit(1)


if __name__ == "__main__":
    retrieve_vehicle_types()