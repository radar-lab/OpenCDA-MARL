'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 16:45:55
FilePath     : /OpenCDA-MARL/opencda_marl/core/traffic/serializer.py
Description  : Event serializer for traffic events
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import h5py
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
import carla

from .events import SpawnEvent


class EventSerializer:
    """Handles serialization of traffic events to/from files."""
    
    VERSION = '1.0'
    
    @classmethod
    def save_events_to_hdf5(cls, 
                           events: List[SpawnEvent], 
                           filepath: str, 
                           config: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save events to HDF5 format for efficient storage and fast loading.
        
        Args:
            events: List of spawn events to save
            filepath: Output file path
            config: Configuration dict to save as metadata
            metadata: Additional metadata to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(filepath, 'w') as f:
                # Save metadata
                f.attrs['version'] = cls.VERSION
                f.attrs['total_events'] = len(events)
                f.attrs['timestamp'] = datetime.now().isoformat()
                
                if config:
                    f.attrs['config'] = json.dumps(config, indent=2)
                    
                if metadata:
                    f.attrs['metadata'] = json.dumps(metadata, indent=2)
                
                if not events:
                    logger.warning("No events to save")
                    return True
                
                # Prepare data arrays
                n = len(events)
                
                # Numeric data
                spawn_steps = np.array([e.spawn_step for e in events], dtype=np.int32)
                junction_ids = np.array([e.junction_id for e in events], dtype=np.int32)
                route_ids = np.array([e.route_id for e in events], dtype=np.int32)
                lane_ids = np.array([e.lane_id for e in events], dtype=np.int32)
                target_speeds = np.array([e.target_speed for e in events], dtype=np.float32)
                
                # Transform data (spawn positions)
                spawn_transforms = np.zeros((n, 6), dtype=np.float32)  # x,y,z,pitch,yaw,roll
                dest_transforms = np.zeros((n, 6), dtype=np.float32)
                
                for i, event in enumerate(events):
                    # Spawn transform
                    spawn_transforms[i] = [
                        event.transform.location.x,
                        event.transform.location.y, 
                        event.transform.location.z,
                        event.transform.rotation.pitch,
                        event.transform.rotation.yaw,
                        event.transform.rotation.roll
                    ]
                    
                    # Destination transform
                    dest_transforms[i] = [
                        event.destination.location.x,
                        event.destination.location.y,
                        event.destination.location.z,
                        event.destination.rotation.pitch,
                        event.destination.rotation.yaw,
                        event.destination.rotation.roll
                    ]
                
                # Save numeric datasets
                f.create_dataset('spawn_steps', data=spawn_steps, compression='gzip')
                f.create_dataset('junction_ids', data=junction_ids, compression='gzip')
                f.create_dataset('route_ids', data=route_ids, compression='gzip')
                f.create_dataset('lane_ids', data=lane_ids, compression='gzip')
                f.create_dataset('target_speeds', data=target_speeds, compression='gzip')
                f.create_dataset('spawn_transforms', data=spawn_transforms, compression='gzip')
                f.create_dataset('dest_transforms', data=dest_transforms, compression='gzip')
                
                # String data (variable length)
                str_dtype = h5py.special_dtype(vlen=str)
                
                event_ids = [e.event_id for e in events]
                vehicle_ids = [e.vehicle_id for e in events]
                flow_names = [e.flow_name for e in events]
                blueprint_ids = [e.blueprint.id for e in events]
                
                f.create_dataset('event_ids', data=event_ids, dtype=str_dtype, compression='gzip')
                f.create_dataset('vehicle_ids', data=vehicle_ids, dtype=str_dtype, compression='gzip')
                f.create_dataset('flow_names', data=flow_names, dtype=str_dtype, compression='gzip')
                f.create_dataset('blueprint_ids', data=blueprint_ids, dtype=str_dtype, compression='gzip')
                
                # Metadata (serialize as JSON strings)
                metadata_strs = [json.dumps(e.metadata) for e in events]
                f.create_dataset('metadata', data=metadata_strs, dtype=str_dtype, compression='gzip')
            
            file_size = Path(filepath).stat().st_size / (1024 * 1024)  # MB
            logger.success(f"Saved {len(events)} events to {filepath} ({file_size:.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save events to {filepath}: {e}")
            return False
    
    @classmethod  
    def load_events_from_hdf5(cls, filepath: str, world: Optional[carla.World] = None) -> Optional[List[SpawnEvent]]:
        """
        Load events from HDF5 file.
        
        Args:
            filepath: Input file path
            world: CARLA world for blueprint lookup
            
        Returns:
            List of loaded events, or None if failed
        """
        try:
            if not Path(filepath).exists():
                logger.error(f"Event file not found: {filepath}")
                return None
                
            with h5py.File(filepath, 'r') as f:
                # Check version compatibility
                version = f.attrs.get('version', '0.0')
                if version != cls.VERSION:
                    logger.warning(f"Loading events from version {version}, "
                                 f"current version is {cls.VERSION}")
                
                total_events = f.attrs.get('total_events', 0)
                if total_events == 0:
                    logger.warning("No events in file")
                    return []
                
                # Load data arrays
                spawn_steps = f['spawn_steps'][:]
                junction_ids = f['junction_ids'][:]
                route_ids = f['route_ids'][:]
                lane_ids = f['lane_ids'][:]
                target_speeds = f['target_speeds'][:]
                spawn_transforms = f['spawn_transforms'][:]
                dest_transforms = f['dest_transforms'][:]
                
                event_ids = f['event_ids'][:]
                vehicle_ids = f['vehicle_ids'][:]
                flow_names = f['flow_names'][:]
                blueprint_ids = f['blueprint_ids'][:]
                metadata_strs = f['metadata'][:]
                
                # Reconstruct events
                events = []
                n = len(spawn_steps)
                
                for i in range(n):
                    # Reconstruct transforms
                    spawn_tf_data = spawn_transforms[i]
                    dest_tf_data = dest_transforms[i]
                    
                    spawn_transform = carla.Transform(
                        carla.Location(x=float(spawn_tf_data[0]), 
                                     y=float(spawn_tf_data[1]), 
                                     z=float(spawn_tf_data[2])),
                        carla.Rotation(pitch=float(spawn_tf_data[3]),
                                     yaw=float(spawn_tf_data[4]),
                                     roll=float(spawn_tf_data[5]))
                    )
                    
                    dest_transform = carla.Transform(
                        carla.Location(x=float(dest_tf_data[0]),
                                     y=float(dest_tf_data[1]),
                                     z=float(dest_tf_data[2])),
                        carla.Rotation(pitch=float(dest_tf_data[3]),
                                     yaw=float(dest_tf_data[4]),
                                     roll=float(dest_tf_data[5]))
                    )
                    
                    # Parse metadata
                    try:
                        metadata = json.loads(metadata_strs[i])
                    except Exception:
                        metadata = {}
                    
                    # Convert blueprint ID to blueprint object
                    blueprint_id = str(blueprint_ids[i])
                    if world is not None:
                        blueprint = world.get_blueprint_library().find(blueprint_id)
                    else:
                        raise ValueError("World instance required for blueprint lookup")
                    
                    # Create event
                    event = SpawnEvent(
                        event_id=str(event_ids[i]),
                        vehicle_id=str(vehicle_ids[i]),
                        flow_name=str(flow_names[i]),
                        spawn_step=int(spawn_steps[i]),
                        junction_id=int(junction_ids[i]),
                        route_id=int(route_ids[i]),
                        lane_id=int(lane_ids[i]),
                        transform=spawn_transform,
                        destination=dest_transform,
                        blueprint=blueprint,
                        target_speed=float(target_speeds[i]),
                        metadata=metadata
                    )
                    
                    events.append(event)
            
            logger.success(f"Loaded {len(events)} events from {filepath}")
            return events
            
        except Exception as e:
            logger.error(f"Failed to load events from {filepath}: {e}")
            return None

    @classmethod
    def load_events_from_json(cls, filepath: str, world: Optional[carla.World] = None) -> Optional[List[SpawnEvent]]:
        """
        Load events from JSON file.
        
        Args:
            filepath: Input file path
            world: CARLA world for blueprint lookup
            
        Returns:
            List of loaded events, or None if failed
        """
        try:
            if not Path(filepath).exists():
                logger.error(f"Event file not found: {filepath}")
                return None
                
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check version compatibility
            version = data.get('version', '0.0')
            if version != cls.VERSION:
                logger.warning(f"Loading events from version {version}, "
                             f"current version is {cls.VERSION}")
            
            total_events = data.get('total_events', 0)
            if total_events == 0:
                logger.warning("No events in file")
                return []
            
            events_data = data.get('events', [])
            if not events_data:
                logger.warning("No events data found in JSON")
                return []
            
            # Reconstruct events
            events = []
            
            for event_data in events_data:
                try:
                    # Extract spawn transform
                    spawn_tf_data = event_data['spawn_transform']
                    spawn_transform = carla.Transform(
                        carla.Location(
                            x=float(spawn_tf_data['location']['x']),
                            y=float(spawn_tf_data['location']['y']),
                            z=float(spawn_tf_data['location']['z'])
                        ),
                        carla.Rotation(
                            pitch=float(spawn_tf_data['rotation']['pitch']),
                            yaw=float(spawn_tf_data['rotation']['yaw']),
                            roll=float(spawn_tf_data['rotation']['roll'])
                        )
                    )
                    
                    # Extract destination transform
                    dest_tf_data = event_data['destination_transform']
                    dest_transform = carla.Transform(
                        carla.Location(
                            x=float(dest_tf_data['location']['x']),
                            y=float(dest_tf_data['location']['y']),
                            z=float(dest_tf_data['location']['z'])
                        ),
                        carla.Rotation(
                            pitch=float(dest_tf_data['rotation']['pitch']),
                            yaw=float(dest_tf_data['rotation']['yaw']),
                            roll=float(dest_tf_data['rotation']['roll'])
                        )
                    )
                    
                    # Convert blueprint ID to blueprint object
                    blueprint_id = str(event_data['blueprint_id'])
                    if world is not None:
                        blueprint = world.get_blueprint_library().find(blueprint_id)
                    else:
                        raise ValueError("World instance required for blueprint lookup")
                    
                    # Create event
                    event = SpawnEvent(
                        event_id=str(event_data['event_id']),
                        vehicle_id=str(event_data['vehicle_id']),
                        flow_name=str(event_data['flow_name']),
                        spawn_step=int(event_data['spawn_step']),
                        junction_id=int(event_data['junction_id']),
                        route_id=int(event_data['route_id']),
                        lane_id=int(event_data['lane_id']),
                        transform=spawn_transform,
                        destination=dest_transform,
                        blueprint=blueprint,
                        target_speed=float(event_data['target_speed']),
                        metadata=event_data.get('metadata', {})
                    )
                    
                    events.append(event)
                    
                except Exception as e:
                    logger.error(f"Failed to parse event {event_data.get('event_id', 'unknown')}: {e}")
                    continue
            
            logger.success(f"Loaded {len(events)} events from {filepath}")
            return events
            
        except Exception as e:
            logger.error(f"Failed to load events from {filepath}: {e}")
            return None

    @classmethod
    def load_events(cls, filepath: str, world: Optional[carla.World] = None) -> Optional[List[SpawnEvent]]:
        """
        Load events from either HDF5 or JSON format (auto-detect).
        
        Args:
            filepath: Input file path
            world: CARLA world for blueprint lookup
            
        Returns:
            List of loaded events, or None if failed
        """
        if not Path(filepath).exists():
            logger.error(f"Event file not found: {filepath}")
            return None
        
        file_format = cls._detect_file_format(filepath)
        
        if file_format == 'hdf5':
            logger.info(f"Loading events from HDF5 file: {filepath}")
            return cls.load_events_from_hdf5(filepath, world)
        elif file_format == 'json':
            logger.info(f"Loading events from JSON file: {filepath}")
            return cls.load_events_from_json(filepath, world)
        else:
            logger.error(f"Unsupported file format for {filepath}. Supported formats: .h5, .hdf5, .json")
            return None

    @classmethod
    def _detect_file_format(cls, filepath: str) -> Optional[str]:
        """
        Detect file format based on extension.
        
        Args:
            filepath: Path to file
            
        Returns:
            'hdf5', 'json', or None if unsupported
        """
        suffix = Path(filepath).suffix.lower()
        
        if suffix in ['.h5', '.hdf5']:
            return 'hdf5'
        elif suffix in ['.json']:
            return 'json'
        else:
            return None
    
    @classmethod
    def export_events_to_json(cls, events: List[SpawnEvent], filepath: str) -> bool:
        """
        Export events to JSON format for human readability.
        
        Args:
            events: List of events to export
            filepath: Output JSON file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                'version': cls.VERSION,
                'timestamp': datetime.now().isoformat(),
                'total_events': len(events),
                'events': []
            }
            
            for event in events:
                event_data = {
                    'event_id': event.event_id,
                    'vehicle_id': event.vehicle_id,
                    'flow_name': event.flow_name,
                    'spawn_step': event.spawn_step,
                    'junction_id': event.junction_id,
                    'route_id': event.route_id,
                    'lane_id': event.lane_id,
                    'blueprint_id': event.blueprint.id,
                    'target_speed': event.target_speed,
                    'spawn_transform': {
                        'location': {
                            'x': event.transform.location.x,
                            'y': event.transform.location.y,
                            'z': event.transform.location.z
                        },
                        'rotation': {
                            'pitch': event.transform.rotation.pitch,
                            'yaw': event.transform.rotation.yaw,
                            'roll': event.transform.rotation.roll
                        }
                    },
                    'destination_transform': {
                        'location': {
                            'x': event.destination.location.x,
                            'y': event.destination.location.y,
                            'z': event.destination.location.z
                        },
                        'rotation': {
                            'pitch': event.destination.rotation.pitch,
                            'yaw': event.destination.rotation.yaw,
                            'roll': event.destination.rotation.roll
                        }
                    },
                    'metadata': event.metadata
                }
                export_data['events'].append(event_data)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            file_size = Path(filepath).stat().st_size / (1024 * 1024)  # MB
            logger.success(f"Exported {len(events)} events to {filepath} ({file_size:.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export events to {filepath}: {e}")
            return False
    
    @classmethod
    def validate_event_file(cls, filepath: str) -> Dict[str, Any]:
        """
        Validate an event file and return metadata (supports both HDF5 and JSON).
        
        Args:
            filepath: Path to event file
            
        Returns:
            Dictionary with validation results and metadata
        """
        result = {
            'valid': False,
            'error': None,
            'format': None,
            'version': None,
            'total_events': 0,
            'file_size_mb': 0,
            'timestamp': None
        }
        
        try:
            if not Path(filepath).exists():
                result['error'] = 'File not found'
                return result
                
            result['file_size_mb'] = Path(filepath).stat().st_size / (1024 * 1024)
            file_format = cls._detect_file_format(filepath)
            result['format'] = file_format
            
            if file_format == 'hdf5':
                return cls._validate_hdf5_file(filepath, result)
            elif file_format == 'json':
                return cls._validate_json_file(filepath, result)
            else:
                result['error'] = f'Unsupported file format: {Path(filepath).suffix}'
                return result
                
        except Exception as e:
            result['error'] = str(e)
            
        return result

    @classmethod
    def _validate_hdf5_file(cls, filepath: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HDF5 event file."""
        try:
            with h5py.File(filepath, 'r') as f:
                result['version'] = f.attrs.get('version', 'unknown')
                result['total_events'] = f.attrs.get('total_events', 0)
                result['timestamp'] = f.attrs.get('timestamp', 'unknown')
                
                # Check required datasets exist
                required_datasets = [
                    'spawn_steps', 'junction_ids', 'route_ids', 'lane_ids',
                    'target_speeds', 'spawn_transforms', 'dest_transforms',
                    'event_ids', 'vehicle_ids', 'flow_names', 'blueprint_ids'
                ]
                
                missing_datasets = [ds for ds in required_datasets if ds not in f]
                if missing_datasets:
                    result['error'] = f'Missing datasets: {missing_datasets}'
                    return result
                
                result['valid'] = True
                
        except Exception as e:
            result['error'] = f'HDF5 validation error: {str(e)}'
            
        return result

    @classmethod
    def _validate_json_file(cls, filepath: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON event file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            result['version'] = data.get('version', 'unknown')
            result['total_events'] = data.get('total_events', 0)
            result['timestamp'] = data.get('timestamp', 'unknown')
            
            # Check required fields
            events_data = data.get('events', [])
            if not events_data and result['total_events'] > 0:
                result['error'] = 'Missing events data'
                return result
            
            # Validate first event structure (if any)
            if events_data:
                first_event = events_data[0]
                required_fields = [
                    'event_id', 'vehicle_id', 'flow_name', 'spawn_step',
                    'junction_id', 'route_id', 'lane_id', 'blueprint_id',
                    'target_speed', 'spawn_transform', 'destination_transform'
                ]
                
                missing_fields = [field for field in required_fields if field not in first_event]
                if missing_fields:
                    result['error'] = f'Missing required fields: {missing_fields}'
                    return result
            
            result['valid'] = True
            
        except json.JSONDecodeError as e:
            result['error'] = f'JSON parsing error: {str(e)}'
        except Exception as e:
            result['error'] = f'JSON validation error: {str(e)}'
            
        return result