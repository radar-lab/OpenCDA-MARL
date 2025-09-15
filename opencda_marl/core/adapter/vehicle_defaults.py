from typing import Dict, Any


def get_sensor_defaults() -> Dict[str, Any]:
    """Get default sensor configuration."""
    return {
        "perception": {
            "activate": False,
            "camera": {
                "visualize": 0,
                "num": 1,
                "positions": [[2.5, 0, 1.0, 0]]
            },
            "lidar": {
                "visualize": False,
                "channels": 32,
                "range": 50,
                "points_per_second": 100000,
                "rotation_frequency": 20,
                "upper_fov": 10.0,
                "lower_fov": -30.0,
                "dropoff_general_rate": 0.0,
                "dropoff_intensity_limit": 1.0,
                "dropoff_zero_intensity": 0.0,
                "noise_stddev": 0.0
            }
        },
        "localization": {
            "activate": False,
            "use_kalman": False,   # Enable fixed Kalman filter
            "dt": 0.1,
            "speed_noise_std": 0.1,    # Standard deviation for speed noise (m/s)
            "pos_noise_std": 0.1,      # Position noise standard deviation (m)
            "heading_noise_std": 0.01, # Heading noise standard deviation (radians)
            "gnss": {
                "noise_alt_stddev": 0.000005,
                "noise_lat_stddev": 0.000005,
                "noise_lon_stddev": 0.000005,
                "heading_direction_stddev": 0.0001,
                "speed_stddev": 0.0001
            },
            "imu": {
                "noise_accel_stddev_x": 0.001,
                "noise_accel_stddev_y": 0.001,
                "noise_accel_stddev_z": 0.015,
                "noise_gyro_stddev_x": 0.001,
                "noise_gyro_stddev_y": 0.001,
                "noise_gyro_stddev_z": 0.001
            }, 
            "debug_helper": {
                "show_animation": False,
                "x_scale": 10.0,
                "y_scale": 10.0
            },
        }
    }


def get_map_manager_defaults() -> Dict[str, Any]:
    """Get default map manager configuration."""
    return {
        "pixels_per_meter": 2,
        "raster_size": [224, 224],
        "lane_sample_resolution": 0.1,
        "visualize": False,
        "activate": False
    }


def get_controller_defaults() -> Dict[str, Any]:
    """Get default controller configuration."""
    return {
        "type": "pid_controller",
        "args": {
            "lat": {
                "k_p": 0.75,
                "k_d": 0.02,
                "k_i": 0.4
            },
            "lon": {
                "k_p": 0.37,
                "k_d": 0.024,
                "k_i": 0.032
            },
            "dynamic": False,
            "dt": 0.1,
            "max_brake": 1.0,
            "max_throttle": 1.0,
            "max_steering": 0.3
        }
    }


def get_v2x_defaults() -> Dict[str, Any]:
    """Get default V2X configuration."""
    return {
        "enabled": False,
        "communication_range": 100,
        "ego_id": "ego_vehicle",
        "apply_lag": False
    }


def get_safety_manager_defaults() -> Dict[str, Any]:
    """Get default safety manager configuration."""
    return {
        "print_message": False,
        "collision_sensor": {
            "history_size": 30,
            "col_thresh": 1,
        },
        "stuck_dector": {
            "len_thresh": 500,
            "speed_thresh": 0.5
        },
        "offroad_dector": [],
        "traffic_light_detector": {
            "light_dist_thresh": 20
        }
    }


def get_behavior_defaults() -> Dict[str, Any]:
    """Get default behavior agent configuration."""
    return {
        "max_speed": 30,
        "tailgate_speed": 55,
        "speed_lim_dist": 5,
        "speed_decrease": 12,
        "safety_time": 1.8,
        "emergency_param": 0.8,
        "collision_time_ahead": 1.2,
        "overtake_allowed": True,
        "overtake_counter_recover": 25,
        "ignore_traffic_light": False,
        "sample_resolution": 4.0,
        "debug": True,
        "local_planner": {
            "buffer_size": 12,
            "trajectory_update_freq": 15,
            "waypoint_update_freq": 9,
            "min_dist": 5,
            "trajectory_dt": 0.20,
            "debug": True,
            "debug_trajectory": True
        }
    }


def get_vehicle_manager_defaults() -> Dict[str, Any]:
    """Get complete default VehicleManager configuration."""
    return {
        "behavior": get_behavior_defaults(),
        "sensing": get_sensor_defaults(),
        "map_manager": get_map_manager_defaults(),
        "controller": get_controller_defaults(),
        "v2x": get_v2x_defaults(),
        "safety_manager": get_safety_manager_defaults()
    }
