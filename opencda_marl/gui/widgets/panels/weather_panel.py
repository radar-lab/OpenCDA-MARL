'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 13:41:33
FilePath     : /OpenCDA-MARL/opencda_marl/gui/widgets/panels/weather_panel.py
Description  : Weather settings panel for OpenCDA-MARL GUI.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from PySide6.QtWidgets import QWidget, QFormLayout, QLabel
from typing import Dict, Any


class WeatherPanel(QWidget):
    """Panel for displaying CARLA weather and environment parameters."""
    
    def __init__(self, parent=None):
        """
        Initialize weather panel.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self._parameters = {}  # Store parameter display widgets
        self._setup_ui()
        
    # --------------------------------------------------------------------- #
    # UI Materials
    # --------------------------------------------------------------------- #        
    def _setup_ui(self):
        """Setup the weather panel UI."""
        layout = QFormLayout()
        self.setLayout(layout)
        
        # Weather condition parameters
        self._add_parameter("cloudiness", "N/A", "Cloudiness (0-100)")
        self._add_parameter("precipitation", "N/A", "Precipitation (0-100)")
        self._add_parameter("precipitation_deposits", "N/A", "Precipitation Deposits (0-100)")
        self._add_parameter("wind_intensity", "N/A", "Wind Intensity (0-100)")
        self._add_parameter("wetness", "N/A", "Wetness (0-100)")
        
        # Sun parameters
        self._add_parameter("sun_azimuth_angle", "N/A", "Sun Azimuth Angle (°)")
        self._add_parameter("sun_altitude_angle", "N/A", "Sun Altitude Angle (°)")
        
        # Fog parameters
        self._add_parameter("fog_density", "N/A", "Fog Density (0-100)")
        self._add_parameter("fog_distance", "N/A", "Fog Distance")
        self._add_parameter("fog_falloff", "N/A", "Fog Falloff")
        
        # Scattering parameters
        self._add_parameter("scattering_intensity", "N/A", "Scattering Intensity")
        self._add_parameter("mie_scattering_scale", "N/A", "Mie Scattering Scale")
        self._add_parameter("rayleigh_scattering_scale", "N/A", "Rayleigh Scattering Scale")
        
    def _add_parameter(self, name: str, default_value: str, label: str):
        """
        Add a display parameter to the form.
        
        Parameters
        ----------
        name : str
            Parameter name
        default_value : str
            Default display value
        label : str
            Display label
        """
        display_label = QLabel(str(default_value))
        display_label.setStyleSheet("QLabel { background-color: white; color: black; padding: 4px; border: 1px solid black; border-radius: 2px; }")
        
        self._parameters[name] = display_label
        self.layout().addRow(f"{label}:", display_label)
        
    def update_parameters(self, weather_data: Dict[str, Any]):
        """
        Update displayed weather parameter values.
        
        Parameters
        ----------
        weather_data : dict
            Weather parameter data from monitor
        """
        # Weather parameters mapping
        param_mapping = {
            'cloudiness': 'cloudiness',
            'precipitation': 'precipitation',
            'precipitation_deposits': 'precipitation_deposits',
            'wind_intensity': 'wind_intensity',
            'wetness': 'wetness',
            'sun_azimuth_angle': 'sun_azimuth_angle',
            'sun_altitude_angle': 'sun_altitude_angle',
            'fog_density': 'fog_density',
            'fog_distance': 'fog_distance',
            'fog_falloff': 'fog_falloff',
            'scattering_intensity': 'scattering_intensity',
            'mie_scattering_scale': 'mie_scattering_scale',
            'rayleigh_scattering_scale': 'rayleigh_scattering_scale'
        }
        
        for data_key, param_key in param_mapping.items():
            if data_key in weather_data and param_key in self._parameters:
                value = weather_data[data_key]
                
                # Format the value for display
                if isinstance(value, float):
                    # Different precision for different types
                    if 'angle' in data_key:
                        display_value = f"{value:.1f}°"
                    elif 'scale' in data_key or 'intensity' in data_key:
                        display_value = f"{value:.3f}"
                    else:
                        display_value = f"{value:.1f}"
                else:
                    display_value = str(value)
                    
                self._parameters[param_key].setText(display_value)
                
    # --------------------------------------------------------------------- #
    # Clean up
    # --------------------------------------------------------------------- #            
    def reset_parameters(self):
        """Reset all parameters to default N/A values."""
        for param_widget in self._parameters.values():
            param_widget.setText("N/A")
            
    def get_current_values(self) -> Dict[str, str]:
        """
        Get current displayed values for all parameters.
        
        Returns
        -------
        dict
            Dictionary of parameter name -> displayed value
        """
        return {name: widget.text() for name, widget in self._parameters.items()}