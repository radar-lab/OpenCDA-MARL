'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 13:41:56
FilePath     : /OpenCDA-MARL/opencda_marl/gui/widgets/panels/traffic_panel.py
Description  : Traffic and vehicle settings panel for OpenCDA-MARL GUI.

GUI component for displaying traffic manager and vehicle parameters.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from PySide6.QtWidgets import QWidget, QFormLayout, QLabel
from typing import Dict, Any


class TrafficPanel(QWidget):
    """Panel for displaying traffic manager and vehicle parameters."""

    def __init__(self, parent=None):
        """
        Initialize traffic panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self._parameters = {}  # Store parameter display widgets
        self._setup_ui()

    # --------------------------------------------------------------------- #
    # UI Material
    # --------------------------------------------------------------------- #
    def _setup_ui(self):
        """Setup the traffic panel UI."""
        layout = QFormLayout()
        self.setLayout(layout)

        # add a separator
        layout.addRow(QLabel("Map Information"))
        # Map information
        self._add_parameter("map_name", "N/A", "Current Map")
        self._add_parameter("waypoints_count", "N/A", "Waypoints Count")
        self._add_parameter("map_layers", "N/A", "Map Topology Layers")

        # add a separator
        layout.addRow(QLabel("Traffic Manager Settings"))
        
        self._add_parameter("n_flows", "N/A", "Number of Traffic Flows")
        self._add_parameter("n_events", "N/A", "Number of Traffic Events")
        self._add_parameter("queue_count", "N/A", "Number of Traffic Queue")
        self._add_parameter("spawned_vehicles", "N/A", "Number of Spawned Vehicles")
        # Traffic Manager settings
        self._add_parameter("tm_port", "N/A", "Traffic Manager Port")
        self._add_parameter("tm_sync_mode", "N/A", "TM Synchronous Mode")
        self._add_parameter("global_distance", "N/A", "Global Distance (m)")

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
        display_label.setStyleSheet(
            "QLabel { background-color: white; color: black; padding: 4px; border: 1px solid black; border-radius: 2px; }")

        self._parameters[name] = display_label
        self.layout().addRow(f"{label}:", display_label)

    def update_parameters(self, traffic_data: Dict[str, Any]):
        """
        Update displayed traffic parameter values.

        Parameters
        ----------
        traffic_data : dict
            Traffic parameter data from monitor
        """
        # Handle visibility of traffic manager parameters
        if 'carla_tm' in traffic_data:
            # Get the form layout
            form_layout = self.layout()
            
            # Find and hide/show the rows for TM parameters
            for param_name in ['tm_port', 'tm_sync_mode', 'global_distance']:
                if param_name in self._parameters:
                    widget = self._parameters[param_name]
                    # Find the row containing this widget
                    for i in range(form_layout.rowCount()):
                        item = form_layout.itemAt(i, QFormLayout.FieldRole)
                        if item and item.widget() == widget:
                            # Get both label and field widgets
                            label_item = form_layout.itemAt(i, QFormLayout.LabelRole)
                            if label_item and label_item.widget():
                                label_item.widget().setVisible(traffic_data['carla_tm'])
                            widget.setVisible(traffic_data['carla_tm'])
                            break

        for param_key, param_value in traffic_data.items():
            if param_key in self._parameters:
                # Format the value for display
                if isinstance(param_value, bool):
                    display_value = "✅" if param_value else "❌"
                elif isinstance(param_value, float):
                    display_value = f"{param_value:.2f}"
                elif isinstance(param_value, int):
                    display_value = str(param_value)
                else:
                    display_value = str(param_value)

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
