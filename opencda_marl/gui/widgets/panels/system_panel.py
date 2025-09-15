'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 13:41:08
FilePath     : /OpenCDA-MARL/opencda_marl/gui/widgets/panels/system_panel.py
Description  : System settings panel for OpenCDA-MARL GUI.

GUI component for displaying CARLA simulation system parameters.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from PySide6.QtWidgets import QWidget, QFormLayout, QLabel
from typing import Dict, Any


class SystemPanel(QWidget):
    """Panel for displaying CARLA simulation system parameters."""

    def __init__(self, parent=None):
        """
        Initialize system panel.

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
        """Setup the system panel UI."""
        layout = QFormLayout()
        self.setLayout(layout)

        # Add system parameters
        self._add_parameter("max_steps", "N/A", "Max Steps")
        self._add_parameter("max_episodes", "N/A", "Max Episodes")
        self._add_parameter("agent_type", "N/A", "Agent Type")
        self._add_parameter("sync_mode", "N/A", "Synchronous Mode")
        self._add_parameter("fixed_delta_seconds", "N/A",
                            "Fixed Delta Time (s)")
        self._add_parameter("max_substeps", "N/A", "Max Substeps")
        self._add_parameter("max_substep_delta_time", "N/A",
                            "Max Substep Delta Time (s)")
        self._add_parameter("substepping", "N/A", "Substepping Enabled")
        self._add_parameter("spectator_as_ego", "N/A", "Spectator as Ego")
        self._add_parameter("no_rendering_mode", "N/A", "No Rendering Mode")

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

    def update_parameters(self, data: Dict[str, Any]):
        """
        Update displayed system parameter values.

        Parameters
        ----------
        data : dict
            System parameter data from monitor
        """
        for param_key, param_value in data.items():
            if param_key in self._parameters:
                value = param_value

                # Format the value for display
                if isinstance(value, bool):
                    display_value = "✅" if value else "❌"
                elif isinstance(value, float):
                    display_value = f"{value:.3f}"
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
