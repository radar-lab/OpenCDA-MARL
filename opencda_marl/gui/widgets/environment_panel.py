'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 13:42:32
FilePath     : /OpenCDA-MARL/opencda_marl/gui/widgets/environment_panel.py
Description  : Environment Panel Widget

GUI component for displaying environment state and information.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QTabWidget
from PySide6.QtGui import QFont
from typing import Dict, Any
from .metrics_display import MetricsDisplay
from .panels import SystemPanel, WeatherPanel, TrafficPanel, RewardPanel


class EnvironmentPanel(QWidget):
    """Widget for displaying environment state and information."""

    def __init__(self, parent=None):
        """Initialize environment panel."""
        super().__init__(parent)
        self._setup_ui()

    # --------------------------------------------------------------------- #
    # UI Material
    # --------------------------------------------------------------------- #
    def _setup_ui(self):
        """Setup the environment panel UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title_label = QLabel("Environment State")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title_label)

        # Global metrics display
        self.metrics_display = MetricsDisplay()
        layout.addWidget(self.metrics_display)

        # Settings tabs with different panels
        group_panel = QGroupBox("Debug Settings")
        self.settings_tabs = QTabWidget()

        # Add panels as tabs
        self.system_panel = SystemPanel()
        self.settings_tabs.addTab(self.system_panel, "System")

        self.weather_panel = WeatherPanel()
        self.settings_tabs.addTab(self.weather_panel, "Weather")

        self.traffic_panel = TrafficPanel()
        self.settings_tabs.addTab(self.traffic_panel, "Traffic")

        self.reward_panel = RewardPanel()
        self.settings_tabs.addTab(self.reward_panel, "Reward")

        group_panel.setLayout(QVBoxLayout())
        group_panel.layout().addWidget(self.settings_tabs)
        layout.addWidget(group_panel)

        # Add stretch to push content to top while filling available height
        layout.addStretch()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def update_monitor_data(self, monitor_data: Dict[str, Any]):
        """
        Update panels with monitor data from CARLA.
        """
        if "system" in monitor_data:
            self.system_panel.update_parameters(monitor_data["system"])

        if 'weather' in monitor_data:
            self.weather_panel.update_parameters(monitor_data['weather'])

        if "traffic" in monitor_data:
            self.traffic_panel.update_parameters(monitor_data["traffic"])

        if "reward" in monitor_data:
            self.reward_panel.update_parameters(monitor_data["reward"])

    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update global metrics display.
        """
        self.metrics_display.update_metrics(metrics)

    # --------------------------------------------------------------------- #
    # Clean up
    # --------------------------------------------------------------------- #
    def reset(self):
        """Reset all metrics to default values."""
        self.metrics_display.reset()
