'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 13:42:47
FilePath     : /OpenCDA-MARL/opencda_marl/gui/observation_viewer.py
Description  : Observation Viewer Widget: GUI component for displaying simulation observations.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from typing import Dict, Any
from .widgets import AgentObservationPanel, EnvironmentPanel
from datetime import datetime


class ObservationViewer(QWidget):
    """
    Widget for displaying real-time simulation observations.
    """
    observation_updated = Signal(dict)

    def __init__(self, coordinator, parent=None):
        """
        Initialize Observation Viewer.
        """
        super().__init__(parent)
        self.coordinator = coordinator
        self._event_history = []  # Store all events for tracing
        self._setup_ui()

        if not coordinator:
            raise ValueError("Coordinator is required")
        self._connect_coordinator()

    # --------------------------------------------------------------------- #
    # UI Materials
    # --------------------------------------------------------------------- #
    def _setup_ui(self):
        """Setup the observation viewer UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create splitter for multi-panel layout
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)

        # Agent observations panel (left side)
        self.agent_panel = AgentObservationPanel()
        self.splitter.addWidget(self.agent_panel)

        # Environment state panel (right side) - full height
        self.environment_panel = EnvironmentPanel()
        self.splitter.addWidget(self.environment_panel)

        # Set initial splitter proportions (agent panel wider)
        self.splitter.setSizes([700, 400])

    # --------------------------------------------------------------------- #
    # Private methods
    # --------------------------------------------------------------------- #
    def _connect_coordinator(self):
        """Connect to coordinator for observation updates."""
        if not self.coordinator:
            raise ValueError("Coordinator is required")

        # Register for step completion callbacks
        self.coordinator.register_post_step_callback(self._on_step_completed)
        self.coordinator.register_episode_callback(self._on_episode_started)
        self.coordinator.register_pre_step_callback(self._on_step_begin)

        self.update_monitor_data()

    def _on_step_begin(self):
        """Callback when coordinator begins a step."""
        pass

    def _on_step_completed(self):
        """Callback when coordinator completes a step."""
        self.update_monitor_data()
        self.update_metrics()
        self.update_observations()

    def _on_episode_started(self):
        """Callback when coordinator starts a new episode."""
        # Episode started
        pass

    # --------------------------------------------------------------------- #
    # main thread
    # --------------------------------------------------------------------- #
    def update_monitor_data(self):
        """
        Update panels with monitor data from CARLA.
        """
        monitor_data = self.coordinator.get_monitor_data()
        self.environment_panel.update_monitor_data(monitor_data)
        self.update_environment_info(monitor_data)

    def update_metrics(self):
        """
        Update global metrics display.
        """
        metrics = self.coordinator.get_metrics()
        self.environment_panel.update_metrics(metrics)

    def update_observations(self):
        """
        Update the observation display.
        """
        observations = self.coordinator.get_observations()
        self.agent_panel.update_all_observations(observations)

        recent_events = self.coordinator.marl_env.get_current_events()

        if recent_events:
            timestamp = datetime.now().strftime("%H:%M:%S")
            for event in recent_events:
                step_info = f"Step: {event.step}, Event Type: {event.event_type}, Vehicle ID: {event.vehicle_id}"
                self._event_history.append(f"[{timestamp}] {step_info}")

    def update_environment_info(self, monitor_data: Dict[str, Any]):
        """
        Update environment information display.
        """
        if 'system' in monitor_data:
            # Log vehicle count changes as events
            if hasattr(self, '_last_vehicle_count'):
                current_count = monitor_data.get(
                    'traffic', {}).get('spawned_vehicles', 0)
                if current_count != self._last_vehicle_count:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if current_count > self._last_vehicle_count:
                        self._event_history.append(
                            f"[{timestamp}] Vehicle spawned (Total: {current_count})")
                    else:
                        self._event_history.append(
                            f"[{timestamp}] Vehicle removed (Total: {current_count})")
                    self._last_vehicle_count = current_count
            else:
                self._last_vehicle_count = monitor_data.get(
                    'traffic', {}).get('spawned_vehicles', 0)

            # Combine system status + event history and update display
            env_info = self._format_env_info(monitor_data)
            event_log = "\n\n=== Event Log ===\n" + \
                "\n".join(self._event_history) if self._event_history else ""
            complete_info = env_info + event_log
            self.agent_panel.update_environment_info(complete_info)

    # --------------------------------------------------------------------- #
    # Helper Functions
    # --------------------------------------------------------------------- #
    def _format_env_info(self, monitor_data: Dict[str, Any]) -> str:
        """Format environment info for display."""
        lines = []

        if 'traffic' in monitor_data:
            traffic = monitor_data['traffic']
            lines.append("=== Traffic Information ===")
            lines.append(
                f"Spawned Vehicles: {traffic.get('spawned_vehicles', 'N/A')}")
            lines.append(
                f"Failure Events Queue: {traffic.get('queue_count', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # Clean up
    # --------------------------------------------------------------------- #
    def reset(self):
        """Reset the observation viewer."""
        self.agent_panel.reset()
        self.environment_panel.reset()
        self._event_history.clear()
