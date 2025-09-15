'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 14:50:25
FilePath     : /OpenCDA-MARL/opencda_marl/gui/widgets/metrics_display.py
Description  : Global metrics display widget for OpenCDA-MARL GUI.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from PySide6.QtWidgets import QGroupBox, QFormLayout, QLabel, QHBoxLayout
from typing import Dict, Any


class MetricsDisplay(QGroupBox):
    """Widget for displaying global simulation metrics."""

    def __init__(self, parent=None):
        """
        Initialize metrics display widget.
        """
        super().__init__("Global Metrics", parent)
        self._parameters = {}  # Store parameter display widgets
        self._setup_ui()

    # --------------------------------------------------------------------- #
    # UI Material
    # --------------------------------------------------------------------- #
    def _setup_ui(self):
        """Setup the metrics display UI."""
        layout = QHBoxLayout()

        # Column 1
        column1 = QFormLayout()
        layout.addLayout(column1)

        # Add metric parameters
        self._add_parameter("step", "0", "Step", column1)
        self._add_parameter("episode", "0", "Episode", column1)
        self._add_parameter("total_vehicles", "0", "Total Vehicles", column1)
        self._add_parameter("total_reward", "0.000", "Total Reward", column1)
        self._add_parameter("avg_reward", "0.000", "Average Reward", column1)
        self._add_parameter("max_reward_episode", "0.000",
                            "Best Ep Reward", column1)
        

        # Column 2
        column2 = QFormLayout()
        layout.addLayout(column2)

        # Add metric parameters
        self._add_parameter("success", "0", "Success", column2)
        self._add_parameter("collision", "0", "Collision", column2)
        self._add_parameter("active_agents", "0", "Actives", column2)

        # Real-time evaluation
        self._add_parameter("episode_success_rate", "0",
                            "Success Rate", column2)
        self._add_parameter("episode_collision_rate", "0",
                            "Collision Rate", column2)
        self._add_parameter("episode_throughput", "0", "Throughput(vph)", column2)

        self.setLayout(layout)

    def _add_parameter(self, name: str, default_value: str, label: str, layout: QFormLayout):
        """
        Add a display parameter to the form.
        """
        display_label = QLabel(str(default_value))
        display_label.setStyleSheet(
            "QLabel { background-color: white; color: black; padding: 4px; "
            "border: 1px solid black; border-radius: 2px; }"
        )

        self._parameters[name] = display_label
        layout = layout or self.layout()
        layout.addRow(f"{label}:", display_label)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update the displayed metrics.
        """
        for key, widget in self._parameters.items():
            if key in metrics:
                value = metrics[key]

                # Format based on type
                if isinstance(value, float):
                    display_value = f"{value:.3f}"
                else:
                    display_value = str(value)

                widget.setText(display_value)

        self._update_real_time_evaluation(metrics)

    def get_current_values(self) -> Dict[str, str]:
        """
        Get current displayed values.
        """
        return {name: widget.text() for name, widget in self._parameters.items()}

    # --------------------------------------------------------------------- #
    # Helper Functions
    # --------------------------------------------------------------------- #
    def _update_real_time_evaluation(self, metrics: Dict[str, Any]):
        """Update real-time evaluation."""
        total_vehicles = metrics['active_agents'] + \
            metrics['collision'] + metrics['success']
        self._parameters["total_vehicles"].setText(str(total_vehicles))
        if total_vehicles == 0:
            return
        success_rate = metrics['success'] / total_vehicles
        collision_rate = metrics['collision'] / total_vehicles
        # convert to percentage
        success_rate = success_rate * 100
        collision_rate = collision_rate * 100
        self._parameters["episode_success_rate"].setText(
            f"{success_rate:.3f}%")
        self._parameters["episode_collision_rate"].setText(
            f"{collision_rate:.3f}%")

        # throughput: how many vehicles have completed per hour
        elapsed_time = metrics['step']
        vps = metrics['success'] / elapsed_time / metrics['fixed_dt']
        vph = vps * 3600
        self._parameters["episode_throughput"].setText(f"{vph:.3f}")

    # --------------------------------------------------------------------- #
    # Clean up
    # --------------------------------------------------------------------- #
    def reset(self):
        """Reset all metrics to default values."""
        defaults = {
            "step": "0",
            "episode": "0",
            "total_vehicles": "0",
            "total_reward": "0.000",
            "episode_reward": "0.000",
            "avg_reward": "0.000",
            "success": "0",
            "collision": "0",
            "active_agents": "0",
            "episode_success_rate": "0",
            "episode_collision_rate": "0",
            "episode_throughput": "0"
        }

        for name, widget in self._parameters.items():
            widget.setText(defaults.get(name, "0"))
