"""
Agent observation panel widget for OpenCDA-MARL GUI.

Author: AXIBA leolihao@arizona.edu
Date: 2025-08-28
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget, QGroupBox, QTextEdit, QGridLayout
from PySide6.QtGui import QFont
from typing import Dict, Any, List
from datetime import datetime
import numpy as np


class AgentObservationPanel(QWidget):
    """Widget for displaying agent-specific observations in tabs."""

    def __init__(self, parent=None):
        """
        Initialize agent observation panel.
        """
        super().__init__(parent)
        self.agent_panels = {}  # Store agent-specific panels
        self._setup_ui()

    # --------------------------------------------------------------------- #
    # UI Materials
    # --------------------------------------------------------------------- #
    def _setup_ui(self):
        """Setup the agent observation panel UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Set tight spacing and margins for main layout
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel("Agent Observations")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title_label)

        # Tab widget for different agents
        self.agent_tabs = QTabWidget()
        layout.addWidget(self.agent_tabs)

        # Add placeholder tab
        self._add_placeholder_tab()

        # Environment info below agent tabs
        self.env_info_group = self._create_env_info_group()
        layout.addWidget(self.env_info_group)

    def _add_placeholder_tab(self):
        """Add placeholder tab when no agents are connected."""
        placeholder = QLabel("No agents connected")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.agent_tabs.addTab(placeholder, "No Agents")

    def _create_agent_panel(self, agent_id: str) -> QWidget:
        """
        Create observation panel for a specific agent.
        """
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Agent info section
        info_label = QLabel(f"Agent {agent_id} Information")
        info_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(info_label)

        # Multi-column grid layout for better space utilization
        grid_layout = QGridLayout()
        layout.addWidget(QWidget())  # Add some spacing

        # Column 1 - Basic Info
        vehicle_id_label = QLabel("N/A")
        vehicle_id_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Vehicle ID:"), 0, 0)
        grid_layout.addWidget(vehicle_id_label, 0, 1)

        vehicle_blueprint_label = QLabel("N/A")
        vehicle_blueprint_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Vehicle Blueprint:"), 1, 0)
        grid_layout.addWidget(vehicle_blueprint_label, 1, 1)

        status_label = QLabel("N/A")
        status_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Status:"), 2, 0)
        grid_layout.addWidget(status_label, 2, 1)
        
        waypoint_label = QLabel("N/A")
        waypoint_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Waypoint Buffer:"), 3, 0)
        grid_layout.addWidget(waypoint_label, 3, 1)
        
        
        # Column 2 - MARL Feature Info
        speed_label = QLabel("N/A")
        speed_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Speed (km/h):"), 0, 2)
        grid_layout.addWidget(speed_label, 0, 3)

        target_speed_label = QLabel("N/A")
        target_speed_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Target Speed:"), 1, 2)
        grid_layout.addWidget(target_speed_label, 1, 3)

        position_label = QLabel("N/A")
        position_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Position:"), 2, 2)
        grid_layout.addWidget(position_label, 2, 3)

        lane_id_label = QLabel("N/A")
        lane_id_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Lane ID:"), 3, 2)
        grid_layout.addWidget(lane_id_label, 3, 3)

        distance_intersection_label = QLabel("N/A")
        distance_intersection_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Dist to Intersection:"), 4, 2)
        grid_layout.addWidget(distance_intersection_label, 4, 3)

        distance_front_vehicle_label = QLabel("N/A")
        distance_front_vehicle_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Dist to Front Vehicle:"), 5, 2)
        grid_layout.addWidget(distance_front_vehicle_label, 5, 3)
        
        # Column 3 - MARL specific info
        reward_label = QLabel("N/A")
        reward_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Reward:"), 0, 4)
        grid_layout.addWidget(reward_label, 0, 5)

        #relative position to intersection
        relative_position_to_intersection_label = QLabel("N/A")
        relative_position_to_intersection_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Relative Position(Intersection):"), 1, 4)
        grid_layout.addWidget(relative_position_to_intersection_label, 1, 5)

        #heading angle
        heading_angle_label = QLabel("N/A")
        heading_angle_label.setFont(QFont("Arial", 10))
        grid_layout.addWidget(QLabel("Heading Angle:"), 2, 4)
        grid_layout.addWidget(heading_angle_label, 2, 5)



        # Add grid to panel layout
        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)
        layout.addWidget(grid_widget)

        # Add stretch to push content to top
        layout.addStretch()

        # Store references to value labels for updates
        panel.vehicle_id_label = vehicle_id_label
        panel.vehicle_blueprint_label = vehicle_blueprint_label
        panel.speed_label = speed_label
        panel.target_speed_label = target_speed_label
        panel.waypoint_label = waypoint_label
        panel.position_label = position_label
        panel.lane_id_label = lane_id_label
        panel.status_label = status_label
        panel.reward_label = reward_label
        panel.distance_intersection_label = distance_intersection_label
        panel.distance_front_vehicle_label = distance_front_vehicle_label
        panel.relative_position_to_intersection_label = relative_position_to_intersection_label
        panel.heading_angle_label = heading_angle_label
        return panel

    def _create_env_info_group(self) -> QGroupBox:
        """
        Create environment information group.
        """
        group = QGroupBox("Environment Info")
        layout = QVBoxLayout()

        # Set minimal margins and no spacing for clean appearance
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(0)

        self.env_info_text = QTextEdit()
        self.env_info_text.setReadOnly(True)
        self.env_info_text.setFont(QFont("Consolas", 9))
        self.env_info_text.setPlainText(
            "Waiting for environment information...")
        self.env_info_text.setMinimumHeight(100)
        self.env_info_text.setStyleSheet(
            "QTextEdit { background-color: white; color: black; margin: 0; padding: 2px; border: 1px solid black; }")

        layout.addWidget(self.env_info_text)
        group.setLayout(layout)

        return group

    def add_agent(self, agent_id: str):
        """
        Add a new agent tab.
        """
        if agent_id in self.agent_panels:
            return  # Agent already exists

        # Remove placeholder if this is the first real agent
        if len(self.agent_panels) == 0:
            self.agent_tabs.clear()

        # Create and add agent panel
        panel = self._create_agent_panel(agent_id)
        self.agent_panels[agent_id] = panel
        self.agent_tabs.addTab(panel, f"Agent {agent_id}")

    def remove_agent(self, agent_id: str):
        """
        Remove an agent tab.
        """
        if agent_id not in self.agent_panels:
            return

        # Find and remove the tab
        panel = self.agent_panels[agent_id]
        for i in range(self.agent_tabs.count()):
            if self.agent_tabs.widget(i) == panel:
                self.agent_tabs.removeTab(i)
                break

        del self.agent_panels[agent_id]

        # Add placeholder if no agents left
        if len(self.agent_panels) == 0:
            self._add_placeholder_tab()

    # --------------------------------------------------------------------- #
    # Information Display
    # --------------------------------------------------------------------- #
    def update_agent_observation(self, agent_id: str, observation: Dict[str, Any]):
        """
        Update observation display for a specific agent.
        """
        if agent_id not in self.agent_panels:
            self.add_agent(agent_id)

        # CRITICAL: use this to print the observation for debugging
        # print(f"DEBUG: agent {agent_id}: {observation}")

        panel = self.agent_panels[agent_id]

        # Update individual form fields
        if 'vehicle_id' in observation:
            panel.vehicle_id_label.setText(str(observation['vehicle_id']))

        if 'vehicle_blueprint' in observation:
            panel.vehicle_blueprint_label.setText(
                str(observation['vehicle_blueprint']))

        if 'speed' in observation:
            speed_value = observation['speed']
            panel.speed_label.setText(f"{speed_value:.1f}")

        if 'waypoint_buffer_size' in observation:
            buffer_size = observation['waypoint_buffer_size']
            panel.waypoint_label.setText(str(buffer_size))

        if 'position_x' in observation and 'position_y' in observation:
            panel.position_label.setText(
                f"({observation['position_x']:.1f}, {observation['position_y']:.1f})")

        if 'lane_position' in observation:
            panel.lane_id_label.setText(str(observation['lane_position']))

        if 'status' in observation:
            panel.status_label.setText(str(observation['status']))

        if 'current_reward' in observation:
            panel.reward_label.setText(str(observation['current_reward']))
            
        if 'target_speed' in observation:
            target_speed = observation['target_speed']
            if isinstance(target_speed, (int, float)):
                panel.target_speed_label.setText(f"{target_speed:.1f} km/h")
            else:
                panel.target_speed_label.setText(str(target_speed))

        if 'distance_to_intersection' in observation:
            distance = observation['distance_to_intersection']
            panel.distance_intersection_label.setText(f"{distance:.1f}m")

        if 'distance_to_front_vehicle' in observation:
            distance = observation['distance_to_front_vehicle']
            panel.distance_front_vehicle_label.setText(f"{distance:.1f}m")

        if 'relative_position_to_intersection' in observation:
            rel_pos = observation['relative_position_to_intersection']
            panel.relative_position_to_intersection_label.setText(f"({rel_pos['x']:.1f}, {rel_pos['y']:.1f})m")

        if 'heading_angle' in observation:
            heading_rad = observation['heading_angle']
            heading_deg = np.degrees(heading_rad)
            panel.heading_angle_label.setText(f"{heading_deg:.1f}°")



    def update_all_observations(self, observations: Dict[str, Dict[str, Any]]):
        """
        Update observations for all agents.
        """
        # Get current active agent IDs from observations
        active_agent_ids = set(observations.keys())
        
        # Get existing panel agent IDs
        existing_agent_ids = set(self.agent_panels.keys())
        
        # Remove panels for agents that are no longer active
        agents_to_remove = existing_agent_ids - active_agent_ids
        for agent_id in agents_to_remove:
            self.remove_agent(agent_id)
        
        # Update or add panels for active agents
        for agent_id, observation in observations.items():
            self.update_agent_observation(agent_id, observation)

    def update_environment_info(self, env_info: str = ""):
        """Update environment information display without jumping to top."""
        self.env_info_text.setPlainText(env_info)

    def clear_all_observations(self):
        """Clear all agent observations."""
        for agent_id in list(self.agent_panels.keys()):
            self.remove_agent(agent_id)

    def log_event(self, message: str):
        """
        Add a timestamped event message to the environment info log.
        """
        if hasattr(self, 'env_info_text'):
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}"

            # Move cursor to end and append new line
            cursor = self.env_info_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(f"\n{log_entry}")

            # Auto-scroll to bottom
            self.env_info_text.verticalScrollBar().setValue(
                self.env_info_text.verticalScrollBar().maximum()
            )
    # --------------------------------------------------------------------- #
    # Helper Functions
    # --------------------------------------------------------------------- #

    def get_agent_ids(self) -> List[str]:
        """
        Get list of current agent IDs.
        """
        return list(self.agent_panels.keys())

    def _format_observation(self, observation: Dict[str, Any]) -> str:
        """
        Format observation data for display.
        """
        if not observation:
            return "No observation data available"

        lines = []
        for key, value in observation.items():
            if isinstance(value, (list, tuple)):
                if len(value) > 10:  # Truncate long arrays
                    value_str = f"[{', '.join(map(str, value[:5]))} ... ({len(value)} items)]"
                else:
                    value_str = str(value)
            elif isinstance(value, dict):
                value_str = f"{{...}} ({len(value)} keys)"
            else:
                value_str = str(value)

            lines.append(f"{key}: {value_str}")

        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # Clean up
    # --------------------------------------------------------------------- #
    def reset(self):
        """Reset the agent observation panel."""
        self.clear_all_observations()
        self.env_info_text.setPlainText(
            "Waiting for environment information...")
