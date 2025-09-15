'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28
FilePath: \OpenCDA-MARL\opencda_marl\gui\step_controller.py
Description  : Step Controller Widget: GUI component for controlling simulation steps.

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QToolBar, QLabel
from PySide6.QtGui import QAction


class StepController(QToolBar):
    """
    Widget for controlling simulation execution step-by-step.
    """

    def __init__(self, coordinator, parent=None):
        """
        Initialize Step Controller.

        Parameters
        ----------
        coordinator : MARLCoordinator
            The coordinator instance to control
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.coordinator = coordinator
        self.spectator = coordinator.carla_spectator
        self.QTimer = QTimer()
        self.QTimer.timeout.connect(self._single_step)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the step controller UI."""
        self.addWidget(QLabel("Step Controller"))
        self.addSeparator()

        # Add control buttons
        self.step_action = QAction("↩︎ Step", self)
        self.step_action.triggered.connect(self._single_step)
        self.step_action.setShortcut("Ctrl+S")
        self.addAction(self.step_action)
        self.start_action = QAction("▶ Start", self)
        self.start_action.triggered.connect(self._start_continuous)
        self.start_action.setShortcut("Ctrl+R")
        self.addAction(self.start_action)
        self.pause_action = QAction("⏸ Pause", self)
        self.pause_action.triggered.connect(self._pause)
        self.pause_action.setDisabled(True)
        self.pause_action.setShortcut("Ctrl+P")
        self.addAction(self.pause_action)
        self.reset_action = QAction("Reset Episode", self)
        self.reset_action.triggered.connect(self.coordinator.reset_episode)
        self.reset_action.setShortcut("Ctrl+E")
        self.addAction(self.reset_action)

        # Add view control buttons
        self.addSeparator()
        self.addWidget(QLabel("View Control"))
        self.world_sync = QAction("⚲ Current View", self)
        self.world_sync.triggered.connect(self._get_current_view)
        self.addAction(self.world_sync)

        self.addSeparator()
        # Add status display
        self.status_label = QLabel("Ready")
        self.addWidget(self.status_label)

    def _get_current_view(self):
        view = self.spectator.get_current_view()
        position = view['position']
        rotation = view['rotation']
        x, y, z = position.x, position.y, position.z
        pitch, yaw, roll = rotation.pitch, rotation.yaw, rotation.roll
        info = f"x: {x:.1f}, y: {y:.1f}, z: {z:.1f}, pitch: {pitch:.1f}, yaw: {yaw:.1f}, roll: {roll:.1f}"
        print(f"Current view: {info}")
        self.status_label.setText(f"Current view: {info}")

    def _start_continuous(self):
        """Start continuous execution."""
        if not self.QTimer.isActive():
            self.status_label.setText("Running...")
            self.start_action.setDisabled(True)
            self.pause_action.setDisabled(False)

            # get the fixed delta time from the coordinator
            fixed_dt = self.coordinator.states['fixed_dt']
            self.QTimer.start(fixed_dt * 1000)

    def _pause(self):
        """Pause continuous execution."""
        if self.QTimer.isActive():
            self.status_label.setText("Paused")
            self.start_action.setDisabled(False)
            self.pause_action.setDisabled(True)
            self.QTimer.stop()

    # --------------------------------------------------------------------- #
    # Single step
    # --------------------------------------------------------------------- #

    def _single_step(self):
        """Execute a single simulation step."""
        self.status_label.setText("Executing step...")
        max_steps = self.coordinator.states['max_steps']
        max_episodes = self.coordinator.states['max_episodes']

        current_step = self.coordinator.states['step']
        current_episode = self.coordinator.states['episode']
        if current_step >= max_steps and current_episode < max_episodes:
            self.coordinator.reset_episode()
        elif current_step >= max_steps and current_episode == max_episodes:
            self.status_label.setText(
                f"All episodes completed ({max_episodes}/{max_episodes})")
            self._pause()
        else:
            self.coordinator.step()

        updated_step = self.coordinator.states['step']
        updated_active_agents = self.coordinator.states.get(
            'active_agents', 0)

        self.status_label.setText(
            f"Episode {current_episode}/{max_episodes} "
            f"- Step {updated_step}/{max_steps} "
            f"({updated_active_agents} agents)")
