'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28
FilePath: \OpenCDA-MARL\opencda_marl\gui\dashboard.py
Description  : Main GUI window for MARL training and simulation control.
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget
from .observation_viewer import ObservationViewer
from .step_controller import StepController


class Dashboard(QMainWindow):
    """
    Main GUI dashboard for MARL experiments.

    Provides unified interface for:
    - Step-by-step simulation control
    - Real-time observation visualization
    - Episode management
    """

    def __init__(self, coordinator, app):
        """
        Initialize MARL Dashboard.

        Parameters
        ----------
        coordinator : MARLCoordinator
            The coordinator instance to control
        app : QApplication
            The QApplication instance (must be created before Dashboard)
        """
        super().__init__()
        self.coordinator = coordinator
        self.app = app
        self._setup_ui()

        logger.info("OpenCDA-MARL Dashboard initialized")

    # --------------------------------------------------------------------- #
    # UI Materials
    # --------------------------------------------------------------------- #
    def _setup_ui(self):
        """Setup the main dashboard UI."""
        self.setWindowTitle("MARL Control Center")
        self.setFixedSize(1200, 800)

        # Create central widget with horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        self.obs_viewer = ObservationViewer(self.coordinator)
        layout.addWidget(self.obs_viewer)

        self.step_controller = StepController(self.coordinator, parent=self)
        self.addToolBar(self.step_controller)

    # --------------------------------------------------------------------- #
    # GUI control methods
    # --------------------------------------------------------------------- #
    def reset(self):
        """Reset the dashboard."""
        self.coordinator.scenario_manager.reset()
        self.coordinator.marl_env.reset()
        self.obs_viewer.reset()

    # --------------------------------------------------------------------- #
    # main thread
    # --------------------------------------------------------------------- #

    def run(self):
        """
        Run the dashboard application.

        Returns
        -------
        int
            Application exit code
        """
        # Ensure coordinator stays alive during GUI execution
        self.app.coordinator = self.coordinator
        self.show()

        self._print_startup_message()

        # Run event loop
        try:
            return self.app.exec_()
        except KeyboardInterrupt:
            print("\nGUI interrupted by user")
            return 0
        finally:
            if hasattr(self.app, 'coordinator'):
                self.app.coordinator.close()
                delattr(self.app, 'coordinator')

    # --------------------------------------------------------------------- #
    # private methods
    # --------------------------------------------------------------------- #
    def _print_startup_message(self):
        """Print startup information to console."""
        print("=" * 60)
        print("GUI INITIALIZED SUCCESSFULLY")
        print("=" * 60)
        print("MARL Control Center is ready!")
        print("Use the Step Controller to control simulation:")
        print("  - Click '↩︎ Step' to execute one step")
        print("  - Click '▶ Start' to run continuously")
        print("  - Click '⏸ Pause' to pause continuous execution")
        print("  - Use 'Reset Episode' to start a new episode")
        print("=" * 60)
