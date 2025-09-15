'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 17:49:26
FilePath     : /OpenCDA-MARL/opencda_marl/gui/widgets/panels/reward_panel.py
Description  : 
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from PySide6.QtWidgets import QWidget, QFormLayout, QLabel
from typing import Dict, Any


class RewardPanel(QWidget):
    """Panel for displaying traffic manager and vehicle parameters."""

    def __init__(self, parent=None):
        """
        Initialize reward panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self._parameters = {}  # Store parameter display widgets
        self._labels = {}  # Store form label widgets for dynamic updating
        self._current_algorithm = None  # Track current algorithm for label updates
        self._setup_ui()

    # --------------------------------------------------------------------- #
    # UI Material
    # --------------------------------------------------------------------- #
    def _setup_ui(self):
        """Setup the reward panel UI."""
        layout = QFormLayout()
        self.setLayout(layout)

        # Reward settings
        self._add_parameter("collision", "N/A", "Collision Reward")
        self._add_parameter("success", "N/A", "Success Reward")
        self._add_parameter("step_penalty", "N/A", "Step Penalty")

        # Training information
        self._add_parameter("episode_count", "N/A", "Episode Count")
        self._add_parameter("algorithm", "N/A", "Algorithm Type")
        self._add_parameter("epsilon", "N/A", "Epsilon")
        self._add_parameter("training_mode", "N/A", "Training Mode")
        
        # DQN-specific metrics
        self._add_parameter("memory_size", "N/A", "Memory Used")
        self._add_parameter("memory_capacity", "N/A", "Memory Capacity")
        self._add_parameter("target_updates", "N/A", "Target Updates")
        self._add_parameter("device", "N/A", "Device")
        
        # Loss metrics (for TD3, DQN, etc.)
        self._add_parameter("actor_loss", "N/A", "Actor Loss")
        self._add_parameter("critic_loss", "N/A", "Critic Loss")

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

        # Create label widget and store both
        label_widget = QLabel(f"{label}:")
        self._parameters[name] = display_label
        self._labels[name] = label_widget
        
        self.layout().addRow(label_widget, display_label)

    def update_parameters(self, reward_data: Dict[str, Any]):
        """
        Update displayed reward parameter values.

        Parameters
        ----------
        reward_data : dict
            Reward parameter data from monitor
        """
        # Update labels dynamically based on algorithm type
        current_algorithm = reward_data.get('algorithm_type', reward_data.get('algorithm', 'Unknown'))
        if current_algorithm != self._current_algorithm:
            self._current_algorithm = current_algorithm
            self._update_labels_for_algorithm(current_algorithm)
        
        for param_key, param_value in reward_data.items():
            if param_key in self._parameters:
                # Format the value for display
                if isinstance(param_value, bool):
                    display_value = "✅" if param_value else "❌"
                elif param_key in ['actor_loss', 'critic_loss'] and isinstance(param_value, float):
                    # Format losses with more precision
                    display_value = f"{param_value:.4f}"
                elif isinstance(param_value, float):
                    display_value = f"{param_value:.2f}"
                elif isinstance(param_value, int):
                    display_value = str(param_value)
                else:
                    display_value = str(param_value)

                self._parameters[param_key].setText(display_value)
        
        # Special formatting for memory usage ratio
        if 'memory_size' in reward_data and 'memory_capacity' in reward_data:
            memory_used = reward_data['memory_size']
            memory_cap = reward_data['memory_capacity']
            if memory_cap > 0:
                memory_ratio = f"{memory_used}/{memory_cap}"
                self._parameters['memory_size'].setText(memory_ratio)
    
    def _update_labels_for_algorithm(self, algorithm: str):
        """Update labels based on the current algorithm."""
        if algorithm == 'TD3' or algorithm == 'twin_delayed_ddpg':
            # TD3-specific labels
            if 'epsilon' in self._labels:
                self._labels['epsilon'].setText("Exploration Noise:")
            if 'target_updates' in self._labels:
                self._labels['target_updates'].setText("Policy Updates:")
        else:
            # Default labels for DQN, Q-learning, etc.
            if 'epsilon' in self._labels:
                self._labels['epsilon'].setText("Epsilon:")
            if 'target_updates' in self._labels:
                self._labels['target_updates'].setText("Target Updates:")

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
