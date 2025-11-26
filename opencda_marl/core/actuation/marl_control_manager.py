# -*- coding: utf-8 -*-
"""
MARL-specific Controller Manager

Uses the fixed MARL PID controller instead of the default OpenCDA controller.
"""
# Author: AXIBA <leolihao@arizona.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencda_marl.core.actuation.marl_pid_controller import Controller


class MARLControlManager(object):
    """
    Controller manager for MARL scenarios.

    Uses the fixed PID controller that correctly applies longitudinal gains
    for speed control.

    Parameters
    ----------
    control_config : dict
        The configuration dictionary of the control manager module.

    Attributes
    ----------
    controller : Controller
        The MARL-specific PID controller.
    """

    def __init__(self, control_config):
        # Always use the fixed MARL PID controller
        self.controller = Controller(control_config['args'])

    def update_info(self, ego_pos, ego_speed):
        """
        Update ego vehicle information for controller.
        """
        self.controller.update_info(ego_pos, ego_speed)

    def run_step(self, target_speed, waypoint):
        """
        Execute current controller step.
        """
        control_command = self.controller.run_step(target_speed, waypoint)
        return control_command
