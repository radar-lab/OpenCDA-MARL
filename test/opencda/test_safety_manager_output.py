"""The safety manager prints a warning once, then again only when the set of
triggered warnings changes."""
from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("carla", reason="the safety sensors import carla")

from opencda.core.safety.safety_manager import SafetyManager  # noqa: E402


class _Sensor:
    def __init__(self):
        self.status = {}

    def tick(self, data_dict):
        pass

    def return_status(self):
        return self.status


def _manager(sensor):
    manager = SafetyManager.__new__(SafetyManager)
    manager.cav_world = SimpleNamespace(global_clock=0)
    manager.print_message = True
    manager._printed_warnings = frozenset()
    manager.status_queue = deque()
    manager.sensors = [sensor]
    return manager


def test_an_unchanged_warning_prints_once_and_a_changed_set_prints_again(capsys):
    sensor = _Sensor()
    manager = _manager(sensor)

    def tick(status):
        sensor.status = status
        manager.update_info({})
        return capsys.readouterr().out.count("Safety Warning")

    assert tick({'offroad': False, 'stuck': False}) == 0
    assert tick({'offroad': True, 'stuck': False}) == 1
    assert tick({'offroad': True, 'stuck': False}) == 0
    assert tick({'offroad': True, 'stuck': True}) == 1
    # Cleared, then raised again: a new episode of the warning prints.
    assert tick({'offroad': False, 'stuck': False}) == 0
    assert tick({'offroad': True, 'stuck': False}) == 1
