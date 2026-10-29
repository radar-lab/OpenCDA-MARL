"""A CARLA world build runs under a longer client timeout than ordinary calls.

Generating the OpenDRIVE world takes 20 to 37 s on a windowed server, longer
than ``world.timeout``; a retry would queue behind the still-running first
request, so the first request gets the generous timeout instead.
"""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("carla", reason="the vendored scenario manager imports carla")

from opencda.scenario_testing.utils import sim_api  # noqa: E402

TIMEOUT = RuntimeError(
    "time-out of 20000ms while waiting for the simulator, make sure the "
    "simulator is ready and connected to 127.0.0.1:2000")


class _Build:
    """A world build that raises ``error`` if given, else returns a world."""

    def __init__(self, error=None):
        self.error = error
        self.calls = 0
        self.world = MagicMock(name="world")

    def __call__(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.world


def _timeouts(client):
    return [c.args[0] for c in client.set_timeout.call_args_list]


def test_the_build_runs_once_under_the_build_timeout_then_restores_the_configured_one():
    client = MagicMock()
    build = _Build()
    assert sim_api.build_world_with_retry(build, client, 20.0) is build.world
    assert build.calls == 1
    assert _timeouts(client) == [sim_api.WORLD_BUILD_TIMEOUT, 20.0]


def test_a_configured_timeout_above_the_build_timeout_is_kept():
    client = MagicMock()
    sim_api.build_world_with_retry(_Build(), client, 300.0)
    assert _timeouts(client) == [300.0, 300.0]


@pytest.mark.parametrize("error", [TIMEOUT, RuntimeError("map not found")])
def test_an_error_propagates_without_a_retry_and_restores_the_timeout(error):
    client = MagicMock()
    build = _Build(error)
    with pytest.raises(RuntimeError, match=str(error)[:8]):
        sim_api.build_world_with_retry(build, client, 20)
    assert build.calls == 1
    assert _timeouts(client) == [sim_api.WORLD_BUILD_TIMEOUT, 20]


@pytest.mark.parametrize("error, expected", [
    (TIMEOUT, True),
    (RuntimeError("rpc call timed out"), True),
    (RuntimeError("map not found"), False),
    (ValueError("time-out"), False),
])
def test_client_timeout_detection(error, expected):
    assert sim_api.is_client_timeout(error) is expected


def _world(timeout=None):
    world = {'sync_mode': True, 'weather': {}}
    if timeout is not None:
        world['timeout'] = timeout
    return world


def test_scenario_manager_builds_the_opendrive_world_under_the_build_timeout(
        monkeypatch, tmp_path):
    client = MagicMock()
    monkeypatch.setattr(sim_api.carla, 'Client', lambda *_: client)
    monkeypatch.setattr(sim_api.ScenarioManager, 'set_weather', lambda *_: object())
    build = _Build()
    monkeypatch.setattr(sim_api, 'load_customized_world', lambda *a, **k: build())
    xodr = tmp_path / 'roads.xodr'
    xodr.write_text('<OpenDRIVE/>')

    manager = sim_api.ScenarioManager(
        {'world': _world(timeout=25.0)}, apply_ml=False, xodr_path=str(xodr))

    assert manager.world is build.world
    assert build.calls == 1
    # world.timeout configures the client; the build raises it, then restores.
    assert _timeouts(client) == [25.0, sim_api.WORLD_BUILD_TIMEOUT, 25.0]


def test_scenario_manager_reports_a_town_timeout_as_a_timeout(monkeypatch):
    client = MagicMock()
    client.load_world.side_effect = TIMEOUT
    monkeypatch.setattr(sim_api.carla, 'Client', lambda *_: client)
    monkeypatch.setattr(sim_api.ScenarioManager, 'set_weather', lambda *_: object())

    with pytest.raises(RuntimeError, match="time-out"):
        sim_api.ScenarioManager({'world': _world()}, apply_ml=False, town='Town05')
    assert client.load_world.call_count == 1
