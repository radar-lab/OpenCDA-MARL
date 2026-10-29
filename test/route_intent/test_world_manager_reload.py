"""World reload: generous time-out, and a failed reset stops the run."""
from types import SimpleNamespace

import pytest

from opencda_marl._kernel.lifecycle import world_manager as wm_module
from opencda_marl._kernel.lifecycle.world_manager import (
    RELOAD_TIMEOUT_S,
    WorldManager,
    WorldResetError,
)


class FakeClient:
    def __init__(self, reload_error=None):
        self.timeouts = []
        self.timeout_during_reload = None
        self.reload_error = reload_error

    def set_timeout(self, seconds):
        self.timeouts.append(seconds)

    def reload_world(self):
        self.timeout_during_reload = self.timeouts[-1]
        if self.reload_error:
            raise RuntimeError(self.reload_error)

    def get_world(self):
        return object()


def _manager(client, configured_timeout=20.0, **cfg):
    coordinator = SimpleNamespace(
        carla_client=client, carla_world=None,
        config={"world": {"timeout": configured_timeout}})
    return WorldManager({"reset_frequency": 1, **cfg}, coordinator)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(wm_module.time, "sleep", lambda _s: None)


def test_reload_raises_the_client_timeout_then_restores_it():
    client = FakeClient()
    manager = _manager(client, configured_timeout=20.0)

    assert manager._reload_carla_world() is True
    assert client.timeout_during_reload == RELOAD_TIMEOUT_S
    assert client.timeouts[-1] == 20.0


def test_reload_never_lowers_a_larger_configured_timeout():
    client = FakeClient()
    manager = _manager(client, configured_timeout=300.0)

    manager._reload_carla_world()
    assert client.timeout_during_reload == 300.0


def test_reload_restores_the_timeout_when_it_fails():
    client = FakeClient(reload_error="time-out of 120000ms while waiting")
    manager = _manager(client)

    assert manager._reload_carla_world() is False
    assert client.timeouts[-1] == 20.0


def test_failed_reset_raises_instead_of_continuing():
    manager = _manager(FakeClient())
    manager.reload_world = lambda: False

    with pytest.raises(WorldResetError, match="stopping the run"):
        manager.on_episode_end()


def test_successful_reset_does_not_raise():
    manager = _manager(FakeClient())
    manager.reload_world = lambda: True
    manager.on_episode_end()


def test_episode_without_a_due_reset_does_not_raise():
    manager = _manager(FakeClient(), reset_frequency=5, auto_reset={"enabled": False})
    manager.reload_world = lambda: pytest.fail("no reset was due")
    manager.on_episode_end()


def test_failed_reset_ends_the_run_through_the_coordinator():
    from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator

    manager = _manager(FakeClient())
    manager.reload_world = lambda: False
    closed = []

    coord = object.__new__(MARLCoordinator)
    coord.config = {"runtime": {"backend": "carla"}}
    coord.pre_step_callbacks = []
    coord.post_step_callbacks = []
    coord.episode_callbacks = []
    coord.world_manager = manager
    coord.trace_writer = SimpleNamespace(close=lambda: closed.append(True))
    coord.scenario_manager = None
    coord.start_episode = 0
    coord.states = {"max_steps": 1, "max_episodes": 5}
    steps = {"n": 0}

    def step():
        steps["n"] += 1
        return {"done": False}

    coord.marl_env = SimpleNamespace(
        step=step, get_episode_metrics=lambda: {},
        get_current_step_rewards=lambda: {}, reset_episode=lambda: {})
    coord.evaluation_manager = SimpleNamespace(update_step=lambda m, r: None)

    with pytest.raises(WorldResetError):
        coord.run()
    assert steps["n"] == 1      # the second episode never started
    assert closed == [True]     # run()'s own cleanup still ran
