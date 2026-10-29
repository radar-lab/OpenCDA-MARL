"""ActorRegistry / SensorRig lifecycle tests with fake CARLA objects."""
import pytest

from opencda_marl._kernel.lifecycle.actor_registry import (
    ActorLeakError,
    ActorRegistry,
    SpawnCollisionError,
)
from opencda_marl._kernel.lifecycle.sensor_rig import SensorRig


class FakeActor:
    _next_id = 1

    def __init__(self, type_id="vehicle.test", parent=None, sticky=False):
        self.id = FakeActor._next_id
        FakeActor._next_id += 1
        self.type_id = type_id
        self.parent = parent
        self.is_alive = True
        self.stopped = False
        # A sticky actor survives destroy() — simulates a genuine leak.
        self.sticky = sticky

    def stop(self):
        self.stopped = True

    def destroy(self):
        """CARLA semantics: True only when a live actor was destroyed."""
        was_alive = self.is_alive
        if not self.sticky:
            self.is_alive = False
        return was_alive


class FakeActorList(list):
    def filter(self, pattern):
        prefix = pattern.rstrip("*").rstrip(".")
        return FakeActorList(a for a in self if a.type_id.startswith(prefix))


class FakeWorld:
    def __init__(self):
        self.actors = []
        self.fail_spawn_with = None

    def tick(self):
        return None

    def spawn_actor(self, blueprint, transform):
        if self.fail_spawn_with is not None:
            raise RuntimeError(self.fail_spawn_with)
        actor = FakeActor("vehicle.test")
        self.actors.append(actor)
        return actor

    def attach_sensor(self, vehicle):
        sensor = FakeActor("sensor.other.gnss", parent=vehicle)
        self.actors.append(sensor)
        return sensor

    def get_actors(self, actor_ids=None):
        alive = [a for a in self.actors if a.is_alive]
        if actor_ids is None:
            return FakeActorList(alive)
        wanted = {int(i) for i in actor_ids}
        return FakeActorList(a for a in alive if a.id in wanted)


class FakeClient:
    def __init__(self, world, fail=False):
        self.world = world
        self.fail = fail
        self.batches = []

    def apply_batch_sync(self, commands, tick):
        if self.fail:
            raise RuntimeError("rpc timeout")
        self.batches.append(commands)
        wanted = {c.actor_id for c in commands}
        responses = []
        for actor in self.world.actors:
            if actor.id in wanted and actor.is_alive:
                actor.destroy()
        for _ in commands:
            responses.append(type("Resp", (), {"error": ""})())
        return responses


def make_registry(strict=False, client_fail=False, with_client=True):
    world = FakeWorld()
    client = FakeClient(world, fail=client_fail) if with_client else None
    return ActorRegistry(world, client, strict=strict), world


def test_spawn_registers_before_return():
    registry, world = make_registry()
    vehicle = registry.spawn_vehicle(object(), object())
    assert registry.tracked_vehicle_ids() == [vehicle.id]


def test_spawn_collision_is_translated_and_untracked():
    registry, world = make_registry()
    world.fail_spawn_with = "Spawn failed because of collision at spawn position"
    with pytest.raises(SpawnCollisionError):
        registry.spawn_vehicle(object(), object())
    assert registry.tracked_vehicle_ids() == []


def test_spawn_other_errors_propagate_unchanged():
    registry, world = make_registry()
    world.fail_spawn_with = "connection reset"
    with pytest.raises(RuntimeError, match="connection reset"):
        registry.spawn_vehicle(object(), object())


def test_abort_spawn_destroys_vehicle_and_attached_sensors():
    registry, world = make_registry()
    vehicle = registry.spawn_vehicle(object(), object())
    sensor = world.attach_sensor(vehicle)
    registry.abort_spawn(vehicle)
    assert not vehicle.is_alive
    assert not sensor.is_alive
    assert sensor.stopped  # stop() must precede destroy() for sensors
    assert registry.tracked_vehicle_ids() == []


def test_successful_teardown_produces_no_leak_report():
    """Verification must not fire on actors the owner chain destroyed — in
    sync mode that would produce false-positive 'leaks' plus a failed
    re-destroy per terminal vehicle."""
    registry, world = make_registry()
    vehicle = registry.spawn_vehicle(object(), object())
    sensor = world.attach_sensor(vehicle)
    registry.adopt_sensors(vehicle.id, [sensor.id])
    # Owner's destroy chain worked.
    sensor.destroy()
    vehicle.destroy()
    registry.finalize_vehicle(vehicle.id)
    assert registry.sweep_pending() == []
    assert registry.sweep_pending() == []
    assert registry.stats["leaks_reclaimed"] == 0


def test_stale_snapshot_listing_is_not_a_leak():
    """A stale snapshot case: the world snapshot can still list an
    actor that is already gone (destroy() returns False). That must not be
    counted or reported as a leak."""
    registry, world = make_registry()
    vehicle = registry.spawn_vehicle(object(), object())
    registry.finalize_vehicle(vehicle.id)
    # Simulate the stale snapshot: still listed as alive, but the server
    # refuses the destroy (already gone).
    vehicle.sticky = True

    def stale_destroy():
        return False  # CARLA: "unable to destroy actor: not found"

    vehicle.destroy = stale_destroy
    assert registry.sweep_pending(force=True) == []
    assert registry.stats["leaks_reclaimed"] == 0


def test_finalize_defers_two_sweeps_then_reclaims_genuine_leak():
    registry, world = make_registry()
    vehicle = registry.spawn_vehicle(object(), object())
    sensor = world.attach_sensor(vehicle)
    registry.adopt_sensors(vehicle.id, [sensor.id])
    # Owner's destroy path "ran" but destroyed nothing.
    registry.finalize_vehicle(vehicle.id)
    assert registry.tracked_vehicle_ids() == []
    # Age gate: the first sweep after the destroy is too early to trust the
    # snapshot; nothing is touched.
    assert registry.sweep_pending() == []
    assert vehicle.is_alive and sensor.is_alive
    # Second sweep: snapshot is definitely fresh — genuine leak reclaimed.
    reclaimed = registry.sweep_pending()
    assert sorted(reclaimed) == sorted([vehicle.id, sensor.id])
    assert not vehicle.is_alive
    assert not sensor.is_alive
    assert registry.stats["leaks_reclaimed"] == 2
    assert registry.sweep_pending() == []


def test_destroy_all_uses_batch_and_forced_sweep_reports_clean():
    registry, world = make_registry()
    v1 = registry.spawn_vehicle(object(), object())
    v2 = registry.spawn_vehicle(object(), object())
    registry.destroy_all()
    world.tick()
    assert registry.sweep_pending(force=True) == []
    assert not v1.is_alive and not v2.is_alive
    assert registry.stats["leaks_reclaimed"] == 0
    assert registry.tracked_vehicle_ids() == []


def test_destroy_all_batch_failure_is_recovered_by_sweep():
    registry, world = make_registry(client_fail=True)
    v1 = registry.spawn_vehicle(object(), object())
    s1 = world.attach_sensor(v1)
    registry.adopt_sensors(v1.id, [s1.id])
    registry.destroy_all()
    world.tick()
    # Batch threw, but the forced verification sweep destroys everything.
    reclaimed = registry.sweep_pending(force=True)
    assert sorted(reclaimed) == sorted([v1.id, s1.id])
    assert not v1.is_alive
    assert not s1.is_alive
    assert registry.tracked_vehicle_ids() == []


def test_destroy_all_without_client_is_recovered_by_sweep():
    registry, world = make_registry(with_client=False)
    v1 = registry.spawn_vehicle(object(), object())
    registry.destroy_all()
    registry.sweep_pending(force=True)
    assert not v1.is_alive


def test_strict_mode_raises_at_sweep_on_genuine_leak():
    registry, world = make_registry(strict=True, with_client=False)
    registry.spawn_vehicle(object(), object())
    registry.destroy_all()
    with pytest.raises(ActorLeakError):
        registry.sweep_pending(force=True)


# --------------------------------------------------------------------- #
# MARLAgentManager spawn paths (bare construction, repo test style)
# --------------------------------------------------------------------- #
def _bare_agent_manager(world):
    from collections import deque
    from opencda_marl._kernel.agent_manager import MARLAgentManager

    mgr = object.__new__(MARLAgentManager)
    mgr.world = world
    mgr.client = None
    mgr.registry = ActorRegistry(world, None)
    mgr._failure_events = deque()
    mgr._spawn_retry_counts = {}
    mgr._current_snapshot = None
    mgr._vehicle_adapters = []
    mgr._spawn_metadata_by_actor_id = {}
    mgr.config = {}
    mgr.map = None
    mgr.cav_world = None
    mgr.agent_type = "simple"
    mgr.planner = None
    mgr.intent_provider = None
    mgr.marl_manager = None
    mgr.debug = False
    return mgr


class _FakeSpawnEvent:
    def __init__(self):
        self.blueprint = type("BP", (), {"id": "vehicle.test"})()
        loc = type("Loc", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
        self.transform = type("TF", (), {"location": loc})()
        self.destination = type("TF", (), {"location": loc})()
        self.target_speed = 30.0
        self.vehicle_id = "veh_test"
        self.route_id = "route_test"
        self.metadata = {}


def test_spawn_setup_failure_aborts_and_destroys_vehicle(monkeypatch):
    import opencda_marl._kernel.agent_manager as am

    world = FakeWorld()
    mgr = _bare_agent_manager(world)

    class ExplodingAdapter:
        def __init__(self, **kwargs):
            # A sensor got attached before the failure — must die with the
            # vehicle.
            world.attach_sensor(kwargs["vehicle"])
            raise RuntimeError("adapter build failed")

    monkeypatch.setattr(am, "MARLVehicleAdapter", ExplodingAdapter)
    with pytest.raises(RuntimeError, match="adapter build failed"):
        mgr._spawn_vehicles([_FakeSpawnEvent()])

    assert mgr.registry.tracked_vehicle_ids() == []
    assert mgr._vehicle_adapters == []
    # Nothing survives in the world: vehicle and its sensor both destroyed.
    assert all(not a.is_alive for a in world.actors)


def test_spawn_collision_requeues_event():
    world = FakeWorld()
    world.fail_spawn_with = "Spawn failed because of collision at spawn position"
    mgr = _bare_agent_manager(world)
    event = _FakeSpawnEvent()
    mgr._spawn_vehicles([event])
    assert list(mgr._failure_events) == [event]
    assert mgr.registry.tracked_vehicle_ids() == []


def test_spawn_retry_gives_up_after_bounded_attempts():
    """_spawn_vehicles re-feeds its own failure queue each step; a permanently
    blocked event must be dropped after MAX_SPAWN_RETRIES, not spin forever."""
    world = FakeWorld()
    world.fail_spawn_with = "Spawn failed because of collision at spawn position"
    mgr = _bare_agent_manager(world)
    mgr.MAX_SPAWN_RETRIES = 3
    event = _FakeSpawnEvent()
    mgr._spawn_vehicles([event])          # attempt 1 -> deferred
    assert list(mgr._failure_events) == [event]
    mgr._spawn_vehicles([])               # attempt 2 -> deferred
    assert list(mgr._failure_events) == [event]
    mgr._spawn_vehicles([])               # attempt 3 == limit -> dropped
    assert list(mgr._failure_events) == []
    assert mgr._spawn_retry_counts == {}


def test_occupied_spawn_point_is_deferred_without_a_server_call():
    from opencda_marl._kernel.adapter.world_snapshot import VehicleKinematics, WorldSnapshot

    world = FakeWorld()
    mgr = _bare_agent_manager(world)
    # A live vehicle sits 1 m from the spawn transform.
    blocker = VehicleKinematics(actor_id=5, x=1.0, y=0.0, z=0.0, vx=0.0, vy=0.0, yaw_rad=0.0)
    mgr._current_snapshot = WorldSnapshot([blocker])
    mgr._vehicle_adapters = [object()]  # non-empty so occupancy is consulted
    event = _FakeSpawnEvent()
    spawn_calls = {"n": 0}
    real_spawn = world.spawn_actor

    def counting_spawn(bp, tf):
        spawn_calls["n"] += 1
        return real_spawn(bp, tf)

    world.spawn_actor = counting_spawn
    mgr._spawn_vehicles([event])
    assert spawn_calls["n"] == 0            # no RPC spent on a blocked point
    assert list(mgr._failure_events) == [event]


# --------------------------------------------------------------------- #
# SensorRig
# --------------------------------------------------------------------- #
class _Wrapper:
    def __init__(self, sensor):
        self.sensor = sensor


def _fake_vm(world, vehicle):
    gnss = world.attach_sensor(vehicle)
    imu = world.attach_sensor(vehicle)
    collision = world.attach_sensor(vehicle)
    safety_imu = world.attach_sensor(vehicle)

    class Detector:  # pure-python detector without a CARLA actor
        pass

    vm = type("VM", (), {})()
    vm.localizer = type("Loc", (), {})()
    vm.localizer.gnss = _Wrapper(gnss)
    vm.localizer.imu = _Wrapper(imu)
    vm.safety_manager = type("Safety", (), {})()
    vm.safety_manager.imu_sensor = _Wrapper(safety_imu)
    vm.safety_manager.sensors = [_Wrapper(collision), Detector(),
                                 _Wrapper(safety_imu)]  # duplicate on purpose
    vm.perception_manager = type("Perception", (), {})()
    vm.perception_manager.rgb_camera = []
    vm.perception_manager.lidar = None
    return vm, [gnss, imu, collision, safety_imu]


def test_sensor_rig_collects_and_dedupes():
    world = FakeWorld()
    vehicle = FakeActor("vehicle.test")
    vm, sensors = _fake_vm(world, vehicle)
    rig = SensorRig.from_vehicle_manager(vm)
    assert sorted(rig.sensor_ids()) == sorted(s.id for s in sensors)
    assert len(rig) == 4  # duplicate safety IMU counted once


def test_sensor_rig_stop_precedes_destroy_and_isolates_failures():
    world = FakeWorld()
    vehicle = FakeActor("vehicle.test")
    vm, sensors = _fake_vm(world, vehicle)

    def boom():
        raise RuntimeError("stop failed")

    sensors[0].stop = boom  # first sensor's stop() fails
    rig = SensorRig.from_vehicle_manager(vm)
    failures = rig.stop_and_destroy()
    assert len(failures) == 1
    # Every sensor is destroyed regardless of the stop() failure.
    assert all(not s.is_alive for s in sensors)
    assert all(s.stopped for s in sensors[1:])


# --------------------------------------------------------------------- #
# Sensor release: the handle that listen()ed must be stopped and destroyed
# --------------------------------------------------------------------- #
class _ListeningSensor:
    """A sensor handle like CARLA's: ``is_alive`` is per handle, and dropping
    a handle that is still alive and listening makes the client warn."""

    _next_id = 9000

    def __init__(self):
        self.id = _ListeningSensor._next_id
        _ListeningSensor._next_id += 1
        self.is_alive = True
        self.calls = []

    def stop(self):
        self.calls.append("stop")

    def destroy(self):
        self.calls.append("destroy")
        self.is_alive = False
        return True


class _InertManager:
    """A vendored manager whose destroy() swallowed its own failure."""

    def destroy(self):
        pass


def _vm_with_sensors():
    sensors = [_ListeningSensor() for _ in range(3)]
    gnss, imu, collision = sensors
    vm = type("VM", (), {})()
    vm.localizer = _InertManager()
    vm.localizer.gnss = _Wrapper(gnss)
    vm.localizer.imu = _Wrapper(imu)
    vm.safety_manager = _InertManager()
    vm.safety_manager.imu_sensor = None
    vm.safety_manager.sensors = [_Wrapper(collision)]
    vm.perception_manager = _InertManager()
    vm.perception_manager.rgb_camera = []
    vm.perception_manager.lidar = None
    vm.destroy = lambda: None  # VehicleManager.destroy over inert managers
    return vm, sensors


def _assert_stopped_then_destroyed(sensors):
    for sensor in sensors:
        assert sensor.calls == ["stop", "destroy"]


def _adapter_cls_with(vm, fail_in_set_destination=False):
    from opencda_marl._kernel.adapter.vehicle_adapter import MARLVehicleAdapter

    class FakeAdapter(MARLVehicleAdapter):
        def __init__(self, **kwargs):
            self.vehicle = kwargs["vehicle"]
            self.actor_id = self.vehicle.id
            self.cav_world = type("CW", (), {
                "remove_vehicle_manager": lambda self, vm: None})()
            self.vm = vm

        def set_destination(self, *args, **kwargs):
            if fail_in_set_destination:
                raise RuntimeError("no route")

    return FakeAdapter


def _agent_manager_with_adapter(world, vm):
    mgr = _bare_agent_manager(world)
    mgr._dropped_spawns = 0
    mgr._event_logs = []
    mgr.cav_world = type("CW", (), {
        "remove_vehicle_manager": lambda self, vm: None})()
    vehicle = mgr.registry.spawn_vehicle(object(), object())
    mgr._vehicle_adapters.append(_adapter_cls_with(vm)(vehicle=vehicle))
    return mgr


def test_agent_manager_cleanup_stops_and_destroys_sensors_its_managers_left():
    world = FakeWorld()
    vm, sensors = _vm_with_sensors()
    mgr = _agent_manager_with_adapter(world, vm)

    mgr.cleanup()
    _assert_stopped_then_destroyed(sensors)


def test_episode_reset_stops_and_destroys_sensors():
    world = FakeWorld()
    vm, sensors = _vm_with_sensors()
    mgr = _agent_manager_with_adapter(world, vm)

    mgr.reset()
    _assert_stopped_then_destroyed(sensors)


def test_adapter_destroy_stops_and_destroys_sensors_its_managers_left():
    vm, sensors = _vm_with_sensors()
    vehicle = FakeActor("vehicle.test")
    adapter = _adapter_cls_with(vm)(vehicle=vehicle)

    adapter.destroy()
    _assert_stopped_then_destroyed(sensors)
    assert adapter.vm is None


def test_spawn_setup_failure_after_the_vm_exists_destroys_its_sensors(monkeypatch):
    import opencda_marl._kernel.agent_manager as am

    world = FakeWorld()
    mgr = _bare_agent_manager(world)
    vm, sensors = _vm_with_sensors()
    vm.update_info = lambda: None
    monkeypatch.setattr(am, "MARLVehicleAdapter",
                        _adapter_cls_with(vm, fail_in_set_destination=True))

    with pytest.raises(RuntimeError, match="no route"):
        mgr._spawn_vehicles([_FakeSpawnEvent()])
    _assert_stopped_then_destroyed(sensors)


def test_world_reload_cleanup_reaches_the_sensors():
    from types import SimpleNamespace
    from opencda_marl._kernel.lifecycle.world_manager import WorldManager

    world = FakeWorld()
    vm, sensors = _vm_with_sensors()
    mgr = _agent_manager_with_adapter(world, vm)
    coordinator = SimpleNamespace(
        carla_world=world,
        scenario_manager=SimpleNamespace(agent_manager=mgr, traffic_manager=None))

    WorldManager({}, coordinator)._cleanup_carla_resources()
    _assert_stopped_then_destroyed(sensors)
