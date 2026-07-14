import asyncio
import math
import threading

import pytest

from core.actuators.actuator_registry import get_actuator_registry
from core.adaptation.immune_executor import ImmuneHeuristicExecutor
from core.sensors.sensor_registry import get_sensor_registry


def test_sensor_registry_sync_and_read():
    registry = get_sensor_registry()
    registry.sync_from_world_model()

    data = registry.read_all()
    assert "port_east_load" in data
    assert "port_west_load" in data
    assert "warehouse_load" in data
    assert "system_cpu_usage" in data

    reliability = registry.get_reliability_vector()
    assert reliability["port_east_load"] == 1.0
    assert reliability["system_cpu_usage"] == 1.0


def test_sensor_registry_rejects_non_finite_readings():
    registry = get_sensor_registry()
    before = registry.read_all()["port_east_load"]

    assert registry.record_reading("port_east_load", math.nan) is False
    assert registry.read_all()["port_east_load"] == before


def test_actuator_registry_actions():
    registry = get_actuator_registry()

    # Test valid RerouteVesselActuator call
    res = registry.execute_action(
        "reroute_vessel", {"vessel_id": "Vessel_Alpha", "heading": 120.0, "speed": 18.0}
    )
    assert res.success is True
    assert "Vessel_Alpha" in res.message

    # Test invalid actuator name
    res_invalid = registry.execute_action("invalid_actuator", {})
    assert res_invalid.success is False

    # Test invalid parameters (speed exceeding max)
    res_speed = registry.execute_action(
        "reroute_vessel", {"vessel_id": "Vessel_Alpha", "heading": 90.0, "speed": 100.0}
    )
    assert res_speed.success is False

    res_nan = registry.execute_action(
        "reroute_vessel", {"vessel_id": "Vessel_Alpha", "heading": math.nan, "speed": 10.0}
    )
    assert res_nan.success is False


def test_actuator_registry_forwards_scoped_authority_context(monkeypatch):
    from types import SimpleNamespace

    from core.actuators.actuator_registry import ActuatorRegistry, ActuatorResult, BaseActuator

    captured = {}

    class FakeGateway:
        async def authorize_tool_execution(self, name, params, **kwargs):
            captured["name"] = name
            captured["params"] = params
            captured["kwargs"] = kwargs
            return SimpleNamespace(approved=True, reason="approved", capability_token_id="cap-test")

        def verify_tool_access(self, name, capability_token_id):
            captured["verified"] = (name, capability_token_id)
            return True

        def finalize_tool_execution(self, **kwargs):
            captured["finalized"] = kwargs
            return {"standing_authority_closed": True}

    class NeedsAuthorityActuator(BaseActuator):
        requires_authority = True

        @property
        def name(self):
            return "needs_authority"

        @property
        def description(self):
            return "Test actuator requiring AuthorityGateway."

        def validate_params(self, params):
            return True

        def execute(self, params):
            return ActuatorResult(True, "ok", {"ran": True})

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: FakeGateway(),
    )
    registry = ActuatorRegistry()
    registry.register(NeedsAuthorityActuator())

    result = registry.execute_action(
        "needs_authority",
        {"value": 1},
        context={
            "source": "overt_action_loop",
            "priority": 0.45,
            "requested_authority_scope": "overt_action_loop:abc123:needs_authority",
            "authorization": "governed_autonomous_overt_action",
        },
    )

    assert result.success is True
    assert captured["verified"] == ("needs_authority", "cap-test")
    assert captured["kwargs"]["source"] == "overt_action_loop"
    assert captured["kwargs"]["priority"] == 0.45
    assert captured["kwargs"]["context"]["requested_authority_scope"] == (
        "overt_action_loop:abc123:needs_authority"
    )
    assert captured["kwargs"]["context"]["authorization"] == "governed_autonomous_overt_action"
    assert captured["finalized"]["capability_token_id"] == "cap-test"
    assert captured["finalized"]["success"] is True


def test_actuator_registry_finalizes_authority_when_token_verification_fails(monkeypatch):
    from types import SimpleNamespace

    from core.actuators.actuator_registry import ActuatorRegistry, ActuatorResult, BaseActuator

    captured = {}

    class FakeGateway:
        async def authorize_tool_execution(self, *_args, **_kwargs):
            return SimpleNamespace(
                approved=True,
                executive_intent_id="intent-test",
                capability_token_id="cap-test",
                standing_authority_token="standing-test",
            )

        def verify_tool_access(self, *_args):
            return False

        def finalize_tool_execution(self, **kwargs):
            captured.update(kwargs)
            return {"standing_authority_closed": True}

    class NeedsAuthorityActuator(BaseActuator):
        requires_authority = True

        @property
        def name(self):
            return "needs_authority"

        @property
        def description(self):
            return "Test actuator requiring AuthorityGateway."

        def validate_params(self, params):
            return True

        def execute(self, params):
            return ActuatorResult(True, "must not run", {})

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: FakeGateway(),
    )
    registry = ActuatorRegistry()
    registry.register(NeedsAuthorityActuator())

    result = registry.execute_action("needs_authority", {"value": 1})

    assert result.success is False
    assert captured == {
        "executive_intent_id": "intent-test",
        "capability_token_id": "cap-test",
        "standing_authority_token": "standing-test",
        "success": False,
        "result": {"success": False},
    }


@pytest.mark.asyncio
async def test_async_actuator_keeps_authority_lifecycle_on_owner_thread(monkeypatch):
    from types import SimpleNamespace

    from core.actuators.actuator_registry import ActuatorRegistry, ActuatorResult, BaseActuator

    owner_thread = threading.get_ident()
    observed = {}

    class FakeGateway:
        async def authorize_tool_execution(self, *_args, **_kwargs):
            observed["authorize_thread"] = threading.get_ident()
            return SimpleNamespace(
                approved=True,
                reason="approved",
                executive_intent_id="intent-async",
                capability_token_id="cap-async",
                standing_authority_token="standing-async",
            )

        def verify_tool_access(self, *_args):
            observed["verify_thread"] = threading.get_ident()
            return True

        def finalize_tool_execution(self, **kwargs):
            observed["finalize_thread"] = threading.get_ident()
            observed["closure"] = kwargs
            return {"closed": True, "standing_authority_closed": True}

    class BlockingActuator(BaseActuator):
        requires_authority = True

        @property
        def name(self):
            return "blocking_authority_probe"

        @property
        def description(self):
            return "Records execution-thread ownership."

        def validate_params(self, params):
            return True

        def execute(self, params):
            observed["execute_thread"] = threading.get_ident()
            return ActuatorResult(True, "ok", {"ran": True})

    gateway = FakeGateway()
    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: gateway,
    )
    registry = ActuatorRegistry()
    registry.register(BlockingActuator())

    result = await registry.execute_action_async(
        "blocking_authority_probe",
        {"value": 1},
        context={"source": "overt_action_loop"},
    )

    assert result.success is True
    assert observed["authorize_thread"] == owner_thread
    assert observed["verify_thread"] == owner_thread
    assert observed["finalize_thread"] == owner_thread
    assert observed["execute_thread"] != owner_thread
    assert observed["closure"]["standing_authority_token"] == "standing-async"


@pytest.mark.asyncio
async def test_sync_actuator_bridge_rejects_active_event_loop():
    from core.actuators.actuator_registry import ActuatorRegistry

    registry = ActuatorRegistry()
    with pytest.raises(RuntimeError, match="await execute_action_async"):
        registry.execute_action(
            "reroute_vessel",
            {"vessel_id": "Vessel_Alpha", "heading": 90.0, "speed": 10.0},
        )


@pytest.mark.asyncio
async def test_blocking_actuator_does_not_stall_event_loop():
    from core.actuators.actuator_registry import ActuatorRegistry, ActuatorResult, BaseActuator

    release = threading.Event()

    class BlockingActuator(BaseActuator):
        @property
        def name(self):
            return "event_loop_probe"

        @property
        def description(self):
            return "Waits for an event-loop callback from a worker thread."

        def validate_params(self, params):
            return True

        def execute(self, params):
            released = release.wait(timeout=0.5)
            return ActuatorResult(released, "released" if released else "timed out", {})

    registry = ActuatorRegistry()
    registry.register(BlockingActuator())
    loop = asyncio.get_running_loop()
    loop.call_later(0.02, release.set)
    started = loop.time()

    result = await registry.execute_action_async("event_loop_probe", {})

    assert result.success is True
    assert loop.time() - started < 0.3


@pytest.mark.asyncio
async def test_cancelled_actuator_closes_authority_after_real_completion(monkeypatch):
    from types import SimpleNamespace

    from core.actuators.actuator_registry import ActuatorRegistry, ActuatorResult, BaseActuator

    started = threading.Event()
    release = threading.Event()
    observed = {}

    class FakeGateway:
        async def authorize_tool_execution(self, *_args, **_kwargs):
            return SimpleNamespace(
                approved=True,
                reason="approved",
                executive_intent_id="intent-cancel",
                capability_token_id="cap-cancel",
                standing_authority_token="standing-cancel",
            )

        def verify_tool_access(self, *_args):
            return True

        def finalize_tool_execution(self, **kwargs):
            observed["closure"] = kwargs
            observed["body_finished_before_closure"] = release.is_set()
            return {"closed": True}

    class BlockingActuator(BaseActuator):
        requires_authority = True

        @property
        def name(self):
            return "cancel_completion_probe"

        @property
        def description(self):
            return "Waits until the test releases its non-cancellable worker."

        def validate_params(self, params):
            return True

        def execute(self, params):
            started.set()
            release.wait(timeout=1.0)
            return ActuatorResult(True, "completed", {"completed": True})

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: FakeGateway(),
    )
    registry = ActuatorRegistry()
    registry.register(BlockingActuator())

    execution = asyncio.create_task(
        registry.execute_action_async("cancel_completion_probe", {})
    )
    assert await asyncio.to_thread(started.wait, 0.5)
    execution.cancel()
    await asyncio.sleep(0)
    assert "closure" not in observed
    execution.cancel()
    await asyncio.sleep(0)
    assert "closure" not in observed

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await execution

    assert observed["body_finished_before_closure"] is True
    assert observed["closure"]["success"] is True
    assert observed["closure"]["standing_authority_token"] == "standing-cancel"


@pytest.mark.asyncio
async def test_actuator_refuses_new_execution_after_shutdown_request():
    from core.actuators.actuator_registry import ActuatorRegistry, ActuatorResult, BaseActuator
    from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

    calls = 0

    class ProbeActuator(BaseActuator):
        @property
        def name(self):
            return "shutdown-refusal-probe"

        @property
        def description(self):
            return "Must not execute after the runtime shutdown latch."

        def validate_params(self, params):
            return True

        def execute(self, params):
            nonlocal calls
            calls += 1
            return ActuatorResult(True, "unexpected", {})

    registry = ActuatorRegistry()
    registry.register(ProbeActuator())
    request_shutdown("actuator-refusal-unit")
    try:
        result = await registry.execute_action_async("shutdown-refusal-probe", {})
    finally:
        clear_shutdown_request()

    assert result.success is False
    assert "runtime shutdown" in result.message
    assert calls == 0


@pytest.mark.asyncio
async def test_actuator_admitted_before_shutdown_drains_worker_to_completion():
    from core.actuators.actuator_registry import ActuatorRegistry, ActuatorResult, BaseActuator
    from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

    started = threading.Event()
    release = threading.Event()

    class ProbeActuator(BaseActuator):
        @property
        def name(self):
            return "shutdown-crossing-probe"

        @property
        def description(self):
            return "Finishes work admitted immediately before shutdown."

        def validate_params(self, params):
            return True

        def execute(self, params):
            started.set()
            release.wait(timeout=1.0)
            return ActuatorResult(True, "completed", {"completed": True})

    registry = ActuatorRegistry()
    registry.register(ProbeActuator())
    execution = asyncio.create_task(
        registry.execute_action_async("shutdown-crossing-probe", {})
    )
    assert await asyncio.to_thread(started.wait, 0.5)
    request_shutdown("actuator-crossing-unit")
    release.set()
    try:
        result = await execution
    finally:
        clear_shutdown_request()

    assert result.success is True
    assert result.updates == {"completed": True}


@pytest.mark.asyncio
async def test_shutdown_during_authorization_closes_lease_without_executing(monkeypatch):
    from types import SimpleNamespace

    from core.actuators.actuator_registry import ActuatorRegistry, BaseActuator
    from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

    authorization_started = asyncio.Event()
    release_authorization = asyncio.Event()
    observed: dict[str, object] = {}

    class FakeGateway:
        async def authorize_tool_execution(self, *_args, **_kwargs):
            authorization_started.set()
            await release_authorization.wait()
            return SimpleNamespace(
                approved=True,
                reason="approved",
                executive_intent_id="intent-shutdown",
                capability_token_id="cap-shutdown",
                standing_authority_token="standing-shutdown",
            )

        def verify_tool_access(self, *_args):
            return True

        def finalize_tool_execution(self, **kwargs):
            observed["closure"] = kwargs
            return {"closed": True}

    class ProbeActuator(BaseActuator):
        requires_authority = True

        @property
        def name(self):
            return "authorization-shutdown-probe"

        @property
        def description(self):
            return "Does not cross a shutdown that begins during authorization."

        def validate_params(self, params):
            return True

        def execute(self, params):
            raise AssertionError("actuator body must not execute")

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: FakeGateway(),
    )
    registry = ActuatorRegistry()
    registry.register(ProbeActuator())
    execution = asyncio.create_task(
        registry.execute_action_async("authorization-shutdown-probe", {})
    )
    await authorization_started.wait()
    request_shutdown("actuator-authorization-unit")
    release_authorization.set()
    try:
        result = await execution
    finally:
        clear_shutdown_request()

    assert result.success is False
    assert "shutdown began during authorization" in result.message
    assert observed["closure"]["success"] is False
    assert observed["closure"]["standing_authority_token"] == "standing-shutdown"


def test_immune_executor_uses_safe_arithmetic_resolver():
    executor = ImmuneHeuristicExecutor()
    sensors = {"port_east_load": 800.0}

    resolved = executor.resolve_params({"amount": "$port_east_load * 0.25"}, sensors)
    assert resolved["amount"] == 200.0

    blocked = executor.resolve_params({"amount": "$port_east_load / 0"}, sensors)
    assert blocked["amount"] == "$port_east_load / 0"


@pytest.mark.asyncio
async def test_async_immune_authority_decision_stays_on_owner_thread(monkeypatch):
    from types import SimpleNamespace

    owner_thread = threading.get_ident()
    observed = {}
    executor = ImmuneHeuristicExecutor()
    monkeypatch.setattr(
        executor,
        "_authorization_preflight",
        lambda _context: (None, "authority_required", "authority required"),
    )

    class FakeGateway:
        async def authorize_state_mutation(self, *_args, **_kwargs):
            observed["authority_thread"] = threading.get_ident()
            return SimpleNamespace(approved=True)

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: FakeGateway(),
    )

    authorized, status, _message = await executor._authorize_execution_async(
        {"source": "adaptive_immune_system"}
    )

    assert authorized is True
    assert status == "authorized"
    assert observed["authority_thread"] == owner_thread


def test_immune_executor_requires_authorized_context_before_actuation():
    from core.world.world_model import get_physics_world_model

    model = get_physics_world_model()
    east_before = model.get_entity("Port_East").load
    west_before = model.get_entity("Port_West").load

    executor = ImmuneHeuristicExecutor()
    result = executor.execute_rule(
        {
            "conditions": [{"sensor": "port_east_load", "operator": ">", "value": 1.0}],
            "actions": [
                {
                    "actuator": "reallocate_flow",
                    "params": {
                        "source_id": "Port_East",
                        "target_id": "Port_West",
                        "amount": 100.0,
                    },
                }
            ],
        }
    )

    assert result["success"] is False
    assert result["status"] == "governance_denied"
    assert result["actions_executed"] == []
    assert model.get_entity("Port_East").load == east_before
    assert model.get_entity("Port_West").load == west_before


def test_immune_executor_defers_maintenance_rule_during_foreground_turn(monkeypatch):
    from core.runtime import foreground_guard

    # Pin the maintenance environment so the higher-precedence desktop-safe-boot policy can't
    # mask the foreground reason this test asserts. core/architect/safe_boot_harness.py does
    # os.environ.setdefault("AURA_SAFE_BOOT_DESKTOP", "1") at import time, which leaks into a
    # broad test run; this makes the foreground-deferral path deterministic regardless.
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "0")
    monkeypatch.setenv("AURA_ENABLE_BACKGROUND_COGNITION", "1")

    foreground_guard._reset_for_tests()
    lease = foreground_guard.begin_foreground_turn(owner="test", source="chat_api")
    try:
        result = ImmuneHeuristicExecutor().execute_rule(
            {
                "conditions": [{"sensor": "port_east_load", "operator": ">", "value": 1.0}],
                "actions": [
                    {
                        "actuator": "reallocate_flow",
                        "params": {
                            "source_id": "Port_East",
                            "target_id": "Port_West",
                            "amount": 100.0,
                        },
                    }
                ],
            },
            context={"source": "adaptive_immune_system", "priority": 0.8},
        )
    finally:
        lease.close()
        foreground_guard._reset_for_tests()

    assert result["success"] is False
    assert result["status"] == "deferred"
    assert result["deferred"] is True
    assert "foreground_chat_active" in result["message"]
