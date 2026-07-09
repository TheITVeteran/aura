import math

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
            "scoped_authority": "overt_action_loop:abc123:needs_authority",
            "authorization": "governed_autonomous_overt_action",
        },
    )

    assert result.success is True
    assert captured["verified"] == ("needs_authority", "cap-test")
    assert captured["kwargs"]["source"] == "overt_action_loop"
    assert captured["kwargs"]["priority"] == 0.45
    assert captured["kwargs"]["context"]["scoped_authority"] == "overt_action_loop:abc123:needs_authority"
    assert captured["kwargs"]["context"]["authorization"] == "governed_autonomous_overt_action"


def test_immune_executor_uses_safe_arithmetic_resolver():
    executor = ImmuneHeuristicExecutor()
    sensors = {"port_east_load": 800.0}

    resolved = executor.resolve_params({"amount": "$port_east_load * 0.25"}, sensors)
    assert resolved["amount"] == 200.0

    blocked = executor.resolve_params({"amount": "$port_east_load / 0"}, sensors)
    assert blocked["amount"] == "$port_east_load / 0"


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
