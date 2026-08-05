from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest


def _clear_hass_environment(monkeypatch) -> None:
    for name in (
        "AURA_HASS_URL",
        "AURA_HASS_TOKEN",
        "AURA_HASS_ALLOW_HTTP",
        "HASS_URL",
        "HASS_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_world_bridge_handler_runs_under_will_governance_and_consumes_token(
    monkeypatch,
    tmp_path,
) -> None:
    from core.agency import capability_token as token_module
    from core.embodiment import world_bridge as world_module
    from core.ethics import conscience as conscience_module
    from core.governance.will import WillDecision, WillOutcome
    from core.governance_context import require_governance
    from core.runtime import action_executor as executor_module
    from core.runtime.post_action_receipt import PostActionReceiptStore

    permission = world_module.Permission(
        channel=world_module.Channel.ENVIRONMENTAL_CHANGE.value,
        granted=True,
    )
    monkeypatch.setattr(
        world_module,
        "_PERMS",
        SimpleNamespace(status=lambda _channel: permission),
    )
    monkeypatch.setattr(world_module, "get_runtime_setting", lambda *_args: "standard")
    monkeypatch.setattr(
        conscience_module,
        "get_conscience",
        lambda: SimpleNamespace(
            evaluate=lambda **_kwargs: SimpleNamespace(
                verdict=conscience_module.Verdict.APPROVE,
                rule_id="",
            )
        ),
    )
    class Will:
        def __init__(self) -> None:
            self.outcomes: list[tuple[str, object]] = []

        def decide(self, **kwargs):
            return WillDecision(
                receipt_id="will-iot-1",
                outcome=WillOutcome.PROCEED,
                domain=kwargs["domain"],
                reason="test_approved",
                source=str(kwargs.get("source") or "world_bridge_test"),
            )

        def record_outcome(self, receipt_id, outcome) -> None:
            self.outcomes.append((receipt_id, outcome))

    will = Will()
    monkeypatch.setattr(executor_module, "get_will", lambda: will)
    monkeypatch.setattr(
        executor_module.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor_module.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    receipt_store = PostActionReceiptStore(tmp_path / "world_action_receipts.jsonl")
    monkeypatch.setattr(
        executor_module,
        "get_post_action_receipt_store",
        lambda: receipt_store,
    )
    token_store = token_module.CapabilityTokenStore()
    monkeypatch.setattr(token_module, "get_token_store", lambda: token_store)

    observed: dict[str, str] = {}

    async def _handler(payload, *, capability_token: str):
        governance = require_governance(
            "test.environmental_change",
            strict=True,
            allowed_domains=("environment_action",),
        )
        observed.update(
            {
                "receipt": governance.receipt_id,
                "domain": governance.domain,
                "capability_token": capability_token,
                "value": str(payload.get("value")),
            }
        )
        return {"applied": True, "effect_verified": True}

    bridge = world_module.WorldBridge()
    bridge.register(world_module.Channel.ENVIRONMENTAL_CHANGE, _handler)

    result = await bridge.call(
        world_module.Channel.ENVIRONMENTAL_CHANGE,
        action="iot:light.office:turn_on",
        intent="test governed effect",
        payload={"value": 1},
    )

    assert result.ok is True
    assert observed["receipt"] == "will-iot-1"
    assert observed["domain"] == "environment_action"
    assert observed["value"] == "1"
    assert result.receipt_id.startswith("post-")
    issued = token_store.get(observed["capability_token"])
    assert issued is not None and issued.is_consumed()
    assert issued.domain == "environment_action"
    assert issued.parent_receipt == "will-iot-1"
    assert issued.side_effects == ["iot:light.office:turn_on"]
    assert receipt_store.get_receipt(result.receipt_id) is not None
    assert len(will.outcomes) == 1

    async def _accepted_but_unverified(_payload, *, capability_token: str):
        assert capability_token
        return {
            "transport_succeeded": True,
            "effect_verified": False,
            "observed_state": {"state": "off"},
        }

    bridge.register(
        world_module.Channel.ENVIRONMENTAL_CHANGE,
        _accepted_but_unverified,
    )
    unverified = await bridge.call(
        world_module.Channel.ENVIRONMENTAL_CHANGE,
        action="iot:light.office:turn_off",
        intent="test unverified effect",
        payload={"value": 0},
    )

    assert unverified.ok is False
    assert unverified.transport_succeeded is True
    assert unverified.effect_verified is False
    assert unverified.manual_reconciliation_required is True
    assert unverified.error.startswith("world_effect_unverified")
    assert receipt_store.get_receipt(unverified.receipt_id) is not None

    original_consume = token_store.consume

    def _fail_consume(*_args, **_kwargs):
        raise RuntimeError("receipt store unavailable")

    monkeypatch.setattr(token_store, "consume", _fail_consume)
    bridge.register(world_module.Channel.ENVIRONMENTAL_CHANGE, _handler)
    receipt_failure = await bridge.call(
        world_module.Channel.ENVIRONMENTAL_CHANGE,
        action="iot:light.office:set_brightness",
        intent="test token receipt closure",
        payload={"value": 120},
    )
    monkeypatch.setattr(token_store, "consume", original_consume)

    assert receipt_failure.ok is False
    assert receipt_failure.transport_succeeded is True
    assert receipt_failure.effect_verified is False
    assert receipt_failure.manual_reconciliation_required is True
    assert "capability_receipt_completion_failed" in receipt_failure.error
    assert receipt_store.get_receipt(receipt_failure.receipt_id) is not None
    assert len(will.outcomes) == 3


def test_iot_bridge_without_credentials_has_no_fake_noop_transport(
    monkeypatch,
    caplog,
) -> None:
    from core.embodiment.iot_bridge import IoTBridge

    _clear_hass_environment(monkeypatch)
    caplog.set_level(logging.INFO, logger="Aura.IoTBridge")

    bridge = IoTBridge()

    assert bridge.get_status() == {
        "running": False,
        "configured": False,
        "transports": [],
        "policy_rules": 3,
        "reality_reach_bound": False,
        "reality_adapter_count": 0,
        "observation_router_bound": False,
        "attachment_broker_bound": False,
        "home_assistant_connector": False,
    }
    assert "Home Assistant IoT transport disabled" in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_home_assistant_transport_rejects_implicit_plaintext_and_maps_percent_brightness(
    monkeypatch,
) -> None:
    from core.embodiment import iot_bridge as iot_module
    from core.embodiment.home_assistant_reality import state_matches_effect

    _clear_hass_environment(monkeypatch)
    monkeypatch.setenv("AURA_HASS_URL", "http://hass.example.test:8123")
    monkeypatch.setenv("AURA_HASS_TOKEN", "secret-token")
    with pytest.raises(RuntimeError, match="explicit_opt_in"):
        iot_module.HassTransport()

    effect = iot_module.IoTEffect(
        target="light.office",
        op="turn_on",
        payload={"brightness_pct": 80},
    )
    assert state_matches_effect(
        {
            "entity_id": "light.office",
            "state": "on",
            "attributes": {"brightness": 204},
        },
        effect,
    )


@pytest.mark.asyncio
async def test_home_assistant_transport_uses_canonical_network_gateway(
    monkeypatch,
) -> None:
    from core.embodiment import home_assistant_reality as hass_module
    from core.embodiment import iot_bridge as iot_module

    _clear_hass_environment(monkeypatch)
    monkeypatch.setenv("AURA_HASS_URL", "https://hass.example.test:8123")
    monkeypatch.setenv("AURA_HASS_TOKEN", "secret-token")
    calls: list[dict] = []

    async def _network_request(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "status_code": 200,
            "content": (
                b'{"entity_id":"light.office","state":"on","attributes":'
                b'{"brightness":120}}'
            ),
        }

    monkeypatch.setattr(
        hass_module.ActionExecutor,
        "request_network_transport",
        staticmethod(_network_request),
    )
    transport = iot_module.HassTransport()

    result = await transport.read_state("light.office")

    assert result["state"] == "on"
    request = calls[0]
    assert request["method"] == "GET"
    assert request["url"] == "https://hass.example.test:8123/api/states/light.office"
    assert request["source"] == "world_bridge:iot.home_assistant.readback"
    assert request["read_only"] is True
    assert request["timeout_s"] == 8.0

    with pytest.raises(RuntimeError, match="requires_reality_reach_transaction"):
        await transport.apply(
            iot_module.IoTEffect(
                target="light.office",
                op="turn_on",
                payload={"brightness": 120},
                reason="test",
            )
        )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_action_executor_network_transport_requires_world_governance(
    monkeypatch,
) -> None:
    from core import governance_context as governance_module
    from core.governance_context import (
        GovernanceViolation,
        LocalGovernanceDecision,
        governed_scope,
    )
    from core.runtime import action_executor as executor_module

    calls: list[dict] = []

    class Gateway:
        async def request_async(self, **kwargs):
            calls.append(kwargs)
            return {"ok": True, "status_code": 200, "content": b"{}"}

    monkeypatch.setattr(executor_module, "get_network_gateway", lambda: Gateway())
    decision = LocalGovernanceDecision(
        receipt_id="world-network-test",
        domain="environment_action",
        source="test",
    )
    async with governed_scope(decision):
        result = await executor_module.ActionExecutor.request_network_transport(
            method="GET",
            url="https://hass.example.test/api/states/light.office",
            timeout_s=2.0,
            source="world_bridge:test",
            read_only=True,
        )

    assert result["ok"] is True
    assert calls[0]["source"] == "world_bridge:test"
    monkeypatch.setattr(governance_module, "governance_runtime_active", lambda: True)
    with pytest.raises(GovernanceViolation):
        await executor_module.ActionExecutor.request_network_transport(
            method="GET",
            url="https://hass.example.test/api/states/light.office",
            timeout_s=2.0,
            source="world_bridge:test",
            read_only=True,
        )
    with pytest.raises(ValueError, match="owned by world_bridge"):
        await executor_module.ActionExecutor.request_network_transport(
            method="GET",
            url="https://hass.example.test/api/states/light.office",
            timeout_s=2.0,
            source="unowned:test",
            read_only=True,
        )


@pytest.mark.asyncio
async def test_home_assistant_discovery_is_read_only(
    monkeypatch,
) -> None:
    from core.embodiment import home_assistant_reality as hass_module
    from core.embodiment import iot_bridge as iot_module

    _clear_hass_environment(monkeypatch)
    monkeypatch.setenv("AURA_HASS_URL", "https://hass.example.test:8123")
    monkeypatch.setenv("AURA_HASS_TOKEN", "secret-token")

    async def _network_request(**kwargs):
        return {
            "ok": True,
            "status_code": 200,
            "content": b'[{"entity_id":"light.office","state":"on"}]',
        }

    calls: list[dict] = []

    async def _recording_network_request(**kwargs):
        calls.append(kwargs)
        return await _network_request(**kwargs)

    monkeypatch.setattr(
        hass_module.ActionExecutor,
        "request_network_transport",
        staticmethod(_recording_network_request),
    )
    transport = iot_module.HassTransport()

    result = await transport.discover()

    assert result[0]["entity_id"] == "light.office"
    assert calls == [
        {
            "method": "GET",
            "url": "https://hass.example.test:8123/api/states",
            "headers": {"Authorization": "Bearer secret-token"},
            "timeout_s": 8.0,
            "source": "world_bridge:iot.home_assistant.discover",
            "read_only": True,
        }
    ]


@pytest.mark.asyncio
async def test_environmental_handler_fails_when_no_physical_transport_exists(
    monkeypatch,
) -> None:
    from core.embodiment import iot_bridge as iot_module

    _clear_hass_environment(monkeypatch)
    bridge = iot_module.IoTBridge()
    monkeypatch.setattr(iot_module, "_BRIDGE", bridge)

    with pytest.raises(RuntimeError, match="iot_transport_unavailable"):
        await iot_module._environmental_change_handler(
            {
                "operation": "apply",
                "target": "light.office",
                "op": "turn_on",
                "effect": {"brightness": 100},
            },
            capability_token="CT-test",
        )


@pytest.mark.asyncio
async def test_environmental_handler_compiles_registered_target_after_authority(
    monkeypatch,
) -> None:
    from core.embodiment import iot_bridge as iot_module

    compiled = SimpleNamespace(
        sha256="sha256:" + "c" * 64,
    )

    class Adapter:
        adapter_id = "mqtt.office.temperature.adapter"

        async def compile_target(self, target, **kwargs):
            assert target == 22.5
            assert kwargs == {
                "inventory_sha256": "sha256:" + "a" * 64,
                "deadline_s": 12.0,
                "idempotency_key": "demo.target.22-5",
                "source": "embodiment_skill",
            }
            return compiled

    class Reality:
        @staticmethod
        def register_adapter(_adapter):
            return None

        @staticmethod
        def actuator_adapter(channel_id):
            assert channel_id == "mqtt.office.temperature.command"
            return Adapter()

        @staticmethod
        def status():
            return {"registry_sha256": "sha256:" + "a" * 64}

    class Coordinator:
        @staticmethod
        async def execute(command):
            assert command is compiled
            return {
                "transport_succeeded": True,
                "effect_verified": True,
                "reality_reach_transaction": {"transaction_id": "tx-1"},
            }

    bridge = iot_module.IoTBridge()
    bridge.bind_reality_reach(Reality(), Coordinator())
    monkeypatch.setattr(iot_module, "_BRIDGE", bridge)

    result = await iot_module._environmental_change_handler(
        {
            "operation": "reality_target",
            "channel_id": "MQTT.OFFICE.TEMPERATURE.COMMAND",
            "target_value": 22.5,
            "timeout_s": 12,
            "idempotency_key": "demo.target.22-5",
            "source": "embodiment_skill",
        },
        capability_token="CT-test",
    )

    assert result["transport_succeeded"] is True
    assert result["effect_verified"] is True
    assert result["channel_id"] == "mqtt.office.temperature.command"
    assert result["target_value"] == 22.5
    assert result["effects"][0]["adapter_id"] == (
        "mqtt.office.temperature.adapter"
    )


@pytest.mark.asyncio
async def test_generic_reality_target_requires_world_bridge_capability_token() -> None:
    from core.embodiment.iot_bridge import IoTBridge

    with pytest.raises(PermissionError, match="iot_capability_token_required"):
        await IoTBridge().apply_target_authorized(
            "mqtt.office.temperature.command",
            22.5,
            capability_token="",
        )


@pytest.mark.asyncio
async def test_legacy_affect_actuator_delegates_to_world_bridge(monkeypatch) -> None:
    from core.autonomic import iot_bridge as legacy_module

    monkeypatch.setenv("AURA_HASS_TOKEN", "test-token")
    calls: list[tuple[tuple, dict]] = []

    class Bridge:
        async def call(self, *args, **kwargs):
            calls.append((args, kwargs))
            return legacy_module.WorldActionResult(
                channel="environmental_change",
                ok=False,
                receipt_id="",
                error="permission_denied",
            )

    monkeypatch.setattr(legacy_module, "get_world_bridge", lambda: Bridge())
    actuator = legacy_module.PhysicalActuator("https://hass.example.test:8123")

    result = await actuator.broadcast_affect_state({"P": 2.0, "A": 2.0})

    assert result["ok"] is False
    assert result["error"] == "permission_denied"
    assert calls[0][0] == (legacy_module.Channel.ENVIRONMENTAL_CHANGE,)
    payload = calls[0][1]["payload"]
    assert payload["effect"] == {"brightness": 255, "color_temp": 250}
    assert calls[0][1]["action"] == "iot:light.office_ambient:turn_on"
