from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from core.embodiment import home_assistant_reality as hass_module
from core.embodiment.home_assistant_connector import HomeAssistantConnector
from core.embodiment.home_assistant_reality import (
    HomeAssistantEffect,
    HomeAssistantRealityAdapter,
    HomeAssistantRealityError,
    HomeAssistantTransport,
)
from core.embodiment.iot_bridge import IoTBridge
from core.reality_reach.actuation import ActuationLease
from core.reality_reach.attachments import AttachmentAccess
from core.reality_reach.live import RealityReachService
from core.reality_reach.transactions import RealityActuationCoordinator


class HomeAssistantNetwork:
    def __init__(
        self,
        state: dict[str, Any],
        *,
        apply_updates: bool = True,
    ) -> None:
        self.state = dict(state)
        self.state["attributes"] = dict(state.get("attributes") or {})
        self.apply_updates = apply_updates
        self.calls: list[dict[str, Any]] = []

    async def request(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        method = str(kwargs["method"])
        url = str(kwargs["url"])
        if method == "GET" and url.endswith("/api/states"):
            return {
                "ok": True,
                "status_code": 200,
                "content": json.dumps([self.state]).encode(),
            }
        if method == "GET" and "/api/states/" in url:
            return {
                "ok": True,
                "status_code": 200,
                "content": json.dumps(self.state).encode(),
            }
        if method != "POST" or "/api/services/" not in url:
            raise AssertionError(f"unexpected Home Assistant request: {method} {url}")
        payload = json.loads(str(kwargs.get("data") or "{}"))
        operation = url.rsplit("/", 1)[-1]
        if self.apply_updates:
            if operation == "turn_on":
                self.state["state"] = "on"
            elif operation == "turn_off":
                self.state["state"] = "off"
            attributes = self.state["attributes"]
            for key, value in payload.items():
                if key == "entity_id" or key == "transition":
                    continue
                if key == "brightness_pct":
                    attributes["brightness"] = round(float(value) * 255.0 / 100.0)
                else:
                    attributes[key] = value
        return {"ok": True, "status_code": 200, "content": b"[]"}

    @property
    def post_calls(self) -> list[dict[str, Any]]:
        return [call for call in self.calls if call.get("method") == "POST"]


@pytest.fixture
def hass_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURA_HASS_URL", "https://hass.example.test:8123")
    monkeypatch.setenv("AURA_HASS_TOKEN", "secret-token")
    monkeypatch.delenv("AURA_HASS_ALLOW_HTTP", raising=False)
    monkeypatch.delenv("HASS_URL", raising=False)
    monkeypatch.delenv("HASS_TOKEN", raising=False)


async def _executor(**kwargs: Any) -> dict[str, Any]:
    try:
        dispatched = dict(
            await kwargs["effect_handler"]({"will_receipt_id": "test.will.hass"})
        )
    except Exception as exc:  # noqa: BLE001 - emulate ActionExecutor containment
        return {
            "ok": False,
            "effect_verified": False,
            "error": f"effect_handler_failed:{type(exc).__name__}:{exc}",
        }
    verified = dict(await kwargs["effect_verifier"]({"result": dispatched}))
    return {**dispatched, **verified, "ok": verified.get("effect_verified") is True}


def _install_network(
    monkeypatch: pytest.MonkeyPatch,
    network: HomeAssistantNetwork,
) -> None:
    monkeypatch.setattr(
        hass_module.ActionExecutor,
        "request_network_transport",
        staticmethod(network.request),
    )


@pytest.mark.asyncio
async def test_actuator_readback_preserves_home_assistant_event_lineage(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del hass_environment
    state = {
        "entity_id": "light.office",
        "state": "off",
        "attributes": {"brightness": 10},
        "last_updated": "2026-08-02T08:00:00Z",
        "context": {"id": "shared-context"},
    }
    network = HomeAssistantNetwork(state)
    _install_network(monkeypatch, network)
    adapter = HomeAssistantRealityAdapter(
        HomeAssistantTransport(),
        "light.office",
        initial_state=state,
    )

    first = adapter.read()[1]
    assert first.wall_clock_source == "home_assistant.last_updated"
    assert first.source_epoch == "hass.light.office"
    assert first.source_quality == "good"
    assert first.source_event_id.startswith("sha256:")

    network.state["last_updated"] = "2026-08-02T08:00:01Z"
    second = await adapter.refresh_readback()
    assert second.captured_at_ns > first.captured_at_ns
    assert second.source_event_id != first.source_event_id


@pytest.mark.asyncio
async def test_iot_bridge_routes_home_assistant_through_durable_reality_transaction(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del hass_environment
    network = HomeAssistantNetwork(
        {
            "entity_id": "light.office",
            "state": "off",
            "attributes": {"brightness": 10},
        }
    )
    _install_network(monkeypatch, network)
    service = RealityReachService(session_id="test.hass.session")
    coordinator = RealityActuationCoordinator(service, root=tmp_path, executor=_executor)
    bridge = IoTBridge()
    bridge.bind_reality_reach(service, coordinator)

    result = await bridge.apply_authorized(
        HomeAssistantEffect(
            "light.office",
            "turn_on",
            {"brightness_pct": 80, "rgb_color": [10, 20, 30]},
            "integration_test",
        ),
        capability_token="CT-test-hass-1",
        transport_name="home_assistant",
        idempotency_key="test.hass.light.office.1",
    )

    assert result["transport_succeeded"] is True
    assert result["effect_verified"] is True
    assert result["failures"] == []
    assert result["reality_reach_transaction"]["state"] == "effect_verified"
    assert network.state["state"] == "on"
    assert network.state["attributes"]["brightness"] == 204
    assert network.state["attributes"]["rgb_color"] == [10, 20, 30]
    assert len(network.post_calls) == 1
    assert network.post_calls[0]["source"] == "world_bridge:iot.home_assistant.actuate"
    assert network.post_calls[0]["read_only"] is False
    assert all(
        call["read_only"] is True
        for call in network.calls
        if call["method"] == "GET"
    )
    assert service.executable_actuator_channels()

    await bridge.stop()
    assert service.executable_actuator_channels() == ()


@pytest.mark.asyncio
async def test_accepted_transport_without_matching_readback_rolls_back_and_fails(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del hass_environment
    network = HomeAssistantNetwork(
        {
            "entity_id": "light.office",
            "state": "off",
            "attributes": {"brightness": 10},
        },
        apply_updates=False,
    )
    _install_network(monkeypatch, network)
    service = RealityReachService(session_id="test.hass.session")
    coordinator = RealityActuationCoordinator(service, root=tmp_path, executor=_executor)
    bridge = IoTBridge()
    bridge.bind_reality_reach(service, coordinator)

    result = await bridge.apply_authorized(
        HomeAssistantEffect("light.office", "turn_on", {"brightness": 120}),
        capability_token="CT-test-hass-2",
        transport_name="home_assistant",
        idempotency_key="test.hass.light.office.2",
    )

    assert result["transport_succeeded"] is True
    assert result["effect_verified"] is False
    assert result["failures"]
    assert result["reality_reach_transaction"]["state"] == "rolled_back"
    assert [call["url"].rsplit("/", 1)[-1] for call in network.post_calls] == [
        "turn_on",
        "turn_off",
    ]


@pytest.mark.asyncio
async def test_state_race_refuses_before_transport(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del hass_environment
    network = HomeAssistantNetwork(
        {
            "entity_id": "light.office",
            "state": "off",
            "attributes": {"brightness": 10},
        }
    )
    _install_network(monkeypatch, network)
    transport = HomeAssistantTransport()
    adapter = await transport.create_adapter("light.office")
    service = RealityReachService(session_id="test.hass.session")
    service.register_adapter(adapter)
    service.refresh()
    command = await adapter.compile_effect(
        HomeAssistantEffect("light.office", "turn_on", {"brightness": 120}),
        inventory_sha256=str(service.status()["registry_sha256"]),
        deadline_s=10.0,
        idempotency_key="test.hass.race.1",
        source="test",
    )
    wall_now = time.time_ns()
    mono_now = time.monotonic_ns()
    lease = ActuationLease(
        lease_id="lease.test.hass.race.1",
        command_sha256=command.sha256,
        adapter_id=adapter.adapter_id,
        session_id="test.hass.session",
        authority_receipt_id="test.will.hass",
        issued_at_ns=wall_now,
        expires_at_ns=wall_now + 10_000_000_000,
        issued_monotonic_ns=mono_now,
        expires_monotonic_ns=mono_now + 10_000_000_000,
    )
    prepared = await adapter.prepare(command, lease)
    network.state["attributes"]["brightness"] = 11

    with pytest.raises(HomeAssistantRealityError, match="state_changed_before_dispatch"):
        await adapter.actuate(command, lease, prepared)

    assert network.post_calls == []


@pytest.mark.asyncio
async def test_manifest_rejects_parameter_smuggling_and_unmodeled_domains(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del hass_environment
    network = HomeAssistantNetwork(
        {
            "entity_id": "light.office",
            "state": "off",
            "attributes": {"brightness": 10},
        }
    )
    _install_network(monkeypatch, network)
    transport = HomeAssistantTransport()
    adapter = await transport.create_adapter("light.office")
    service = RealityReachService(session_id="test.hass.session")
    service.register_adapter(adapter)
    service.refresh()

    with pytest.raises(HomeAssistantRealityError, match="parameters_not_manifested"):
        await adapter.compile_effect(
            HomeAssistantEffect(
                "light.office",
                "turn_on",
                {"brightness": 120, "service": "unlock"},
            ),
            inventory_sha256=str(service.status()["registry_sha256"]),
            deadline_s=10.0,
            idempotency_key="test.hass.smuggling.1",
            source="test",
        )
    network.state = {
        "entity_id": "lock.front_door",
        "state": "locked",
        "attributes": {},
    }
    with pytest.raises(HomeAssistantRealityError, match="no_physical_manifest"):
        await transport.create_adapter("lock.front_door")
    assert network.post_calls == []


@pytest.mark.asyncio
async def test_climate_setpoint_is_typed_and_range_bounded(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del hass_environment
    network = HomeAssistantNetwork(
        {
            "entity_id": "climate.studio",
            "state": "heat",
            "attributes": {
                "temperature": 22.0,
                "min_temp": 5.0,
                "max_temp": 35.0,
                "temperature_unit": "celsius",
            },
        }
    )
    _install_network(monkeypatch, network)
    service = RealityReachService(session_id="test.hass.session")
    coordinator = RealityActuationCoordinator(service, root=tmp_path, executor=_executor)
    bridge = IoTBridge()
    bridge.bind_reality_reach(service, coordinator)

    result = await bridge.apply_authorized(
        HomeAssistantEffect(
            "climate.studio",
            "set_temperature",
            {"temperature": 21.0},
        ),
        capability_token="CT-test-hass-climate",
        transport_name="home_assistant",
        idempotency_key="test.hass.climate.1",
    )

    assert result["effect_verified"] is True
    assert network.state["attributes"]["temperature"] == 21.0
    adapter = next(iter(bridge._reality_adapters.values()))
    with pytest.raises(ValueError, match="outside"):
        await adapter.compile_effect(
            HomeAssistantEffect(
                "climate.studio",
                "set_temperature",
                {"temperature": 50.0},
            ),
            inventory_sha256=str(service.status()["registry_sha256"]),
            deadline_s=10.0,
            idempotency_key="test.hass.climate.out-of-range",
            source="test",
        )


@pytest.mark.asyncio
async def test_direct_transport_apply_refuses_without_network_side_effect(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del hass_environment
    network = HomeAssistantNetwork(
        {"entity_id": "switch.pump", "state": "off", "attributes": {}}
    )
    _install_network(monkeypatch, network)
    transport = HomeAssistantTransport()

    with pytest.raises(HomeAssistantRealityError, match="requires_reality_reach"):
        await transport.apply(HomeAssistantEffect("switch.pump", "turn_on", {}))

    assert network.calls == []


@pytest.mark.asyncio
async def test_numeric_and_binary_entities_become_typed_read_only_sensors(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del hass_environment
    network = HomeAssistantNetwork(
        {
            "entity_id": "sensor.office_temperature",
            "state": "21.5",
            "last_updated": "2026-08-01T20:15:30.123456+00:00",
            "context": {"id": "01K1KQHOMEASSISTANTPROOF"},
            "attributes": {
                "device_class": "temperature",
                "unit_of_measurement": "\u00b0C",
                "suggested_display_precision": 1,
            },
        }
    )
    _install_network(monkeypatch, network)
    transport = HomeAssistantTransport()
    numeric = await transport.create_sensor_adapter("sensor.office_temperature")

    declaration = numeric.declarations()[0]
    assert declaration.unit == "celsius"
    assert declaration.observable == "home_assistant_temperature"
    assert numeric.read()[0].value == 21.5
    assert numeric.read()[0].wall_clock_source == "home_assistant.last_updated"
    assert numeric.read()[0].source_event_id.startswith("sha256:")
    assert numeric.read()[0].source_event_id != "01K1KQHOMEASSISTANTPROOF"
    assert numeric.read()[0].source_quality == "good"
    assert numeric.read()[0].captured_at_ns == 1_785_615_330_123_456_000
    assert declaration.compliance_tags == (
        "home_assistant",
        "read_only_sensor",
        "device_class_standard",
    )
    first_event_id = numeric.read()[0].source_event_id
    network.state = {
        "entity_id": "sensor.office_temperature",
        "state": "22.0",
        "last_updated": "2026-08-01T20:15:31.123456+00:00",
        "context": {"id": "01K1KQHOMEASSISTANTPROOF"},
        "attributes": {
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "suggested_display_precision": 1,
        },
    }
    second_reading = await numeric.refresh_readback()
    assert second_reading.source_event_id != first_event_id

    network.state = {
        "entity_id": "binary_sensor.front_door",
        "state": "on",
        "attributes": {"device_class": "door"},
    }
    binary = await transport.create_sensor_adapter("binary_sensor.front_door")
    assert binary.declarations()[0].unit == "binary"
    assert binary.read()[0].value == 1.0
    assert "binary_sensor" in binary.declarations()[0].compliance_tags


@pytest.mark.asyncio
async def test_home_assistant_connector_separates_discovery_trust_and_control(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del hass_environment
    monkeypatch.setenv("AURA_HASS_INSTALLATION_ID", "test-installation")
    state = {
        "entity_id": "light.office",
        "state": "off",
        "attributes": {"friendly_name": "Office Light"},
    }
    network = HomeAssistantNetwork(state)
    _install_network(monkeypatch, network)
    transport = HomeAssistantTransport()

    async def discover() -> list[dict[str, Any]]:
        return [dict(network.state)]

    connector = HomeAssistantConnector(transport, discover_callback=discover)
    candidate = (await connector.discover())[0]

    assert candidate.persistent_identity is True
    assert candidate.display_name == "Office Light"
    assert candidate.access == (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
    with pytest.raises(PermissionError, match="requires_observation"):
        await connector.attach(candidate, (AttachmentAccess.CONTROL,))
    sensor = await connector.attach(candidate, (AttachmentAccess.OBSERVE,))
    assert sensor.adapter_id.endswith("sensor.adapter")
    actuator = await connector.attach(
        candidate,
        (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL),
    )
    assert actuator.actuator_capabilities()


@pytest.mark.asyncio
async def test_home_assistant_connector_refuses_manifest_drift_and_weak_identity(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del hass_environment
    monkeypatch.delenv("AURA_HASS_INSTALLATION_ID", raising=False)
    network = HomeAssistantNetwork(
        {
            "entity_id": "climate.studio",
            "state": "heat",
            "attributes": {
                "temperature": 21.0,
                "min_temp": 5.0,
                "max_temp": 35.0,
                "temperature_unit": "\u00b0C",
            },
        }
    )
    _install_network(monkeypatch, network)
    transport = HomeAssistantTransport()

    async def discover() -> list[dict[str, Any]]:
        return [dict(network.state)]

    connector = HomeAssistantConnector(transport, discover_callback=discover)
    candidate = (await connector.discover())[0]
    assert candidate.persistent_identity is False

    network.state["attributes"]["max_temp"] = 45.0
    with pytest.raises(RuntimeError, match="changed_before_attachment"):
        await connector.attach(candidate, (AttachmentAccess.OBSERVE,))


@pytest.mark.asyncio
async def test_unclassified_sensor_manifest_is_stable_within_magnitude_bucket(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del hass_environment
    monkeypatch.setenv("AURA_HASS_INSTALLATION_ID", "test-installation")
    network = HomeAssistantNetwork(
        {
            "entity_id": "sensor.experimental_signal",
            "state": "10.0",
            "attributes": {"unit_of_measurement": "arb"},
        }
    )
    _install_network(monkeypatch, network)
    transport = HomeAssistantTransport()

    async def discover() -> list[dict[str, Any]]:
        return [dict(network.state)]

    connector = HomeAssistantConnector(transport, discover_callback=discover)
    first = (await connector.discover())[0]
    network.state["state"] = "11.0"
    second = (await connector.discover())[0]

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.metadata["sensor_domain_source"] == "inferred_magnitude_bucket"


@pytest.mark.asyncio
async def test_background_sensor_readback_gets_narrow_read_only_governance(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del hass_environment
    from core import governance_context as governance_module

    observed = []

    async def network_request(**_kwargs):
        observed.append(governance_module.get_active_governance())
        return {
            "ok": True,
            "status_code": 200,
            "content": b'{"entity_id":"sensor.office_temperature","state":"21.5",'
            b'"attributes":{"device_class":"temperature","unit_of_measurement":"C"}}',
        }

    monkeypatch.setattr(governance_module, "governance_runtime_active", lambda: True)
    monkeypatch.setattr(
        hass_module.ActionExecutor,
        "request_network_transport",
        staticmethod(network_request),
    )
    transport = HomeAssistantTransport()

    state = await transport.read_state("sensor.office_temperature")

    assert state["state"] == "21.5"
    assert observed[0] is not None
    assert observed[0].domain == "environment_action"
    constraints = dict(observed[0].constraints)
    assert constraints["read_only"] is True
    assert "target_sha256" in constraints


@pytest.mark.asyncio
async def test_iot_bridge_registers_home_assistant_with_attachment_fabric(
    hass_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del hass_environment
    network = HomeAssistantNetwork(
        {"entity_id": "light.office", "state": "off", "attributes": {}}
    )
    _install_network(monkeypatch, network)
    service = RealityReachService(session_id="test.hass.fabric")
    coordinator = RealityActuationCoordinator(service, root=tmp_path, executor=_executor)

    class Router:
        def register_sampler(self, _adapter) -> None:
            return None

        def unregister_sampler(self, _adapter_id) -> None:
            return None

    class Broker:
        def __init__(self) -> None:
            self.connectors = []

        def register_connector(self, connector) -> None:
            self.connectors.append(connector)

    router = Router()
    broker = Broker()
    bridge = IoTBridge()
    bridge.bind_reality_reach(service, coordinator)
    bridge.bind_sensory_fabric(router, broker)

    await bridge.start(interval=60.0)
    try:
        assert len(broker.connectors) == 1
        assert isinstance(broker.connectors[0], HomeAssistantConnector)
        assert bridge.get_status()["home_assistant_connector"] is True
    finally:
        await bridge.stop()
