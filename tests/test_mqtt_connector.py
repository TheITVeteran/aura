from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from core.embodiment.mqtt_connector import (
    MQTTConnector,
    MQTTConnectorError,
    MQTTResourceSpec,
    PahoMQTTScalarTransport,
    parse_mqtt_resource_manifest,
)
from core.embodiment.reality_connectors import (
    build_configured_reality_connector_catalog,
)
from core.reality_reach.attachments import AttachmentAccess
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.scalar_adapter import ScalarSample, ScalarWriteResult
from core.runtime.audit_chain import canonical_json, sha256_hex


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _resource(*, writable: bool = True) -> MQTTResourceSpec:
    return MQTTResourceSpec(
        resource_id="tank.level",
        device_id="tank.alpha",
        observable="tank_fill_level",
        unit="percent",
        state_topic="plant/tank-alpha/level/state",
        command_topic="plant/tank-alpha/level/set" if writable else "",
        domain=NumericDomain(0.0, 100.0),
        resolution=0.1,
        tolerance=0.5,
        safe_value=0.0 if writable else None,
        device_reported_feedback=writable,
    )


class _Transport:
    transport_id = "mqtt.test"
    broker_identity_sha256 = _digest("broker-alpha")

    def __init__(self, value: float = 25.0) -> None:
        self.value = value
        self.sequence = 0
        self.writes: list[tuple[str, float, str, bool]] = []

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        self.sequence += 1
        return ScalarSample(
            value=self.value,
            captured_at_ns=time.time_ns(),
            source_event_id=_digest(
                {"resource": resource_id, "value": self.value, "sequence": self.sequence}
            ),
            quality="device_reported",
        )

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        self.writes.append((resource_id, value, idempotency_key, recovery))
        self.value = value
        return ScalarWriteResult(
            accepted=True,
            transport_completed=True,
            receipt={"resource_id": resource_id, "recovery": recovery},
        )


def test_mqtt_manifest_is_typed_stable_and_secret_free() -> None:
    raw = [
        {
            "resource_id": "tank.level",
            "device_id": "tank.alpha",
            "observable": "tank_fill_level",
            "unit": "percent",
            "state_topic": "plant/tank-alpha/level/state",
            "command_topic": "plant/tank-alpha/level/set",
            "minimum": 0,
            "maximum": 100,
            "resolution": 0.1,
            "tolerance": 0.5,
            "safe_value": 0,
            "device_reported_feedback": True,
        }
    ]

    first = parse_mqtt_resource_manifest(json.dumps(raw))[0]
    second = parse_mqtt_resource_manifest(raw)[0]

    assert first == second
    assert first.sha256 == second.sha256
    assert first.decode(b"42.5") == 42.5
    assert first.encode(42.5) == b"42.5"
    assert "password" not in json.dumps(first.to_dict()).lower()


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"state_topic": "plant/+/state"}, "concrete MQTT topic"),
        (
            {
                "command_topic": "plant/tank-alpha/level/state",
                "device_reported_feedback": True,
            },
            "distinct device-reported",
        ),
        (
            {
                "command_topic": "plant/tank-alpha/level/set",
                "device_reported_feedback": False,
            },
            "distinct device-reported",
        ),
    ],
)
def test_mqtt_manifest_rejects_ambiguous_or_unverified_control(
    updates,
    match,
) -> None:
    values = {
        "resource_id": "tank.level",
        "device_id": "tank.alpha",
        "observable": "tank_fill_level",
        "unit": "percent",
        "state_topic": "plant/tank-alpha/level/state",
        "command_topic": "plant/tank-alpha/level/set",
        "domain": NumericDomain(0.0, 100.0),
        "resolution": 0.1,
        "device_reported_feedback": True,
    }
    values.update(updates)
    with pytest.raises(ValueError, match=match):
        MQTTResourceSpec(**values)


@pytest.mark.asyncio
async def test_mqtt_connector_discovers_and_attaches_verified_resource() -> None:
    transport = _Transport()
    connector = MQTTConnector(transport, (_resource(),))

    candidates = await connector.discover()
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.access == (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
    assert candidate.metadata["independent_readback"] is True
    assert candidate.metadata["control_available"] is True

    adapter = await connector.attach(
        candidate,
        (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL),
    )
    assert adapter.physical_identity_sha256 == candidate.identity_fingerprint
    assert len(adapter.actuator_capabilities()) == 1
    assert (await adapter.refresh_readback()).value == 25.0


@pytest.mark.asyncio
async def test_mqtt_connector_does_not_discover_unmeasured_resource() -> None:
    class Missing(_Transport):
        async def read_scalar(self, resource_id: str) -> ScalarSample:
            del resource_id
            raise TimeoutError("no retained device state")

    connector = MQTTConnector(Missing(), (_resource(),))
    assert await connector.discover() == ()


def test_mqtt_catalog_reports_partial_configuration(monkeypatch) -> None:
    monkeypatch.delenv("AURA_OPENHAB_URL", raising=False)
    monkeypatch.delenv("AURA_OPENHAB_TOKEN", raising=False)
    monkeypatch.setenv("AURA_MQTT_BROKER_URL", "mqtts://broker.example.test")
    monkeypatch.delenv("AURA_MQTT_RESOURCES_JSON", raising=False)
    monkeypatch.delenv("AURA_MQTT_INSTALLATION_ID", raising=False)

    status = build_configured_reality_connector_catalog().status()
    mqtt = next(item for item in status["connectors"] if item["connector_id"] == "mqtt.manifest")

    assert status["ready"] is False
    assert mqtt["configured"] is True
    assert mqtt["state"] == "invalid"
    assert "AURA_MQTT_RESOURCES_JSON" in mqtt["error"]
    assert "AURA_MQTT_INSTALLATION_ID" in mqtt["error"]


def test_mqtt_resource_manifest_rejects_duplicate_topics() -> None:
    base = {
        "device_id": "tank.alpha",
        "observable": "tank_fill_level",
        "unit": "percent",
        "state_topic": "plant/tank-alpha/level/state",
        "minimum": 0,
        "maximum": 100,
        "resolution": 0.1,
    }
    with pytest.raises(MQTTConnectorError, match="state_topic_duplicate"):
        parse_mqtt_resource_manifest(
            [
                {**base, "resource_id": "tank.level"},
                {**base, "resource_id": "tank.level.backup"},
            ]
        )


@pytest.mark.asyncio
async def test_paho_transport_uses_v5_qos_idempotency_and_clean_shutdown(
    monkeypatch,
) -> None:
    import paho.mqtt.client as mqtt

    class PublishInfo:
        mid = 41

        @staticmethod
        def wait_for_publish(_timeout):
            return None

        @staticmethod
        def is_published():
            return True

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.subscriptions = []
            self.published = []
            self.disconnected = False
            self.stopped = False

        def tls_set(self, **kwargs):
            self.tls = kwargs

        def username_pw_set(self, username, password):
            self.credentials = (username, password)

        def connect_async(self, host, port, *, keepalive):
            self.endpoint = (host, port, keepalive)

        def loop_start(self):
            self.started = True

        def subscribe(self, topic, *, qos):
            self.subscriptions.append((topic, qos))

        def publish(self, topic, **kwargs):
            self.published.append((topic, kwargs))
            return PublishInfo()

        def disconnect(self):
            self.disconnected = True

        def loop_stop(self):
            self.stopped = True

    client = Client()
    monkeypatch.setattr(mqtt, "Client", lambda **_kwargs: client)
    monkeypatch.setenv("AURA_MQTT_BROKER_URL", "mqtts://broker.example.test")
    monkeypatch.setenv("AURA_MQTT_INSTALLATION_ID", "plant-alpha")
    monkeypatch.setenv("AURA_MQTT_USERNAME", "aura")
    monkeypatch.setenv("AURA_MQTT_PASSWORD", "not-retained")
    transport = PahoMQTTScalarTransport((_resource(),))

    await transport._ensure_started()
    client.on_connect(client, None, None, SimpleNamespace(is_failure=False), None)
    client.on_message(
        client,
        None,
        SimpleNamespace(topic="plant/tank-alpha/level/state", payload=b"31.5"),
    )
    sample = await transport.read_scalar("tank.level")
    result = await transport.write_scalar(
        "tank.level",
        45.0,
        idempotency_key="mqtt.test.45",
    )

    assert sample.value == 31.5
    assert sample.wall_clock_source == "system.time_ns"
    assert sample.source_epoch == transport.broker_identity_sha256
    assert sample.source_sequence == 1
    assert result.accepted is True
    topic, publish = client.published[0]
    assert topic == "plant/tank-alpha/level/set"
    assert publish["qos"] == 1
    assert publish["retain"] is False
    assert ("aura-idempotency-key", "mqtt.test.45") in publish[
        "properties"
    ].UserProperty
    assert "not-retained" not in json.dumps(result.receipt)

    await transport.stop()
    assert client.disconnected is True
    assert client.stopped is True
