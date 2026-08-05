from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import asyncua
import pytest

import core.embodiment.opcua_connector as opcua_module
from core.embodiment.opcua_connector import (
    AsyncUaScalarTransport,
    OPCUAConnector,
    OPCUAConnectorError,
    OPCUAResourceSpec,
    parse_opcua_resource_manifest,
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


def _resource(*, writable: bool = True) -> OPCUAResourceSpec:
    return OPCUAResourceSpec(
        resource_id="tank.level",
        device_id="tank.alpha",
        observable="tank_fill_level",
        unit="percent",
        state_node_id="ns=2;s=Tank.Alpha.LevelFeedback",
        command_node_id="ns=2;s=Tank.Alpha.LevelSetpoint" if writable else "",
        domain=NumericDomain(0.0, 100.0),
        resolution=0.1,
        tolerance=0.5,
        safe_value=0.0 if writable else None,
    )


class _Transport:
    transport_id = "opcua.test"
    server_identity_sha256 = _digest("server-alpha")

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
            quality="server_reported",
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


def test_opcua_manifest_is_typed_stable_and_requires_distinct_feedback() -> None:
    raw = [
        {
            "resource_id": "tank.level",
            "device_id": "tank.alpha",
            "observable": "tank_fill_level",
            "unit": "percent",
            "state_node_id": "ns=2;s=Tank.Alpha.LevelFeedback",
            "command_node_id": "ns=2;s=Tank.Alpha.LevelSetpoint",
            "minimum": 0,
            "maximum": 100,
            "resolution": 0.1,
            "tolerance": 0.5,
            "safe_value": 0,
        }
    ]

    first = parse_opcua_resource_manifest(json.dumps(raw))[0]
    second = parse_opcua_resource_manifest(raw)[0]

    assert first == second
    assert first.sha256 == second.sha256
    assert first.decode(42.5) == 42.5
    assert first.encode(42.5) == 42.5
    assert "password" not in json.dumps(first.to_dict()).lower()

    with pytest.raises(ValueError, match="distinct state node"):
        replace(first, command_node_id=first.state_node_id)


def test_opcua_manifest_rejects_node_aliasing_across_resources() -> None:
    base = {
        "device_id": "tank.alpha",
        "observable": "tank_fill_level",
        "unit": "percent",
        "state_node_id": "ns=2;s=Tank.SharedFeedback",
        "minimum": 0,
        "maximum": 100,
        "resolution": 0.1,
    }
    with pytest.raises(OPCUAConnectorError, match="state_node_duplicate"):
        parse_opcua_resource_manifest(
            [
                {**base, "resource_id": "tank.level"},
                {**base, "resource_id": "tank.level.backup"},
            ]
        )


@pytest.mark.asyncio
async def test_opcua_connector_discovers_and_attaches_verified_resource() -> None:
    transport = _Transport()
    connector = OPCUAConnector(transport, (_resource(),))

    candidates = await connector.discover()
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.access == (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
    assert candidate.metadata["independent_readback"] is True
    assert candidate.metadata["control_available"] is True
    assert "Tank.Alpha" not in json.dumps(candidate.to_dict())

    adapter = await connector.attach(
        candidate,
        (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL),
    )
    assert adapter.physical_identity_sha256 == candidate.identity_fingerprint
    assert len(adapter.actuator_capabilities()) == 1
    assert (await adapter.refresh_readback()).value == 25.0


@pytest.mark.asyncio
async def test_opcua_connector_excludes_unmeasured_resource() -> None:
    class Missing(_Transport):
        async def read_scalar(self, resource_id: str) -> ScalarSample:
            del resource_id
            raise TimeoutError("server did not return a current value")

    assert await OPCUAConnector(Missing(), (_resource(),)).discover() == ()


def test_opcua_secure_session_is_default_and_insecure_requires_opt_in(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AURA_OPCUA_ENDPOINT", "opc.tcp://plc.example.test:4840")
    monkeypatch.setenv("AURA_OPCUA_INSTALLATION_ID", "plant-alpha")
    monkeypatch.delenv("AURA_OPCUA_SECURITY_POLICY", raising=False)
    with pytest.raises(OPCUAConnectorError, match="security_material_missing"):
        AsyncUaScalarTransport((_resource(),))

    monkeypatch.setenv("AURA_OPCUA_SECURITY_POLICY", "None")
    monkeypatch.setenv("AURA_OPCUA_SECURITY_MODE", "None")
    monkeypatch.setattr(opcua_module, "_allow_insecure", lambda: False)
    with pytest.raises(OPCUAConnectorError, match="explicit_opt_in"):
        AsyncUaScalarTransport((_resource(),))

    monkeypatch.setattr(opcua_module, "_allow_insecure", lambda: True)
    monkeypatch.setenv("AURA_OPCUA_TIMEOUT_S", "nan")
    with pytest.raises(ValueError, match="must be finite"):
        AsyncUaScalarTransport((_resource(),))


@pytest.mark.asyncio
async def test_asyncua_transport_reads_writes_idempotently_and_redacts_secrets(
    monkeypatch,
) -> None:
    class Status:
        @staticmethod
        def is_good() -> bool:
            return True

        def __str__(self) -> str:
            return "Good"

    class Node:
        def __init__(self, node_id: str) -> None:
            self.node_id = node_id
            self.writes: list[tuple[object, object]] = []

        async def read_data_value(self):
            return SimpleNamespace(
                StatusCode=Status(),
                Value=SimpleNamespace(Value=31.5),
                SourceTimestamp=datetime(2026, 8, 5, tzinfo=UTC),
                ServerTimestamp=None,
            )

        async def write_value(self, value, *, varianttype):
            self.writes.append((value, varianttype))

    class Client:
        instance = None

        def __init__(self, endpoint, **kwargs):
            type(self).instance = self
            self.endpoint = endpoint
            self.kwargs = kwargs
            self.nodes: dict[str, Node] = {}
            self.connected = False
            self.disconnected = False

        async def connect(self, **kwargs):
            self.connect_kwargs = kwargs
            self.connected = True

        async def disconnect(self):
            self.disconnected = True

        def set_user(self, username):
            self.username = username

        def set_password(self, password):
            self.password = password

        def get_node(self, node_id):
            return self.nodes.setdefault(node_id, Node(node_id))

    monkeypatch.setattr(asyncua, "Client", Client)
    monkeypatch.setattr(opcua_module, "_allow_insecure", lambda: True)
    monkeypatch.setenv("AURA_OPCUA_ENDPOINT", "opc.tcp://plc.example.test:4840")
    monkeypatch.setenv("AURA_OPCUA_INSTALLATION_ID", "plant-alpha")
    monkeypatch.setenv("AURA_OPCUA_SECURITY_POLICY", "None")
    monkeypatch.setenv("AURA_OPCUA_SECURITY_MODE", "None")
    monkeypatch.setenv("AURA_OPCUA_USERNAME", "aura")
    monkeypatch.setenv("AURA_OPCUA_PASSWORD", "not-retained")
    transport = AsyncUaScalarTransport((_resource(),))

    sample = await transport.read_scalar("tank.level")
    first = await transport.write_scalar(
        "tank.level",
        45.0,
        idempotency_key="opcua.test.45",
    )
    second = await transport.write_scalar(
        "tank.level",
        45.0,
        idempotency_key="opcua.test.45",
    )

    assert sample.value == 31.5
    assert sample.wall_clock_source == "opcua.source_timestamp"
    assert sample.source_epoch == transport.server_identity_sha256
    assert first is second
    command = Client.instance.nodes["ns=2;s=Tank.Alpha.LevelSetpoint"]
    assert len(command.writes) == 1
    assert "not-retained" not in json.dumps(first.receipt)
    assert "aura" not in json.dumps(first.receipt)

    await transport.stop()
    assert Client.instance.disconnected is True


def test_opcua_catalog_reports_partial_configuration(monkeypatch) -> None:
    for name in (
        "AURA_OPENHAB_URL",
        "AURA_OPENHAB_TOKEN",
        "AURA_MQTT_BROKER_URL",
        "AURA_MQTT_RESOURCES_JSON",
        "AURA_MQTT_INSTALLATION_ID",
        "AURA_OPCUA_RESOURCES_JSON",
        "AURA_OPCUA_INSTALLATION_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AURA_OPCUA_ENDPOINT", "opc.tcp://plc.example.test:4840")

    status = build_configured_reality_connector_catalog().status()
    opcua = next(
        item for item in status["connectors"] if item["connector_id"] == "opcua.manifest"
    )

    assert status["ready"] is False
    assert opcua["configured"] is True
    assert opcua["state"] == "invalid"
    assert "AURA_OPCUA_RESOURCES_JSON" in opcua["error"]
    assert "AURA_OPCUA_INSTALLATION_ID" in opcua["error"]
