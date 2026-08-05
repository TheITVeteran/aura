from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import replace
from typing import Any

import pytest

import core.embodiment.scpi_connector as scpi_module
from core.embodiment.reality_connectors import (
    build_configured_reality_connector_catalog,
)
from core.embodiment.scpi_connector import (
    SCPIConnector,
    SCPIConnectorError,
    SCPIResourceSpec,
    SCPIScalarTransport,
    SCPIStreamTransport,
    parse_scpi_resource_manifest,
)
from core.reality_reach.attachments import AttachmentAccess
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.scalar_adapter import ScalarSample, ScalarWriteResult
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.network_gateway import StreamAdmission


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _resource(*, writable: bool = True) -> SCPIResourceSpec:
    return SCPIResourceSpec(
        resource_id="supply.voltage",
        device_id="supply.alpha",
        observable="output_voltage",
        unit="volt",
        read_query="MEAS:VOLT?",
        command_template="SOUR:VOLT {value}" if writable else "",
        domain=NumericDomain(0.0, 30.0),
        resolution=0.001,
        tolerance=0.01,
        uncertainty=0.002,
        safe_value=0.0 if writable else None,
    )


class _Transport:
    transport_id = "scpi.test"

    def __init__(self, *, stable: bool = True, value: float = 5.0) -> None:
        self.identity_stable = stable
        self.instrument_identity_sha256 = _digest("instrument-alpha")
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
            quality="instrument_reported",
            uncertainty=0.002,
            source_epoch=self.instrument_identity_sha256,
            source_sequence=self.sequence,
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


class _StalledTransport(_Transport):
    async def read_scalar(self, resource_id: str) -> ScalarSample:
        await asyncio.sleep(1.0)
        return await super().read_scalar(resource_id)


def test_scpi_manifest_is_stable_numeric_and_injection_closed() -> None:
    raw = [
        {
            "resource_id": "supply.voltage",
            "device_id": "supply.alpha",
            "observable": "output_voltage",
            "unit": "volt",
            "read_query": "MEAS:VOLT?",
            "command_template": "SOUR:VOLT {value}",
            "minimum": 0,
            "maximum": 30,
            "resolution": 0.001,
            "tolerance": 0.01,
            "uncertainty": 0.002,
            "safe_value": 0,
        }
    ]

    first = parse_scpi_resource_manifest(json.dumps(raw))[0]
    second = parse_scpi_resource_manifest(raw)[0]

    assert first == second
    assert first.sha256 == second.sha256
    assert first.decode("1.250E+01") == 12.5
    assert first.command(12.5) == "SOUR:VOLT 12.5"
    assert "password" not in json.dumps(first.to_dict()).lower()
    assert replace(first, read_query="MEAS:VOLT? (@1)").read_query == "MEAS:VOLT? (@1)"

    with pytest.raises(ValueError, match="single SCPI query"):
        replace(first, read_query="MEAS:VOLT?;SYST:ERR?")
    with pytest.raises(ValueError, match=r"one \{value\}"):
        replace(first, command_template="SOUR:VOLT {value} {value}")
    with pytest.raises(ValueError, match=r"one \{value\}"):
        replace(first, command_template="SOUR:VOLT {value}}")
    with pytest.raises(SCPIConnectorError, match="outside_manifest_contract"):
        first.command(31.0)


def test_scpi_manifest_rejects_aliased_queries() -> None:
    base = {
        "device_id": "supply.alpha",
        "observable": "output_voltage",
        "unit": "volt",
        "read_query": "MEAS:VOLT?",
        "minimum": 0,
        "maximum": 30,
        "resolution": 0.001,
    }
    with pytest.raises(SCPIConnectorError, match="read_query_duplicate"):
        parse_scpi_resource_manifest(
            [
                {**base, "resource_id": "supply.voltage"},
                {**base, "resource_id": "supply.voltage.backup"},
            ]
        )


@pytest.mark.asyncio
async def test_scpi_connector_only_exposes_control_for_stable_identity() -> None:
    stable = SCPIConnector(_Transport(stable=True), (_resource(),))
    stable_candidate = (await stable.discover())[0]
    assert stable_candidate.access == (
        AttachmentAccess.OBSERVE,
        AttachmentAccess.CONTROL,
    )
    assert stable_candidate.metadata["independent_readback"] is True
    adapter = await stable.attach(
        stable_candidate,
        (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL),
    )
    assert len(adapter.actuator_capabilities()) == 1
    reading = await adapter.refresh_readback()
    assert reading.value == 5.0
    assert reading.uncertainty == 0.002

    unstable = SCPIConnector(_Transport(stable=False), (_resource(),))
    unstable_candidate = (await unstable.discover())[0]
    assert unstable_candidate.access == (AttachmentAccess.OBSERVE,)
    assert unstable_candidate.persistent_identity is False
    with pytest.raises(PermissionError, match="not_declared"):
        await unstable.attach(
            unstable_candidate,
            (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL),
        )


@pytest.mark.asyncio
async def test_scpi_discovery_has_one_connector_wide_deadline() -> None:
    connector = SCPIConnector(
        _StalledTransport(),
        (_resource(writable=False),),
        discovery_budget_s=0.01,
    )
    started = time.monotonic()

    assert await connector.discover() == ()
    assert time.monotonic() - started < 0.2


class _Reader:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = deque(responses)

    async def readline(self) -> bytes:
        if not self.responses:
            raise RuntimeError("unexpected SCPI query")
        return self.responses.popleft()


class _Writer:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, payload: bytes) -> None:
        self.writes.append(payload)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closed


class _Gateway:
    def __init__(self, responses: list[bytes]) -> None:
        self.reader = _Reader(responses)
        self.writer = _Writer()
        self.calls: list[dict[str, object]] = []

    async def connect_stream(self, endpoint: str, **kwargs: object) -> StreamAdmission:
        self.calls.append({"endpoint": endpoint, **kwargs})
        return StreamAdmission(
            reader=self.reader,  # type: ignore[arg-type]
            writer=self.writer,  # type: ignore[arg-type]
            destination_host="instrument.local",
            destination_port=5025,
            peer_address="192.168.1.50",
            secure=True,
            peer_certificate_sha256=_digest(b"certificate"),
            source="reality_reach:scpi.stream",
            read_only=bool(kwargs.get("read_only", False)),
        )


def _configure_tls(monkeypatch: pytest.MonkeyPatch, idn: str) -> None:
    monkeypatch.setenv("AURA_SCPI_ENDPOINT", "tls://instrument.local:5025")
    monkeypatch.setenv("AURA_SCPI_INSTALLATION_ID", "lab-alpha")
    monkeypatch.setenv("AURA_SCPI_EXPECTED_IDN_SHA256", sha256_hex(idn.encode("utf-8")))
    monkeypatch.setenv("AURA_SCPI_SERVER_CERT_SHA256", sha256_hex(b"certificate"))


@pytest.mark.asyncio
async def test_scpi_stream_command_is_completed_error_checked_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idn = "AURA-LABS,PSU-30V,SN123,1.0"
    gateway = _Gateway([f"{idn}\n".encode(), b"1\n", b"0,No error\n"])
    _configure_tls(monkeypatch, idn)
    monkeypatch.setattr(scpi_module, "get_network_gateway", lambda: gateway)
    transport = SCPIStreamTransport((_resource(),))

    first = await transport.write_scalar(
        "supply.voltage",
        12.5,
        idempotency_key="set-12.5",
    )
    second = await transport.write_scalar(
        "supply.voltage",
        12.5,
        idempotency_key="set-12.5",
    )

    assert first is second
    assert gateway.writer.writes == [
        b"*IDN?\n",
        b"SOUR:VOLT 12.5\n",
        b"*OPC?\n",
        b"SYST:ERR?\n",
    ]
    assert first.receipt["operation_complete"] is True
    assert first.receipt["error_queue_clear"] is True
    assert gateway.calls[0]["expected_certificate_sha256"] == sha256_hex(b"certificate")
    assert gateway.calls[0]["read_only"] is False
    assert "SN123" not in json.dumps(first.receipt)


@pytest.mark.asyncio
async def test_scpi_stream_refuses_nonzero_instrument_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idn = "AURA-LABS,PSU-30V,SN123,1.0"
    gateway = _Gateway([f"{idn}\n".encode(), b"1\n", b"-222,Data out of range\n"])
    _configure_tls(monkeypatch, idn)
    monkeypatch.setattr(scpi_module, "get_network_gateway", lambda: gateway)

    with pytest.raises(SCPIConnectorError, match="error_queue_nonzero"):
        await SCPIStreamTransport((_resource(),)).write_scalar(
            "supply.voltage",
            12.5,
            idempotency_key="set-12.5",
        )


@pytest.mark.asyncio
async def test_scpi_stream_closes_admission_when_identity_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_idn = "AURA-LABS,PSU-30V,SN123,1.0"
    gateway = _Gateway([b"different-instrument\n"])
    _configure_tls(monkeypatch, expected_idn)
    monkeypatch.setattr(scpi_module, "get_network_gateway", lambda: gateway)

    with pytest.raises(SCPIConnectorError, match="identity_mismatch"):
        await SCPIStreamTransport((_resource(writable=False),)).read_scalar(
            "supply.voltage"
        )

    assert gateway.writer.closed is True
    assert gateway.calls[0]["read_only"] is True


def test_scpi_transport_refuses_plaintext_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_SCPI_ENDPOINT", "tcp://instrument.local:5025")
    monkeypatch.setenv("AURA_SCPI_INSTALLATION_ID", "lab-alpha")
    monkeypatch.setenv(
        "AURA_SCPI_EXPECTED_IDN_SHA256",
        sha256_hex(b"instrument"),
    )
    monkeypatch.setattr(scpi_module, "_allow_plaintext", lambda: True)

    with pytest.raises(SCPIConnectorError, match="control_requires_tls_certificate_pin"):
        SCPIStreamTransport((_resource(),))


@pytest.mark.asyncio
async def test_real_plaintext_scpi_loopback_reads_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idn = "AURA-LABS,DMM,SN-LOOP,1.0"
    observed: list[str] = []

    async def instrument(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                command = line.decode("ascii").strip()
                observed.append(command)
                response = idn if command == "*IDN?" else "2.500E+00"
                writer.write((response + "\n").encode("ascii"))
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(instrument, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv("AURA_SCPI_ENDPOINT", f"tcp://127.0.0.1:{port}")
    monkeypatch.setenv("AURA_SCPI_INSTALLATION_ID", "loopback-lab")
    monkeypatch.setenv(
        "AURA_SCPI_EXPECTED_IDN_SHA256",
        sha256_hex(idn.encode("utf-8")),
    )
    monkeypatch.setattr(scpi_module, "_allow_plaintext", lambda: True)
    resource = replace(_resource(writable=False), read_query="MEAS:VOLT?")
    transport = SCPIStreamTransport((resource,))
    try:
        sample = await transport.read_scalar(resource.resource_id)
    finally:
        await transport.stop()
        server.close()
        await server.wait_closed()

    assert sample.value == 2.5
    assert sample.quality == "instrument_reported"
    assert transport.identity_stable is False
    assert observed == ["*IDN?", "MEAS:VOLT?"]


def test_scpi_catalog_reports_partial_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AURA_SCPI_RESOURCES_JSON",
        "AURA_SCPI_INSTALLATION_ID",
        "AURA_SCPI_EXPECTED_IDN_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AURA_SCPI_ENDPOINT", "tcp://instrument.local:5025")

    status = build_configured_reality_connector_catalog().status()
    scpi = next(
        item for item in status["connectors"] if item["connector_id"] == "scpi.manifest"
    )

    assert status["ready"] is False
    assert scpi["configured"] is True
    assert scpi["state"] == "invalid"
    assert "AURA_SCPI_RESOURCES_JSON" in scpi["error"]
    assert "AURA_SCPI_EXPECTED_IDN_SHA256" in scpi["error"]


def test_scpi_transport_protocol_is_runtime_checkable() -> None:
    assert isinstance(_Transport(), SCPIScalarTransport)
