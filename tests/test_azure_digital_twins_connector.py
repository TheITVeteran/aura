from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any

import pytest

import core.embodiment.azure_digital_twins_connector as azure_module
from core.embodiment.azure_digital_twins_connector import (
    AzureDigitalTwinResourceSpec,
    AzureDigitalTwinsConnector,
    AzureDigitalTwinsConnectorError,
    AzureDigitalTwinsScalarTransport,
    EntraClientCredentialsTokenProvider,
    parse_azure_digital_twin_manifest,
)
from core.embodiment.reality_connectors import (
    build_configured_reality_connector_catalog,
)
from core.reality_reach.attachments import AttachmentAccess
from core.reality_reach.contracts import NumericDomain
from core.runtime.audit_chain import sha256_hex


class _TokenProvider:
    identity_sha256 = sha256_hex(b"credential-identity")

    def __init__(self, token: str = "private-access-token") -> None:
        self.token = token
        self.calls = 0

    async def access_token(self) -> str:
        self.calls += 1
        return self.token


def _resource(*, writable: bool = True) -> AzureDigitalTwinResourceSpec:
    return AzureDigitalTwinResourceSpec(
        resource_id="pump.temperature",
        twin_id="pump-alpha",
        observable="temperature",
        unit="celsius",
        reported_path="/telemetry/temperature",
        desired_path="/commands/temperatureSetpoint" if writable else "",
        expected_model_id="dtmi:aura:Pump;1",
        domain=NumericDomain(-40.0, 180.0),
        resolution=0.1,
        tolerance=0.5,
        uncertainty=0.2,
        safe_value=20.0 if writable else None,
    )


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AURA_AZURE_DT_ENDPOINT",
        "https://aura-lab.api.westus2.digitaltwins.azure.net",
    )
    monkeypatch.setenv("AURA_AZURE_DT_INSTANCE_ID", "aura-lab")


def _twin(value: float = 21.5) -> bytes:
    return json.dumps(
        {
            "$dtId": "pump-alpha",
            "$metadata": {
                "$model": "dtmi:aura:Pump;1",
                "telemetry": {"temperature": {"lastUpdateTime": "2026-08-05T07:08:09Z"}},
            },
            "telemetry": {"temperature": value},
            "commands": {"temperatureSetpoint": 20.0},
        }
    ).encode()


def test_azure_manifest_is_strict_stable_and_injection_closed() -> None:
    raw = [
        {
            "resource_id": "pump.temperature",
            "twin_id": "pump-alpha",
            "observable": "temperature",
            "unit": "celsius",
            "reported_path": "/telemetry/temperature",
            "desired_path": "/commands/temperatureSetpoint",
            "expected_model_id": "dtmi:aura:Pump;1",
            "minimum": -40,
            "maximum": 180,
            "resolution": 0.1,
            "tolerance": 0.5,
            "uncertainty": 0.2,
            "safe_value": 20,
        }
    ]
    first = parse_azure_digital_twin_manifest(json.dumps(raw))[0]
    second = parse_azure_digital_twin_manifest(raw)[0]

    assert first == second
    assert first.sha256 == second.sha256
    assert first.decode({"telemetry": {"temperature": 21.5}}) == 21.5
    assert "credential" not in json.dumps(first.to_dict()).lower()

    for pointer in (
        "telemetry/temperature",
        "/$metadata/$model",
        "/telemetry/~2temperature",
        "/telemetry/temperature\ncommands",
    ):
        mutated = dict(raw[0], reported_path=pointer)
        with pytest.raises(ValueError):
            parse_azure_digital_twin_manifest([mutated])

    with pytest.raises(ValueError, match="distinct desired and reported"):
        parse_azure_digital_twin_manifest([dict(raw[0], desired_path="/telemetry/temperature")])
    with pytest.raises(AzureDigitalTwinsConnectorError, match="duplicate_json_key"):
        parse_azure_digital_twin_manifest('[{"resource_id":"a","resource_id":"b"}]')


@pytest.mark.asyncio
async def test_azure_transport_reads_bound_model_and_emits_secret_free_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)
    calls: list[dict[str, Any]] = []

    async def request(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "ok": True,
            "status_code": 200,
            "headers": {"etag": 'W/"twin-version-7"'},
            "content": _twin(),
        }

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    token = _TokenProvider()
    transport = AzureDigitalTwinsScalarTransport((_resource(),), token)

    sample = await transport.read_scalar("pump.temperature")

    assert sample.value == 21.5
    assert sample.uncertainty == 0.2
    assert sample.quality == "device_reported_cloud_twin"
    assert sample.wall_clock_source == "azure_digital_twins.metadata.lastUpdateTime"
    assert calls[0]["method"] == "GET"
    assert calls[0]["read_only"] is True
    assert calls[0]["url"].endswith("/digitaltwins/pump-alpha?api-version=2023-10-31")
    assert calls[0]["headers"]["Authorization"] == "Bearer private-access-token"
    assert "private-access-token" not in json.dumps(
        sample.__dict__ if hasattr(sample, "__dict__") else {"event": sample.source_event_id}
    )


@pytest.mark.asyncio
async def test_azure_transport_etag_fences_patch_and_does_not_claim_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)
    responses = deque(
        [
            {
                "ok": True,
                "status_code": 200,
                "headers": {"ETag": 'W/"twin-version-7"'},
                "content": _twin(),
            },
            {"ok": True, "status_code": 204, "headers": {}, "content": b""},
        ]
    )
    calls: list[dict[str, Any]] = []

    async def request(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return responses.popleft()

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    transport = AzureDigitalTwinsScalarTransport((_resource(),), _TokenProvider())
    await transport.read_scalar("pump.temperature")
    first = await transport.write_scalar(
        "pump.temperature",
        24.0,
        idempotency_key="setpoint-24",
    )
    second = await transport.write_scalar(
        "pump.temperature",
        24.0,
        idempotency_key="setpoint-24",
    )

    assert first is second
    assert first.accepted is True
    assert first.receipt["effect_verified"] is False
    patch = calls[1]
    assert patch["method"] == "PATCH"
    assert patch["read_only"] is False
    assert patch["headers"]["If-Match"] == 'W/"twin-version-7"'
    assert json.loads(patch["data"]) == [
        {
            "op": "replace",
            "path": "/commands/temperatureSetpoint",
            "value": 24.0,
        }
    ]
    serialized = json.dumps(dict(first.receipt))
    assert "private-access-token" not in serialized
    assert "pump-alpha" not in serialized

    with pytest.raises(
        AzureDigitalTwinsConnectorError,
        match="fresh_etag_readback",
    ):
        await transport.write_scalar(
            "pump.temperature",
            25.0,
            idempotency_key="setpoint-25",
        )


@pytest.mark.asyncio
async def test_azure_transport_rejects_model_mismatch_and_etag_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)
    wrong_model = json.loads(_twin())
    wrong_model["$metadata"]["$model"] = "dtmi:aura:Other;1"

    async def wrong_read(**_kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "status_code": 200,
            "headers": {"ETag": 'W/"1"'},
            "content": json.dumps(wrong_model).encode(),
        }

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        wrong_read,
    )
    transport = AzureDigitalTwinsScalarTransport((_resource(),), _TokenProvider())
    with pytest.raises(AzureDigitalTwinsConnectorError, match="identity_mismatch"):
        await transport.read_scalar("pump.temperature")

    responses = deque(
        [
            {
                "ok": True,
                "status_code": 200,
                "headers": {"ETag": 'W/"1"'},
                "content": _twin(),
            },
            {"ok": False, "status_code": 412, "headers": {}, "content": b""},
        ]
    )

    async def conflict(**_kwargs: Any) -> dict[str, Any]:
        return responses.popleft()

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        conflict,
    )
    transport = AzureDigitalTwinsScalarTransport((_resource(),), _TokenProvider())
    await transport.read_scalar("pump.temperature")
    with pytest.raises(AzureDigitalTwinsConnectorError, match="etag_precondition_failed"):
        await transport.write_scalar(
            "pump.temperature",
            24.0,
            idempotency_key="setpoint-conflict",
        )


@pytest.mark.asyncio
async def test_azure_connector_exposes_verified_scalar_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)

    async def request(**_kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "status_code": 200,
            "headers": {"ETag": 'W/"1"'},
            "content": _twin(),
        }

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    resource = _resource()
    connector = AzureDigitalTwinsConnector(
        AzureDigitalTwinsScalarTransport((resource,), _TokenProvider()),
        (resource,),
    )

    candidate = (await connector.discover())[0]
    assert candidate.access == (
        AttachmentAccess.OBSERVE,
        AttachmentAccess.CONTROL,
    )
    assert candidate.metadata["independent_device_reported_readback"] is True
    adapter = await connector.attach(
        candidate,
        (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL),
    )
    assert len(adapter.declarations()) == 2
    assert len(adapter.actuator_capabilities()) == 1


@pytest.mark.asyncio
async def test_entra_token_provider_refreshes_once_and_never_returns_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = "11111111-1111-4111-8111-111111111111"
    client = "22222222-2222-4222-8222-222222222222"
    secret = "super-private-client-secret"
    monkeypatch.setenv("AURA_AZURE_DT_TENANT_ID", tenant)
    monkeypatch.setenv("AURA_AZURE_DT_CLIENT_ID", client)
    monkeypatch.setenv("AURA_AZURE_DT_CLIENT_SECRET", secret)
    calls: list[dict[str, Any]] = []

    async def request(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "ok": True,
            "status_code": 200,
            "headers": {},
            "content": json.dumps(
                {"access_token": "refreshed-access-token", "expires_in": 3600}
            ).encode(),
        }

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    provider = EntraClientCredentialsTokenProvider()

    assert await provider.access_token() == "refreshed-access-token"
    assert await provider.access_token() == "refreshed-access-token"
    assert len(calls) == 1
    assert calls[0]["source"] == "world_bridge:azure_digital_twins.oauth"
    assert calls[0]["read_only"] is False
    assert secret in calls[0]["data"]
    assert secret not in provider.identity_sha256


@pytest.mark.asyncio
async def test_entra_token_provider_adopts_rotated_secret_after_token_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant = "11111111-1111-4111-8111-111111111111"
    client = "22222222-2222-4222-8222-222222222222"
    first_secret = "first-private-client-secret"
    second_secret = "second-private-client-secret"
    monkeypatch.setenv("AURA_AZURE_DT_TENANT_ID", tenant)
    monkeypatch.setenv("AURA_AZURE_DT_CLIENT_ID", client)
    monkeypatch.setenv("AURA_AZURE_DT_CLIENT_SECRET", first_secret)
    clock = [100.0]
    calls: list[dict[str, Any]] = []

    async def request(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "ok": True,
            "status_code": 200,
            "headers": {},
            "content": json.dumps(
                {
                    "access_token": f"rotated-access-token-{len(calls)}",
                    "expires_in": 3600,
                }
            ).encode(),
        }

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    provider = EntraClientCredentialsTokenProvider(monotonic_clock=lambda: clock[0])

    assert await provider.access_token() == "rotated-access-token-1"
    monkeypatch.setenv("AURA_AZURE_DT_CLIENT_SECRET", second_secret)
    assert await provider.access_token() == "rotated-access-token-1"
    clock[0] += 3541.0
    assert await provider.access_token() == "rotated-access-token-2"

    assert len(calls) == 2
    assert first_secret in calls[0]["data"]
    assert second_secret not in calls[0]["data"]
    assert second_secret in calls[1]["data"]
    assert first_secret not in calls[1]["data"]
    assert first_secret not in provider.identity_sha256
    assert second_secret not in provider.identity_sha256


@pytest.mark.asyncio
async def test_entra_token_provider_fails_closed_when_rotated_secret_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AURA_AZURE_DT_TENANT_ID",
        "11111111-1111-4111-8111-111111111111",
    )
    monkeypatch.setenv(
        "AURA_AZURE_DT_CLIENT_ID",
        "22222222-2222-4222-8222-222222222222",
    )
    monkeypatch.setenv("AURA_AZURE_DT_CLIENT_SECRET", "initial-secret")
    clock = [10.0]
    calls = 0

    async def request(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "ok": True,
            "status_code": 200,
            "headers": {},
            "content": json.dumps({"access_token": "temporary-token", "expires_in": 120}).encode(),
        }

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    provider = EntraClientCredentialsTokenProvider(monotonic_clock=lambda: clock[0])
    assert await provider.access_token() == "temporary-token"
    monkeypatch.delenv("AURA_AZURE_DT_CLIENT_SECRET")
    clock[0] += 61.0

    with pytest.raises(AzureDigitalTwinsConnectorError, match="client_secret_invalid"):
        await provider.access_token()
    assert calls == 1


@pytest.mark.asyncio
async def test_entra_token_refresh_is_singleflight_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AURA_AZURE_DT_TENANT_ID",
        "11111111-1111-4111-8111-111111111111",
    )
    monkeypatch.setenv(
        "AURA_AZURE_DT_CLIENT_ID",
        "22222222-2222-4222-8222-222222222222",
    )
    monkeypatch.setenv("AURA_AZURE_DT_CLIENT_SECRET", "concurrent-secret")
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def request(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return {
            "ok": True,
            "status_code": 200,
            "headers": {},
            "content": json.dumps(
                {"access_token": "singleflight-token", "expires_in": 3600}
            ).encode(),
        }

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    provider = EntraClientCredentialsTokenProvider()
    tasks = [asyncio.create_task(provider.access_token()) for _ in range(12)]
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    release.set()
    tokens = await asyncio.gather(*tasks)

    assert tokens == ["singleflight-token"] * 12
    assert calls == 1


@pytest.mark.asyncio
async def test_azure_transport_rejects_non_mapping_network_gateway_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)

    async def request(**_kwargs: Any) -> Any:
        return b"not-a-network-response"

    monkeypatch.setattr(
        azure_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    transport = AzureDigitalTwinsScalarTransport((_resource(),), _TokenProvider())

    with pytest.raises(AzureDigitalTwinsConnectorError, match="non_mapping"):
        await transport.read_scalar("pump.temperature")


def test_azure_catalog_reports_partial_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AURA_AZURE_DT_INSTANCE_ID",
        "AURA_AZURE_DT_RESOURCES_JSON",
        "AURA_AZURE_DT_TENANT_ID",
        "AURA_AZURE_DT_CLIENT_ID",
        "AURA_AZURE_DT_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "AURA_AZURE_DT_ENDPOINT",
        "https://aura-lab.api.westus2.digitaltwins.azure.net",
    )

    status = build_configured_reality_connector_catalog().status()
    azure = next(
        item for item in status["connectors"] if item["connector_id"] == "azure.digital_twins"
    )

    assert status["ready"] is False
    assert azure["configured"] is True
    assert azure["state"] == "invalid"
    assert "AURA_AZURE_DT_CLIENT_SECRET" in azure["error"]
    assert "super-private" not in azure["error"]


def test_azure_timestamp_fallback_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch)
    document = json.loads(_twin())
    document["$metadata"]["telemetry"]["temperature"]["lastUpdateTime"] = "invalid"
    started = time.time_ns()
    captured, source = azure_module._captured_at_ns(
        document,
        ("telemetry", "temperature"),
    )
    assert captured >= started
    assert source == "system.time_ns.response_receipt"
