from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from typing import Any

import pytest

import core.embodiment.aws_twinmaker_connector as twinmaker_module
from core.embodiment.aws_twinmaker_connector import (
    AwsCredentials,
    AwsTwinMakerConnector,
    AwsTwinMakerConnectorError,
    AwsTwinMakerResourceSpec,
    AwsTwinMakerScalarTransport,
    EnvironmentAwsCredentialProvider,
    parse_aws_twinmaker_manifest,
    sign_aws_v4_headers,
)
from core.embodiment.reality_connectors import (
    build_configured_reality_connector_catalog,
)
from core.reality_reach.attachments import AttachmentAccess
from core.reality_reach.contracts import NumericDomain

_WORKSPACE_ARN = "arn:aws:iottwinmaker:us-west-2:123456789012:workspace/aura-lab"
_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


class _Credentials:
    def credentials(self) -> AwsCredentials:
        return AwsCredentials(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key=_SECRET,
            session_token="temporary-session-token",
        )


def _resource(*, writable: bool = True) -> AwsTwinMakerResourceSpec:
    return AwsTwinMakerResourceSpec(
        resource_id="pump.temperature",
        entity_id="pump-alpha",
        component_name="Telemetry",
        component_path="Plant/Pump",
        observable="temperature",
        unit="celsius",
        reported_property="temperatureReported",
        desired_property="temperatureDesired" if writable else "",
        value_type="double",
        domain=NumericDomain(-40.0, 180.0),
        resolution=0.1,
        tolerance=0.5,
        uncertainty=0.2,
        safe_value=20.0 if writable else None,
    )


def _workspace() -> dict[str, Any]:
    return {"arn": _WORKSPACE_ARN, "workspaceId": "aura-lab"}


def _history(value: float = 21.5) -> dict[str, Any]:
    return {
        "propertyValues": [
            {
                "entityPropertyReference": {
                    "entityId": "pump-alpha",
                    "componentName": "Telemetry",
                    "componentPath": "Plant/Pump",
                    "propertyName": "temperatureReported",
                },
                "values": [
                    {
                        "time": "2026-08-05T07:08:09.123456Z",
                        "value": {"doubleValue": value},
                    }
                ],
            }
        ]
    }


def _response(content: object, *, status: int = 200) -> dict[str, Any]:
    return {
        "ok": status < 400,
        "status_code": status,
        "headers": {"x-amzn-requestid": "request-123"},
        "content": json.dumps(content).encode(),
    }


def test_sigv4_matches_the_published_aws_s3_get_object_vector() -> None:
    headers = sign_aws_v4_headers(
        method="GET",
        url="https://examplebucket.s3.amazonaws.com/test.txt",
        payload=b"",
        headers={"Range": "bytes=0-9"},
        credentials=AwsCredentials(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key=_SECRET,
        ),
        region="us-east-1",
        service="s3",
        now=datetime(2013, 5, 24, tzinfo=UTC),
    )

    assert headers["Authorization"] == (
        "AWS4-HMAC-SHA256 "
        "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,"
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    )


def test_sigv4_binds_temporary_session_token_without_exposing_secret() -> None:
    headers = sign_aws_v4_headers(
        method="POST",
        url=(
            "https://data.iottwinmaker.us-west-2.amazonaws.com/"
            "workspaces/aura-lab/entity-properties/history"
        ),
        payload=b"{}",
        headers={"Content-Type": "application/json"},
        credentials=_Credentials().credentials(),
        region="us-west-2",
        service="iottwinmaker",
        now=datetime(2026, 8, 5, tzinfo=UTC),
    )

    assert headers["X-Amz-Security-Token"] == "temporary-session-token"
    assert "x-amz-security-token" in headers["Authorization"]
    assert _SECRET not in json.dumps(headers)


def test_environment_aws_provider_adopts_rotated_credentials_without_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EnvironmentAwsCredentialProvider()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7FIRST1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "first-secret-access-key-material")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "first-session-token")

    first = provider.credentials()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7SECOND")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "second-secret-access-key-material")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "second-session-token")
    second = provider.credentials()

    assert first.access_key_id == "AKIAIOSFODNN7FIRST1"
    assert first.session_token == "first-session-token"
    assert second.access_key_id == "AKIAIOSFODNN7SECOND"
    assert second.session_token == "second-session-token"
    assert first.identity_sha256 != second.identity_sha256
    serialized = json.dumps({"first": first.identity_sha256, "second": second.identity_sha256})
    assert "secret-access-key-material" not in serialized
    assert "session-token" not in serialized


def test_twinmaker_manifest_is_strict_typed_and_injection_closed() -> None:
    raw = [
        {
            "resource_id": "pump.temperature",
            "entity_id": "pump-alpha",
            "component_name": "Telemetry",
            "component_path": "Plant/Pump",
            "observable": "temperature",
            "unit": "celsius",
            "reported_property": "temperatureReported",
            "desired_property": "temperatureDesired",
            "value_type": "double",
            "minimum": -40,
            "maximum": 180,
            "resolution": 0.1,
            "tolerance": 0.5,
            "uncertainty": 0.2,
            "safe_value": 20,
        }
    ]
    first = parse_aws_twinmaker_manifest(json.dumps(raw))[0]
    second = parse_aws_twinmaker_manifest(raw)[0]

    assert first == second
    assert first.sha256 == second.sha256
    assert first.decode({"doubleValue": 21.5}) == 21.5
    assert first.encode(24.0) == {"doubleValue": 24.0}
    with pytest.raises(ValueError, match="distinct desired and reported"):
        parse_aws_twinmaker_manifest([dict(raw[0], desired_property="temperatureReported")])
    with pytest.raises(ValueError, match="canonical AWS TwinMaker identifier"):
        parse_aws_twinmaker_manifest([dict(raw[0], entity_id="pump/other")])
    with pytest.raises(AwsTwinMakerConnectorError, match="duplicate_json_key"):
        parse_aws_twinmaker_manifest('[{"resource_id":"a","resource_id":"b"}]')
    with pytest.raises(AwsTwinMakerConnectorError, match="ambiguous"):
        first.decode({"doubleValue": 1.0, "integerValue": 1})


@pytest.mark.asyncio
async def test_transport_verifies_workspace_then_reads_timestamped_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    responses = deque([_response(_workspace()), _response(_history())])

    async def request(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return responses.popleft()

    monkeypatch.setattr(
        twinmaker_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    transport = AwsTwinMakerScalarTransport(
        (_resource(),),
        _Credentials(),
        workspace_arn=_WORKSPACE_ARN,
    )

    sample = await transport.read_scalar("pump.temperature")

    assert sample.value == 21.5
    assert sample.uncertainty == 0.2
    assert sample.quality == "device_reported_cloud_twin"
    assert sample.wall_clock_source == "aws_iot_twinmaker.property_value.time"
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"].endswith("/workspaces/aura-lab")
    assert calls[0]["read_only"] is True
    assert calls[1]["method"] == "POST"
    assert calls[1]["read_only"] is True
    read_body = json.loads(calls[1]["data"])
    assert read_body == {
        "componentName": "Telemetry",
        "componentPath": "Plant/Pump",
        "entityId": "pump-alpha",
        "maxResults": 1,
        "orderByTime": "DESCENDING",
        "selectedProperties": ["temperatureReported"],
    }
    assert _SECRET not in json.dumps(
        {
            "source_event_id": sample.source_event_id,
            "source_epoch": sample.source_epoch,
        }
    )


@pytest.mark.asyncio
async def test_transport_writes_distinct_desired_property_without_claiming_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    responses = deque(
        [_response(_workspace()), _response(_history()), _response({"errorEntries": []})]
    )

    async def request(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return responses.popleft()

    monkeypatch.setattr(
        twinmaker_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    transport = AwsTwinMakerScalarTransport(
        (_resource(),),
        _Credentials(),
        workspace_arn=_WORKSPACE_ARN,
    )
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
    write = calls[2]
    assert write["read_only"] is False
    body = json.loads(write["data"])
    reference = body["entries"][0]["entityPropertyReference"]
    assert reference["propertyName"] == "temperatureDesired"
    assert body["entries"][0]["propertyValues"][0]["value"] == {"doubleValue": 24.0}
    receipt = json.dumps(dict(first.receipt))
    assert _SECRET not in receipt
    assert "pump-alpha" not in receipt
    assert "temperatureDesired" not in receipt


@pytest.mark.asyncio
async def test_transport_rejects_workspace_property_and_batch_identity_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def wrong_workspace(**_kwargs: Any) -> dict[str, Any]:
        return _response(
            {
                "arn": ("arn:aws:iottwinmaker:us-west-2:999999999999:workspace/aura-lab"),
                "workspaceId": "aura-lab",
            }
        )

    monkeypatch.setattr(
        twinmaker_module.ActionExecutor,
        "request_network_transport",
        wrong_workspace,
    )
    transport = AwsTwinMakerScalarTransport(
        (_resource(),),
        _Credentials(),
        workspace_arn=_WORKSPACE_ARN,
    )
    with pytest.raises(AwsTwinMakerConnectorError, match="identity_mismatch"):
        await transport.read_scalar("pump.temperature")

    wrong_history = _history()
    wrong_history["propertyValues"][0]["entityPropertyReference"]["propertyName"] = (
        "temperatureDesired"
    )
    responses = deque([_response(_workspace()), _response(wrong_history)])

    async def wrong_property(**_kwargs: Any) -> dict[str, Any]:
        return responses.popleft()

    monkeypatch.setattr(
        twinmaker_module.ActionExecutor,
        "request_network_transport",
        wrong_property,
    )
    transport = AwsTwinMakerScalarTransport(
        (_resource(),),
        _Credentials(),
        workspace_arn=_WORKSPACE_ARN,
    )
    with pytest.raises(AwsTwinMakerConnectorError, match="property_identity_mismatch"):
        await transport.read_scalar("pump.temperature")

    responses = deque(
        [
            _response(_workspace()),
            _response(_history()),
            _response({"errorEntries": [{"errors": [{"errorCode": "Rejected"}]}]}),
        ]
    )

    async def batch_error(**_kwargs: Any) -> dict[str, Any]:
        return responses.popleft()

    monkeypatch.setattr(
        twinmaker_module.ActionExecutor,
        "request_network_transport",
        batch_error,
    )
    transport = AwsTwinMakerScalarTransport(
        (_resource(),),
        _Credentials(),
        workspace_arn=_WORKSPACE_ARN,
    )
    await transport.read_scalar("pump.temperature")
    with pytest.raises(AwsTwinMakerConnectorError, match="batch_error"):
        await transport.write_scalar(
            "pump.temperature",
            24.0,
            idempotency_key="rejected",
        )


@pytest.mark.asyncio
async def test_connector_exposes_observe_and_verified_control_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def request(**kwargs: Any) -> dict[str, Any]:
        if kwargs["url"].endswith("/workspaces/aura-lab"):
            return _response(_workspace())
        return _response(_history())

    monkeypatch.setattr(
        twinmaker_module.ActionExecutor,
        "request_network_transport",
        request,
    )
    resource = _resource()
    connector = AwsTwinMakerConnector(
        AwsTwinMakerScalarTransport(
            (resource,),
            _Credentials(),
            workspace_arn=_WORKSPACE_ARN,
        ),
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


def test_catalog_reports_partial_twinmaker_configuration_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AURA_AWS_TWINMAKER_RESOURCES_JSON",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AURA_AWS_TWINMAKER_WORKSPACE_ARN", _WORKSPACE_ARN)

    status = build_configured_reality_connector_catalog().status()
    twinmaker = next(
        item for item in status["connectors"] if item["connector_id"] == "aws.iot_twinmaker"
    )

    assert status["ready"] is False
    assert twinmaker["configured"] is True
    assert twinmaker["state"] == "invalid"
    assert "AWS_SECRET_ACCESS_KEY" in twinmaker["error"]
    assert _SECRET not in twinmaker["error"]


def test_generic_aws_credentials_do_not_enable_twinmaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AURA_AWS_TWINMAKER_WORKSPACE_ARN", raising=False)
    monkeypatch.delenv("AURA_AWS_TWINMAKER_RESOURCES_JSON", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", _SECRET)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "temporary-session-token")

    status = build_configured_reality_connector_catalog().status()
    twinmaker = next(
        item for item in status["connectors"] if item["connector_id"] == "aws.iot_twinmaker"
    )

    assert twinmaker["configured"] is False
    assert twinmaker["state"] == "not_configured"


def test_reference_identity_requires_exact_composite_component_path() -> None:
    spec = _resource(writable=False)
    reference = {
        "entityId": spec.entity_id,
        "componentName": spec.component_name,
        "componentPath": "Plant/OtherPump",
        "propertyName": spec.reported_property,
    }
    assert (
        twinmaker_module._reference_matches(
            reference,
            spec,
            spec.reported_property,
        )
        is False
    )


def test_workspace_arn_partition_and_region_are_bound() -> None:
    resource = _resource(writable=False)
    transport = AwsTwinMakerScalarTransport(
        (resource,),
        _Credentials(),
        workspace_arn=("arn:aws-cn:iottwinmaker:cn-north-1:123456789012:workspace/w"),
    )
    assert transport.instance_identity_sha256.startswith("sha256:")

    with pytest.raises(AwsTwinMakerConnectorError, match="workspace_arn_invalid"):
        AwsTwinMakerScalarTransport(
            (resource,),
            _Credentials(),
            workspace_arn=("arn:aws:iottwinmaker:us-west-2:123456789012:workspace/aura/lab"),
        )
