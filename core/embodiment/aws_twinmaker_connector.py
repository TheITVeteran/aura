"""Manifest-bound AWS IoT TwinMaker sensing and verified desired-state writes.

TwinMaker is a remote representation. A successful BatchPutPropertyValues call
proves only that AWS accepted a desired-state sample. Writable resources must
therefore bind a distinct device-reported property, whose fresh time-series
sample is verified by the shared scalar effect coordinator.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Never, Protocol, runtime_checkable

from core.governance_context import (
    get_active_governance,
    governance_runtime_active,
    governed_scope,
    local_internal_decision,
    require_governance,
)
from core.reality_reach.attachments import AttachmentAccess, DeviceCandidate
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.live import LiveChannelAdapter
from core.reality_reach.scalar_adapter import (
    ScalarRealityAdapter,
    ScalarResourceProfile,
    ScalarSample,
    ScalarWriteResult,
)
from core.runtime.action_executor import ActionExecutor
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock

_SERVICE = "iottwinmaker"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_WORKSPACE_ARN = re.compile(
    r"^arn:(aws|aws-cn|aws-us-gov):iottwinmaker:([a-z0-9-]+):([0-9]{12}):workspace/([A-Za-z_0-9](?:[A-Za-z_0-9-]{0,126}[A-Za-z0-9])?)$"
)
_ENTITY_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z_0-9.:-]{0,126}[A-Za-z0-9])?$")
_COMPONENT_NAME = re.compile(r"^[A-Za-z_0-9-]{1,256}$")
_COMPONENT_PATH = re.compile(r"^[A-Za-z_0-9/-]{1,2048}$")
_PROPERTY_NAME = re.compile(r"^[A-Za-z_0-9-]{1,256}$")
_REGION = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z0-9-]+-[0-9]+$")
_ACCESS_KEY = re.compile(r"^[A-Z0-9]{16,128}$")
_VALUE_TYPES = frozenset({"boolean", "double", "integer", "long"})
_MAX_JSON_BYTES = 512 * 1024
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_IDEMPOTENCY_RECEIPTS = 4096
_CONTROL_DOMAINS = ("environment_action", "external_action", "tool_execution")


class AwsTwinMakerConnectorError(RuntimeError):
    """A TwinMaker identity, payload, signature, or effect contract failed."""


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hmac(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _bounded_match(value: object, *, name: str, pattern: re.Pattern[str]) -> str:
    normalized = str(value or "").strip()
    if not pattern.fullmatch(normalized):
        raise ValueError(f"{name} is not a canonical AWS TwinMaker identifier")
    return normalized


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _strict_json(payload: bytes, *, role: str) -> Any:
    if len(payload) > _MAX_JSON_BYTES:
        raise AwsTwinMakerConnectorError(f"aws_twinmaker_{role}_too_large")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AwsTwinMakerConnectorError(f"aws_twinmaker_{role}_duplicate_json_key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> Never:
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except AwsTwinMakerConnectorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AwsTwinMakerConnectorError(f"aws_twinmaker_{role}_invalid_json") from exc


def _workspace_identity(arn: object) -> tuple[str, str, str, str, str]:
    normalized = str(arn or "").strip()
    match = _WORKSPACE_ARN.fullmatch(normalized)
    if match is None:
        raise AwsTwinMakerConnectorError("aws_twinmaker_workspace_arn_invalid")
    partition, region, account_id, workspace_id = match.groups()
    if not _REGION.fullmatch(region):
        raise AwsTwinMakerConnectorError("aws_twinmaker_region_invalid")
    return normalized, partition, region, account_id, workspace_id


def _endpoint_suffix(partition: str) -> str:
    return "amazonaws.com.cn" if partition == "aws-cn" else "amazonaws.com"


def _canonical_query(query: str) -> str:
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True, strict_parsing=False)
    encoded = [
        (
            urllib.parse.quote(key, safe="-_.~"),
            urllib.parse.quote(value, safe="-_.~"),
        )
        for key, value in pairs
    ]
    return "&".join(f"{key}={value}" for key, value in sorted(encoded))


def _canonical_headers(headers: Mapping[str, str]) -> tuple[str, str]:
    normalized: dict[str, str] = {}
    for key, value in headers.items():
        name = str(key).strip().lower()
        if not name or any(character.isspace() for character in name):
            raise ValueError("AWS signing header name is invalid")
        collapsed = " ".join(str(value).strip().split())
        if "\n" in collapsed or "\r" in collapsed:
            raise ValueError("AWS signing header value is invalid")
        normalized[name] = collapsed
    ordered = sorted(normalized.items())
    return (
        "".join(f"{name}:{value}\n" for name, value in ordered),
        ";".join(name for name, _value in ordered),
    )


@dataclass(frozen=True, slots=True)
class AwsCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str = ""

    def __post_init__(self) -> None:
        access_key = str(self.access_key_id or "").strip()
        secret = str(self.secret_access_key or "").strip()
        token = str(self.session_token or "").strip()
        if not _ACCESS_KEY.fullmatch(access_key):
            raise AwsTwinMakerConnectorError("aws_access_key_id_invalid")
        if not 16 <= len(secret.encode("utf-8")) <= 4096:
            raise AwsTwinMakerConnectorError("aws_secret_access_key_invalid")
        if len(token.encode("utf-8")) > 16_384:
            raise AwsTwinMakerConnectorError("aws_session_token_invalid")
        object.__setattr__(self, "access_key_id", access_key)
        object.__setattr__(self, "secret_access_key", secret)
        object.__setattr__(self, "session_token", token)

    @property
    def identity_sha256(self) -> str:
        return _digest({"access_key_id": self.access_key_id})


@runtime_checkable
class AwsCredentialProvider(Protocol):
    def credentials(self) -> AwsCredentials: ...


class EnvironmentAwsCredentialProvider:
    """Read every request so externally rotated environment credentials work."""

    def credentials(self) -> AwsCredentials:
        return AwsCredentials(
            access_key_id=str(os.getenv("AWS_ACCESS_KEY_ID") or ""),
            secret_access_key=str(os.getenv("AWS_SECRET_ACCESS_KEY") or ""),
            session_token=str(os.getenv("AWS_SESSION_TOKEN") or ""),
        )


def sign_aws_v4_headers(
    *,
    method: str,
    url: str,
    payload: bytes,
    headers: Mapping[str, str],
    credentials: AwsCredentials,
    region: str,
    service: str,
    now: datetime,
) -> dict[str, str]:
    """Return deterministic SigV4 headers for one fully buffered request."""

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("AWS SigV4 URL must be an authenticated HTTPS origin")
    if parsed.port not in {None, 443}:
        raise ValueError("AWS SigV4 URL must use the default HTTPS port")
    normalized_method = str(method or "").strip().upper()
    if not re.fullmatch(r"[A-Z]+", normalized_method):
        raise ValueError("AWS SigV4 method is invalid")
    if not _REGION.fullmatch(str(region or "")):
        raise ValueError("AWS SigV4 region is invalid")
    service_name = str(service or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9-]{1,64}", service_name):
        raise ValueError("AWS SigV4 service is invalid")
    instant = now.astimezone(UTC)
    amz_date = instant.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = instant.strftime("%Y%m%d")
    payload_hash = _sha256(payload)
    signed = {str(key): str(value) for key, value in headers.items()}
    signed["Host"] = parsed.hostname.lower()
    signed["X-Amz-Content-Sha256"] = payload_hash
    signed["X-Amz-Date"] = amz_date
    if credentials.session_token:
        signed["X-Amz-Security-Token"] = credentials.session_token
    canonical_headers, signed_headers = _canonical_headers(signed)
    canonical_uri = urllib.parse.quote(
        urllib.parse.unquote(parsed.path or "/"),
        safe="/-_.~",
    )
    canonical_request = "\n".join(
        (
            normalized_method,
            canonical_uri,
            _canonical_query(parsed.query),
            canonical_headers,
            signed_headers,
            payload_hash,
        )
    )
    scope = f"{date_stamp}/{region}/{service_name}/aws4_request"
    string_to_sign = "\n".join(
        ("AWS4-HMAC-SHA256", amz_date, scope, _sha256(canonical_request.encode()))
    )
    date_key = _hmac(("AWS4" + credentials.secret_access_key).encode(), date_stamp)
    region_key = _hmac(date_key, region)
    service_key = _hmac(region_key, service_name)
    signing_key = _hmac(service_key, "aws4_request")
    signature = hmac.new(
        signing_key,
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    signed["Authorization"] = (
        "AWS4-HMAC-SHA256 "
        f"Credential={credentials.access_key_id}/{scope},"
        f"SignedHeaders={signed_headers},Signature={signature}"
    )
    return signed


@dataclass(frozen=True, slots=True)
class AwsTwinMakerResourceSpec:
    resource_id: str
    entity_id: str
    component_name: str
    observable: str
    unit: str
    reported_property: str
    domain: NumericDomain
    resolution: float
    component_path: str = ""
    desired_property: str = ""
    value_type: str = "double"
    safe_value: float | None = None
    tolerance: float | None = None
    uncertainty: float | None = None
    max_commands_per_minute: int = 12
    cooldown_s: float = 0.0
    stale_after_s: float = 60.0

    def __post_init__(self) -> None:
        for name in ("resource_id", "observable", "unit"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        for name, pattern in (
            ("entity_id", _ENTITY_ID),
            ("component_name", _COMPONENT_NAME),
            ("reported_property", _PROPERTY_NAME),
        ):
            object.__setattr__(
                self,
                name,
                _bounded_match(getattr(self, name), name=name, pattern=pattern),
            )
        component_path = str(self.component_path or "").strip()
        if component_path:
            component_path = _bounded_match(
                component_path,
                name="component_path",
                pattern=_COMPONENT_PATH,
            )
        object.__setattr__(self, "component_path", component_path)
        desired = str(self.desired_property or "").strip()
        if desired:
            desired = _bounded_match(
                desired,
                name="desired_property",
                pattern=_PROPERTY_NAME,
            )
            if desired == self.reported_property:
                raise ValueError(
                    "writable TwinMaker resources require distinct desired and reported properties"
                )
        object.__setattr__(self, "desired_property", desired)
        value_type = str(self.value_type or "double").strip().lower()
        if value_type not in _VALUE_TYPES:
            raise ValueError("value_type is not a supported TwinMaker scalar type")
        object.__setattr__(self, "value_type", value_type)
        if not isinstance(self.domain, NumericDomain):
            raise TypeError("domain must be NumericDomain")
        resolution = _finite(self.resolution, name="resolution")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        object.__setattr__(self, "resolution", resolution)
        tolerance = (
            resolution
            if self.tolerance is None
            else _finite(
                self.tolerance,
                name="tolerance",
            )
        )
        if tolerance < resolution:
            raise ValueError("tolerance must not be smaller than resolution")
        object.__setattr__(self, "tolerance", tolerance)
        if self.uncertainty is not None:
            uncertainty = _finite(self.uncertainty, name="uncertainty")
            if uncertainty < 0.0:
                raise ValueError("uncertainty must be non-negative")
            object.__setattr__(self, "uncertainty", uncertainty)
        if self.safe_value is not None:
            safe = _finite(self.safe_value, name="safe_value")
            if not desired or not self.domain.contains(safe):
                raise ValueError("safe_value requires a writable in-domain resource")
            object.__setattr__(self, "safe_value", safe)
        if not 1 <= int(self.max_commands_per_minute) <= 600:
            raise ValueError("max_commands_per_minute must lie inside [1, 600]")
        cooldown = _finite(self.cooldown_s, name="cooldown_s")
        stale = _finite(self.stale_after_s, name="stale_after_s")
        if cooldown < 0.0 or not 0.1 <= stale <= 86_400.0:
            raise ValueError("TwinMaker timing bounds are invalid")
        object.__setattr__(self, "cooldown_s", cooldown)
        object.__setattr__(self, "stale_after_s", stale)

    @property
    def writable(self) -> bool:
        return bool(self.desired_property)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "entity_id": self.entity_id,
            "component_name": self.component_name,
            "component_path": self.component_path,
            "observable": self.observable,
            "unit": self.unit,
            "reported_property": self.reported_property,
            "desired_property": self.desired_property,
            "value_type": self.value_type,
            "domain": self.domain.to_dict(),
            "resolution": self.resolution,
            "safe_value": self.safe_value,
            "tolerance": self.tolerance,
            "uncertainty": self.uncertainty,
            "max_commands_per_minute": self.max_commands_per_minute,
            "cooldown_s": self.cooldown_s,
            "stale_after_s": self.stale_after_s,
        }

    def decode(self, value: object) -> float:
        if not isinstance(value, Mapping) or len(value) != 1:
            raise AwsTwinMakerConnectorError("aws_twinmaker_data_value_ambiguous")
        expected_key = {
            "boolean": "booleanValue",
            "double": "doubleValue",
            "integer": "integerValue",
            "long": "longValue",
        }[self.value_type]
        if expected_key not in value:
            raise AwsTwinMakerConnectorError("aws_twinmaker_data_value_type_mismatch")
        raw = value[expected_key]
        if self.value_type == "boolean":
            if not isinstance(raw, bool):
                raise AwsTwinMakerConnectorError("aws_twinmaker_boolean_value_type_mismatch")
            number = 1.0 if raw else 0.0
        elif isinstance(raw, bool):
            raise AwsTwinMakerConnectorError("aws_twinmaker_numeric_value_type_mismatch")
        else:
            number = _finite(raw, name="TwinMaker reported property")
        if self.value_type in {"integer", "long"} and float(int(number)) != number:
            raise AwsTwinMakerConnectorError("aws_twinmaker_integer_value_not_integral")
        if not self.domain.contains(number):
            raise AwsTwinMakerConnectorError(
                "aws_twinmaker_reported_property_outside_manifest_domain"
            )
        return number

    def encode(self, value: float) -> dict[str, bool | float | int]:
        number = _finite(value, name="TwinMaker desired property")
        if not self.domain.contains(number):
            raise AwsTwinMakerConnectorError(
                "aws_twinmaker_desired_property_outside_manifest_domain"
            )
        if self.value_type == "boolean":
            if number not in {0.0, 1.0}:
                raise AwsTwinMakerConnectorError(
                    "aws_twinmaker_boolean_command_requires_zero_or_one"
                )
            return {"booleanValue": bool(number)}
        if self.value_type in {"integer", "long"}:
            integer = int(number)
            if float(integer) != number:
                raise AwsTwinMakerConnectorError(
                    "aws_twinmaker_integer_command_requires_integer_value"
                )
            key = "integerValue" if self.value_type == "integer" else "longValue"
            return {key: integer}
        return {"doubleValue": number}


def parse_aws_twinmaker_manifest(raw: object) -> tuple[AwsTwinMakerResourceSpec, ...]:
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > _MAX_MANIFEST_BYTES:
            raise AwsTwinMakerConnectorError("aws_twinmaker_manifest_too_large")
        raw = _strict_json(raw.encode(), role="manifest")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise AwsTwinMakerConnectorError("aws_twinmaker_manifest_must_be_a_list")
    if not 1 <= len(raw) <= 512:
        raise AwsTwinMakerConnectorError("aws_twinmaker_manifest_size_invalid")
    resources: list[AwsTwinMakerResourceSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise AwsTwinMakerConnectorError("aws_twinmaker_manifest_entry_invalid")
        resources.append(
            AwsTwinMakerResourceSpec(
                resource_id=str(item.get("resource_id") or ""),
                entity_id=str(item.get("entity_id") or ""),
                component_name=str(item.get("component_name") or ""),
                component_path=str(item.get("component_path") or ""),
                observable=str(item.get("observable") or ""),
                unit=str(item.get("unit") or ""),
                reported_property=str(item.get("reported_property") or ""),
                desired_property=str(item.get("desired_property") or ""),
                value_type=str(item.get("value_type") or "double"),
                domain=NumericDomain(
                    _finite(item.get("minimum"), name="minimum"),
                    _finite(item.get("maximum"), name="maximum"),
                ),
                resolution=_finite(item.get("resolution"), name="resolution"),
                safe_value=(
                    None
                    if item.get("safe_value") is None
                    else _finite(item.get("safe_value"), name="safe_value")
                ),
                tolerance=(
                    None
                    if item.get("tolerance") is None
                    else _finite(item.get("tolerance"), name="tolerance")
                ),
                uncertainty=(
                    None
                    if item.get("uncertainty") is None
                    else _finite(item.get("uncertainty"), name="uncertainty")
                ),
                max_commands_per_minute=int(item.get("max_commands_per_minute") or 12),
                cooldown_s=_finite(item.get("cooldown_s") or 0.0, name="cooldown_s"),
                stale_after_s=_finite(
                    item.get("stale_after_s") or 60.0,
                    name="stale_after_s",
                ),
            )
        )
    identities = [
        (
            item.entity_id,
            item.component_name,
            item.component_path,
            item.reported_property,
        )
        for item in resources
    ]
    if len(set(identities)) != len(identities):
        raise AwsTwinMakerConnectorError("aws_twinmaker_reported_property_duplicate")
    if len({item.resource_id for item in resources}) != len(resources):
        raise AwsTwinMakerConnectorError("aws_twinmaker_resource_id_duplicate")
    return tuple(sorted(resources, key=lambda item: item.resource_id))


def _parse_time_ns(value: object) -> int:
    if not isinstance(value, str) or not 20 <= len(value) <= 35:
        raise AwsTwinMakerConnectorError("aws_twinmaker_sample_time_missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        timestamp = int(parsed.timestamp() * 1_000_000_000)
    except (OverflowError, ValueError) as exc:
        raise AwsTwinMakerConnectorError("aws_twinmaker_sample_time_invalid") from exc
    if timestamp <= 0:
        raise AwsTwinMakerConnectorError("aws_twinmaker_sample_time_invalid")
    return timestamp


def _reference(spec: AwsTwinMakerResourceSpec, property_name: str) -> dict[str, str]:
    reference = {
        "entityId": spec.entity_id,
        "componentName": spec.component_name,
        "propertyName": property_name,
    }
    if spec.component_path:
        reference["componentPath"] = spec.component_path
    return reference


def _reference_matches(
    actual: object,
    spec: AwsTwinMakerResourceSpec,
    property_name: str,
) -> bool:
    if not isinstance(actual, Mapping):
        return False
    expected = _reference(spec, property_name)
    return all(
        str(actual.get(key) or "") == str(expected.get(key) or "")
        for key in ("entityId", "componentName", "componentPath", "propertyName")
    ) and not any(
        key in actual and actual.get(key) not in {None, ""} for key in ("externalIdProperty",)
    )


class AwsTwinMakerScalarTransport:
    """SigV4-authenticated TwinMaker time-series scalar transport."""

    transport_id = "aws.iot_twinmaker.rest"

    def __init__(
        self,
        resources: tuple[AwsTwinMakerResourceSpec, ...],
        credential_provider: AwsCredentialProvider,
        *,
        workspace_arn: str,
        workspace_verification_ttl_s: float = 300.0,
    ) -> None:
        if not resources or not isinstance(credential_provider, AwsCredentialProvider):
            raise TypeError("resources and credential_provider must satisfy contracts")
        (
            self._workspace_arn,
            self._partition,
            self._region,
            self._account_id,
            self._workspace_id,
        ) = _workspace_identity(workspace_arn)
        suffix = _endpoint_suffix(self._partition)
        self._control_endpoint = f"https://iottwinmaker.{self._region}.{suffix}"
        self._data_endpoint = f"https://data.iottwinmaker.{self._region}.{suffix}"
        self._resources = {item.resource_id: item for item in resources}
        self._credentials = credential_provider
        self._identity = _digest(
            {
                "workspace_arn": self._workspace_arn,
                "control_endpoint": self._control_endpoint,
                "data_endpoint": self._data_endpoint,
            }
        )
        ttl = _finite(workspace_verification_ttl_s, name="workspace_verification_ttl_s")
        if not 1.0 <= ttl <= 3600.0:
            raise ValueError("workspace verification TTL must lie inside [1, 3600]")
        self._workspace_verification_ttl_s = ttl
        self._workspace_verified_until = 0.0
        self._sequences: dict[str, int] = {}
        self._idempotency: dict[str, tuple[str, float, ScalarWriteResult]] = {}
        self._verification_lock = checked_async_lock("aws_twinmaker.workspace")
        self._write_lock = checked_async_lock("aws_twinmaker.transport")

    @property
    def instance_identity_sha256(self) -> str:
        return self._identity

    @property
    def identity_stable(self) -> bool:
        return True

    def _path(self, suffix: str) -> str:
        workspace = urllib.parse.quote(self._workspace_id, safe="-_.~")
        return f"/workspaces/{workspace}{suffix}"

    async def _request(
        self,
        *,
        method: str,
        endpoint: str,
        path: str,
        payload: bytes,
        read_only: bool,
    ) -> dict[str, Any]:
        credentials = self._credentials.credentials()
        url = endpoint + path
        base_headers = {"Content-Type": "application/json"}
        headers = sign_aws_v4_headers(
            method=method,
            url=url,
            payload=payload,
            headers=base_headers,
            credentials=credentials,
            region=self._region,
            service=_SERVICE,
            now=datetime.now(UTC),
        )
        result = await ActionExecutor.request_network_transport(
            method=method,
            url=url,
            headers=headers,
            data=payload or None,
            timeout_s=10.0,
            source="world_bridge:aws_iot_twinmaker",
            read_only=read_only,
        )
        if not isinstance(result, Mapping):
            raise AwsTwinMakerConnectorError("aws_twinmaker_network_gateway_returned_non_mapping")
        return dict(result)

    async def _ensure_workspace_verified(self) -> None:
        if time.monotonic() < self._workspace_verified_until:
            return
        async with self._verification_lock:
            if time.monotonic() < self._workspace_verified_until:
                return
            response = await self._request(
                method="GET",
                endpoint=self._control_endpoint,
                path=self._path(""),
                payload=b"",
                read_only=True,
            )
            if not response.get("ok") or int(response.get("status_code") or 0) != 200:
                raise AwsTwinMakerConnectorError("aws_twinmaker_workspace_verification_failed")
            document = _strict_json(
                bytes(response.get("content") or b""),
                role="workspace",
            )
            if not isinstance(document, Mapping) or (
                document.get("arn") != self._workspace_arn
                or document.get("workspaceId") != self._workspace_id
            ):
                raise AwsTwinMakerConnectorError("aws_twinmaker_workspace_identity_mismatch")
            self._workspace_verified_until = time.monotonic() + self._workspace_verification_ttl_s

    async def _read_governed(self, spec: AwsTwinMakerResourceSpec) -> ScalarSample:
        await self._ensure_workspace_verified()
        body: dict[str, Any] = {
            "entityId": spec.entity_id,
            "componentName": spec.component_name,
            "selectedProperties": [spec.reported_property],
            "maxResults": 1,
            "orderByTime": "DESCENDING",
        }
        if spec.component_path:
            body["componentPath"] = spec.component_path
        payload = canonical_json(body)
        response = await self._request(
            method="POST",
            endpoint=self._data_endpoint,
            path=self._path("/entity-properties/history"),
            payload=payload,
            read_only=True,
        )
        if not response.get("ok") or int(response.get("status_code") or 0) != 200:
            raise AwsTwinMakerConnectorError("aws_twinmaker_property_read_failed")
        document = _strict_json(
            bytes(response.get("content") or b""),
            role="property_history",
        )
        histories = document.get("propertyValues") if isinstance(document, Mapping) else None
        if not isinstance(histories, list) or len(histories) != 1:
            raise AwsTwinMakerConnectorError("aws_twinmaker_property_history_ambiguous")
        history = histories[0]
        if not isinstance(history, Mapping) or not _reference_matches(
            history.get("entityPropertyReference"),
            spec,
            spec.reported_property,
        ):
            raise AwsTwinMakerConnectorError("aws_twinmaker_reported_property_identity_mismatch")
        values = history.get("values")
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], Mapping):
            raise AwsTwinMakerConnectorError("aws_twinmaker_property_value_ambiguous")
        item = values[0]
        value = spec.decode(item.get("value"))
        captured_at_ns = _parse_time_ns(item.get("time"))
        sequence = self._sequences.get(spec.resource_id, 0) + 1
        self._sequences[spec.resource_id] = sequence
        return ScalarSample(
            value=value,
            captured_at_ns=captured_at_ns,
            source_event_id=_digest(
                {
                    "instance": self._identity,
                    "entity": spec.entity_id,
                    "component": spec.component_name,
                    "component_path": spec.component_path,
                    "reported_property": spec.reported_property,
                    "captured_at_ns": captured_at_ns,
                    "value": value,
                }
            ),
            quality="device_reported_cloud_twin",
            uncertainty=spec.uncertainty,
            wall_clock_source="aws_iot_twinmaker.property_value.time",
            source_epoch=self._identity,
            source_sequence=sequence,
        )

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        spec = self._resources.get(resource_id)
        if spec is None:
            raise LookupError("aws_twinmaker_resource_not_bound")
        if get_active_governance() is None and governance_runtime_active():
            decision = local_internal_decision(
                "aws_iot_twinmaker.device_reported_readback",
                domain="environment_action",
                constraints={
                    "read_only": True,
                    "resource_sha256": _digest(resource_id),
                    "workspace_sha256": _digest(self._workspace_arn),
                },
            )
            async with governed_scope(decision):
                return await self._read_governed(spec)
        return await self._read_governed(spec)

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        spec = self._resources.get(resource_id)
        if spec is None or not spec.writable:
            raise PermissionError("aws_twinmaker_resource_not_writable")
        encoded = spec.encode(value)
        target = _finite(value, name="TwinMaker desired property")
        key = str(idempotency_key or "").strip()
        if not key or len(key.encode("utf-8")) > 256:
            raise ValueError("aws_twinmaker_idempotency_key_invalid")
        require_governance(
            f"aws_iot_twinmaker.write_scalar:{resource_id}",
            strict=True,
            allowed_domains=_CONTROL_DOMAINS,
        )
        await self._ensure_workspace_verified()
        async with self._write_lock:
            previous = self._idempotency.get(key)
            if previous is not None:
                old_resource, old_value, result = previous
                if old_resource != resource_id or old_value != target:
                    raise AwsTwinMakerConnectorError("aws_twinmaker_idempotency_key_conflict")
                return result
            now = (
                datetime.now(UTC)
                .isoformat(timespec="microseconds")
                .replace(
                    "+00:00",
                    "Z",
                )
            )
            payload = canonical_json(
                {
                    "entries": [
                        {
                            "entityPropertyReference": _reference(
                                spec,
                                spec.desired_property,
                            ),
                            "propertyValues": [{"time": now, "value": encoded}],
                        }
                    ]
                }
            )
            response = await self._request(
                method="POST",
                endpoint=self._data_endpoint,
                path=self._path("/entity-properties"),
                payload=payload,
                read_only=False,
            )
            status = int(response.get("status_code") or 0)
            if not response.get("ok") or status != 200:
                if status == 0 or status >= 500 or status in {408, 429}:
                    raise AwsTwinMakerConnectorError(
                        "aws_twinmaker_desired_property_effect_indeterminate"
                    )
                raise AwsTwinMakerConnectorError("aws_twinmaker_desired_property_rejected")
            document = _strict_json(
                bytes(response.get("content") or b"{}"),
                role="batch_put",
            )
            errors = document.get("errorEntries") if isinstance(document, Mapping) else None
            if errors is not None and errors != []:
                raise AwsTwinMakerConnectorError("aws_twinmaker_desired_property_batch_error")
            result = ScalarWriteResult(
                accepted=True,
                transport_completed=True,
                receipt={
                    "protocol": self.transport_id,
                    "resource_id": resource_id,
                    "instance_identity_sha256": self._identity,
                    "workspace_arn_sha256": _digest(self._workspace_arn),
                    "entity_id_sha256": _digest(spec.entity_id),
                    "component_name_sha256": _digest(spec.component_name),
                    "desired_property_sha256": _digest(spec.desired_property),
                    "target_sha256": _digest(target),
                    "idempotency_sha256": _digest(key),
                    "recovery": recovery,
                    "transport_status": status,
                    "effect_verified": False,
                },
            )
            if len(self._idempotency) >= _MAX_IDEMPOTENCY_RECEIPTS:
                self._idempotency.pop(next(iter(self._idempotency)))
            self._idempotency[key] = (resource_id, target, result)
            return result


class AwsTwinMakerConnector:
    """Discover and attach declared AWS IoT TwinMaker properties."""

    connector_id = "aws.iot_twinmaker"

    def __init__(
        self,
        transport: AwsTwinMakerScalarTransport,
        resources: tuple[AwsTwinMakerResourceSpec, ...],
        *,
        candidate_ttl_s: float = 180.0,
        discovery_budget_s: float = 30.0,
    ) -> None:
        if not resources:
            raise ValueError("resources must not be empty")
        self._transport = transport
        self._resources = {item.resource_id: item for item in resources}
        self._ttl_s = max(
            30.0,
            min(_finite(candidate_ttl_s, name="candidate_ttl_s"), 3600.0),
        )
        budget = _finite(discovery_budget_s, name="discovery_budget_s")
        if not 0.01 <= budget <= 300.0:
            raise ValueError("discovery_budget_s must lie inside [0.01, 300]")
        self._discovery_budget_s = budget

    def _profile(self, spec: AwsTwinMakerResourceSpec) -> ScalarResourceProfile:
        return ScalarResourceProfile(
            resource_id=spec.resource_id,
            observable=spec.observable,
            unit=spec.unit,
            domain=spec.domain,
            resolution=spec.resolution,
            writable=spec.writable,
            physical_identity_sha256=_digest(
                {
                    "instance": self._transport.instance_identity_sha256,
                    "entity": spec.entity_id,
                    "component": spec.component_name,
                    "component_path": spec.component_path,
                    "reported_property": spec.reported_property,
                    "desired_property": spec.desired_property,
                }
            ),
            owner="core.embodiment.aws_twinmaker_connector",
            protocol="aws_iot_twinmaker",
            safe_value=spec.safe_value,
            tolerance=spec.tolerance,
            max_commands_per_minute=spec.max_commands_per_minute,
            cooldown_s=spec.cooldown_s,
            stale_after_s=spec.stale_after_s,
            readback_distinct_from_command=spec.writable,
        )

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        candidates: list[DeviceCandidate] = []
        now_ns = max(1, time.time_ns())
        deadline = time.monotonic() + self._discovery_budget_s
        for spec in self._resources.values():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                async with asyncio.timeout(remaining):
                    sample = await self._transport.read_scalar(spec.resource_id)
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                continue
            profile = self._profile(spec)
            if not profile.domain.contains(sample.value):
                continue
            manifest = _digest(
                {
                    "spec_sha256": spec.sha256,
                    "profile_sha256": profile.sha256,
                    "instance_identity_sha256": self._transport.instance_identity_sha256,
                }
            )
            access = (
                (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
                if profile.writable
                else (AttachmentAccess.OBSERVE,)
            )
            candidates.append(
                DeviceCandidate(
                    candidate_id="aws.twinmaker.candidate." + manifest.removeprefix("sha256:")[:32],
                    connector_id=self.connector_id,
                    device_id=f"aws.twinmaker.{spec.resource_id}",
                    display_name=(
                        f"AWS TwinMaker {spec.entity_id}/{spec.component_name}: {spec.observable}"
                    )[:160],
                    transport=self._transport.transport_id,
                    identity_fingerprint=profile.physical_identity_sha256,
                    manifest_sha256=manifest,
                    access=access,
                    discovered_at_ns=now_ns,
                    expires_at_ns=now_ns + int(self._ttl_s * 1_000_000_000),
                    persistent_identity=True,
                    proposal_salience=0.3,
                    metadata={
                        "resource_id": spec.resource_id,
                        "entity_id_sha256": _digest(spec.entity_id),
                        "component_name_sha256": _digest(spec.component_name),
                        "spec_sha256": spec.sha256,
                        "profile_sha256": profile.sha256,
                        "control_available": profile.writable,
                        "independent_device_reported_readback": spec.writable,
                    },
                )
            )
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> LiveChannelAdapter:
        if candidate.connector_id != self.connector_id:
            raise ValueError("aws_twinmaker_candidate_connector_mismatch")
        requested = set(access)
        if not requested or not requested.issubset(set(candidate.access)):
            raise PermissionError("aws_twinmaker_attachment_access_not_declared")
        if AttachmentAccess.CONTROL in requested and AttachmentAccess.OBSERVE not in requested:
            raise PermissionError("aws_twinmaker_control_requires_observation")
        resource_id = str(candidate.metadata.get("resource_id") or "")
        spec = self._resources.get(resource_id)
        if spec is None:
            raise LookupError("aws_twinmaker_candidate_resource_missing")
        current = next(
            (item for item in await self.discover() if item.candidate_id == candidate.candidate_id),
            None,
        )
        if current is None or (
            current.identity_fingerprint != candidate.identity_fingerprint
            or current.manifest_sha256 != candidate.manifest_sha256
        ):
            raise RuntimeError("aws_twinmaker_candidate_changed_before_attachment")
        profile = self._profile(spec)
        if AttachmentAccess.CONTROL not in requested:
            profile = replace(profile, writable=False, safe_value=None)
        sample = await self._transport.read_scalar(resource_id)
        return ScalarRealityAdapter(self._transport, profile, initial_sample=sample)

    async def detach(self, _adapter: LiveChannelAdapter) -> None:
        return None


def build_configured_aws_twinmaker_connector() -> AwsTwinMakerConnector:
    workspace_arn = str(os.getenv("AURA_AWS_TWINMAKER_WORKSPACE_ARN") or "").strip()
    raw = str(os.getenv("AURA_AWS_TWINMAKER_RESOURCES_JSON") or "").strip()
    if not raw:
        raise AwsTwinMakerConnectorError("aws_twinmaker_resource_manifest_missing")
    resources = parse_aws_twinmaker_manifest(raw)
    provider = EnvironmentAwsCredentialProvider()
    provider.credentials()
    return AwsTwinMakerConnector(
        AwsTwinMakerScalarTransport(
            resources,
            provider,
            workspace_arn=workspace_arn,
        ),
        resources,
    )


__all__ = [
    "AwsCredentialProvider",
    "AwsCredentials",
    "AwsTwinMakerConnector",
    "AwsTwinMakerConnectorError",
    "AwsTwinMakerResourceSpec",
    "AwsTwinMakerScalarTransport",
    "EnvironmentAwsCredentialProvider",
    "build_configured_aws_twinmaker_connector",
    "parse_aws_twinmaker_manifest",
    "sign_aws_v4_headers",
]
