"""Manifest-bound ROS 2 sensing, services, and verified actions.

This adapter speaks the rosbridge 2.1 JSON protocol without importing a ROS
distribution into Aura's process.  A bounded manifest is the authority-neutral
description of one robot.  The attachment broker still owns trust, and the
Reality Middleware runtime still owns lifecycle, QoS, idempotency, deadlines,
cancellation, and restart reconciliation.

Action transport success is deliberately insufficient.  Every declared action
must name a separate read-only verification service and predicate.  Aura only
reports a successful physical action after that independent readback matches.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import urllib.parse
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from core.bus.qos import Durability, History, QosProfile, Reliability
from core.reality_reach.attachments import AttachmentAccess, DeviceCandidate
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.live import ChannelReading, LiveChannelAdapter, ReadingStatus
from core.reality_reach.middleware_contracts import (
    ActionContext,
    ActionEndpoint,
    ActionState,
    ManagedAdapterDeclaration,
    PhysicalEffectIndeterminateError,
    RestartPolicy,
    ServiceEndpoint,
    TelemetryEndpoint,
    TelemetryMode,
    bounded_payload,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.errors import NetworkEffectDenied
from core.runtime.lockdep import checked_async_lock
from core.runtime.network_gateway import get_network_gateway
from core.utils.task_tracker import get_task_tracker

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_ROS_NAME = re.compile(r"^/(?:[A-Za-z0-9_]+/?)+$")
_ROS_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*/(?:(?:msg|srv|action)/)?[A-Za-z][A-Za-z0-9_]*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TERMINAL_ACTION_STATUS = frozenset({4, 5, 6})
_MAX_WIRE_BYTES = 1_048_576
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_PENDING = 256
logger = logging.getLogger("Aura.RealityReach.ROS2")


class ROS2ConnectorError(RuntimeError):
    """A ROS graph, transport, payload, or effect contract failed."""


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _ros_name(value: object, *, name: str) -> str:
    normalized = str(value or "").strip()
    if not _ROS_NAME.fullmatch(normalized) or "//" in normalized:
        raise ValueError(f"{name} must be an absolute ROS graph name")
    return normalized.rstrip("/") or "/"


def _ros_type(value: object, *, name: str, category: str) -> str:
    normalized = str(value or "").strip()
    if not _ROS_TYPE.fullmatch(normalized):
        raise ValueError(f"{name} must be a ROS interface type")
    parts = normalized.split("/")
    if len(parts) == 3 and parts[1] != category:
        raise ValueError(f"{name} must name a ROS {category} interface")
    return normalized


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _json_pointer(value: object, *, name: str) -> str:
    pointer = str(value or "").strip()
    if not pointer.startswith("/") or len(pointer.encode("utf-8")) > 512:
        raise ValueError(f"{name} must be a bounded JSON pointer")
    for segment in pointer.split("/")[1:]:
        if "~" in re.sub(r"~[01]", "", segment):
            raise ValueError(f"{name} contains an invalid JSON pointer escape")
    return pointer


def _pointer_get(document: object, pointer: str) -> object:
    current = document
    for raw in pointer.split("/")[1:]:
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if key not in current:
                raise ROS2ConnectorError(f"ros_payload_field_missing:{pointer}")
            current = current[key]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            if not key.isdigit() or int(key) >= len(current):
                raise ROS2ConnectorError(f"ros_payload_index_missing:{pointer}")
            current = current[int(key)]
        else:
            raise ROS2ConnectorError(f"ros_payload_path_not_traversable:{pointer}")
    return current


def _bounded_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return bounded_payload(value, name=name, maximum=_MAX_WIRE_BYTES)


def _manifest_mapping(item: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = item.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ROS2ConnectorError(f"ros2_manifest_{key}_must_be_an_object")
    return value


def _qos_from_manifest(value: object) -> QosProfile:
    raw = {} if value is None else _bounded_mapping(value, name="ROS QoS")
    reliability = str(raw.get("reliability") or "reliable").lower()
    durability = str(raw.get("durability") or "volatile").lower()
    history = str(raw.get("history") or "keep_last").lower()
    try:
        profile = QosProfile(
            reliability=Reliability[reliability.upper()],
            durability=Durability[durability.upper()],
            history=History(history),
            depth=int(raw.get("depth") or 10),
            lifespan_s=_finite(raw.get("lifespan_s") or 0.0, name="lifespan_s"),
            deadline_s=_finite(raw.get("deadline_s") or 0.0, name="deadline_s"),
            liveliness_lease_s=_finite(
                raw.get("liveliness_lease_s") or 0.0,
                name="liveliness_lease_s",
            ),
        )
    except (KeyError, ValueError) as exc:
        raise ValueError("ROS QoS policy is invalid") from exc
    if not 1 <= profile.depth <= 4096:
        raise ValueError("ROS QoS depth must lie inside [1, 4096]")
    for timing in (
        profile.lifespan_s,
        profile.deadline_s,
        profile.liveliness_lease_s,
    ):
        if not 0.0 <= timing <= 86_400.0:
            raise ValueError("ROS QoS timing must lie inside [0, 86400]")
    return profile


def _rosbridge_qos(profile: QosProfile) -> dict[str, Any]:
    return {
        "reliability": profile.reliability.name.lower(),
        "durability": profile.durability.name.lower(),
        "history": profile.history.value,
        "depth": profile.depth,
    }


@dataclass(frozen=True, slots=True)
class ROS2TelemetrySpec:
    endpoint_id: str
    channel_id: str
    topic: str
    message_type: str
    value_pointer: str
    observable: str
    unit: str
    domain: NumericDomain
    resolution: float
    qos: QosProfile
    sample_period_s: float = 0.5
    stale_after_s: float = 5.0
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        for name in ("endpoint_id", "channel_id", "observable", "unit"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        object.__setattr__(self, "topic", _ros_name(self.topic, name="topic"))
        object.__setattr__(
            self,
            "message_type",
            _ros_type(self.message_type, name="message_type", category="msg"),
        )
        object.__setattr__(
            self,
            "value_pointer",
            _json_pointer(self.value_pointer, name="value_pointer"),
        )
        if not isinstance(self.domain, NumericDomain):
            raise TypeError("domain must be a NumericDomain")
        resolution = _finite(self.resolution, name="resolution")
        period = _finite(self.sample_period_s, name="sample_period_s")
        stale = _finite(self.stale_after_s, name="stale_after_s")
        if resolution <= 0.0 or not 0.05 <= period <= 3600.0 or stale < period:
            raise ValueError("ROS telemetry timing or resolution is invalid")
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "sample_period_s", period)
        object.__setattr__(self, "stale_after_s", stale)
        if self.uncertainty is not None:
            uncertainty = _finite(self.uncertainty, name="uncertainty")
            if uncertainty < 0.0:
                raise ValueError("uncertainty must be non-negative")
            object.__setattr__(self, "uncertainty", uncertainty)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "channel_id": self.channel_id,
            "topic": self.topic,
            "message_type": self.message_type,
            "value_pointer": self.value_pointer,
            "observable": self.observable,
            "unit": self.unit,
            "domain": self.domain.to_dict(),
            "resolution": self.resolution,
            "qos": self.qos.to_dict(),
            "sample_period_s": self.sample_period_s,
            "stale_after_s": self.stale_after_s,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class ROS2ServiceSpec:
    endpoint_id: str
    service: str
    service_type: str
    read_only: bool = True
    timeout_s: float = 10.0
    max_inflight: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.read_only, bool):
            raise TypeError("read_only must be boolean")
        if not self.read_only:
            raise ValueError("state-changing ROS services must be declared as verified actions")
        object.__setattr__(self, "endpoint_id", _identifier(self.endpoint_id, name="endpoint_id"))
        object.__setattr__(self, "service", _ros_name(self.service, name="service"))
        object.__setattr__(
            self,
            "service_type",
            _ros_type(self.service_type, name="service_type", category="srv"),
        )
        timeout = _finite(self.timeout_s, name="timeout_s")
        if not 0.05 <= timeout <= 300.0 or not 1 <= int(self.max_inflight) <= 64:
            raise ValueError("ROS service bounds are invalid")
        object.__setattr__(self, "timeout_s", timeout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "service": self.service,
            "service_type": self.service_type,
            "read_only": self.read_only,
            "timeout_s": self.timeout_s,
            "max_inflight": self.max_inflight,
        }


@dataclass(frozen=True, slots=True)
class ROS2ActionSpec:
    endpoint_id: str
    verification_service: str
    verification_service_type: str
    verification_request: Mapping[str, Any]
    verification_pointer: str
    verification_expected: Any
    transport_kind: str = "action"
    action: str = ""
    action_type: str = ""
    command_service: str = ""
    command_service_type: str = ""
    timeout_s: float = 300.0
    cancel_timeout_s: float = 10.0
    preemptible: bool = True
    feedback_progress_pointer: str = ""
    reconciliation_service: str = ""
    reconciliation_service_type: str = ""
    reconciliation_state_pointer: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.preemptible, bool):
            raise TypeError("preemptible must be boolean")
        object.__setattr__(self, "endpoint_id", _identifier(self.endpoint_id, name="endpoint_id"))
        transport_kind = str(self.transport_kind or "action").strip().lower()
        if transport_kind not in {"action", "service"}:
            raise ValueError("ROS action transport_kind must be action or service")
        object.__setattr__(self, "transport_kind", transport_kind)
        action = str(self.action or "").strip()
        action_type = str(self.action_type or "").strip()
        command_service = str(self.command_service or "").strip()
        command_service_type = str(self.command_service_type or "").strip()
        if transport_kind == "action":
            action = _ros_name(action, name="action")
            action_type = _ros_type(action_type, name="action_type", category="action")
            if command_service or command_service_type:
                raise ValueError("native ROS actions cannot also declare a command service")
        else:
            command_service = _ros_name(command_service, name="command_service")
            command_service_type = _ros_type(
                command_service_type,
                name="command_service_type",
                category="srv",
            )
            if action or action_type:
                raise ValueError("ROS service actions cannot also declare a native action")
            if self.preemptible:
                raise ValueError("ROS service actions cannot claim preemption")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "command_service", command_service)
        object.__setattr__(self, "command_service_type", command_service_type)
        object.__setattr__(
            self,
            "verification_service",
            _ros_name(self.verification_service, name="verification_service"),
        )
        object.__setattr__(
            self,
            "verification_service_type",
            _ros_type(
                self.verification_service_type,
                name="verification_service_type",
                category="srv",
            ),
        )
        object.__setattr__(
            self,
            "verification_request",
            _bounded_mapping(self.verification_request, name="verification_request"),
        )
        object.__setattr__(
            self,
            "verification_pointer",
            _json_pointer(self.verification_pointer, name="verification_pointer"),
        )
        progress = str(self.feedback_progress_pointer or "").strip()
        if progress:
            progress = _json_pointer(progress, name="feedback_progress_pointer")
        object.__setattr__(self, "feedback_progress_pointer", progress)
        reconcile_service = str(self.reconciliation_service or "").strip()
        reconcile_type = str(self.reconciliation_service_type or "").strip()
        reconcile_pointer = str(self.reconciliation_state_pointer or "").strip()
        if any((reconcile_service, reconcile_type, reconcile_pointer)) and not all(
            (reconcile_service, reconcile_type, reconcile_pointer)
        ):
            raise ValueError("ROS action reconciliation configuration must be complete")
        if reconcile_service:
            reconcile_service = _ros_name(reconcile_service, name="reconciliation_service")
            reconcile_type = _ros_type(
                reconcile_type,
                name="reconciliation_service_type",
                category="srv",
            )
            reconcile_pointer = _json_pointer(
                reconcile_pointer,
                name="reconciliation_state_pointer",
            )
        object.__setattr__(self, "reconciliation_service", reconcile_service)
        object.__setattr__(self, "reconciliation_service_type", reconcile_type)
        object.__setattr__(self, "reconciliation_state_pointer", reconcile_pointer)
        timeout = _finite(self.timeout_s, name="timeout_s")
        cancel = _finite(self.cancel_timeout_s, name="cancel_timeout_s")
        if not 0.1 <= timeout <= 86_400.0 or not 0.05 <= cancel <= 300.0:
            raise ValueError("ROS action timing bounds are invalid")
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "cancel_timeout_s", cancel)

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "transport_kind": self.transport_kind,
            "action": self.action,
            "action_type": self.action_type,
            "command_service": self.command_service,
            "command_service_type": self.command_service_type,
            "verification_service": self.verification_service,
            "verification_service_type": self.verification_service_type,
            "verification_request": dict(self.verification_request),
            "verification_pointer": self.verification_pointer,
            "verification_expected": self.verification_expected,
            "timeout_s": self.timeout_s,
            "cancel_timeout_s": self.cancel_timeout_s,
            "preemptible": self.preemptible,
            "feedback_progress_pointer": self.feedback_progress_pointer,
            "reconciliation_service": self.reconciliation_service,
            "reconciliation_service_type": self.reconciliation_service_type,
            "reconciliation_state_pointer": self.reconciliation_state_pointer,
        }


@dataclass(frozen=True, slots=True)
class ROS2NodeSpec:
    node_id: str
    device_id: str
    display_name: str
    telemetry: tuple[ROS2TelemetrySpec, ...]
    services: tuple[ROS2ServiceSpec, ...] = ()
    actions: tuple[ROS2ActionSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identifier(self.node_id, name="node_id"))
        object.__setattr__(self, "device_id", _identifier(self.device_id, name="device_id"))
        if not str(self.display_name or "").strip() or len(self.display_name) > 160:
            raise ValueError("display_name must be present and bounded")
        if not self.telemetry:
            raise ValueError("a ROS device requires at least one measurable telemetry channel")
        endpoint_ids = [
            *(item.endpoint_id for item in self.telemetry),
            *(item.endpoint_id for item in self.services),
            *(item.endpoint_id for item in self.actions),
        ]
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("ROS endpoint ids must be unique")
        if len({item.channel_id for item in self.telemetry}) != len(self.telemetry):
            raise ValueError("ROS telemetry channel ids must be unique")
        if len({item.topic for item in self.telemetry}) != len(self.telemetry):
            raise ValueError("ROS telemetry topics must be unique")

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "device_id": self.device_id,
            "display_name": self.display_name,
            "telemetry": [item.to_dict() for item in self.telemetry],
            "services": [item.to_dict() for item in self.services],
            "actions": [item.to_dict() for item in self.actions],
        }


def parse_ros2_node_manifest(raw: object) -> ROS2NodeSpec:
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > _MAX_MANIFEST_BYTES:
            raise ROS2ConnectorError("ros2_manifest_too_large")
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ROS2ConnectorError("ros2_manifest_invalid_json") from exc
    body = _bounded_mapping(raw, name="ROS node manifest")
    telemetry_raw = body.get("telemetry") or []
    services_raw = body.get("services") or []
    actions_raw = body.get("actions") or []
    for name, value in (
        ("telemetry", telemetry_raw),
        ("services", services_raw),
        ("actions", actions_raw),
    ):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ROS2ConnectorError(f"ros2_manifest_{name}_must_be_a_list")
        if len(value) > 512:
            raise ROS2ConnectorError(f"ros2_manifest_{name}_too_large")
    telemetry = tuple(
        ROS2TelemetrySpec(
            endpoint_id=str(item.get("endpoint_id") or ""),
            channel_id=str(item.get("channel_id") or ""),
            topic=str(item.get("topic") or ""),
            message_type=str(item.get("message_type") or ""),
            value_pointer=str(item.get("value_pointer") or ""),
            observable=str(item.get("observable") or ""),
            unit=str(item.get("unit") or ""),
            domain=NumericDomain(
                _finite(item.get("minimum"), name="minimum"),
                _finite(item.get("maximum"), name="maximum"),
            ),
            resolution=_finite(item.get("resolution"), name="resolution"),
            qos=_qos_from_manifest(item.get("qos")),
            sample_period_s=_finite(
                item.get("sample_period_s") or 0.5,
                name="sample_period_s",
            ),
            stale_after_s=_finite(
                item.get("stale_after_s") or 5.0,
                name="stale_after_s",
            ),
            uncertainty=(
                None
                if item.get("uncertainty") is None
                else _finite(item.get("uncertainty"), name="uncertainty")
            ),
        )
        for item in telemetry_raw
        if isinstance(item, Mapping)
    )
    if len(telemetry) != len(telemetry_raw):
        raise ROS2ConnectorError("ros2_manifest_telemetry_entry_invalid")
    services = tuple(
        ROS2ServiceSpec(
            endpoint_id=str(item.get("endpoint_id") or ""),
            service=str(item.get("service") or ""),
            service_type=str(item.get("service_type") or ""),
            read_only=item.get("read_only", True),
            timeout_s=_finite(item.get("timeout_s") or 10.0, name="timeout_s"),
            max_inflight=int(item.get("max_inflight") or 1),
        )
        for item in services_raw
        if isinstance(item, Mapping)
    )
    if len(services) != len(services_raw):
        raise ROS2ConnectorError("ros2_manifest_service_entry_invalid")
    actions = tuple(
        ROS2ActionSpec(
            endpoint_id=str(item.get("endpoint_id") or ""),
            verification_service=str(item.get("verification_service") or ""),
            verification_service_type=str(item.get("verification_service_type") or ""),
            verification_request=_manifest_mapping(item, "verification_request"),
            verification_pointer=str(item.get("verification_pointer") or ""),
            verification_expected=item.get("verification_expected"),
            transport_kind=str(item.get("transport_kind") or "action"),
            action=str(item.get("action") or ""),
            action_type=str(item.get("action_type") or ""),
            command_service=str(item.get("command_service") or ""),
            command_service_type=str(item.get("command_service_type") or ""),
            timeout_s=_finite(item.get("timeout_s") or 300.0, name="timeout_s"),
            cancel_timeout_s=_finite(
                item.get("cancel_timeout_s") or 10.0,
                name="cancel_timeout_s",
            ),
            preemptible=item.get("preemptible", True),
            feedback_progress_pointer=str(item.get("feedback_progress_pointer") or ""),
            reconciliation_service=str(item.get("reconciliation_service") or ""),
            reconciliation_service_type=str(item.get("reconciliation_service_type") or ""),
            reconciliation_state_pointer=str(item.get("reconciliation_state_pointer") or ""),
        )
        for item in actions_raw
        if isinstance(item, Mapping)
    )
    if len(actions) != len(actions_raw):
        raise ROS2ConnectorError("ros2_manifest_action_entry_invalid")
    return ROS2NodeSpec(
        node_id=str(body.get("node_id") or ""),
        device_id=str(body.get("device_id") or ""),
        display_name=str(body.get("display_name") or ""),
        telemetry=telemetry,
        services=services,
        actions=actions,
    )


@dataclass(frozen=True, slots=True)
class ROSTopicSample:
    topic: str
    message: Mapping[str, Any]
    captured_at_ns: int
    source_sequence: int


@dataclass(frozen=True, slots=True)
class ROSGraphSnapshot:
    topics: Mapping[str, str]
    services: Mapping[str, str]


@runtime_checkable
class ROS2Transport(Protocol):
    @property
    def transport_id(self) -> str: ...

    @property
    def server_identity_sha256(self) -> str: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def subscribe(self, spec: ROS2TelemetrySpec) -> None: ...

    async def unsubscribe(self, spec: ROS2TelemetrySpec) -> None: ...

    async def latest(self, spec: ROS2TelemetrySpec, *, timeout_s: float) -> ROSTopicSample: ...

    async def graph_snapshot(self, spec: ROS2NodeSpec) -> ROSGraphSnapshot: ...

    async def call_service(
        self,
        service: str,
        service_type: str,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
        request_id: str | None = None,
    ) -> Mapping[str, Any]: ...

    async def send_action_goal(
        self,
        spec: ROS2ActionSpec,
        goal_id: str,
        request: Mapping[str, Any],
    ) -> None: ...

    async def next_action_event(self, goal_id: str, *, timeout_s: float) -> Mapping[str, Any]: ...

    async def cancel_action_goal(
        self,
        spec: ROS2ActionSpec,
        goal_id: str,
        *,
        timeout_s: float,
    ) -> bool: ...


class RosbridgeWebSocketTransport:
    """One bounded, pinned rosbridge WebSocket session with correlation."""

    transport_id = "rosbridge.v2.1"

    def __init__(self) -> None:
        url = str(os.getenv("AURA_ROSBRIDGE_URL") or "").strip()
        installation = _identifier(
            os.getenv("AURA_ROSBRIDGE_INSTALLATION_ID"),
            name="AURA_ROSBRIDGE_INSTALLATION_ID",
        )
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme not in {"ws", "wss"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ROS2ConnectorError("rosbridge_url_invalid")
        if parsed.scheme == "ws" and not self._allow_plaintext():
            raise ROS2ConnectorError("rosbridge_plaintext_requires_explicit_opt_in")
        pin = str(os.getenv("AURA_ROSBRIDGE_SERVER_CERT_SHA256") or "").strip().lower()
        if parsed.scheme == "wss" and not _DIGEST.fullmatch(pin):
            raise ROS2ConnectorError("rosbridge_server_certificate_pin_required")
        version = str(os.getenv("AURA_ROSBRIDGE_PROTOCOL_VERSION") or "2.1.0").strip()
        try:
            version_tuple = tuple(int(part) for part in version.split("."))
        except ValueError as exc:
            raise ROS2ConnectorError("rosbridge_protocol_version_invalid") from exc
        if len(version_tuple) != 3 or version_tuple < (2, 1, 0):
            raise ROS2ConnectorError("rosbridge_protocol_2_1_required")
        self._url = url
        self._secure = parsed.scheme == "wss"
        self._certificate_pin = pin
        self._server_identity = _digest(
            {
                "installation": installation,
                "endpoint": f"{parsed.scheme}://{parsed.hostname}:{parsed.port or (443 if self._secure else 80)}{parsed.path or '/'}",
                "certificate_pin": pin,
            }
        )
        self._connect_lock = checked_async_lock("rosbridge.connect")
        self._send_lock = checked_async_lock("rosbridge.send")
        self._socket: Any | None = None
        self._receive_task: asyncio.Task[Any] | None = None
        self._service_futures: dict[str, asyncio.Future[Mapping[str, Any]]] = {}
        self._action_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._action_results: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._latest: dict[str, ROSTopicSample] = {}
        self._topic_events: dict[str, asyncio.Event] = {}
        self._subscriptions: dict[str, str] = {}
        self._sequence = 0

    @staticmethod
    def _allow_plaintext() -> bool:
        from core.runtime.flags import FlagKind, declare

        return str(
            declare(
                "AURA_ROSBRIDGE_ALLOW_PLAINTEXT",
                kind=FlagKind.STRING,
                default="",
                description="Permit a plaintext development rosbridge session",
                owner="core.embodiment.ros2_connector",
            ).value()
        ).strip().lower() in {"1", "true", "yes", "on"}

    @property
    def server_identity_sha256(self) -> str:
        return self._server_identity

    @property
    def identity_stable(self) -> bool:
        return self._secure

    def _authorization_headers(self) -> dict[str, str]:
        authorization = str(os.getenv("AURA_ROSBRIDGE_AUTHORIZATION") or "").strip()
        return {"Authorization": authorization} if authorization else {}

    async def connect(self) -> None:
        async with self._connect_lock:
            if (
                self._socket is not None
                and self._receive_task is not None
                and not self._receive_task.done()
            ):
                return
            try:
                admission = await get_network_gateway().connect_websocket(
                    self._url,
                    headers=self._authorization_headers(),
                    open_timeout=10.0,
                    close_timeout=5.0,
                    ping_interval=20.0,
                    ping_timeout=10.0,
                    max_size=_MAX_WIRE_BYTES,
                    max_queue=64,
                    source="reality_reach:ros2.rosbridge",
                    read_only=True,
                    allow_private_target=True,
                )
                socket = admission.connection
                if self._secure:
                    ssl_object = socket.transport.get_extra_info("ssl_object")
                    certificate = ssl_object.getpeercert(binary_form=True) if ssl_object else None
                    actual = (
                        "sha256:" + __import__("hashlib").sha256(certificate or b"").hexdigest()
                    )
                    if certificate is None or actual != self._certificate_pin:
                        await socket.close(code=1008, reason="certificate pin mismatch")
                        raise ROS2ConnectorError("rosbridge_server_certificate_pin_mismatch")
            except ROS2ConnectorError:
                raise
            except (ImportError, NetworkEffectDenied, OSError, RuntimeError, TimeoutError) as exc:
                raise ROS2ConnectorError("rosbridge_connect_failed") from exc
            self._socket = socket
            self._receive_task = get_task_tracker().create_task(
                self._receive_loop(socket),
                name="ROSBridgeReceive",
            )

    async def _send(self, body: Mapping[str, Any]) -> None:
        encoded = canonical_json(body)
        if len(encoded) > _MAX_WIRE_BYTES:
            raise ROS2ConnectorError("rosbridge_message_too_large")
        await self.connect()
        async with self._send_lock:
            socket = self._socket
            if socket is None:
                raise ROS2ConnectorError("rosbridge_not_connected")
            try:
                await socket.send(encoded.decode("utf-8"))
            except (OSError, RuntimeError) as exc:
                await self._invalidate(socket, exc)
                raise ROS2ConnectorError("rosbridge_send_failed") from exc

    async def _receive_loop(self, socket: Any) -> None:
        failure: BaseException | None = None
        try:
            async for raw in socket:
                if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_WIRE_BYTES:
                    raise ROS2ConnectorError("rosbridge_non_json_or_oversize_message")
                decoded = json.loads(raw)
                if not isinstance(decoded, Mapping):
                    raise ROS2ConnectorError("rosbridge_message_not_an_object")
                self._dispatch(dict(decoded))
        except asyncio.CancelledError:
            raise
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            failure = exc
        finally:
            await self._invalidate(socket, failure or ConnectionError("rosbridge_closed"))

    def _dispatch(self, body: dict[str, Any]) -> None:
        operation = str(body.get("op") or "")
        correlation_id = str(body.get("id") or "")
        if operation == "publish":
            topic = str(body.get("topic") or "")
            message = _bounded_mapping(body.get("msg"), name="ROS topic message")
            self._sequence += 1
            self._latest[topic] = ROSTopicSample(
                topic=topic,
                message=message,
                captured_at_ns=time.time_ns(),
                source_sequence=self._sequence,
            )
            self._topic_events.setdefault(topic, asyncio.Event()).set()
            return
        if operation == "service_response" and correlation_id:
            future = self._service_futures.pop(correlation_id, None)
            if future is not None and not future.done():
                if body.get("result") is True:
                    future.set_result(
                        _bounded_mapping(body.get("values"), name="ROS service response")
                    )
                else:
                    future.set_exception(ROS2ConnectorError("ros_service_call_failed"))
            return
        if operation == "action_feedback" and correlation_id:
            queue = self._action_queues.get(correlation_id)
            if queue is not None:
                self._put_action_event(queue, body)
            return
        if operation == "action_result" and correlation_id:
            future = self._action_results.get(correlation_id)
            if future is not None and not future.done():
                future.set_result(body)
            return
        if operation == "status" and str(body.get("level") or "").lower() in {"error", "fatal"}:
            error = ROS2ConnectorError(f"rosbridge_status_error:{str(body.get('msg') or '')[:240]}")
            future = self._service_futures.pop(correlation_id, None)
            if future is not None and not future.done():
                future.set_exception(error)
            queue = self._action_queues.get(correlation_id)
            if queue is not None:
                self._put_action_event(queue, {"op": "error", "error": str(error)})

    @staticmethod
    def _put_action_event(
        queue: asyncio.Queue[dict[str, Any]],
        event: dict[str, Any],
    ) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(event)

    @staticmethod
    def _fail_future(future: asyncio.Future[Any], error: BaseException) -> None:
        if future.done():
            return
        future.set_exception(error)
        # Mark the exception observed even when shutdown has no remaining waiter.
        future.exception()

    async def _invalidate(self, socket: Any, reason: BaseException) -> None:
        async with self._connect_lock:
            if socket is not self._socket:
                return
            self._socket = None
            self._receive_task = None
            self._subscriptions.clear()
            error = ROS2ConnectorError(f"rosbridge_connection_lost:{type(reason).__name__}")
            for future in self._service_futures.values():
                self._fail_future(future, error)
            self._service_futures.clear()
            for queue in self._action_queues.values():
                self._put_action_event(queue, {"op": "error", "error": str(error)})
            for future in self._action_results.values():
                self._fail_future(future, error)

    async def subscribe(self, spec: ROS2TelemetrySpec) -> None:
        prior = self._subscriptions.get(spec.topic)
        if prior is not None:
            if prior != spec.sha256:
                raise ROS2ConnectorError("ros_topic_redeclared_with_different_contract")
            return
        subscription_id = f"sub-{_digest(spec.to_dict()).removeprefix('sha256:')[:32]}"
        await self._send(
            {
                "op": "subscribe",
                "id": subscription_id,
                "topic": spec.topic,
                "type": spec.message_type,
                "qos": _rosbridge_qos(spec.qos),
                "throttle_rate": max(0, int(spec.sample_period_s * 1000)),
                "queue_length": max(1, min(spec.qos.depth, 1024)),
                "compression": "none",
            }
        )
        self._subscriptions[spec.topic] = spec.sha256

    async def unsubscribe(self, spec: ROS2TelemetrySpec) -> None:
        if spec.topic not in self._subscriptions:
            return
        subscription_id = f"sub-{_digest(spec.to_dict()).removeprefix('sha256:')[:32]}"
        await self._send({"op": "unsubscribe", "id": subscription_id, "topic": spec.topic})
        self._subscriptions.pop(spec.topic, None)

    async def latest(self, spec: ROS2TelemetrySpec, *, timeout_s: float) -> ROSTopicSample:
        await self.subscribe(spec)
        sample = self._latest.get(spec.topic)
        if sample is None or time.time_ns() - sample.captured_at_ns > int(spec.stale_after_s * 1e9):
            event = self._topic_events.setdefault(spec.topic, asyncio.Event())
            event.clear()
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_s)
            except TimeoutError as exc:
                raise ROS2ConnectorError("ros_topic_sample_timeout") from exc
            sample = self._latest.get(spec.topic)
        if sample is None:
            raise ROS2ConnectorError("ros_topic_sample_missing")
        return sample

    async def call_service(
        self,
        service: str,
        service_type: str,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
        request_id: str | None = None,
    ) -> Mapping[str, Any]:
        if len(self._service_futures) >= _MAX_PENDING:
            raise ROS2ConnectorError("ros_service_pending_limit_reached")
        correlation = _identifier(
            request_id or f"service-{uuid.uuid4().hex}",
            name="request_id",
        )
        if correlation in self._service_futures:
            raise ROS2ConnectorError("ros_service_request_id_in_flight")
        future: asyncio.Future[Mapping[str, Any]] = asyncio.get_running_loop().create_future()
        self._service_futures[correlation] = future
        try:
            await self._send(
                {
                    "op": "call_service",
                    "id": correlation,
                    "service": _ros_name(service, name="service"),
                    "type": _ros_type(service_type, name="service_type", category="srv"),
                    "args": _bounded_mapping(request, name="ROS service request"),
                    "timeout": float(timeout_s),
                }
            )
            return await asyncio.wait_for(future, timeout=float(timeout_s))
        except TimeoutError as exc:
            raise ROS2ConnectorError("ros_service_response_timeout") from exc
        finally:
            self._service_futures.pop(correlation, None)

    async def graph_snapshot(self, spec: ROS2NodeSpec) -> ROSGraphSnapshot:
        topics_result = await self.call_service(
            "/rosapi/topics",
            "rosapi_msgs/srv/Topics",
            {},
            timeout_s=5.0,
        )
        services_result = await self.call_service(
            "/rosapi/services",
            "rosapi_msgs/srv/Services",
            {},
            timeout_s=5.0,
        )
        topic_names = list(topics_result.get("topics") or [])
        topic_types = list(topics_result.get("types") or [])
        topics = {
            str(name): str(topic_types[index]) if index < len(topic_types) else ""
            for index, name in enumerate(topic_names)
        }
        service_names = [str(item) for item in list(services_result.get("services") or [])]
        required_services = {item.service for item in spec.services}
        required_services.update(item.verification_service for item in spec.actions)
        required_services.update(
            item.command_service for item in spec.actions if item.transport_kind == "service"
        )
        required_services.update(
            item.reconciliation_service for item in spec.actions if item.reconciliation_service
        )
        services: dict[str, str] = {}
        for service in sorted(required_services & set(service_names)):
            result = await self.call_service(
                "/rosapi/service_type",
                "rosapi_msgs/srv/ServiceType",
                {"service": service},
                timeout_s=5.0,
            )
            services[service] = str(result.get("type") or "")
        for action in spec.actions:
            for suffix in ("send_goal", "get_result", "cancel_goal"):
                name = f"{action.action}/_action/{suffix}".replace("//", "/")
                if name in service_names:
                    services[name] = "action_transport"
        return ROSGraphSnapshot(topics=topics, services=services)

    async def send_action_goal(
        self,
        spec: ROS2ActionSpec,
        goal_id: str,
        request: Mapping[str, Any],
    ) -> None:
        goal_id = _identifier(goal_id, name="goal_id")
        completed = [item for item, future in self._action_results.items() if future.done()]
        while len(self._action_results) >= _MAX_PENDING and completed:
            retired = completed.pop(0)
            self._action_results.pop(retired, None)
            self._action_queues.pop(retired, None)
        if goal_id in self._action_queues or len(self._action_results) >= _MAX_PENDING:
            raise ROS2ConnectorError("ros_action_goal_id_in_flight_or_limit_reached")
        self._action_queues[goal_id] = asyncio.Queue(maxsize=1024)
        self._action_results[goal_id] = asyncio.get_running_loop().create_future()
        try:
            await self._send(
                {
                    "op": "send_action_goal",
                    "id": goal_id,
                    "action": spec.action,
                    "action_type": spec.action_type,
                    "args": _bounded_mapping(request, name="ROS action request"),
                    "feedback": True,
                }
            )
        except BaseException:
            self._action_queues.pop(goal_id, None)
            self._action_results.pop(goal_id, None)
            raise

    async def next_action_event(self, goal_id: str, *, timeout_s: float) -> Mapping[str, Any]:
        queue = self._action_queues.get(_identifier(goal_id, name="goal_id"))
        result = self._action_results.get(goal_id)
        if queue is None or result is None:
            raise ROS2ConnectorError("ros_action_goal_unknown")
        if not queue.empty():
            return queue.get_nowait()
        if result.done():
            return result.result()
        feedback_task = get_task_tracker().create_task(
            queue.get(),
            name=f"ROS2Feedback:{goal_id}",
        )
        result_task = asyncio.shield(result)
        try:
            done, pending = await asyncio.wait(
                {feedback_task, result_task},
                timeout=float(timeout_s),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise ROS2ConnectorError("ros_action_event_timeout")
            if feedback_task in done:
                return feedback_task.result()
            return result_task.result()
        finally:
            for task in (feedback_task, result_task):
                if not task.done():
                    task.cancel()

    async def cancel_action_goal(
        self,
        spec: ROS2ActionSpec,
        goal_id: str,
        *,
        timeout_s: float,
    ) -> bool:
        goal_id = _identifier(goal_id, name="goal_id")
        result = self._action_results.get(goal_id)
        if goal_id not in self._action_queues or result is None:
            return False
        await self._send({"op": "cancel_action_goal", "id": goal_id, "action": spec.action})
        try:
            event = await asyncio.wait_for(asyncio.shield(result), timeout=float(timeout_s))
        except TimeoutError:
            return False
        return event.get("op") == "action_result" and int(event.get("status") or 0) == 5

    async def close(self) -> None:
        async with self._connect_lock:
            socket = self._socket
            task = self._receive_task
            error = ROS2ConnectorError("rosbridge_transport_closed")
            for future in self._service_futures.values():
                self._fail_future(future, error)
            for future in self._action_results.values():
                self._fail_future(future, error)
            self._service_futures.clear()
            self._socket = None
            self._receive_task = None
            self._subscriptions.clear()
            self._action_queues.clear()
            self._action_results.clear()
        if task is not None:
            task.cancel()
        if socket is not None:
            try:
                await socket.close(code=1000, reason="Aura ROS adapter closed")
            except (OSError, RuntimeError) as exc:
                logger.debug(
                    "ROS bridge close failed after local state was fenced: %s",
                    exc,
                )
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


class ROS2ManagedAdapter:
    """One attached robot as both a live sensor and managed physical node."""

    def __init__(
        self,
        transport: ROS2Transport,
        spec: ROS2NodeSpec,
        *,
        installation_id: str,
        initial_samples: Mapping[str, ROSTopicSample],
        physical_identity_sha256: str | None = None,
    ) -> None:
        if not isinstance(transport, ROS2Transport):
            raise TypeError("transport must satisfy ROS2Transport")
        self._transport = transport
        self._spec = spec
        self.adapter_id = _identifier(
            f"ros2.{installation_id}.{spec.device_id}",
            name="adapter_id",
        )
        identity = str(physical_identity_sha256 or "").strip() or _digest(
            {
                "server": transport.server_identity_sha256,
                "device_id": spec.device_id,
                "manifest": spec.sha256,
            }
        )
        if not _DIGEST.fullmatch(identity):
            raise ValueError("physical_identity_sha256 must be a sha256 digest")
        self.physical_identity_sha256 = identity
        self._telemetry = {item.endpoint_id: item for item in spec.telemetry}
        self._services = {item.endpoint_id: item for item in spec.services}
        self._actions = {item.endpoint_id: item for item in spec.actions}
        self._readings: dict[str, ChannelReading] = {}
        for item in spec.telemetry:
            sample = initial_samples.get(item.topic)
            if sample is not None:
                self._readings[item.channel_id] = self._decode(item, sample)

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return tuple(
            ChannelDeclaration(
                channel_id=item.channel_id,
                kind=ChannelKind.SENSOR,
                observable=item.observable,
                unit=item.unit,
                domain=item.domain,
                coupling=CouplingClass.NETWORK,
                reality_layers=(RealityLayer.EFFECTIVE,),
                evidence_level=EvidenceLevel.P2,
                owner="core.embodiment.ros2_connector",
                resolution=item.resolution,
                sample_rate_hz=1.0 / item.sample_period_s,
                max_latency_s=item.sample_period_s,
                stale_after_s=item.stale_after_s,
                reference_id=f"ros2.topic.{_digest(item.topic).removeprefix('sha256:')[:24]}",
                coupling_validated=True,
            )
            for item in self._spec.telemetry
        )

    def read(self) -> tuple[ChannelReading, ...]:
        return tuple(
            self._readings[item.channel_id]
            for item in self._spec.telemetry
            if item.channel_id in self._readings
        )

    def lifecycle_declaration(self) -> ManagedAdapterDeclaration:
        return ManagedAdapterDeclaration(
            node_id=self._spec.node_id,
            adapter_id=self.adapter_id,
            adapter_identity_sha256=self.physical_identity_sha256,
            telemetry=tuple(
                TelemetryEndpoint(
                    endpoint_id=item.endpoint_id,
                    channel_ids=(item.channel_id,),
                    qos=item.qos,
                    mode=TelemetryMode.PULL,
                    sample_period_s=item.sample_period_s,
                    sample_timeout_s=min(
                        item.sample_period_s, max(0.05, item.sample_period_s * 0.8)
                    ),
                )
                for item in self._spec.telemetry
            ),
            services=tuple(
                ServiceEndpoint(
                    endpoint_id=item.endpoint_id,
                    timeout_s=item.timeout_s,
                    max_inflight=item.max_inflight,
                    read_only=item.read_only,
                )
                for item in self._spec.services
            ),
            actions=tuple(
                ActionEndpoint(
                    endpoint_id=item.endpoint_id,
                    timeout_s=item.timeout_s,
                    cancel_timeout_s=item.cancel_timeout_s,
                    preemptible=item.preemptible,
                    restart_policy=RestartPolicy.RECONCILE,
                    requires_effect_verification=True,
                )
                for item in self._spec.actions
            ),
        )

    def _decode(self, spec: ROS2TelemetrySpec, sample: ROSTopicSample) -> ChannelReading:
        value = _finite(
            _pointer_get(sample.message, spec.value_pointer), name="ROS telemetry value"
        )
        if not spec.domain.contains(value):
            raise ROS2ConnectorError("ros_telemetry_outside_manifest_domain")
        return ChannelReading(
            channel_id=spec.channel_id,
            value=value,
            unit=spec.unit,
            captured_at_ns=sample.captured_at_ns,
            status=ReadingStatus.AVAILABLE,
            source=f"ros2:{spec.topic}",
            uncertainty=spec.uncertainty,
            source_sequence=sample.source_sequence,
            source_event_id=_digest(
                {
                    "topic": spec.topic,
                    "sequence": sample.source_sequence,
                    "message": dict(sample.message),
                }
            ),
            source_quality="rosbridge_live_topic",
        )

    async def on_configure(self) -> bool:
        await self._transport.connect()
        return True

    async def on_activate(self) -> bool:
        for spec in self._spec.telemetry:
            await self._transport.subscribe(spec)
        return True

    async def on_deactivate(self) -> bool:
        for spec in reversed(self._spec.telemetry):
            await self._transport.unsubscribe(spec)
        return True

    async def on_cleanup(self) -> bool:
        await self._transport.close()
        return True

    async def on_shutdown(self) -> bool:
        await self._transport.close()
        return True

    async def on_error(self) -> bool:
        await self._transport.close()
        return True

    async def read_telemetry(self, endpoint_id: str) -> tuple[ChannelReading, ...]:
        spec = self._telemetry.get(endpoint_id)
        if spec is None:
            raise LookupError(endpoint_id)
        sample = await self._transport.latest(spec, timeout_s=min(spec.sample_period_s, 5.0))
        reading = self._decode(spec, sample)
        self._readings[spec.channel_id] = reading
        return (reading,)

    async def handle_service(
        self, endpoint_id: str, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        spec = self._services.get(endpoint_id)
        if spec is None:
            raise LookupError(endpoint_id)
        return await self._transport.call_service(
            spec.service,
            spec.service_type,
            request,
            timeout_s=spec.timeout_s,
        )

    async def execute_action(
        self,
        endpoint_id: str,
        request: Mapping[str, Any],
        context: ActionContext,
    ) -> Mapping[str, Any]:
        spec = self._actions.get(endpoint_id)
        if spec is None:
            raise LookupError(endpoint_id)
        dispatched = False
        try:
            if spec.transport_kind == "service":
                await context.publish_feedback(0.25, {"phase": "command_submitted"})
                dispatched = True
                result = await self._transport.call_service(
                    spec.command_service,
                    spec.command_service_type,
                    request,
                    timeout_s=spec.timeout_s,
                    request_id=f"command-{context.goal_id}",
                )
            else:
                dispatched = True
                await self._transport.send_action_goal(spec, context.goal_id, request)
                while True:
                    event = await self._transport.next_action_event(
                        context.goal_id,
                        timeout_s=spec.timeout_s,
                    )
                    operation = str(event.get("op") or "")
                    if operation == "error":
                        raise ROS2ConnectorError(
                            str(event.get("error") or "ros_action_transport_error")
                        )
                    if operation == "action_feedback":
                        values = _bounded_mapping(
                            event.get("values"),
                            name="ROS action feedback",
                        )
                        progress = 0.0
                        if spec.feedback_progress_pointer:
                            progress = _finite(
                                _pointer_get(values, spec.feedback_progress_pointer),
                                name="ROS action progress",
                            )
                            if not 0.0 <= progress <= 1.0:
                                raise ROS2ConnectorError(
                                    "ros_action_progress_outside_unit_interval"
                                )
                        await context.publish_feedback(progress, values)
                        continue
                    if operation != "action_result":
                        continue
                    status = int(event.get("status") or 0)
                    if (
                        status not in _TERMINAL_ACTION_STATUS
                        or status != 4
                        or event.get("result") is not True
                    ):
                        raise ROS2ConnectorError(f"ros_action_not_successful:status={status}")
                    result = _bounded_mapping(
                        event.get("values"),
                        name="ROS action result",
                    )
                    break
            verification = await self._transport.call_service(
                spec.verification_service,
                spec.verification_service_type,
                spec.verification_request,
                timeout_s=min(spec.cancel_timeout_s, 30.0),
                request_id=f"verify-{context.goal_id}",
            )
            observed = _pointer_get(verification, spec.verification_pointer)
            if observed != spec.verification_expected:
                raise ROS2ConnectorError("ros_action_effect_verification_failed")
            effect_receipt = {
                "adapter_identity_sha256": self.physical_identity_sha256,
                "goal_id": context.goal_id,
                "transport_kind": spec.transport_kind,
                "operation": spec.action or spec.command_service,
                "verification_service": spec.verification_service,
                "verification_response": dict(verification),
            }
            await context.publish_feedback(1.0, {"phase": "effect_verified"})
            return {
                "effect_verified": True,
                "effect_receipt_sha256": _digest(effect_receipt),
                "result": dict(result),
                "verification": dict(verification),
            }
        except PhysicalEffectIndeterminateError:
            raise
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            if dispatched:
                raise PhysicalEffectIndeterminateError(
                    f"ros_physical_effect_unproven:{type(exc).__name__}:{exc}"
                ) from exc
            raise

    async def cancel_action(self, endpoint_id: str, goal_id: str, reason: str) -> bool:
        del reason
        spec = self._actions.get(endpoint_id)
        if spec is None:
            raise LookupError(endpoint_id)
        if spec.transport_kind == "service":
            return False
        return await self._transport.cancel_action_goal(
            spec,
            goal_id,
            timeout_s=spec.cancel_timeout_s,
        )

    async def reconcile_action(
        self,
        endpoint_id: str,
        record: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        spec = self._actions.get(endpoint_id)
        if spec is None:
            raise LookupError(endpoint_id)
        if not spec.reconciliation_service:
            return {
                "state": ActionState.INDETERMINATE.value,
                "error": "robot_does_not_expose_durable_goal_reconciliation",
                "result": {},
            }
        goal_id = _identifier(record.get("goal_id"), name="goal_id")
        response = await self._transport.call_service(
            spec.reconciliation_service,
            spec.reconciliation_service_type,
            {"goal_id": goal_id},
            timeout_s=spec.cancel_timeout_s,
            request_id=f"reconcile-{goal_id}",
        )
        state = str(_pointer_get(response, spec.reconciliation_state_pointer)).lower()
        try:
            claimed = ActionState(state)
        except ValueError as exc:
            raise ROS2ConnectorError("ros_action_reconciliation_state_invalid") from exc
        if not claimed.terminal:
            raise ROS2ConnectorError("ros_action_reconciliation_not_terminal")
        if claimed is not ActionState.SUCCEEDED:
            return {"state": claimed.value, "result": dict(response)}
        verification = await self._transport.call_service(
            spec.verification_service,
            spec.verification_service_type,
            spec.verification_request,
            timeout_s=spec.cancel_timeout_s,
            request_id=f"reconcile-verify-{goal_id}",
        )
        if _pointer_get(verification, spec.verification_pointer) != spec.verification_expected:
            return {
                "state": ActionState.INDETERMINATE.value,
                "error": "reconciled_success_effect_not_verified",
                "result": {},
            }
        receipt = _digest(
            {
                "adapter_identity_sha256": self.physical_identity_sha256,
                "goal_id": goal_id,
                "reconciliation": dict(response),
                "verification": dict(verification),
            }
        )
        return {
            "state": ActionState.SUCCEEDED.value,
            "result": {
                "effect_verified": True,
                "effect_receipt_sha256": receipt,
                "reconciled": True,
                "verification": dict(verification),
            },
        }


class ROS2Connector:
    """Discover and attach one manifest-declared ROS 2 physical node."""

    connector_id = "ros2.rosbridge"

    def __init__(
        self,
        transport: ROS2Transport,
        spec: ROS2NodeSpec,
        *,
        installation_id: str,
        candidate_ttl_s: float = 180.0,
        discovery_timeout_s: float = 5.0,
    ) -> None:
        if not isinstance(transport, ROS2Transport):
            raise TypeError("transport must satisfy ROS2Transport")
        self._transport = transport
        self._spec = spec
        self._installation_id = _identifier(installation_id, name="installation_id")
        self._ttl_s = max(30.0, min(float(candidate_ttl_s), 3600.0))
        self._discovery_timeout_s = max(0.1, min(float(discovery_timeout_s), 30.0))
        self._initial_samples: dict[str, ROSTopicSample] = {}

    def _identity(self) -> str:
        return _digest(
            {
                "server": self._transport.server_identity_sha256,
                "device_id": self._spec.device_id,
                "manifest": self._spec.sha256,
            }
        )

    @staticmethod
    def _action_services(action: str) -> set[str]:
        root = action.rstrip("/")
        return {
            f"{root}/_action/send_goal",
            f"{root}/_action/get_result",
            f"{root}/_action/cancel_goal",
        }

    async def _probe(self) -> ROSGraphSnapshot:
        await self._transport.connect()
        graph = await self._transport.graph_snapshot(self._spec)
        for telemetry in self._spec.telemetry:
            if graph.topics.get(telemetry.topic) != telemetry.message_type:
                raise ROS2ConnectorError(f"ros_topic_missing_or_type_mismatch:{telemetry.topic}")
        required_services = {item.service: item.service_type for item in self._spec.services}
        for action in self._spec.actions:
            required_services[action.verification_service] = action.verification_service_type
            if action.reconciliation_service:
                required_services[action.reconciliation_service] = (
                    action.reconciliation_service_type
                )
            if action.transport_kind == "service":
                required_services[action.command_service] = action.command_service_type
            elif not self._action_services(action.action).issubset(set(graph.services)):
                raise ROS2ConnectorError(f"ros_action_server_unavailable:{action.action}")
        for service, interface in required_services.items():
            if graph.services.get(service) != interface:
                raise ROS2ConnectorError(f"ros_service_missing_or_type_mismatch:{service}")
        samples: dict[str, ROSTopicSample] = {}
        for telemetry in self._spec.telemetry:
            samples[telemetry.topic] = await self._transport.latest(
                telemetry,
                timeout_s=self._discovery_timeout_s,
            )
        self._initial_samples = samples
        return graph

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        try:
            graph = await self._probe()
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            return ()
        now_ns = time.time_ns()
        manifest = _digest(
            {
                "node_spec_sha256": self._spec.sha256,
                "server_identity_sha256": self._transport.server_identity_sha256,
                "topics": dict(sorted(graph.topics.items())),
                "services": dict(sorted(graph.services.items())),
            }
        )
        control = bool(self._spec.actions)
        return (
            DeviceCandidate(
                candidate_id=f"ros2.candidate.{manifest.removeprefix('sha256:')[:32]}",
                connector_id=self.connector_id,
                device_id=f"ros2.{self._installation_id}.{self._spec.device_id}",
                display_name=self._spec.display_name,
                transport=self._transport.transport_id,
                identity_fingerprint=self._identity(),
                manifest_sha256=manifest,
                access=(
                    (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
                    if control
                    else (AttachmentAccess.OBSERVE,)
                ),
                discovered_at_ns=now_ns,
                expires_at_ns=now_ns + int(self._ttl_s * 1_000_000_000),
                persistent_identity=bool(getattr(self._transport, "identity_stable", True)),
                privacy_sensitive=True,
                proposal_salience=0.55,
                metadata={
                    "node_id": self._spec.node_id,
                    "node_spec_sha256": self._spec.sha256,
                    "telemetry_count": len(self._spec.telemetry),
                    "service_count": len(self._spec.services),
                    "action_count": len(self._spec.actions),
                    "verified_action_count": len(self._spec.actions),
                    "control_available": control,
                },
            ),
        )

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> LiveChannelAdapter:
        if candidate.connector_id != self.connector_id:
            raise ValueError("ros2_candidate_connector_mismatch")
        requested = set(access)
        if not requested or not requested.issubset(set(candidate.access)):
            raise PermissionError("ros2_attachment_access_not_declared")
        if AttachmentAccess.CONTROL in requested and AttachmentAccess.OBSERVE not in requested:
            raise PermissionError("ros2_control_requires_observation")
        current = next(
            (item for item in await self.discover() if item.candidate_id == candidate.candidate_id),
            None,
        )
        if current is None or (
            current.identity_fingerprint != candidate.identity_fingerprint
            or current.manifest_sha256 != candidate.manifest_sha256
        ):
            raise RuntimeError("ros2_candidate_changed_before_attachment")
        if AttachmentAccess.CONTROL not in requested and (self._spec.actions):
            services = tuple(item for item in self._spec.services if item.read_only)
            spec = ROS2NodeSpec(
                node_id=self._spec.node_id,
                device_id=self._spec.device_id,
                display_name=self._spec.display_name,
                telemetry=self._spec.telemetry,
                services=services,
                actions=(),
            )
        else:
            spec = self._spec
        return ROS2ManagedAdapter(
            self._transport,
            spec,
            installation_id=self._installation_id,
            initial_samples=self._initial_samples,
            physical_identity_sha256=self._identity(),
        )

    async def detach(self, adapter: LiveChannelAdapter) -> None:
        if adapter.adapter_id.startswith(f"ros2.{self._installation_id}."):
            await self._transport.close()

    async def stop(self) -> None:
        await self._transport.close()


def build_configured_ros2_connector() -> ROS2Connector:
    raw = str(os.getenv("AURA_ROSBRIDGE_NODE_MANIFEST_JSON") or "").strip()
    installation = str(os.getenv("AURA_ROSBRIDGE_INSTALLATION_ID") or "").strip()
    if not raw:
        raise ROS2ConnectorError("ros2_node_manifest_missing")
    if not installation:
        raise ROS2ConnectorError("ros2_installation_id_missing")
    return ROS2Connector(
        RosbridgeWebSocketTransport(),
        parse_ros2_node_manifest(raw),
        installation_id=installation,
    )


__all__ = [
    "ROS2ActionSpec",
    "ROS2Connector",
    "ROS2ConnectorError",
    "ROS2ManagedAdapter",
    "ROS2NodeSpec",
    "ROS2ServiceSpec",
    "ROS2TelemetrySpec",
    "ROS2Transport",
    "ROSGraphSnapshot",
    "ROSTopicSample",
    "RosbridgeWebSocketTransport",
    "build_configured_ros2_connector",
    "parse_ros2_node_manifest",
]
