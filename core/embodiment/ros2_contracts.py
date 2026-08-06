"""Immutable contracts for Aura's manifest-bound ROS 2 integration.

This module owns validation and transport-neutral value objects. Runtime
networking, attachment lifecycle, and physical effect verification remain in
ros2_connector so contract importers do not acquire network side effects.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from core.bus.qos import Durability, History, QosProfile, Reliability
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.middleware_contracts import bounded_payload

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_ROS_NAME = re.compile(r"^/(?:[A-Za-z0-9_]+/?)+$")
_ROS_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*/(?:(?:msg|srv|action)/)?[A-Za-z][A-Za-z0-9_]*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TERMINAL_ACTION_STATUS = frozenset({4, 5, 6})
_MAX_WIRE_BYTES = 1_048_576
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_PENDING = 256

class ROS2ConnectorError(RuntimeError):
    """A ROS graph, transport, payload, or effect contract failed."""


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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

__all__ = [
    "ROS2ActionSpec",
    "ROS2ConnectorError",
    "ROS2NodeSpec",
    "ROS2ServiceSpec",
    "ROS2TelemetrySpec",
    "ROS2Transport",
    "ROSGraphSnapshot",
    "ROSTopicSample",
    "parse_ros2_node_manifest",
]
