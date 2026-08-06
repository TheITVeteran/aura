"""Runtime ROS 2 transport, attachment, and verified action integration.

A bounded manifest defines one robot through the transport-neutral contracts in
ros2_contracts. This runtime owns rosbridge networking, graph discovery,
attachment lifecycle, independent effect readback, and restart reconciliation.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from typing import Any

from core.embodiment.ros2_contracts import (
    _DIGEST,
    _TERMINAL_ACTION_STATUS,
    ROS2ActionSpec,
    ROS2ConnectorError,
    ROS2NodeSpec,
    ROS2ServiceSpec,
    ROS2TelemetrySpec,
    ROS2Transport,
    ROSGraphSnapshot,
    ROSTopicSample,
    _bounded_mapping,
    _digest,
    _finite,
    _identifier,
    _pointer_get,
    parse_ros2_node_manifest,
)
from core.embodiment.rosbridge_transport import RosbridgeWebSocketTransport
from core.reality_reach.attachments import AttachmentAccess, DeviceCandidate
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
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
)

logger = logging.getLogger("Aura.RealityReach.ROS2")




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
