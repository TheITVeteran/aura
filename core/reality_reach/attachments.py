"""Portable discovery, trust, and attachment for physical Reality Reach adapters.

The broker separates four events that older Aura paths conflated:

* a connector notices a candidate;
* Aura proposes a relationship with that candidate;
* a stable device identity receives bounded trust;
* an adapter is attached to the live Reality Reach inventory.

Discovery never grants actuation.  Trusted devices may reattach after a host
migration without code changes, but only through the access classes and
manifest digest named by their durable trust grant.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from core.environment.runtime_workspace import environment_runtime_file
from core.reality_reach.body_projection import (
    PhysicalBodyProjection,
    project_adapter_to_body,
    remove_body_projection,
)
from core.reality_reach.live import LiveChannelAdapter, RealityReachService
from core.reality_reach.observation_router import RealityObservationRouter
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

_SCHEMA = "aura.reality-attachments.v1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_METADATA_BYTES = 16 * 1024


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _identifier(value: str, *, name: str) -> str:
    canonical = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(canonical):
        raise ValueError(f"{name} must be a canonical identifier")
    return canonical


def _frozen_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("device metadata must contain canonical JSON") from exc
    if len(encoded) > _MAX_METADATA_BYTES:
        raise ValueError("device metadata exceeds the bounded envelope")
    decoded = json.loads(encoded.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("device metadata must be a mapping")
    return MappingProxyType(decoded)


class AttachmentAccess(StrEnum):
    OBSERVE = "observe"
    CONTROL = "control"


class ConnectionState(StrEnum):
    DISCOVERED = "discovered"
    PENDING_TRUST = "pending_trust"
    ATTACHING = "attaching"
    ATTACHED = "attached"
    REFUSED = "refused"
    LOST = "lost"
    ERROR = "error"
    REVOKED = "revoked"


@dataclass(frozen=True, slots=True)
class DeviceCandidate:
    candidate_id: str
    connector_id: str
    device_id: str
    display_name: str
    transport: str
    identity_fingerprint: str
    manifest_sha256: str
    access: tuple[AttachmentAccess, ...]
    discovered_at_ns: int
    expires_at_ns: int
    persistent_identity: bool = False
    privacy_sensitive: bool = False
    proposal_salience: float = 0.4
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        for name in ("candidate_id", "connector_id", "device_id", "transport"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        if not str(self.display_name or "").strip() or len(self.display_name) > 160:
            raise ValueError("display_name must be present and bounded")
        for name in ("identity_fingerprint", "manifest_sha256"):
            if not _DIGEST.fullmatch(str(getattr(self, name))):
                raise ValueError(f"{name} must be a sha256 digest")
        if not self.access or len(self.access) != len(set(self.access)):
            raise ValueError("candidate access must be non-empty and unique")
        if any(not isinstance(item, AttachmentAccess) for item in self.access):
            raise TypeError("candidate access must contain AttachmentAccess values")
        if self.discovered_at_ns <= 0 or self.expires_at_ns <= self.discovered_at_ns:
            raise ValueError("candidate discovery lifetime is invalid")
        salience = float(self.proposal_salience)
        if not math.isfinite(salience) or not 0.0 <= salience <= 1.0:
            raise ValueError("proposal_salience must lie inside [0, 1]")
        object.__setattr__(self, "proposal_salience", salience)
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "connector_id": self.connector_id,
            "device_id": self.device_id,
            "display_name": self.display_name,
            "transport": self.transport,
            "identity_fingerprint": self.identity_fingerprint,
            "manifest_sha256": self.manifest_sha256,
            "access": [item.value for item in self.access],
            "discovered_at_ns": self.discovered_at_ns,
            "expires_at_ns": self.expires_at_ns,
            "persistent_identity": self.persistent_identity,
            "privacy_sensitive": self.privacy_sensitive,
            "proposal_salience": self.proposal_salience,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ConnectionRequest:
    request_id: str
    candidate_id: str
    candidate_sha256: str
    requested_access: tuple[AttachmentAccess, ...]
    initiated_by: str
    reason: str
    state: ConnectionState
    created_at_ns: int
    updated_at_ns: int
    authority_receipt_id: str = ""
    adapter_id: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        for name in ("request_id", "candidate_id", "initiated_by"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        if not _DIGEST.fullmatch(self.candidate_sha256):
            raise ValueError("candidate_sha256 must be a sha256 digest")
        if not self.requested_access or len(self.requested_access) != len(
            set(self.requested_access)
        ):
            raise ValueError("requested access must be non-empty and unique")
        if any(not isinstance(item, AttachmentAccess) for item in self.requested_access):
            raise TypeError("requested access must contain AttachmentAccess values")
        if not isinstance(self.state, ConnectionState):
            raise TypeError("state must be a ConnectionState")
        if self.created_at_ns <= 0 or self.updated_at_ns < self.created_at_ns:
            raise ValueError("connection request timestamps are invalid")
        if len(self.reason) > 320 or len(self.error) > 320:
            raise ValueError("connection request text exceeds its bound")
        if self.adapter_id:
            object.__setattr__(self, "adapter_id", _identifier(self.adapter_id, name="adapter_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "requested_access": [item.value for item in self.requested_access],
            "initiated_by": self.initiated_by,
            "reason": self.reason,
            "state": self.state.value,
            "created_at_ns": self.created_at_ns,
            "updated_at_ns": self.updated_at_ns,
            "authority_receipt_id": self.authority_receipt_id,
            "adapter_id": self.adapter_id,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class TrustGrant:
    identity_fingerprint: str
    connector_id: str
    manifest_sha256: str
    allowed_access: tuple[AttachmentAccess, ...]
    authority_receipt_id: str
    issued_at_ns: int
    persistent: bool

    def __post_init__(self) -> None:
        if not _DIGEST.fullmatch(self.identity_fingerprint):
            raise ValueError("identity_fingerprint must be a sha256 digest")
        if not _DIGEST.fullmatch(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a sha256 digest")
        object.__setattr__(self, "connector_id", _identifier(self.connector_id, name="connector_id"))
        if not self.allowed_access or len(self.allowed_access) != len(set(self.allowed_access)):
            raise ValueError("allowed_access must be non-empty and unique")
        if not str(self.authority_receipt_id or "").strip():
            raise ValueError("authority_receipt_id must be present")
        if self.issued_at_ns <= 0:
            raise ValueError("issued_at_ns must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_fingerprint": self.identity_fingerprint,
            "connector_id": self.connector_id,
            "manifest_sha256": self.manifest_sha256,
            "allowed_access": [item.value for item in self.allowed_access],
            "authority_receipt_id": self.authority_receipt_id,
            "issued_at_ns": self.issued_at_ns,
            "persistent": self.persistent,
        }


@runtime_checkable
class DeviceConnector(Protocol):
    @property
    def connector_id(self) -> str: ...

    async def discover(self) -> tuple[DeviceCandidate, ...]: ...

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> LiveChannelAdapter: ...

    async def detach(self, adapter: LiveChannelAdapter) -> None: ...


class DeviceAttachmentBroker:
    """Lifecycle owner for portable physical relationships."""

    def __init__(
        self,
        service: RealityReachService,
        observation_router: RealityObservationRouter,
        *,
        state_path: Path | None = None,
        discovery_interval_s: float = 60.0,
        connector_timeout_s: float = 12.0,
        max_candidates: int = 2048,
        max_pending_proposals: int = 8,
    ) -> None:
        if not isinstance(service, RealityReachService):
            raise TypeError("service must be a RealityReachService")
        if not isinstance(observation_router, RealityObservationRouter):
            raise TypeError("observation_router must be a RealityObservationRouter")
        self._service = service
        self._router = observation_router
        self._state_path = state_path or environment_runtime_file(
            "shared",
            "reality_attachment_trust.json",
            purpose="identity",
        )
        self._discovery_interval_s = max(5.0, min(float(discovery_interval_s), 3600.0))
        self._connector_timeout_s = max(1.0, min(float(connector_timeout_s), 60.0))
        self._max_candidates = max(1, min(int(max_candidates), 10_000))
        self._max_pending_proposals = max(0, min(int(max_pending_proposals), 64))
        self._connectors: dict[str, DeviceConnector] = {}
        self._candidates: dict[str, DeviceCandidate] = {}
        self._requests: dict[str, ConnectionRequest] = {}
        self._grants: dict[str, TrustGrant] = {}
        self._attached: dict[str, tuple[str, LiveChannelAdapter]] = {}
        self._body_projections: dict[str, PhysicalBodyProjection] = {}
        self._lock = threading.RLock()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[Any] | None = None
        self._running = False
        self._discoveries = 0
        self._attachment_failures = 0
        self._load_state()

    def register_connector(self, connector: DeviceConnector) -> None:
        if not isinstance(connector, DeviceConnector):
            raise TypeError("connector must satisfy DeviceConnector")
        connector_id = _identifier(connector.connector_id, name="connector_id")
        with self._lock:
            existing = self._connectors.get(connector_id)
            if existing is not None and existing is not connector:
                raise ValueError(f"connector already registered: {connector_id}")
            self._connectors[connector_id] = connector
        if self._running:
            self._wake.set()

    def unregister_connector(self, connector_id: str) -> None:
        canonical = _identifier(connector_id, name="connector_id")
        with self._lock:
            if self._connectors.pop(canonical, None) is None:
                raise LookupError(f"connector is not registered: {canonical}")
        if self._running:
            self._wake.set()

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        with self._lock:
            connectors = tuple(self._connectors.values())
        semaphore = asyncio.Semaphore(8)

        async def _one(connector: DeviceConnector) -> tuple[DeviceCandidate, ...]:
            async with semaphore:
                try:
                    found = await asyncio.wait_for(
                        connector.discover(),
                        timeout=self._connector_timeout_s,
                    )
                    if any(item.connector_id != connector.connector_id for item in found):
                        raise ValueError("connector returned a foreign candidate identity")
                    return found
                except asyncio.CancelledError:
                    raise
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    record_degradation(
                        "reality_attachment.discovery",
                        exc,
                        action=f"retained other connectors after {connector.connector_id} failed",
                    )
                    return ()

        batches = await asyncio.gather(*(_one(item) for item in connectors))
        now_ns = time.time_ns()
        merged = sorted(
            (candidate for batch in batches for candidate in batch),
            key=lambda item: (-item.proposal_salience, item.candidate_id),
        )[: self._max_candidates]
        newly_discovered: list[DeviceCandidate] = []
        with self._lock:
            previous_ids = set(self._candidates)
            self._candidates = {
                item.candidate_id: item
                for item in merged
                if item.expires_at_ns > now_ns
            }
            newly_discovered = [
                item for item in merged if item.candidate_id not in previous_ids
            ]
            self._discoveries += len(newly_discovered)
        await self._restore_trusted_connections()
        await self._propose_new_connections(newly_discovered)
        return tuple(merged)

    async def request_connection(
        self,
        candidate_id: str,
        *,
        requested_access: tuple[AttachmentAccess, ...] = (AttachmentAccess.OBSERVE,),
        initiated_by: str = "aura",
        reason: str = "new physical capability discovered",
    ) -> ConnectionRequest:
        canonical = _identifier(candidate_id, name="candidate_id")
        with self._lock:
            candidate = self._candidates.get(canonical)
        if candidate is None or candidate.expires_at_ns <= time.time_ns():
            raise LookupError(f"candidate is not currently discoverable: {canonical}")
        requested = tuple(dict.fromkeys(requested_access))
        if not requested or not set(requested).issubset(set(candidate.access)):
            raise PermissionError("requested attachment access is not declared by candidate")
        candidate_sha256 = _digest(candidate.to_dict())
        request_id = (
            "reality.connect."
            + _digest(
                {
                    "candidate": candidate_sha256,
                    "access": [item.value for item in requested],
                }
            ).removeprefix("sha256:")[:32]
        )
        with self._lock:
            existing = self._requests.get(request_id)
            if existing is not None and existing.state not in {
                ConnectionState.ERROR,
                ConnectionState.LOST,
                ConnectionState.REVOKED,
            }:
                return existing
            grant = self._grants.get(candidate.identity_fingerprint)
        trusted = bool(
            grant is not None
            and grant.connector_id == candidate.connector_id
            and grant.manifest_sha256 == candidate.manifest_sha256
            and set(requested).issubset(set(grant.allowed_access))
        )
        now_ns = max(1, time.time_ns())
        request = ConnectionRequest(
            request_id=request_id,
            candidate_id=candidate.candidate_id,
            candidate_sha256=candidate_sha256,
            requested_access=requested,
            initiated_by=initiated_by,
            reason=str(reason or "")[:320],
            state=ConnectionState.ATTACHING if trusted else ConnectionState.PENDING_TRUST,
            created_at_ns=now_ns,
            updated_at_ns=now_ns,
            authority_receipt_id=grant.authority_receipt_id if grant is not None else "",
        )
        with self._lock:
            self._requests[request_id] = request
        if trusted:
            return await self._attach(request)
        await self._announce_request(candidate, request)
        await self._persist_state()
        return request

    async def authorize_and_attach(
        self,
        request_id: str,
        *,
        authority_receipt_id: str,
        persistent: bool = True,
    ) -> ConnectionRequest:
        canonical = _identifier(request_id, name="request_id")
        with self._lock:
            request = self._requests.get(canonical)
            candidate = self._candidates.get(request.candidate_id) if request else None
        if request is None or candidate is None:
            raise LookupError("connection request or candidate is unavailable")
        if request.state != ConnectionState.PENDING_TRUST:
            raise RuntimeError(f"connection request is not pending trust: {request.state.value}")
        if persistent and not candidate.persistent_identity:
            raise PermissionError("candidate identity is not stable enough for persistent trust")
        grant = TrustGrant(
            identity_fingerprint=candidate.identity_fingerprint,
            connector_id=candidate.connector_id,
            manifest_sha256=candidate.manifest_sha256,
            allowed_access=request.requested_access,
            authority_receipt_id=str(authority_receipt_id or "")[:192],
            issued_at_ns=max(1, time.time_ns()),
            persistent=bool(persistent),
        )
        with self._lock:
            self._grants[grant.identity_fingerprint] = grant
            self._requests[request.request_id] = replace(
                request,
                state=ConnectionState.ATTACHING,
                authority_receipt_id=grant.authority_receipt_id,
                updated_at_ns=max(request.created_at_ns, time.time_ns()),
            )
            attaching = self._requests[request.request_id]
        await self._persist_state()
        return await self._attach(attaching)

    async def revoke(self, identity_fingerprint: str, *, reason: str) -> None:
        if not _DIGEST.fullmatch(identity_fingerprint):
            raise ValueError("identity_fingerprint must be a sha256 digest")
        with self._lock:
            self._grants.pop(identity_fingerprint, None)
            attached = [
                (request_id, adapter_id, adapter)
                for request_id, (adapter_id, adapter) in self._attached.items()
                if self._candidates.get(
                    self._requests[request_id].candidate_id
                ) is not None
                and self._candidates[
                    self._requests[request_id].candidate_id
                ].identity_fingerprint
                == identity_fingerprint
            ]
        for request_id, adapter_id, adapter in attached:
            connector = self._connectors.get(
                self._candidates[self._requests[request_id].candidate_id].connector_id
            )
            if connector is not None:
                await connector.detach(adapter)
            try:
                self._router.unregister_sampler(adapter_id)
            except LookupError:
                pass
            await asyncio.to_thread(self._service.unregister_adapter, adapter_id)
            self._remove_body_projection(adapter_id)
            with self._lock:
                request = self._requests[request_id]
                self._requests[request_id] = replace(
                    request,
                    state=ConnectionState.REVOKED,
                    updated_at_ns=max(request.created_at_ns, time.time_ns()),
                    error=str(reason or "revoked")[:320],
                )
                self._attached.pop(request_id, None)
        await asyncio.to_thread(self._service.refresh)
        await self._persist_state()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(
            self._discovery_loop(),
            name="RealityAttachmentBroker",
        )

    async def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        with self._lock:
            attached = list(self._attached.items())
        for request_id, (adapter_id, adapter) in attached:
            request = self._requests.get(request_id)
            candidate = self._candidates.get(request.candidate_id) if request else None
            connector = self._connectors.get(candidate.connector_id) if candidate else None
            if connector is not None:
                try:
                    await connector.detach(adapter)
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    record_degradation("reality_attachment.detach", exc)
            try:
                self._router.unregister_sampler(adapter_id)
            except LookupError:
                pass
            try:
                await asyncio.to_thread(self._service.unregister_adapter, adapter_id)
            except LookupError:
                pass
            self._remove_body_projection(adapter_id)
        with self._lock:
            self._attached.clear()
        await asyncio.to_thread(self._service.refresh)

    async def _discovery_loop(self) -> None:
        while self._running:
            try:
                await self.discover()
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                record_degradation(
                    "reality_attachment.loop",
                    exc,
                    action="continued physical discovery after one bounded scan failed",
                )
            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=self._discovery_interval_s,
                )
                self._wake.clear()
            except TimeoutError:
                pass

    async def _restore_trusted_connections(self) -> None:
        with self._lock:
            candidates = tuple(self._candidates.values())
            active_candidates = {
                self._requests[request_id].candidate_id
                for request_id in self._attached
                if request_id in self._requests
            }
        for candidate in candidates:
            grant = self._grants.get(candidate.identity_fingerprint)
            if grant is None or candidate.candidate_id in active_candidates:
                continue
            if (
                not grant.persistent
                or grant.connector_id != candidate.connector_id
                or grant.manifest_sha256 != candidate.manifest_sha256
            ):
                continue
            await self.request_connection(
                candidate.candidate_id,
                requested_access=grant.allowed_access,
                initiated_by="aura.migration",
                reason="reattach trusted physical capability after runtime discovery",
            )

    async def _propose_new_connections(
        self,
        candidates: list[DeviceCandidate],
    ) -> None:
        with self._lock:
            pending_count = sum(
                item.state == ConnectionState.PENDING_TRUST
                for item in self._requests.values()
            )
        remaining = max(0, self._max_pending_proposals - pending_count)
        for candidate in candidates[:remaining]:
            try:
                await self.request_connection(
                    candidate.candidate_id,
                    requested_access=(AttachmentAccess.OBSERVE,),
                    initiated_by="aura",
                    reason="I discovered a physical capability and would like to connect to its bounded sensory surface.",
                )
            except (LookupError, PermissionError, RuntimeError, TypeError, ValueError):
                continue

    async def _attach(self, request: ConnectionRequest) -> ConnectionRequest:
        with self._lock:
            candidate = self._candidates.get(request.candidate_id)
            connector = self._connectors.get(candidate.connector_id) if candidate else None
        if candidate is None or connector is None:
            return await self._fail_request(request, "candidate_or_connector_unavailable")
        adapter: LiveChannelAdapter | None = None
        service_registered = False
        sampler_registered = False
        body_projected = False
        try:
            adapter = await asyncio.wait_for(
                connector.attach(candidate, request.requested_access),
                timeout=self._connector_timeout_s,
            )
            if adapter is None:
                raise TypeError("connector returned no Reality Reach adapter")
            self._service.register_adapter(adapter)
            service_registered = True
            try:
                self._router.register_sampler(adapter)
                sampler_registered = True
            except (TypeError, ValueError):
                # Push-only and synchronous adapters remain valid Reality Reach
                # channels; they simply do not use the async sampling hook.
                pass
            self._body_projections[str(adapter.adapter_id)] = project_adapter_to_body(
                adapter,
                device_id=candidate.device_id,
                display_name=candidate.display_name,
                transport=candidate.transport,
                privacy_sensitive=candidate.privacy_sensitive,
                persistent_identity=candidate.persistent_identity,
                manifest_sha256=candidate.manifest_sha256,
            )
            body_projected = True
            await asyncio.to_thread(self._service.refresh)
            now_ns = max(request.created_at_ns, time.time_ns())
            attached = replace(
                request,
                state=ConnectionState.ATTACHED,
                adapter_id=str(adapter.adapter_id),
                updated_at_ns=now_ns,
                error="",
            )
            with self._lock:
                self._requests[request.request_id] = attached
                self._attached[request.request_id] = (str(adapter.adapter_id), adapter)
            await self._persist_state()
            return attached
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            if adapter is not None:
                adapter_id = str(getattr(adapter, "adapter_id", "") or "")
                if body_projected:
                    self._remove_body_projection(adapter_id)
                if sampler_registered:
                    try:
                        self._router.unregister_sampler(adapter_id)
                    except LookupError:
                        pass
                if service_registered:
                    try:
                        await asyncio.to_thread(
                            self._service.unregister_adapter,
                            adapter_id,
                        )
                    except (LookupError, OSError, RuntimeError, TypeError, ValueError):
                        pass
                try:
                    await connector.detach(adapter)
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                    pass
            self._attachment_failures += 1
            return await self._fail_request(
                request,
                f"{type(exc).__name__}:{exc}"[:320],
            )

    def _remove_body_projection(self, adapter_id: str) -> None:
        projection = self._body_projections.pop(str(adapter_id or ""), None)
        if projection is None:
            return
        try:
            remove_body_projection(projection)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "reality_attachment.body_schema",
                exc,
                action="retained attachment teardown after body-schema projection removal failed",
            )

    async def _fail_request(
        self,
        request: ConnectionRequest,
        error: str,
    ) -> ConnectionRequest:
        failed = replace(
            request,
            state=ConnectionState.ERROR,
            updated_at_ns=max(request.created_at_ns, time.time_ns()),
            error=error[:320],
        )
        with self._lock:
            self._requests[request.request_id] = failed
        await self._persist_state()
        return failed

    async def _announce_request(
        self,
        candidate: DeviceCandidate,
        request: ConnectionRequest,
    ) -> None:
        try:
            from core.observability.neural_feed import get_feed

            get_feed().push(
                content=(
                    f"I discovered {candidate.display_name} via {candidate.transport} and "
                    f"would like to attach its {', '.join(item.value for item in request.requested_access)} "
                    f"surface. Connection request: {request.request_id}."
                ),
                title="PHYSICAL_CONNECTION_PROPOSED",
                category="SOMATIC",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("reality_attachment.announce", exc)
        try:
            from core.advanced_cognition.integration import (
                get_advanced_cognition_runtime,
            )

            runtime = get_advanced_cognition_runtime()
            await asyncio.to_thread(
                runtime.observe_state,
                "physical_attachment",
                {
                    "candidate_id": candidate.candidate_id,
                    "device_id": candidate.device_id,
                    "transport": candidate.transport,
                    "requested_access": [
                        item.value for item in request.requested_access
                    ],
                    "privacy_sensitive": candidate.privacy_sensitive,
                    "connection_state": request.state.value,
                    "request_id": request.request_id,
                },
                source=f"reality_discovery:{candidate.connector_id}",
                confidence=0.8 if candidate.persistent_identity else 0.55,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("reality_attachment.cognition", exc)

    def candidates(self) -> tuple[DeviceCandidate, ...]:
        with self._lock:
            return tuple(sorted(self._candidates.values(), key=lambda item: item.candidate_id))

    def requests(self) -> tuple[ConnectionRequest, ...]:
        with self._lock:
            return tuple(sorted(self._requests.values(), key=lambda item: item.request_id))

    def status(self) -> dict[str, Any]:
        with self._lock:
            state_counts: dict[str, int] = {}
            for request in self._requests.values():
                state_counts[request.state.value] = state_counts.get(request.state.value, 0) + 1
            return {
                "alive": self.is_alive(),
                "ready": self.is_ready(),
                "connectors": len(self._connectors),
                "candidates": len(self._candidates),
                "trust_grants": len(self._grants),
                "attached": len(self._attached),
                "discoveries": self._discoveries,
                "attachment_failures": self._attachment_failures,
                "request_states": state_counts,
            }

    def get_status(self) -> dict[str, Any]:
        return self.status()

    def is_alive(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def is_ready(self) -> bool:
        return self.is_alive()

    def _load_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
                raise ValueError("unsupported attachment trust state schema")
            body = payload.get("body")
            if not isinstance(body, dict) or payload.get("body_sha256") != _digest(body):
                raise ValueError("attachment trust state digest mismatch")
            grants = body.get("grants", [])
            if not isinstance(grants, list):
                raise ValueError("attachment grants must be a list")
            for raw in grants:
                if not isinstance(raw, dict):
                    raise ValueError("attachment grant must be a mapping")
                grant = TrustGrant(
                    identity_fingerprint=str(raw.get("identity_fingerprint") or ""),
                    connector_id=str(raw.get("connector_id") or ""),
                    manifest_sha256=str(raw.get("manifest_sha256") or ""),
                    allowed_access=tuple(
                        AttachmentAccess(str(item))
                        for item in raw.get("allowed_access", [])
                    ),
                    authority_receipt_id=str(raw.get("authority_receipt_id") or ""),
                    issued_at_ns=int(raw.get("issued_at_ns") or 0),
                    persistent=bool(raw.get("persistent")),
                )
                if grant.persistent:
                    self._grants[grant.identity_fingerprint] = grant
        except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "reality_attachment.load",
                exc,
                action="ignored invalid physical trust state instead of granting attachment authority",
            )
            self._grants.clear()

    async def _persist_state(self) -> None:
        with self._lock:
            body = {
                "grants": [
                    item.to_dict()
                    for item in sorted(
                        self._grants.values(),
                        key=lambda item: item.identity_fingerprint,
                    )
                    if item.persistent
                ]
            }
        payload = {
            "schema": _SCHEMA,
            "body": body,
            "body_sha256": _digest(body),
        }
        await asyncio.to_thread(
            atomic_write_text,
            self._state_path,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )


__all__ = [
    "AttachmentAccess",
    "ConnectionRequest",
    "ConnectionState",
    "DeviceAttachmentBroker",
    "DeviceCandidate",
    "DeviceConnector",
    "TrustGrant",
]
