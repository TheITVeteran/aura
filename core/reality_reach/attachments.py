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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from core.reality_reach.attachment_authority import (
    AttachmentAuthorityError,
    AttachmentCapabilityAuthorityVerifier,
    PhysicalAuthorityVerifier,
    build_attachment_authority_intent,
)
from core.reality_reach.body_projection import (
    PhysicalBodyProjection,
    project_adapter_to_body,
    remove_body_projection,
)
from core.reality_reach.live import LiveChannelAdapter, RealityReachService
from core.reality_reach.observation_router import RealityObservationRouter
from core.reality_reach.trust_custody import (
    AttachmentTrustStore,
    AttachmentTrustStoreError,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_METADATA_BYTES = 16 * 1024
_SESSION_GRANT_MAX_S = 8 * 60 * 60
_PERSISTENT_OBSERVE_MAX_S = 90 * 24 * 60 * 60
_PERSISTENT_CONTROL_MAX_S = 30 * 24 * 60 * 60
_DEFAULT_PERSISTENT_OBSERVE_S = 30 * 24 * 60 * 60
_DEFAULT_PERSISTENT_CONTROL_S = 7 * 24 * 60 * 60


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


def _bounded_grant_ttl_s(
    access: tuple[AttachmentAccess, ...],
    *,
    persistent: bool,
    requested_ttl_s: int | None,
) -> int:
    includes_control = AttachmentAccess.CONTROL in access
    if not persistent:
        maximum = _SESSION_GRANT_MAX_S
        default = _SESSION_GRANT_MAX_S
    elif includes_control:
        maximum = _PERSISTENT_CONTROL_MAX_S
        default = _DEFAULT_PERSISTENT_CONTROL_S
    else:
        maximum = _PERSISTENT_OBSERVE_MAX_S
        default = _DEFAULT_PERSISTENT_OBSERVE_S
    if requested_ttl_s is None:
        return default
    if isinstance(requested_ttl_s, bool) or not isinstance(requested_ttl_s, int):
        raise TypeError("grant_ttl_s must be an integer number of seconds")
    if requested_ttl_s <= 0 or requested_ttl_s > maximum:
        raise ValueError(f"grant_ttl_s must lie inside [1, {maximum}]")
    return requested_ttl_s


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
    authority_intent: Mapping[str, Any]
    authority_evidence: Mapping[str, Any]
    private_device_metadata: Mapping[str, Any]
    issued_at_ns: int
    expires_at_ns: int
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
        if self.issued_at_ns <= 0 or self.expires_at_ns <= self.issued_at_ns:
            raise ValueError("trust grant lifetime is invalid")
        object.__setattr__(
            self,
            "authority_intent",
            _frozen_metadata(self.authority_intent),
        )
        object.__setattr__(
            self,
            "authority_evidence",
            _frozen_metadata(self.authority_evidence),
        )
        object.__setattr__(
            self,
            "private_device_metadata",
            _frozen_metadata(self.private_device_metadata),
        )

    def is_valid_at(self, now_ns: int) -> bool:
        return self.issued_at_ns <= now_ns < self.expires_at_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_fingerprint": self.identity_fingerprint,
            "connector_id": self.connector_id,
            "manifest_sha256": self.manifest_sha256,
            "allowed_access": [item.value for item in self.allowed_access],
            "authority_receipt_id": self.authority_receipt_id,
            "authority_intent": dict(self.authority_intent),
            "authority_evidence": dict(self.authority_evidence),
            "private_device_metadata": dict(self.private_device_metadata),
            "issued_at_ns": self.issued_at_ns,
            "expires_at_ns": self.expires_at_ns,
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
        trust_store: AttachmentTrustStore | None = None,
        trust_store_error: str = "",
        authority_verifier: PhysicalAuthorityVerifier | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        discovery_interval_s: float = 60.0,
        connector_timeout_s: float = 12.0,
        max_candidates: int = 2048,
        max_pending_proposals: int = 8,
    ) -> None:
        if not isinstance(service, RealityReachService):
            raise TypeError("service must be a RealityReachService")
        if not isinstance(observation_router, RealityObservationRouter):
            raise TypeError("observation_router must be a RealityObservationRouter")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._service = service
        self._router = observation_router
        self._trust_store: AttachmentTrustStore | None = trust_store
        self._custody_error = str(trust_store_error or "")[:320]
        if self._trust_store is None and not self._custody_error:
            self._custody_error = "attachment_trust_custody_not_provisioned"
        if self._trust_store is not None and not isinstance(
            self._trust_store,
            AttachmentTrustStore,
        ):
            raise TypeError("trust_store must satisfy AttachmentTrustStore")
        self._authority_verifier = (
            authority_verifier or AttachmentCapabilityAuthorityVerifier()
        )
        if not isinstance(self._authority_verifier, PhysicalAuthorityVerifier):
            raise TypeError("authority_verifier must satisfy PhysicalAuthorityVerifier")
        self._clock_ns = clock_ns
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
        self._state_load_lock = asyncio.Lock()
        self._state_loaded = False
        self._wake = asyncio.Event()
        self._task: asyncio.Task[Any] | None = None
        self._running = False
        self._discoveries = 0
        self._attachment_failures = 0
        self._expired_grants = 0
        self._state_needs_compaction = False

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
        await self._ensure_state_loaded()
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
        now_ns = self._clock_ns()
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
        await self._ensure_state_loaded()
        canonical = _identifier(candidate_id, name="candidate_id")
        with self._lock:
            candidate = self._candidates.get(canonical)
        if candidate is None or candidate.expires_at_ns <= self._clock_ns():
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
            current_time_ns = self._clock_ns()
            if grant is not None and not grant.is_valid_at(current_time_ns):
                self._grants.pop(candidate.identity_fingerprint, None)
                self._expired_grants += 1
                grant = None
        trusted = bool(
            grant is not None
            and grant.is_valid_at(current_time_ns)
            and grant.connector_id == candidate.connector_id
            and grant.manifest_sha256 == candidate.manifest_sha256
            and set(requested).issubset(set(grant.allowed_access))
        )
        now_ns = max(1, current_time_ns)
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
        return request

    def authority_intent(
        self,
        request_id: str,
        *,
        persistent: bool,
        grant_ttl_s: int | None = None,
    ) -> dict[str, Any]:
        """Return the deterministic payload the Will must authorize."""

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
        ttl_s = _bounded_grant_ttl_s(
            request.requested_access,
            persistent=persistent,
            requested_ttl_s=grant_ttl_s,
        )
        return build_attachment_authority_intent(
            request_id=request.request_id,
            candidate_sha256=request.candidate_sha256,
            identity_fingerprint=candidate.identity_fingerprint,
            connector_id=candidate.connector_id,
            manifest_sha256=candidate.manifest_sha256,
            requested_access=tuple(item.value for item in request.requested_access),
            persistent=persistent,
            grant_ttl_s=ttl_s,
        )

    async def authorize_and_attach(
        self,
        request_id: str,
        *,
        authority_capability: Mapping[str, Any],
        persistent: bool = True,
        grant_ttl_s: int | None = None,
    ) -> ConnectionRequest:
        await self._ensure_state_loaded()
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
        if persistent and (self._trust_store is None or self._custody_error):
            raise AttachmentTrustStoreError(
                "attachment_trust_custody_unhealthy_for_persistent_grant"
            )
        intent = self.authority_intent(
            request.request_id,
            persistent=persistent,
            grant_ttl_s=grant_ttl_s,
        )
        authority_evidence = self._authority_verifier.verify(
            authority_capability,
            intent=intent,
            persistent=persistent,
        )
        capability = authority_evidence.get("capability")
        authority_receipt_id = (
            str(capability.get("receipt_id") or "")
            if isinstance(capability, Mapping)
            else ""
        )
        if not authority_receipt_id:
            raise AttachmentAuthorityError("attachment_authority_receipt_missing")
        issued_at_ns = max(1, self._clock_ns())
        ttl_s = int(intent["grant_ttl_s"])
        grant = TrustGrant(
            identity_fingerprint=candidate.identity_fingerprint,
            connector_id=candidate.connector_id,
            manifest_sha256=candidate.manifest_sha256,
            allowed_access=request.requested_access,
            authority_receipt_id=authority_receipt_id[:192],
            authority_intent=intent,
            authority_evidence=authority_evidence,
            private_device_metadata={
                "device_id": candidate.device_id,
                "display_name": candidate.display_name,
                "transport": candidate.transport,
                "privacy_sensitive": candidate.privacy_sensitive,
                "metadata": dict(candidate.metadata),
            },
            issued_at_ns=issued_at_ns,
            expires_at_ns=issued_at_ns + ttl_s * 1_000_000_000,
            persistent=bool(persistent),
        )
        with self._lock:
            previous_grant = self._grants.get(grant.identity_fingerprint)
            previous_request = self._requests[request.request_id]
            self._grants[grant.identity_fingerprint] = grant
            self._requests[request.request_id] = replace(
                request,
                state=ConnectionState.ATTACHING,
                authority_receipt_id=grant.authority_receipt_id,
                updated_at_ns=max(request.created_at_ns, self._clock_ns()),
            )
            attaching = self._requests[request.request_id]
        try:
            if persistent:
                await self._persist_state()
        except (OSError, RuntimeError, TypeError, ValueError):
            with self._lock:
                if previous_grant is None:
                    self._grants.pop(grant.identity_fingerprint, None)
                else:
                    self._grants[grant.identity_fingerprint] = previous_grant
                self._requests[request.request_id] = previous_request
            raise
        return await self._attach(attaching)

    async def revoke(self, identity_fingerprint: str, *, reason: str) -> None:
        await self._ensure_state_loaded()
        if not _DIGEST.fullmatch(identity_fingerprint):
            raise ValueError("identity_fingerprint must be a sha256 digest")
        with self._lock:
            previous_grant = self._grants.pop(identity_fingerprint, None)
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
        try:
            if previous_grant is not None and previous_grant.persistent:
                await self._persist_state()
        except (OSError, RuntimeError, TypeError, ValueError):
            if previous_grant is not None:
                with self._lock:
                    self._grants[identity_fingerprint] = previous_grant
            raise
        for request_id, adapter_id, adapter in attached:
            connector = self._connectors.get(
                self._candidates[self._requests[request_id].candidate_id].connector_id
            )
            if connector is not None:
                try:
                    await connector.detach(adapter)
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    record_degradation(
                        "reality_attachment.revoke_detach",
                        exc,
                        action=(
                            "continued local authority and adapter revocation after "
                            "remote connector teardown failed"
                        ),
                    )
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
                request = self._requests[request_id]
                self._requests[request_id] = replace(
                    request,
                    state=ConnectionState.REVOKED,
                    updated_at_ns=max(request.created_at_ns, self._clock_ns()),
                    error=str(reason or "revoked")[:320],
                )
                self._attached.pop(request_id, None)
        await asyncio.to_thread(self._service.refresh)

    async def start(self) -> None:
        if self._running:
            return
        await self._ensure_state_loaded()
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
        now_ns = self._clock_ns()
        with self._lock:
            expired = [
                identity
                for identity, grant in self._grants.items()
                if not grant.is_valid_at(now_ns)
            ]
            for identity in expired:
                self._grants.pop(identity, None)
            self._expired_grants += len(expired)
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
                or not grant.is_valid_at(now_ns)
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
        if expired or self._state_needs_compaction:
            try:
                await self._persist_state()
            except (OSError, RuntimeError, TypeError, ValueError):
                # The expired grant has already been removed from the live
                # authority set.  Persistence stays visibly unhealthy, but a
                # Keychain outage must not blind bounded physical sensing.
                pass

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
            now_ns = max(request.created_at_ns, self._clock_ns())
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
            updated_at_ns=max(request.created_at_ns, self._clock_ns()),
            error=error[:320],
        )
        with self._lock:
            self._requests[request.request_id] = failed
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
            custody = (
                dict(self._trust_store.status())
                if self._trust_store is not None
                else {
                    "healthy": False,
                    "error": "attachment_trust_custody_unavailable",
                }
            )
            if self._custody_error:
                custody["healthy"] = False
                custody["error"] = self._custody_error
            return {
                "alive": self.is_alive(),
                "ready": self.is_ready(),
                "connectors": len(self._connectors),
                "candidates": len(self._candidates),
                "trust_grants": len(self._grants),
                "attached": len(self._attached),
                "discoveries": self._discoveries,
                "attachment_failures": self._attachment_failures,
                "expired_grants": self._expired_grants,
                "request_states": state_counts,
                "persistent_trust_ready": self._state_loaded
                and self._trust_store is not None
                and not self._custody_error,
                "trust_state_loaded": self._state_loaded,
                "trust_custody": custody,
            }

    def get_status(self) -> dict[str, Any]:
        return self.status()

    def is_alive(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def is_ready(self) -> bool:
        return self.is_alive()

    def _load_state(self) -> None:
        if self._trust_store is None:
            return
        try:
            payload = self._trust_store.load()
            if payload is None:
                return
            grants = payload.get("grants", [])
            if not isinstance(grants, list):
                raise ValueError("attachment grants must be a list")
            loaded: dict[str, TrustGrant] = {}
            now_ns = self._clock_ns()
            for raw in grants:
                if not isinstance(raw, dict):
                    raise ValueError("attachment grant must be a mapping")
                authority_intent = raw.get("authority_intent")
                authority_evidence = raw.get("authority_evidence")
                private_device_metadata = raw.get("private_device_metadata")
                if (
                    not isinstance(authority_intent, dict)
                    or not isinstance(authority_evidence, dict)
                    or not isinstance(private_device_metadata, dict)
                ):
                    raise ValueError("attachment grant private evidence is invalid")
                grant = TrustGrant(
                    identity_fingerprint=str(raw.get("identity_fingerprint") or ""),
                    connector_id=str(raw.get("connector_id") or ""),
                    manifest_sha256=str(raw.get("manifest_sha256") or ""),
                    allowed_access=tuple(
                        AttachmentAccess(str(item))
                        for item in raw.get("allowed_access", [])
                    ),
                    authority_receipt_id=str(raw.get("authority_receipt_id") or ""),
                    authority_intent=authority_intent,
                    authority_evidence=authority_evidence,
                    private_device_metadata=private_device_metadata,
                    issued_at_ns=int(raw.get("issued_at_ns") or 0),
                    expires_at_ns=int(raw.get("expires_at_ns") or 0),
                    persistent=bool(raw.get("persistent")),
                )
                if not grant.persistent:
                    raise ValueError("session trust must not be present in durable state")
                self._validate_loaded_grant(grant)
                if grant.is_valid_at(now_ns):
                    loaded[grant.identity_fingerprint] = grant
                else:
                    self._expired_grants += 1
                    self._state_needs_compaction = True
            self._grants = loaded
            self._custody_error = ""
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._custody_error = f"{type(exc).__name__}:{exc}"[:320]
            record_degradation(
                "reality_attachment.load",
                exc,
                action=(
                    "refused invalid physical trust state while retaining discovery "
                    "without attachment authority"
                ),
            )
            self._grants.clear()

    async def _ensure_state_loaded(self) -> None:
        if self._state_loaded:
            return
        async with self._state_load_lock:
            if self._state_loaded:
                return
            await asyncio.to_thread(self._load_state)
            self._state_loaded = True

    def _validate_loaded_grant(self, grant: TrustGrant) -> None:
        intent = dict(grant.authority_intent)
        expected_access = sorted(item.value for item in grant.allowed_access)
        lifetime_ns = grant.expires_at_ns - grant.issued_at_ns
        if (
            intent.get("identity_fingerprint") != grant.identity_fingerprint
            or intent.get("connector_id") != grant.connector_id
            or intent.get("manifest_sha256") != grant.manifest_sha256
            or intent.get("requested_access") != expected_access
            or intent.get("persistent") is not True
            or not isinstance(intent.get("grant_ttl_s"), int)
            or isinstance(intent.get("grant_ttl_s"), bool)
            or lifetime_ns != int(intent["grant_ttl_s"]) * 1_000_000_000
        ):
            raise AttachmentAuthorityError("attachment_authority_grant_binding_invalid")
        evidence = self._authority_verifier.validate_persisted(
            grant.authority_evidence,
            intent=intent,
            persistent=True,
        )
        capability = evidence.get("capability")
        if (
            not isinstance(capability, Mapping)
            or capability.get("receipt_id") != grant.authority_receipt_id
            or int(evidence.get("verified_at_ns") or 0)
            > grant.issued_at_ns + 5_000_000_000
        ):
            raise AttachmentAuthorityError("attachment_authority_grant_evidence_invalid")

    def _persistent_body(self) -> dict[str, Any]:
        now_ns = self._clock_ns()
        with self._lock:
            expired = [
                identity
                for identity, grant in self._grants.items()
                if grant.persistent and not grant.is_valid_at(now_ns)
            ]
            for identity in expired:
                self._grants.pop(identity, None)
            self._expired_grants += len(expired)
            return {
                "grants": [
                    item.to_dict()
                    for item in sorted(
                        self._grants.values(),
                        key=lambda item: item.identity_fingerprint,
                    )
                    if item.persistent
                ]
            }

    async def _persist_state(self) -> None:
        body = self._persistent_body()
        if self._trust_store is None:
            if body["grants"]:
                raise AttachmentTrustStoreError(
                    "attachment_trust_custody_unavailable_for_persistence"
                )
            return
        try:
            await asyncio.to_thread(self._trust_store.save, body)
            self._custody_error = ""
            self._state_needs_compaction = False
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._custody_error = f"{type(exc).__name__}:{exc}"[:320]
            record_degradation(
                "reality_attachment.persist",
                exc,
                action="refused to acknowledge uncommitted physical trust",
            )
            raise

    async def rotate_trust_custody(self) -> Mapping[str, Any]:
        """Rotate the Keychain root and atomically re-encrypt the trust head."""

        await self._ensure_state_loaded()
        if self._trust_store is None:
            raise AttachmentTrustStoreError("attachment_trust_custody_unavailable")
        body = self._persistent_body()
        try:
            receipt = await asyncio.to_thread(
                self._trust_store.rotate_and_save,
                body,
            )
            self._custody_error = ""
            return receipt
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._custody_error = f"{type(exc).__name__}:{exc}"[:320]
            record_degradation(
                "reality_attachment.rotate_custody",
                exc,
                action="retained the prior authenticated trust head",
            )
            raise


__all__ = [
    "AttachmentAccess",
    "ConnectionRequest",
    "ConnectionState",
    "DeviceAttachmentBroker",
    "DeviceCandidate",
    "DeviceConnector",
    "TrustGrant",
]
