"""Supervisor-owned in-memory authority for detached worker result origins."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Never

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    VerifiedCampaignTrustPolicy,
    prepare_role_signature_request,
)
from core.brain.llm.latent_cortex.worker_origin import (
    MAX_WORKER_PROTOCOL_VALUE_BYTES,
    WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    ZERO_SHA256,
    WorkerOriginError,
    assemble_worker_lifecycle_event_origin,
    assemble_worker_result_origin,
    build_worker_authorization_payload,
    build_worker_lifecycle_event_payload,
    build_worker_result_signed_payload,
    compute_allowed_cell_digest,
    verify_worker_authorization,
)


class DetachedWorkerOriginState(StrEnum):
    """Terminally monotonic lifecycle for one worker-origin identity."""

    PREPARED = "prepared"
    AWAITING_EXTERNAL_SIGNATURE = "awaiting_external_signature"
    AUTHORIZED = "authorized"
    RUNNING = "running"
    TERMINAL = "terminal"
    ABANDONED = "abandoned"


class DetachedWorkerOriginError(WorkerOriginError):
    """Stable authority state or policy failure."""


def _fail(code: str) -> Never:
    raise DetachedWorkerOriginError(code)


def _positive_int(value: Any, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{role}_invalid")
    return value


def _canonical_copy(value: Any) -> Any:
    import json

    try:
        payload = canonical_json_bytes(value)
        if len(payload) > MAX_WORKER_PROTOCOL_VALUE_BYTES:
            _fail("worker_origin_value_too_large")
        return json.loads(payload)
    except DetachedWorkerOriginError:
        raise
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail("worker_origin_value_invalid")


class DetachedWorkerOriginAuthority:
    """Own one non-exported ephemeral key and its exact result state machine."""

    def __init__(
        self,
        *,
        policy: VerifiedCampaignTrustPolicy,
        campaign_name: str,
        protocol_sha256: str,
        detached_plan_sha256: str,
        broker_policy_sha256: str,
        executable_binding_sha256: str,
        environment_sha256: str,
        sandbox_sha256: str,
        source_manifest_sha256: str,
        session_id: str,
        supervisor_attempt: int,
        arm: str,
        worker_attempt_slot: int,
        allowed_cells: Sequence[Mapping[str, str]],
        model_identity_sha256: str,
        adapter_identity_sha256: str,
        authorization_ttl_seconds: int = 300,
    ) -> None:
        self._policy = policy
        self._authorization_ttl_seconds = _positive_int(
            authorization_ttl_seconds,
            role="worker_authorization_ttl_seconds",
        )
        cells = _canonical_copy(allowed_cells)
        if not isinstance(cells, list):
            _fail("worker_allowed_cells_invalid")
        try:
            allowed_digest = compute_allowed_cell_digest(cells)
        except WorkerOriginError as exc:
            raise DetachedWorkerOriginError(exc.code) from exc
        self._allowed_cells = tuple(
            (cell["cell_id"], cell["cell_type"]) for cell in cells
        )
        self._signing_key: Ed25519PrivateKey | None = (
            Ed25519PrivateKey.generate()
        )
        public_raw = self._signing_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        try:
            self._authorization_payload = build_worker_authorization_payload(
                campaign_name=campaign_name,
                policy_sha256=policy.policy_sha256,
                protocol_sha256=protocol_sha256,
                detached_plan_sha256=detached_plan_sha256,
                broker_policy_sha256=broker_policy_sha256,
                executable_binding_sha256=executable_binding_sha256,
                environment_sha256=environment_sha256,
                sandbox_sha256=sandbox_sha256,
                source_manifest_sha256=source_manifest_sha256,
                session_id=session_id,
                supervisor_attempt=supervisor_attempt,
                arm=arm,
                worker_attempt_slot=worker_attempt_slot,
                allowed_cell_digest=allowed_digest,
                model_identity_sha256=model_identity_sha256,
                adapter_identity_sha256=adapter_identity_sha256,
                worker_key_custody=WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
                worker_public_key_raw=public_raw,
            )
        except WorkerOriginError as exc:
            raise DetachedWorkerOriginError(exc.code) from exc
        self._state = DetachedWorkerOriginState.PREPARED
        self._authorization_request: dict[str, Any] | None = None
        self._authorization_attestation: dict[str, Any] | None = None
        self._authorization_requested_at_unix: int | None = None
        self._completed_cell_ids: list[str] = []
        self._chain_head = ZERO_SHA256
        self._lifecycle_receipt: dict[str, Any] | None = None

    @property
    def state(self) -> DetachedWorkerOriginState:
        return self._state

    @property
    def authorization_payload(self) -> dict[str, Any]:
        return copy.deepcopy(self._authorization_payload)

    @property
    def authorization_request(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._authorization_request)

    @property
    def authorization_attestation(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._authorization_attestation)

    @property
    def result_count(self) -> int:
        return len(self._completed_cell_ids)

    @property
    def chain_head(self) -> str:
        return self._chain_head

    @property
    def lifecycle_receipt(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._lifecycle_receipt)

    def _require_state(
        self,
        *allowed: DetachedWorkerOriginState,
    ) -> None:
        if self._state not in allowed:
            _fail("worker_origin_state_transition_invalid")

    def _active_key(self) -> Ed25519PrivateKey:
        if self._signing_key is None:
            _fail("worker_origin_authority_finalized")
        return self._signing_key

    def request_authorization(self, *, signed_at_unix: int) -> dict[str, Any]:
        """Publish the exact CAMPAIGN_RUNNER request and begin waiting."""

        self._require_state(DetachedWorkerOriginState.PREPARED)
        requested_at = _positive_int(
            signed_at_unix, role="worker_authorization_requested_at_unix"
        )
        try:
            request = prepare_role_signature_request(
                self._policy,
                role=CAMPAIGN_RUNNER,
                payload=self._authorization_payload,
                signed_at_unix=requested_at,
            )
        except ValueError as exc:
            raise DetachedWorkerOriginError(
                "worker_authorization_request_invalid"
            ) from exc
        self._authorization_requested_at_unix = requested_at
        self._authorization_request = _canonical_copy(request)
        self._state = DetachedWorkerOriginState.AWAITING_EXTERNAL_SIGNATURE
        return copy.deepcopy(self._authorization_request)

    def accept_authorization(
        self,
        attestation: Mapping[str, Any],
        *,
        now_unix: int,
    ) -> dict[str, Any]:
        """Accept only the exact, fresh CAMPAIGN_RUNNER attestation."""

        self._require_state(
            DetachedWorkerOriginState.AWAITING_EXTERNAL_SIGNATURE
        )
        now = _positive_int(now_unix, role="worker_authorization_now_unix")
        requested_at = self._authorization_requested_at_unix
        if requested_at is None:
            _fail("worker_authorization_request_missing")
        if (
            now < requested_at
            or now > requested_at + self._authorization_ttl_seconds
        ):
            _fail("worker_authorization_window_expired")
        try:
            verified = verify_worker_authorization(
                self._policy,
                attestation,
                expected_payload=self._authorization_payload,
                not_before_unix=requested_at,
                not_after_unix=requested_at,
            )
        except WorkerOriginError as exc:
            raise DetachedWorkerOriginError(exc.code) from exc
        self._authorization_attestation = _canonical_copy(attestation)
        self._state = DetachedWorkerOriginState.AUTHORIZED
        return copy.deepcopy(verified)

    def start(self) -> None:
        """Enter the sole result-producing state."""

        self._require_state(DetachedWorkerOriginState.AUTHORIZED)
        if self._authorization_attestation is None:
            _fail("worker_authorization_missing")
        self._state = DetachedWorkerOriginState.RUNNING

    def record_result(
        self,
        result_body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Sign exactly the next allowed typed cell and advance the chain."""

        self._require_state(DetachedWorkerOriginState.RUNNING)
        if not isinstance(result_body, Mapping):
            _fail("worker_result_body_invalid")
        body = _canonical_copy(result_body)
        cell_id = body.get("cell_id")
        cell_type = body.get("cell_type")
        attempt_id = body.get("attempt_id")
        if cell_id in self._completed_cell_ids:
            _fail("worker_result_cell_duplicate")
        if self.result_count >= len(self._allowed_cells):
            _fail("worker_result_cell_exhausted")
        expected_cell_id, expected_cell_type = self._allowed_cells[
            self.result_count
        ]
        allowed_ids = {allowed_id for allowed_id, _kind in self._allowed_cells}
        if cell_id not in allowed_ids:
            _fail("worker_result_cell_not_allowed")
        if cell_id != expected_cell_id:
            _fail("worker_result_cell_out_of_order")
        if cell_type != expected_cell_type:
            _fail("worker_result_cell_type_mismatch")
        if not isinstance(attempt_id, str):
            _fail("worker_result_attempt_id_invalid")
        attestation = self._authorization_attestation
        if attestation is None:
            _fail("worker_authorization_missing")
        sequence = self.result_count + 1
        signed_payload = build_worker_result_signed_payload(
            authorization_attestation=attestation,
            authorization_payload=self._authorization_payload,
            result_body=body,
            cell_id=expected_cell_id,
            cell_type=expected_cell_type,
            attempt_id=attempt_id,
            sequence=sequence,
            previous_origin_sha256=self._chain_head,
        )
        signed_bytes = canonical_json_bytes(signed_payload)
        origin = assemble_worker_result_origin(
            signed_payload,
            signature=self._active_key().sign(signed_bytes),
        )
        result = {**body, "worker_origin": origin}
        self._completed_cell_ids.append(expected_cell_id)
        self._chain_head = origin["origin_sha256"]
        return result

    def _finalize(
        self,
        *,
        event_type: str,
        occurred_at_unix: int,
        return_code: int | None,
        reason: str | None,
    ) -> dict[str, Any]:
        prior_state = self._state.value
        signed_payload = build_worker_lifecycle_event_payload(
            authorization_attestation=self._authorization_attestation,
            authorization_payload=self._authorization_payload,
            event_type=event_type,
            prior_state=prior_state,
            result_count=self.result_count,
            previous_origin_sha256=self._chain_head,
            completed_cell_ids=self._completed_cell_ids,
            occurred_at_unix=occurred_at_unix,
            return_code=return_code,
            reason=reason,
        )
        signed_bytes = canonical_json_bytes(signed_payload)
        receipt = assemble_worker_lifecycle_event_origin(
            signed_payload,
            signature=self._active_key().sign(signed_bytes),
        )
        self._lifecycle_receipt = receipt
        self._signing_key = None
        return copy.deepcopy(receipt)

    def complete(
        self,
        *,
        occurred_at_unix: int,
        return_code: int = 0,
    ) -> dict[str, Any]:
        """Emit the signed terminal receipt after every allowed cell."""

        self._require_state(DetachedWorkerOriginState.RUNNING)
        if self.result_count != len(self._allowed_cells):
            _fail("worker_origin_completion_incomplete")
        receipt = self._finalize(
            event_type="terminal",
            occurred_at_unix=occurred_at_unix,
            return_code=return_code,
            reason=None,
        )
        self._state = DetachedWorkerOriginState.TERMINAL
        return receipt

    def abandon(
        self,
        *,
        reason: str,
        occurred_at_unix: int,
    ) -> dict[str, Any]:
        """Emit a signed abandonment receipt from any nonterminal state."""

        self._require_state(
            DetachedWorkerOriginState.PREPARED,
            DetachedWorkerOriginState.AWAITING_EXTERNAL_SIGNATURE,
            DetachedWorkerOriginState.AUTHORIZED,
            DetachedWorkerOriginState.RUNNING,
        )
        receipt = self._finalize(
            event_type="abandoned",
            occurred_at_unix=occurred_at_unix,
            return_code=None,
            reason=reason,
        )
        self._state = DetachedWorkerOriginState.ABANDONED
        return receipt


__all__ = [
    "DetachedWorkerOriginAuthority",
    "DetachedWorkerOriginError",
    "DetachedWorkerOriginState",
]
