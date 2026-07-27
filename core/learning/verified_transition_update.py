"""Journaled exactly-once mutation for verified recurrent transition groups."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrent_grpo import (
    RecurrentGRPOConfig,
    exact_adjoint_verified_transition_group_value_and_grad,
    recurrent_policy_sha256,
)
from core.learning.verified_transition_episode import (
    TransitionArtifactStore,
    canonical_json_bytes,
)
from core.learning.verified_transition_group_admission import (
    validate_verified_transition_group_admission,
)
from core.learning.verified_transition_reward import VerifiedTransitionEvidence
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.file_write_gateway import FileWriteGateway

VERIFIED_TRANSITION_UPDATE_SCHEMA = "aura.verified_transition.update_receipt.v1"
VERIFIED_TRANSITION_RESERVATION_SCHEMA = "aura.verified_transition.update_reservation.v1"
VERIFIED_TRANSITION_COMMIT_SCHEMA = "aura.verified_transition.update_commit.v1"


class VerifiedTransitionUpdateError(RuntimeError):
    """Raised when an admitted update cannot complete exactly once."""


def _fail(code: str) -> Never:
    raise VerifiedTransitionUpdateError(code)


def _require_sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _require_time(value: Any, *, role: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{role}_invalid")
    return value


def _seal(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(sealed)
    ).hexdigest()
    return sealed


def _validate_seal(document: Mapping[str, Any], *, role: str) -> None:
    observed = _require_sha256(document.get("receipt_sha256"), role=f"{role}_receipt")
    unsigned = dict(document)
    unsigned.pop("receipt_sha256", None)
    if observed != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        _fail(f"{role}_digest_mismatch")


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(document),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class VerifiedTransitionUpdateJournal:
    """Create-once reservation and commit records keyed by admission digest."""

    root: Path
    gateway: FileWriteGateway

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        gateway: FileWriteGateway | None = None,
    ) -> VerifiedTransitionUpdateJournal:
        resolved_gateway = gateway or FileWriteGateway()
        path = Path(
            resolved_gateway.ensure_directory(
                Path(root),
                source="verified_transition_update.journal",
            )
        )
        return cls(root=path, gateway=resolved_gateway)

    def _path(self, admission_sha256: str, suffix: str) -> Path:
        digest = _require_sha256(admission_sha256, role="update_admission")
        return self.root / f"{digest}.{suffix}.json"

    def reserve(
        self,
        *,
        admission_sha256: str,
        policy_before_sha256: str,
        reserved_at_unix_ns: int,
    ) -> dict[str, Any]:
        reservation = _seal(
            {
                "schema": VERIFIED_TRANSITION_RESERVATION_SCHEMA,
                "admission_sha256": _require_sha256(
                    admission_sha256, role="reservation_admission"
                ),
                "policy_before_sha256": _require_sha256(
                    policy_before_sha256, role="reservation_policy_before"
                ),
                "reserved_at_unix_ns": _require_time(
                    reserved_at_unix_ns, role="reservation_time"
                ),
            }
        )
        created = self.gateway.write_bytes_if_absent(
            self._path(admission_sha256, "reserved"),
            _json_bytes(reservation),
            source="verified_transition_update.reserve",
            durable=True,
        )
        if not created:
            _fail("verified_transition_admission_already_reserved")
        return reservation

    def commit(
        self,
        *,
        admission_sha256: str,
        reservation_sha256: str,
        policy_before_sha256: str,
        policy_after_sha256: str,
        objective_receipt_sha256: str,
        committed_at_unix_ns: int,
    ) -> dict[str, Any]:
        commit = _seal(
            {
                "schema": VERIFIED_TRANSITION_COMMIT_SCHEMA,
                "admission_sha256": _require_sha256(
                    admission_sha256, role="commit_admission"
                ),
                "reservation_sha256": _require_sha256(
                    reservation_sha256, role="commit_reservation"
                ),
                "policy_before_sha256": _require_sha256(
                    policy_before_sha256, role="commit_policy_before"
                ),
                "policy_after_sha256": _require_sha256(
                    policy_after_sha256, role="commit_policy_after"
                ),
                "objective_receipt_sha256": _require_sha256(
                    objective_receipt_sha256, role="commit_objective"
                ),
                "committed_at_unix_ns": _require_time(
                    committed_at_unix_ns, role="commit_time"
                ),
            }
        )
        created = self.gateway.write_bytes_if_absent(
            self._path(admission_sha256, "committed"),
            _json_bytes(commit),
            source="verified_transition_update.commit",
            durable=True,
        )
        if not created:
            _fail("verified_transition_admission_already_committed")
        return commit

    def read(self, admission_sha256: str, suffix: str) -> dict[str, Any]:
        if suffix not in {"reserved", "committed"}:
            _fail("verified_transition_journal_suffix_invalid")
        payload = read_stable_bytes(
            self._path(admission_sha256, suffix),
            max_bytes=65_536,
        )
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerifiedTransitionUpdateError(
                "verified_transition_journal_json_invalid"
            ) from exc
        if not isinstance(document, dict) or _json_bytes(document) != payload:
            _fail("verified_transition_journal_noncanonical")
        return document


def apply_verified_transition_group_update(
    model: Any,
    optimizer: Any,
    prompt_tokens: Sequence[int],
    samples: Sequence[Any],
    group_admission_receipt: Mapping[str, Any],
    reward_receipt: Mapping[str, Any],
    transition_evidence: Sequence[VerifiedTransitionEvidence],
    *,
    transition_store: TransitionArtifactStore,
    group_manifest: Mapping[str, Any],
    group_manifest_attestation: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    spec: RLCExecutionSpec,
    journal: VerifiedTransitionUpdateJournal,
    bridge_tokens: Sequence[int] = (),
    config: RecurrentGRPOConfig | None = None,
    now_unix_ns: Callable[[], int] = time.time_ns,
) -> dict[str, Any]:
    """Validate, reserve, rehash, update exactly once, and receipt the result."""

    admission = validate_verified_transition_group_admission(
        transition_store,
        group_admission_receipt,
        reward_receipt,
        transition_evidence,
        samples,
        prompt_tokens,
        group_manifest=group_manifest,
        group_manifest_attestation=group_manifest_attestation,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
    )
    if admission.get("optimizer_admitted") is not True:
        _fail("verified_transition_group_not_admitted")
    if admission.get("recurrent_execution_spec_sha256") != spec.sha256:
        _fail("verified_transition_execution_spec_mismatch")
    policy_before = recurrent_policy_sha256(model, spec)
    if admission.get("policy_sha256") != policy_before:
        _fail("verified_transition_policy_before_mismatch")
    reserved_at = _require_time(now_unix_ns(), role="update_reserved_at")
    reservation = journal.reserve(
        admission_sha256=cast_sha256(admission.get("receipt_sha256")),
        policy_before_sha256=policy_before,
        reserved_at_unix_ns=reserved_at,
    )

    objective = exact_adjoint_verified_transition_group_value_and_grad(
        model,
        prompt_tokens,
        samples,
        admission,
        reward_receipt,
        transition_evidence,
        transition_store=transition_store,
        group_manifest=group_manifest,
        group_manifest_attestation=group_manifest_attestation,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        spec=spec,
        bridge_tokens=bridge_tokens,
        config=config,
    )
    if objective.gradients is None:
        _fail("verified_transition_gradient_missing")
    policy_immediately_before_update = recurrent_policy_sha256(model, spec)
    if policy_immediately_before_update != policy_before:
        _fail("verified_transition_policy_changed_before_update")

    optimizer.update(model, objective.gradients)
    try:
        import mlx.core as mx

        mx.eval(model.trainable_parameters(), optimizer.state)
    except (ImportError, AttributeError):
        pass
    policy_after = recurrent_policy_sha256(model, spec)
    if policy_after == policy_before:
        _fail("verified_transition_optimizer_did_not_change_policy")
    objective_receipt = objective.receipt()
    objective_sha256 = hashlib.sha256(
        _json_bytes(objective_receipt)
    ).hexdigest()
    committed_at = _require_time(now_unix_ns(), role="update_committed_at")
    if committed_at < reserved_at:
        _fail("verified_transition_update_time_reversed")
    commit = journal.commit(
        admission_sha256=cast_sha256(admission["receipt_sha256"]),
        reservation_sha256=cast_sha256(reservation["receipt_sha256"]),
        policy_before_sha256=policy_before,
        policy_after_sha256=policy_after,
        objective_receipt_sha256=objective_sha256,
        committed_at_unix_ns=committed_at,
    )
    return _seal(
        {
            "schema": VERIFIED_TRANSITION_UPDATE_SCHEMA,
            "group_admission_sha256": admission["receipt_sha256"],
            "reservation_sha256": reservation["receipt_sha256"],
            "commit_sha256": commit["receipt_sha256"],
            "objective_receipt_sha256": objective_sha256,
            "policy_before_sha256": policy_before,
            "policy_after_sha256": policy_after,
            "optimizer_update_count": 1,
            "reserved_at_unix_ns": reserved_at,
            "committed_at_unix_ns": committed_at,
        }
    )


def validate_verified_transition_update_receipt(
    journal: VerifiedTransitionUpdateJournal,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay durable reservation and commit bytes against the final receipt."""

    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema",
        "group_admission_sha256",
        "reservation_sha256",
        "commit_sha256",
        "objective_receipt_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "optimizer_update_count",
        "reserved_at_unix_ns",
        "committed_at_unix_ns",
        "receipt_sha256",
    }:
        _fail("verified_transition_update_receipt_schema_invalid")
    if receipt.get("schema") != VERIFIED_TRANSITION_UPDATE_SCHEMA:
        _fail("verified_transition_update_receipt_version_invalid")
    _validate_seal(receipt, role="verified_transition_update")
    admission_sha256 = cast_sha256(receipt.get("group_admission_sha256"))
    reservation = journal.read(admission_sha256, "reserved")
    commit = journal.read(admission_sha256, "committed")
    if reservation.get("schema") != VERIFIED_TRANSITION_RESERVATION_SCHEMA:
        _fail("verified_transition_reservation_schema_invalid")
    if commit.get("schema") != VERIFIED_TRANSITION_COMMIT_SCHEMA:
        _fail("verified_transition_commit_schema_invalid")
    _validate_seal(reservation, role="verified_transition_reservation")
    _validate_seal(commit, role="verified_transition_commit")
    if (
        reservation.get("receipt_sha256") != receipt.get("reservation_sha256")
        or commit.get("receipt_sha256") != receipt.get("commit_sha256")
        or reservation.get("admission_sha256") != admission_sha256
        or commit.get("admission_sha256") != admission_sha256
        or commit.get("reservation_sha256") != reservation.get("receipt_sha256")
        or reservation.get("policy_before_sha256")
        != receipt.get("policy_before_sha256")
        or commit.get("policy_before_sha256") != receipt.get("policy_before_sha256")
        or commit.get("policy_after_sha256") != receipt.get("policy_after_sha256")
        or commit.get("objective_receipt_sha256")
        != receipt.get("objective_receipt_sha256")
        or reservation.get("reserved_at_unix_ns")
        != receipt.get("reserved_at_unix_ns")
        or commit.get("committed_at_unix_ns")
        != receipt.get("committed_at_unix_ns")
        or receipt.get("optimizer_update_count") != 1
    ):
        _fail("verified_transition_update_reconstruction_mismatch")
    if receipt.get("policy_before_sha256") == receipt.get("policy_after_sha256"):
        _fail("verified_transition_update_policy_unchanged")
    if cast(int, receipt["committed_at_unix_ns"]) < cast(
        int, receipt["reserved_at_unix_ns"]
    ):
        _fail("verified_transition_update_time_reversed")
    return dict(receipt)


def cast_sha256(value: Any) -> str:
    return _require_sha256(value, role="verified_transition_digest")


__all__ = [
    "VERIFIED_TRANSITION_COMMIT_SCHEMA",
    "VERIFIED_TRANSITION_RESERVATION_SCHEMA",
    "VERIFIED_TRANSITION_UPDATE_SCHEMA",
    "VerifiedTransitionUpdateError",
    "VerifiedTransitionUpdateJournal",
    "apply_verified_transition_group_update",
    "validate_verified_transition_update_receipt",
]
