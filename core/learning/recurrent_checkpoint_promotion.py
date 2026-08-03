"""Fail-closed promotion state machine for recurrent checkpoint candidates.

Training completion, a lower validation loss, and a valid adapter package are
necessary evidence, but none of them says that recurrent generation improved.
This module owns the missing transition between a completed checkpoint and the
broader permanent-distillation gate.  It is deliberately runtime-free so an
independent verifier can replay every decision from canonical evidence.

``latest`` means most recently trained, ``pending`` means awaiting a decision,
and ``promoted`` means the only checkpoint Aura may activate.  Those identities
must never be aliases: staging a candidate changes latest/pending but leaves the
promoted pointer untouched.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

CANDIDATE_SCHEMA: Final = "aura.rlc.checkpoint_candidate.v1"
GATE_SCHEMA: Final = "aura.rlc.checkpoint_evidence_gate.v1"
DECISION_SCHEMA: Final = "aura.rlc.checkpoint_promotion_decision.v1"
POINTER_SCHEMA: Final = "aura.rlc.checkpoint_pointer.v1"
REGISTRY_SCHEMA: Final = "aura.rlc.checkpoint_registry.v1"

PROMOTE: Final = "promote"
RETAIN_PARENT: Final = "retain_parent"
ROLLBACK_AND_HALT: Final = "rollback_and_halt"
OUTCOMES: Final = frozenset({PROMOTE, RETAIN_PARENT, ROLLBACK_AND_HALT})

PASS: Final = "PASS"
FAIL: Final = "FAIL"
UNMEASURED: Final = "UNMEASURED"
GATE_STATUSES: Final = frozenset({PASS, FAIL, UNMEASURED})

REQUIRED_GATES: Final = (
    "four_arm_semantics",
    "vanilla_no_regression",
    "recurrent_gain",
    "positive_interaction",
    "family_retention",
    "branch_specialization",
    "contract_integrity",
    "halt_revert",
    "broad_retention",
    "independent_replay",
)

# These floors prevent a syntactically passing smoke run from becoming a
# promotion certificate. Statistical power remains owned by the preregistered
# campaign verifier; these are only minimum evidence cardinalities.
MINIMUM_PROBES: Final = {
    "four_arm_semantics": 40,
    "vanilla_no_regression": 40,
    "recurrent_gain": 40,
    "positive_interaction": 40,
    "family_retention": 7,
    "branch_specialization": 40,
    "contract_integrity": 80,
    "halt_revert": 20,
    "broad_retention": 7,
    "independent_replay": 40,
}

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_IDENTIFIER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:-]{0,159}\Z")
_MAX_HISTORY = 65_536


class RecurrentCheckpointPromotionError(ValueError):
    """A candidate, decision, or registry violated the promotion contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise RecurrentCheckpointPromotionError(code)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise RecurrentCheckpointPromotionError(
            "recurrent_checkpoint_noncanonical_value"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _exact(value: Any, keys: set[str], *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(f"recurrent_checkpoint_{role}_schema_invalid")
    return value


def _sha(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"recurrent_checkpoint_{role}_sha256_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"recurrent_checkpoint_{role}_invalid")
    return value


def _commit(value: Any) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        _fail("recurrent_checkpoint_source_commit_invalid")
    return value


def evidence_gate(
    *,
    gate: str,
    status: str,
    probes_graded: int,
    probes_passed: int,
    evidence_sha256: str,
    verifier_sha256: str,
) -> dict[str, Any]:
    """Build one canonical evidence row without converting missing into pass."""

    if gate not in REQUIRED_GATES:
        _fail("recurrent_checkpoint_gate_unknown")
    if status not in GATE_STATUSES:
        _fail("recurrent_checkpoint_gate_status_invalid")
    if (
        type(probes_graded) is not int
        or type(probes_passed) is not int
        or probes_graded < 0
        or not 0 <= probes_passed <= probes_graded
        or (status == UNMEASURED and (probes_graded != 0 or probes_passed != 0))
        or (status == PASS and probes_passed != probes_graded)
    ):
        _fail("recurrent_checkpoint_gate_counts_invalid")
    return {
        "schema": GATE_SCHEMA,
        "gate": gate,
        "status": status,
        "probes_graded": probes_graded,
        "probes_passed": probes_passed,
        "evidence_sha256": _sha(evidence_sha256, role="gate_evidence"),
        "verifier_sha256": _sha(verifier_sha256, role="gate_verifier"),
    }


def _validated_gates(
    value: Any,
    *,
    expected_verifiers: Mapping[str, str],
) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != len(REQUIRED_GATES)
        or set(expected_verifiers) != set(REQUIRED_GATES)
    ):
        _fail("recurrent_checkpoint_gate_set_incomplete")
    rows: dict[str, dict[str, Any]] = {}
    for raw in value:
        record = _exact(
            raw,
            {
                "schema",
                "gate",
                "status",
                "probes_graded",
                "probes_passed",
                "evidence_sha256",
                "verifier_sha256",
            },
            role="gate",
        )
        if record.get("schema") != GATE_SCHEMA:
            _fail("recurrent_checkpoint_gate_schema_invalid")
        row = evidence_gate(
            gate=record.get("gate"),
            status=record.get("status"),
            probes_graded=record.get("probes_graded"),
            probes_passed=record.get("probes_passed"),
            evidence_sha256=record.get("evidence_sha256"),
            verifier_sha256=record.get("verifier_sha256"),
        )
        gate = row["gate"]
        if gate in rows:
            _fail("recurrent_checkpoint_gate_duplicate")
        if row["verifier_sha256"] != _sha(
            expected_verifiers[gate], role=f"expected_{gate}_verifier"
        ):
            _fail("recurrent_checkpoint_gate_verifier_mismatch")
        rows[gate] = row
    if set(rows) != set(REQUIRED_GATES):
        _fail("recurrent_checkpoint_gate_set_incomplete")
    return [rows[gate] for gate in REQUIRED_GATES]


def checkpoint_candidate(
    *,
    candidate_id: str,
    parent_id: str,
    candidate_artifact_sha256: str,
    parent_artifact_sha256: str,
    candidate_active: bool,
    source_commit: str,
    source_closure_sha256: str,
    campaign_grade_sha256: str,
    independent_verdict_sha256: str,
    gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a candidate; semantic validation happens against trusted verifiers."""

    if type(candidate_active) is not bool:
        _fail("recurrent_checkpoint_candidate_active_invalid")
    body = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": _identifier(candidate_id, role="candidate_id"),
        "parent_id": _identifier(parent_id, role="parent_id"),
        "candidate_artifact_sha256": _sha(
            candidate_artifact_sha256, role="candidate_artifact"
        ),
        "parent_artifact_sha256": _sha(parent_artifact_sha256, role="parent_artifact"),
        "candidate_active": candidate_active,
        "source_commit": _commit(source_commit),
        "source_closure_sha256": _sha(source_closure_sha256, role="source_closure"),
        "campaign_grade_sha256": _sha(campaign_grade_sha256, role="campaign_grade"),
        "independent_verdict_sha256": _sha(
            independent_verdict_sha256, role="independent_verdict"
        ),
        "gates": [dict(row) for row in gates],
    }
    return {**body, "candidate_sha256": _digest(body)}


def _validated_candidate(
    value: Any,
    *,
    expected_verifiers: Mapping[str, str],
) -> dict[str, Any]:
    record = dict(
        _exact(
            value,
            {
                "schema",
                "candidate_id",
                "parent_id",
                "candidate_artifact_sha256",
                "parent_artifact_sha256",
                "candidate_active",
                "source_commit",
                "source_closure_sha256",
                "campaign_grade_sha256",
                "independent_verdict_sha256",
                "gates",
                "candidate_sha256",
            },
            role="candidate",
        )
    )
    claimed = _sha(record.pop("candidate_sha256"), role="candidate")
    if record.get("schema") != CANDIDATE_SCHEMA or claimed != _digest(record):
        _fail("recurrent_checkpoint_candidate_identity_invalid")
    normalized = checkpoint_candidate(
        candidate_id=record["candidate_id"],
        parent_id=record["parent_id"],
        candidate_artifact_sha256=record["candidate_artifact_sha256"],
        parent_artifact_sha256=record["parent_artifact_sha256"],
        candidate_active=record["candidate_active"],
        source_commit=record["source_commit"],
        source_closure_sha256=record["source_closure_sha256"],
        campaign_grade_sha256=record["campaign_grade_sha256"],
        independent_verdict_sha256=record["independent_verdict_sha256"],
        gates=_validated_gates(record["gates"], expected_verifiers=expected_verifiers),
    )
    if normalized["candidate_sha256"] != claimed:
        _fail("recurrent_checkpoint_candidate_identity_invalid")
    return normalized


def evaluate_checkpoint_candidate(
    candidate: Mapping[str, Any],
    *,
    expected_verifiers: Mapping[str, str],
) -> dict[str, Any]:
    """Return promote, retain-parent, or rollback-and-halt with named reasons."""

    normalized = _validated_candidate(candidate, expected_verifiers=expected_verifiers)
    reasons: list[dict[str, Any]] = []
    if normalized["candidate_artifact_sha256"] == normalized["parent_artifact_sha256"]:
        reasons.append({"gate": "artifact_identity", "reason": "candidate_equals_parent"})
    if normalized["candidate_active"]:
        reasons.append(
            {"gate": "activation_order", "reason": "candidate_active_before_promotion"}
        )
    for row in normalized["gates"]:
        minimum = MINIMUM_PROBES[row["gate"]]
        if row["status"] != PASS:
            reasons.append(
                {
                    "gate": row["gate"],
                    "reason": (
                        "gate_unmeasured" if row["status"] == UNMEASURED else "gate_failed"
                    ),
                    "probes_graded": row["probes_graded"],
                    "probes_passed": row["probes_passed"],
                }
            )
        elif row["probes_graded"] < minimum:
            reasons.append(
                {
                    "gate": row["gate"],
                    "reason": "gate_underpowered",
                    "probes_graded": row["probes_graded"],
                    "probes_required": minimum,
                }
            )
    if not reasons:
        outcome = PROMOTE
    elif normalized["candidate_active"]:
        outcome = ROLLBACK_AND_HALT
    else:
        outcome = RETAIN_PARENT
    body = {
        "schema": DECISION_SCHEMA,
        "outcome": outcome,
        "candidate_id": normalized["candidate_id"],
        "parent_id": normalized["parent_id"],
        "candidate_artifact_sha256": normalized["candidate_artifact_sha256"],
        "parent_artifact_sha256": normalized["parent_artifact_sha256"],
        "candidate_sha256": normalized["candidate_sha256"],
        "source_commit": normalized["source_commit"],
        "source_closure_sha256": normalized["source_closure_sha256"],
        "campaign_grade_sha256": normalized["campaign_grade_sha256"],
        "independent_verdict_sha256": normalized["independent_verdict_sha256"],
        "gate_evidence_sha256": _digest(normalized["gates"]),
        "reasons": reasons,
    }
    return {**body, "decision_sha256": _digest(body)}


def _pointer(
    *, checkpoint_id: str, artifact_sha256: str, source_commit: str, source_closure_sha256: str
) -> dict[str, Any]:
    return {
        "schema": POINTER_SCHEMA,
        "checkpoint_id": _identifier(checkpoint_id, role="pointer_id"),
        "artifact_sha256": _sha(artifact_sha256, role="pointer_artifact"),
        "source_commit": _commit(source_commit),
        "source_closure_sha256": _sha(source_closure_sha256, role="pointer_source_closure"),
    }


def _validated_pointer(value: Any, *, role: str) -> dict[str, Any]:
    record = _exact(
        value,
        {"schema", "checkpoint_id", "artifact_sha256", "source_commit", "source_closure_sha256"},
        role=role,
    )
    if record.get("schema") != POINTER_SCHEMA:
        _fail(f"recurrent_checkpoint_{role}_schema_invalid")
    return _pointer(
        checkpoint_id=record["checkpoint_id"],
        artifact_sha256=record["artifact_sha256"],
        source_commit=record["source_commit"],
        source_closure_sha256=record["source_closure_sha256"],
    )


def checkpoint_registry(
    *,
    promoted_id: str,
    promoted_artifact_sha256: str,
    source_commit: str,
    source_closure_sha256: str,
) -> dict[str, Any]:
    promoted = _pointer(
        checkpoint_id=promoted_id,
        artifact_sha256=promoted_artifact_sha256,
        source_commit=source_commit,
        source_closure_sha256=source_closure_sha256,
    )
    body = {
        "schema": REGISTRY_SCHEMA,
        "latest": promoted,
        "pending": None,
        "promoted": promoted,
        "rejected_decisions": [],
        "halted_candidates": [],
    }
    return {**body, "registry_sha256": _digest(body)}


def validate_registry(value: Any) -> dict[str, Any]:
    record = dict(
        _exact(
            value,
            {
                "schema",
                "latest",
                "pending",
                "promoted",
                "rejected_decisions",
                "halted_candidates",
                "registry_sha256",
            },
            role="registry",
        )
    )
    claimed = _sha(record.pop("registry_sha256"), role="registry")
    if record.get("schema") != REGISTRY_SCHEMA or claimed != _digest(record):
        _fail("recurrent_checkpoint_registry_identity_invalid")
    latest = _validated_pointer(record["latest"], role="latest_pointer")
    promoted = _validated_pointer(record["promoted"], role="promoted_pointer")
    pending = (
        None
        if record["pending"] is None
        else _validated_pointer(record["pending"], role="pending_pointer")
    )
    rejected = record["rejected_decisions"]
    halted = record["halted_candidates"]
    if (
        not isinstance(rejected, list)
        or not isinstance(halted, list)
        or len(rejected) > _MAX_HISTORY
        or len(halted) > _MAX_HISTORY
        or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in rejected)
        or any(not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None for value in halted)
        or len(set(rejected)) != len(rejected)
        or len(set(halted)) != len(halted)
    ):
        _fail("recurrent_checkpoint_registry_history_invalid")
    normalized_body = {
        "schema": REGISTRY_SCHEMA,
        "latest": latest,
        "pending": pending,
        "promoted": promoted,
        "rejected_decisions": list(rejected),
        "halted_candidates": list(halted),
    }
    normalized = {**normalized_body, "registry_sha256": _digest(normalized_body)}
    if normalized["registry_sha256"] != claimed:
        _fail("recurrent_checkpoint_registry_identity_invalid")
    return normalized


def stage_checkpoint_candidate(
    registry: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Stage a candidate without changing the active promoted checkpoint."""

    current = validate_registry(registry)
    if current["pending"] is not None:
        _fail("recurrent_checkpoint_pending_candidate_exists")
    required_candidate_fields = {
        "candidate_id",
        "parent_id",
        "candidate_artifact_sha256",
        "parent_artifact_sha256",
        "source_commit",
        "source_closure_sha256",
    }
    if not isinstance(candidate, Mapping) or not required_candidate_fields <= set(candidate):
        _fail("recurrent_checkpoint_candidate_schema_invalid")
    if (
        candidate["parent_id"] != current["promoted"]["checkpoint_id"]
        or candidate["parent_artifact_sha256"] != current["promoted"]["artifact_sha256"]
    ):
        _fail("recurrent_checkpoint_candidate_parent_mismatch")
    if (
        candidate["candidate_id"] == current["latest"]["checkpoint_id"]
        or candidate["candidate_id"] in current["halted_candidates"]
    ):
        _fail("recurrent_checkpoint_candidate_id_reused")
    pointer = _pointer(
        checkpoint_id=candidate["candidate_id"],
        artifact_sha256=candidate["candidate_artifact_sha256"],
        source_commit=candidate["source_commit"],
        source_closure_sha256=candidate["source_closure_sha256"],
    )
    body = {
        **{key: value for key, value in current.items() if key != "registry_sha256"},
        "latest": pointer,
        "pending": pointer,
    }
    return {**body, "registry_sha256": _digest(body)}


def _validated_decision(value: Any) -> dict[str, Any]:
    record = dict(
        _exact(
            value,
            {
                "schema",
                "outcome",
                "candidate_id",
                "parent_id",
                "candidate_artifact_sha256",
                "parent_artifact_sha256",
                "candidate_sha256",
                "source_commit",
                "source_closure_sha256",
                "campaign_grade_sha256",
                "independent_verdict_sha256",
                "gate_evidence_sha256",
                "reasons",
                "decision_sha256",
            },
            role="decision",
        )
    )
    claimed = _sha(record.pop("decision_sha256"), role="decision")
    if (
        record.get("schema") != DECISION_SCHEMA
        or record.get("outcome") not in OUTCOMES
        or not isinstance(record.get("reasons"), list)
        or claimed != _digest(record)
    ):
        _fail("recurrent_checkpoint_decision_identity_invalid")
    for role in (
        "candidate_artifact_sha256",
        "parent_artifact_sha256",
        "candidate_sha256",
        "source_closure_sha256",
        "campaign_grade_sha256",
        "independent_verdict_sha256",
        "gate_evidence_sha256",
    ):
        _sha(record[role], role=role)
    _identifier(record["candidate_id"], role="candidate_id")
    _identifier(record["parent_id"], role="parent_id")
    _commit(record["source_commit"])
    return {**record, "decision_sha256": claimed}


def apply_checkpoint_decision(
    registry: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    expected_verifiers: Mapping[str, str],
    restored_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay and apply one decision; rollback requires exact parent bytes."""

    current = validate_registry(registry)
    verdict = _validated_decision(decision)
    replayed = evaluate_checkpoint_candidate(
        candidate, expected_verifiers=expected_verifiers
    )
    if verdict != replayed:
        _fail("recurrent_checkpoint_decision_replay_mismatch")
    pending = current["pending"]
    if (
        pending is None
        or pending["checkpoint_id"] != verdict["candidate_id"]
        or pending["artifact_sha256"] != verdict["candidate_artifact_sha256"]
        or current["promoted"]["checkpoint_id"] != verdict["parent_id"]
        or current["promoted"]["artifact_sha256"] != verdict["parent_artifact_sha256"]
    ):
        _fail("recurrent_checkpoint_decision_stale")
    body = {key: value for key, value in current.items() if key != "registry_sha256"}
    body["pending"] = None
    if verdict["outcome"] == PROMOTE:
        if verdict["reasons"]:
            _fail("recurrent_checkpoint_promote_has_refusals")
        body["promoted"] = pending
    else:
        if not verdict["reasons"]:
            _fail("recurrent_checkpoint_refusal_missing_reasons")
        body["rejected_decisions"] = [
            *body["rejected_decisions"],
            verdict["decision_sha256"],
        ]
    if verdict["outcome"] == ROLLBACK_AND_HALT:
        if restored_artifact_sha256 != current["promoted"]["artifact_sha256"]:
            _fail("recurrent_checkpoint_rollback_not_exact")
        body["halted_candidates"] = [
            *body["halted_candidates"],
            verdict["candidate_id"],
        ]
    elif restored_artifact_sha256 is not None:
        _fail("recurrent_checkpoint_unexpected_rollback_evidence")
    return {**body, "registry_sha256": _digest(body)}


__all__ = [
    "CANDIDATE_SCHEMA",
    "DECISION_SCHEMA",
    "FAIL",
    "GATE_SCHEMA",
    "MINIMUM_PROBES",
    "PASS",
    "POINTER_SCHEMA",
    "PROMOTE",
    "REGISTRY_SCHEMA",
    "REQUIRED_GATES",
    "RETAIN_PARENT",
    "ROLLBACK_AND_HALT",
    "RecurrentCheckpointPromotionError",
    "UNMEASURED",
    "apply_checkpoint_decision",
    "checkpoint_candidate",
    "checkpoint_registry",
    "evaluate_checkpoint_candidate",
    "evidence_gate",
    "stage_checkpoint_candidate",
    "validate_registry",
]
