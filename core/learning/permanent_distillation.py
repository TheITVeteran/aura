"""Versioned promotion transaction for permanently distilled policies.

A recurrent or correction policy that measured well in a campaign is not yet
allowed to become part of Aura. SPARK-064 names the bar: promote through
versioned adapters or base updates *only after* broad anti-interference,
personality, tool, safety, memory, and frontier regressions pass, and support
exact rollback.

The failure this module exists to prevent is the one this codebase keeps
finding: **the absence of a check reported as a passed check**. A promotion
gate that silently skips the memory battery because no memory evidence was
supplied has not protected memory; it has only failed to look. So the gate set
here is complete by declaration. Every required gate must be present, must name
the battery that produced it, must report how many probes actually ran, and
must carry an evidence digest. A missing gate, an empty gate, or an unknown
extra gate refuses the promotion by name. There is no path through this module
that reaches ADMIT without every declared battery having run.

Rollback is exact, not approximate. A rollback generation must name an earlier
generation in the same lineage and present an *observed* artifact manifest —
digests read back off the restored files — that equals the target's recorded
manifest byte for byte. A restore that lands different bytes is refused rather
than recorded as a successful rollback.

The module is deliberately free of Aura runtime dependencies: it is a strict
state machine over data, so the independent verifier can replay a promotion
lineage without importing cognition, and durable writes stay with the caller
that owns the governed scope.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Never

GENERATION_SCHEMA: Final = "aura.rlc.permanent_distillation.generation.v1"
GATE_REPORT_SCHEMA: Final = "aura.rlc.permanent_distillation.gate_report.v1"
ARTIFACT_SCHEMA: Final = "aura.rlc.permanent_distillation.artifact.v1"
DECISION_SCHEMA: Final = "aura.rlc.permanent_distillation.decision.v1"

GENESIS_PARENT: Final = "0" * 64

BASELINE: Final = "baseline"
PROMOTION: Final = "promotion"
ROLLBACK: Final = "rollback"
GENERATION_KINDS: Final = (BASELINE, PROMOTION, ROLLBACK)

ADMIT: Final = "ADMIT"
REFUSE: Final = "REFUSE"

PASS: Final = "PASS"
FAIL: Final = "FAIL"
GATE_VERDICTS: Final = (PASS, FAIL)

# The complete regression surface a permanent change must clear. These are the
# six families SPARK-064 names, split where one battery cannot honestly cover
# two of them. The tuple is the contract: a gate report is valid only when its
# gate ids equal this set exactly.
REQUIRED_GATES: Final = (
    "anti_interference",
    "capability_families",
    "personality_retention",
    "tool_effect_honesty",
    "authority_safety",
    "memory_retention",
    "frontier_regression",
)

# A gate that graded fewer probes than this did not measure its family. The
# floors are deliberately low -- they exist to refuse the empty battery, not to
# set the scientific power of a campaign, which the preregistration owns.
_MINIMUM_PROBES: Final = {
    "anti_interference": 6,
    "capability_families": 12,
    "personality_retention": 4,
    "tool_effect_honesty": 4,
    "authority_safety": 4,
    "memory_retention": 4,
    "frontier_regression": 8,
}

_GATE_FIELDS: Final = frozenset(
    {
        "gate",
        "battery_schema",
        "probes_graded",
        "probes_passed",
        "verdict",
        "evidence_sha256",
    }
)
_ARTIFACT_FILE_FIELDS: Final = frozenset({"name", "sha256", "size_bytes"})
_SHA256_PATTERN: Final = re.compile(r"\A[0-9a-f]{64}\Z")
_NAME_PATTERN: Final = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MAX_ARTIFACT_FILES: Final = 4096
_MAX_LINEAGE_RECORDS: Final = 65_536
_ARTIFACT_READ_CHUNK: Final = 1 << 20


class PermanentDistillationError(ValueError):
    """A promotion, generation, or rollback contract is invalid."""


class PermanentDistillationRefusalError(PermanentDistillationError):
    """A promotion was blocked; the decision names every responsible gate."""

    def __init__(self, decision: Mapping[str, Any]) -> None:
        super().__init__("permanent_distillation_promotion_refused")
        self.decision: dict[str, Any] = dict(decision)


def _fail(code: str) -> Never:
    raise PermanentDistillationError(str(code or "permanent_distillation_invalid"))


def _sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise PermanentDistillationError(
            "permanent_distillation_noncanonical_value"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.match(value))


def _required_text(value: Any, code: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        _fail(code)
    return value


def _required_index(value: Any, code: str) -> int:
    if type(value) is not int or value < 0:
        _fail(code)
    return value


# ---------------------------------------------------------------------------
# Artifact identity
# ---------------------------------------------------------------------------


def artifact_manifest(
    *,
    artifact_id: str,
    base_model_identity: str,
    adapter_identity: str,
    files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize the exact bytes a generation consists of.

    The manifest is what makes rollback checkable: two generations hold the
    same artifact if and only if their manifests are equal.
    """

    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        _fail("permanent_distillation_artifact_files_invalid")
    if not files or len(files) > _MAX_ARTIFACT_FILES:
        _fail("permanent_distillation_artifact_files_invalid")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in files:
        if not isinstance(raw, Mapping) or set(raw) != _ARTIFACT_FILE_FIELDS:
            _fail("permanent_distillation_artifact_file_fields_differ")
        name = raw["name"]
        if not isinstance(name, str) or not _NAME_PATTERN.match(name):
            _fail("permanent_distillation_artifact_file_name_invalid")
        if name in seen:
            _fail("permanent_distillation_artifact_file_duplicate")
        seen.add(name)
        if not _is_sha256(raw["sha256"]):
            _fail("permanent_distillation_artifact_file_digest_invalid")
        size = raw["size_bytes"]
        if type(size) is not int or size < 0:
            _fail("permanent_distillation_artifact_file_size_invalid")
        rows.append(
            {"name": name, "sha256": raw["sha256"], "size_bytes": size}
        )

    rows.sort(key=lambda row: row["name"])
    body = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_id": _required_text(
            artifact_id, "permanent_distillation_artifact_id_invalid"
        ),
        "base_model_identity": _required_text(
            base_model_identity, "permanent_distillation_base_identity_invalid"
        ),
        "adapter_identity": _required_text(
            adapter_identity, "permanent_distillation_adapter_identity_invalid"
        ),
        "files": rows,
        "total_bytes": sum(row["size_bytes"] for row in rows),
    }
    return {**body, "artifact_sha256": _sha256(body)}


def observed_artifact_manifest(
    *,
    artifact_id: str,
    base_model_identity: str,
    adapter_identity: str,
    root: Path | str,
    names: Sequence[str],
) -> dict[str, Any]:
    """Read the named files back off disk and describe what is actually there.

    Rollback verification uses this: the restored bytes are re-hashed rather
    than assumed, so a restore that lands the wrong content cannot be recorded
    as an exact rollback.
    """

    base = Path(root).expanduser()
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        _fail("permanent_distillation_artifact_files_invalid")

    files: list[dict[str, Any]] = []
    for name in names:
        if not isinstance(name, str) or not _NAME_PATTERN.match(name):
            _fail("permanent_distillation_artifact_file_name_invalid")
        target = base / name
        try:
            if not target.is_file() or target.is_symlink():
                _fail("permanent_distillation_artifact_file_missing")
            digest = hashlib.sha256()
            size = 0
            with target.open("rb") as handle:
                while True:
                    chunk = handle.read(_ARTIFACT_READ_CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise PermanentDistillationError(
                "permanent_distillation_artifact_file_unreadable"
            ) from exc
        files.append(
            {"name": name, "sha256": digest.hexdigest(), "size_bytes": size}
        )

    return artifact_manifest(
        artifact_id=artifact_id,
        base_model_identity=base_model_identity,
        adapter_identity=adapter_identity,
        files=files,
    )


def _validated_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("permanent_distillation_artifact_invalid")
    if value.get("schema") != ARTIFACT_SCHEMA:
        _fail("permanent_distillation_artifact_schema_invalid")
    files = value.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        _fail("permanent_distillation_artifact_files_invalid")
    normalized = artifact_manifest(
        artifact_id=value.get("artifact_id"),
        base_model_identity=value.get("base_model_identity"),
        adapter_identity=value.get("adapter_identity"),
        files=[dict(row) if isinstance(row, Mapping) else row for row in files],
    )
    if dict(value) != normalized:
        _fail("permanent_distillation_artifact_differs")
    return normalized


# ---------------------------------------------------------------------------
# The complete regression gate set
# ---------------------------------------------------------------------------


def gate_result(
    *,
    gate: str,
    battery_schema: str,
    probes_graded: int,
    probes_passed: int,
    verdict: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    """Describe one battery that actually ran."""

    if gate not in REQUIRED_GATES:
        _fail("permanent_distillation_gate_unknown")
    graded = _required_index(probes_graded, "permanent_distillation_gate_probes_invalid")
    passed = _required_index(probes_passed, "permanent_distillation_gate_probes_invalid")
    if passed > graded:
        _fail("permanent_distillation_gate_probes_invalid")
    if verdict not in GATE_VERDICTS:
        _fail("permanent_distillation_gate_verdict_invalid")
    if not _is_sha256(evidence_sha256):
        _fail("permanent_distillation_gate_evidence_invalid")
    return {
        "gate": gate,
        "battery_schema": _required_text(
            battery_schema, "permanent_distillation_gate_battery_invalid"
        ),
        "probes_graded": graded,
        "probes_passed": passed,
        "verdict": verdict,
        "evidence_sha256": evidence_sha256,
    }


def gate_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Normalize the gate set and refuse anything short of complete.

    Completeness is checked before any verdict is read. An incomplete report is
    an invalid report, not a failing one -- the caller must not be able to
    convert "we did not measure memory" into "memory did not block us".
    """

    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        _fail("permanent_distillation_gate_report_invalid")

    rows: dict[str, dict[str, Any]] = {}
    for raw in results:
        if not isinstance(raw, Mapping) or set(raw) != _GATE_FIELDS:
            _fail("permanent_distillation_gate_fields_differ")
        row = gate_result(
            gate=raw["gate"],
            battery_schema=raw["battery_schema"],
            probes_graded=raw["probes_graded"],
            probes_passed=raw["probes_passed"],
            verdict=raw["verdict"],
            evidence_sha256=raw["evidence_sha256"],
        )
        if row["gate"] in rows:
            _fail("permanent_distillation_gate_duplicate")
        rows[row["gate"]] = row

    if set(rows) != set(REQUIRED_GATES):
        _fail("permanent_distillation_gate_set_incomplete")

    body = {
        "schema": GATE_REPORT_SCHEMA,
        "gates": [rows[name] for name in REQUIRED_GATES],
    }
    return {**body, "gate_report_sha256": _sha256(body)}


def _validated_gate_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != GATE_REPORT_SCHEMA:
        _fail("permanent_distillation_gate_report_invalid")
    gates = value.get("gates")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)):
        _fail("permanent_distillation_gate_report_invalid")
    normalized = gate_report(
        [dict(row) if isinstance(row, Mapping) else row for row in gates]
    )
    if dict(value) != normalized:
        _fail("permanent_distillation_gate_report_differs")
    return normalized


def evaluate_promotion(
    *,
    report: Mapping[str, Any],
    candidate_artifact: Mapping[str, Any],
    incumbent_artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether a candidate may become a permanent generation.

    Returns a decision either way. Every refusal names the gates responsible,
    so a blocked promotion is engineering evidence rather than a silent stop.
    """

    normalized_report = _validated_gate_report(report)
    candidate = _validated_artifact(candidate_artifact)
    incumbent = (
        None if incumbent_artifact is None else _validated_artifact(incumbent_artifact)
    )

    refusals: list[dict[str, Any]] = []
    for row in normalized_report["gates"]:
        floor = _MINIMUM_PROBES[row["gate"]]
        if row["probes_graded"] < floor:
            refusals.append(
                {
                    "gate": row["gate"],
                    "reason": "gate_did_not_measure",
                    "probes_graded": row["probes_graded"],
                    "probes_required": floor,
                }
            )
            continue
        if row["verdict"] != PASS:
            refusals.append(
                {
                    "gate": row["gate"],
                    "reason": "gate_failed",
                    "probes_graded": row["probes_graded"],
                    "probes_passed": row["probes_passed"],
                }
            )

    if incumbent is not None and incumbent["artifact_sha256"] == candidate["artifact_sha256"]:
        refusals.append(
            {
                "gate": "artifact_identity",
                "reason": "candidate_equals_incumbent",
                "artifact_sha256": candidate["artifact_sha256"],
            }
        )

    body = {
        "schema": DECISION_SCHEMA,
        "decision": REFUSE if refusals else ADMIT,
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "incumbent_artifact_sha256": (
            None if incumbent is None else incumbent["artifact_sha256"]
        ),
        "gate_report_sha256": normalized_report["gate_report_sha256"],
        "gates_required": list(REQUIRED_GATES),
        "refusals": refusals,
    }
    return {**body, "decision_sha256": _sha256(body)}


# ---------------------------------------------------------------------------
# Versioned generations and exact rollback
# ---------------------------------------------------------------------------


def _generation(
    *,
    kind: str,
    generation_index: int,
    parent_generation_sha256: str,
    artifact: Mapping[str, Any],
    provenance: Mapping[str, Any],
    gate_report_sha256: str | None,
    decision_sha256: str | None,
    restores_generation_sha256: str | None,
    created_at_unix: int,
) -> dict[str, Any]:
    if kind not in GENERATION_KINDS:
        _fail("permanent_distillation_generation_kind_invalid")
    if not isinstance(provenance, Mapping):
        _fail("permanent_distillation_provenance_invalid")
    if type(created_at_unix) is not int or created_at_unix <= 0:
        _fail("permanent_distillation_generation_time_invalid")
    if parent_generation_sha256 != GENESIS_PARENT and not _is_sha256(
        parent_generation_sha256
    ):
        _fail("permanent_distillation_generation_parent_invalid")
    body = {
        "schema": GENERATION_SCHEMA,
        "kind": kind,
        "generation_index": _required_index(
            generation_index, "permanent_distillation_generation_index_invalid"
        ),
        "parent_generation_sha256": parent_generation_sha256,
        "artifact": _validated_artifact(artifact),
        "provenance": json.loads(json.dumps(dict(provenance), sort_keys=True)),
        "gate_report_sha256": gate_report_sha256,
        "decision_sha256": decision_sha256,
        "restores_generation_sha256": restores_generation_sha256,
        "created_at_unix": created_at_unix,
    }
    return {**body, "generation_sha256": _sha256(body)}


def baseline_generation(
    *,
    artifact: Mapping[str, Any],
    provenance: Mapping[str, Any],
    created_at_unix: int,
) -> dict[str, Any]:
    """Open a lineage at the frozen pre-treatment artifact."""

    return _generation(
        kind=BASELINE,
        generation_index=0,
        parent_generation_sha256=GENESIS_PARENT,
        artifact=artifact,
        provenance=provenance,
        gate_report_sha256=None,
        decision_sha256=None,
        restores_generation_sha256=None,
        created_at_unix=created_at_unix,
    )


def promote_generation(
    *,
    lineage: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
    provenance: Mapping[str, Any],
    created_at_unix: int,
) -> dict[str, Any]:
    """Append a promotion, or refuse with the decision that blocked it."""

    records = validate_lineage(lineage)
    head = records[-1]
    decision = evaluate_promotion(
        report=report,
        candidate_artifact=artifact,
        incumbent_artifact=head["artifact"],
    )
    if decision["decision"] != ADMIT:
        raise PermanentDistillationRefusalError(decision)
    return _generation(
        kind=PROMOTION,
        generation_index=head["generation_index"] + 1,
        parent_generation_sha256=head["generation_sha256"],
        artifact=artifact,
        provenance=provenance,
        gate_report_sha256=decision["gate_report_sha256"],
        decision_sha256=decision["decision_sha256"],
        restores_generation_sha256=None,
        created_at_unix=created_at_unix,
    )


def rollback_generation(
    *,
    lineage: Sequence[Mapping[str, Any]],
    restores_generation_sha256: str,
    observed_artifact: Mapping[str, Any],
    provenance: Mapping[str, Any],
    created_at_unix: int,
) -> dict[str, Any]:
    """Append an exact rollback to an earlier generation in this lineage.

    ``observed_artifact`` must be read back off the restored files. The record
    is refused unless those bytes equal the target generation's recorded bytes
    exactly, which is what makes the rollback a proof rather than an intention.
    """

    records = validate_lineage(lineage)
    head = records[-1]
    if not _is_sha256(restores_generation_sha256):
        _fail("permanent_distillation_rollback_target_invalid")
    target = next(
        (
            row
            for row in records
            if row["generation_sha256"] == restores_generation_sha256
        ),
        None,
    )
    if target is None:
        _fail("permanent_distillation_rollback_target_unknown")
    if target["generation_sha256"] == head["generation_sha256"]:
        _fail("permanent_distillation_rollback_target_is_head")

    observed = _validated_artifact(observed_artifact)
    if observed != target["artifact"]:
        _fail("permanent_distillation_rollback_not_exact")

    return _generation(
        kind=ROLLBACK,
        generation_index=head["generation_index"] + 1,
        parent_generation_sha256=head["generation_sha256"],
        artifact=observed,
        provenance=provenance,
        gate_report_sha256=None,
        decision_sha256=None,
        restores_generation_sha256=target["generation_sha256"],
        created_at_unix=created_at_unix,
    )


def validate_lineage(records: Any) -> list[dict[str, Any]]:
    """Replay a promotion lineage and refuse any chain, order, or kind defect."""

    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
        or len(records) > _MAX_LINEAGE_RECORDS
    ):
        _fail("permanent_distillation_lineage_invalid")

    replayed: list[dict[str, Any]] = []
    known: dict[str, dict[str, Any]] = {}
    previous = GENESIS_PARENT
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping) or raw.get("schema") != GENERATION_SCHEMA:
            _fail("permanent_distillation_generation_invalid")
        normalized = _generation(
            kind=raw.get("kind"),
            generation_index=raw.get("generation_index"),
            parent_generation_sha256=raw.get("parent_generation_sha256"),
            artifact=raw.get("artifact"),
            provenance=raw.get("provenance"),
            gate_report_sha256=raw.get("gate_report_sha256"),
            decision_sha256=raw.get("decision_sha256"),
            restores_generation_sha256=raw.get("restores_generation_sha256"),
            created_at_unix=raw.get("created_at_unix"),
        )
        if dict(raw) != normalized:
            _fail("permanent_distillation_generation_differs")
        if (
            normalized["generation_index"] != index
            or normalized["parent_generation_sha256"] != previous
        ):
            _fail("permanent_distillation_lineage_chain_differs")

        kind = normalized["kind"]
        if index == 0:
            if kind != BASELINE:
                _fail("permanent_distillation_lineage_genesis_differs")
        elif kind == BASELINE:
            _fail("permanent_distillation_lineage_second_baseline")

        if kind == PROMOTION:
            if not _is_sha256(normalized["gate_report_sha256"]) or not _is_sha256(
                normalized["decision_sha256"]
            ):
                _fail("permanent_distillation_promotion_evidence_missing")
            if normalized["restores_generation_sha256"] is not None:
                _fail("permanent_distillation_promotion_restores_invalid")
        elif kind == ROLLBACK:
            target_sha = normalized["restores_generation_sha256"]
            if not _is_sha256(target_sha):
                _fail("permanent_distillation_rollback_target_invalid")
            target = known.get(target_sha)
            if target is None:
                _fail("permanent_distillation_rollback_target_unknown")
            if normalized["artifact"] != target["artifact"]:
                _fail("permanent_distillation_rollback_not_exact")
            if (
                normalized["gate_report_sha256"] is not None
                or normalized["decision_sha256"] is not None
            ):
                _fail("permanent_distillation_rollback_evidence_invalid")
        else:
            if (
                normalized["gate_report_sha256"] is not None
                or normalized["decision_sha256"] is not None
                or normalized["restores_generation_sha256"] is not None
            ):
                _fail("permanent_distillation_baseline_evidence_invalid")

        if normalized["generation_sha256"] in known:
            _fail("permanent_distillation_generation_duplicate")
        known[normalized["generation_sha256"]] = normalized
        previous = normalized["generation_sha256"]
        replayed.append(normalized)

    return replayed


def active_artifact(lineage: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the artifact a validated lineage currently says is in force."""

    return validate_lineage(lineage)[-1]["artifact"]


__all__ = [
    "ADMIT",
    "ARTIFACT_SCHEMA",
    "BASELINE",
    "DECISION_SCHEMA",
    "FAIL",
    "GATE_REPORT_SCHEMA",
    "GENERATION_SCHEMA",
    "GENERATION_KINDS",
    "GENESIS_PARENT",
    "PASS",
    "PROMOTION",
    "REFUSE",
    "REQUIRED_GATES",
    "ROLLBACK",
    "PermanentDistillationError",
    "PermanentDistillationRefusalError",
    "active_artifact",
    "artifact_manifest",
    "baseline_generation",
    "evaluate_promotion",
    "gate_report",
    "gate_result",
    "observed_artifact_manifest",
    "promote_generation",
    "rollback_generation",
    "validate_lineage",
]
