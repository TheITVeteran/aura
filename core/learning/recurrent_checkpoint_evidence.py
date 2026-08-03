"""Derive recurrent checkpoint promotion gates from native proof artifacts.

The promotion state machine accepts canonical gate rows, but callers must not
hand-author those rows.  This module translates the existing paired-campaign,
independent-replay, lesion, branch, and permanent-retention artifacts into the
ten exact gates required by :mod:`recurrent_checkpoint_promotion`.

An absent branch, halt/revert, or broad-retention report is emitted as
``UNMEASURED``.  This keeps early directional pilots useful for diagnosis while
making them structurally incapable of promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.exact_paired_grade import exact_interaction_proven
from core.learning.permanent_distillation import gate_report
from core.learning.recurrent_checkpoint_promotion import (
    FAIL,
    PASS,
    UNMEASURED,
    evidence_gate,
)

PILOT_RESULT_SCHEMA: Final = "aura.latent_cortex.resident_pilot_result.v1"
PAIRED_GRADE_SCHEMA: Final = "aura.latent_cortex.resident_paired_grade.v2"
INDEPENDENT_VERDICT_SCHEMA: Final = "aura.latent_cortex.independent_evidence_verdict.v2"
BRANCH_REPORT_SCHEMA: Final = "aura.rlc.branch_specialization_report.v1"
HALT_REVERT_REPORT_SCHEMA: Final = "aura.rlc.halt_revert_report.v1"

_SHA256_LENGTH = 64
_RLC_ARMS = ("base_rlc", "adapter_rlc")
_ALL_ARMS = ("base_vanilla", "base_rlc", "adapter_vanilla", "adapter_rlc")
_VALID_SCORE_REASONS = frozenset({"correct", "incorrect_or_schema_mismatch"})


class RecurrentCheckpointEvidenceError(ValueError):
    """A native proof artifact is malformed or internally inconsistent."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise RecurrentCheckpointEvidenceError(code)


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
        raise RecurrentCheckpointEvidenceError(
            "recurrent_checkpoint_evidence_noncanonical"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _verified_document(
    value: Any,
    *,
    schema: str,
    digest_key: str | None,
    role: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != schema:
        _fail(f"recurrent_checkpoint_{role}_schema_invalid")
    document = dict(value)
    if digest_key is not None:
        claimed = document.pop(digest_key, None)
        if not _is_sha(claimed) or claimed != _digest(document):
            _fail(f"recurrent_checkpoint_{role}_digest_invalid")
        document[digest_key] = claimed
    return document


def _count(value: Any, *, role: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"recurrent_checkpoint_{role}_count_invalid")
    return value


def branch_specialization_report(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build per-task structural evidence; wording differences are insufficient."""

    if (
        not isinstance(cases, Sequence)
        or isinstance(cases, (str, bytes, bytearray))
        or not cases
        or len(cases) > 100_000
    ):
        _fail("recurrent_checkpoint_branch_cases_invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in cases:
        required = {
            "task_id",
            "branch_count",
            "distinct_role_count",
            "max_pairwise_cosine",
            "selection_margin_positive",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            _fail("recurrent_checkpoint_branch_case_schema_invalid")
        task_id = raw["task_id"]
        branch_count = raw["branch_count"]
        distinct = raw["distinct_role_count"]
        cosine = raw["max_pairwise_cosine"]
        margin = raw["selection_margin_positive"]
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in seen
            or type(branch_count) is not int
            or type(distinct) is not int
            or branch_count < 2
            or not 2 <= distinct <= branch_count
            or isinstance(cosine, bool)
            or not isinstance(cosine, (int, float))
            or not math.isfinite(float(cosine))
            or not -1.0 <= float(cosine) <= 1.0
            or type(margin) is not bool
        ):
            _fail("recurrent_checkpoint_branch_case_invalid")
        seen.add(task_id)
        normalized.append(
            {
                "task_id": task_id,
                "branch_count": branch_count,
                "distinct_role_count": distinct,
                "max_pairwise_cosine": float(cosine),
                "selection_margin_positive": margin,
                "specialized": float(cosine) <= 0.98 and margin,
            }
        )
    body = {"schema": BRANCH_REPORT_SCHEMA, "cases": normalized}
    return {**body, "report_sha256": _digest(body)}


def halt_revert_report(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build lesion evidence proving containment and exact best-state restore."""

    if (
        not isinstance(cases, Sequence)
        or isinstance(cases, (str, bytes, bytearray))
        or not cases
        or len(cases) > 100_000
    ):
        _fail("recurrent_checkpoint_halt_revert_cases_invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in cases:
        required = {
            "case_id",
            "trigger",
            "halt_observed",
            "reverted_to_best",
            "best_state_sha256",
            "restored_state_sha256",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            _fail("recurrent_checkpoint_halt_revert_case_schema_invalid")
        case_id = raw["case_id"]
        trigger = raw["trigger"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen
            or not isinstance(trigger, str)
            or not trigger
            or type(raw["halt_observed"]) is not bool
            or type(raw["reverted_to_best"]) is not bool
            or not _is_sha(raw["best_state_sha256"])
            or not _is_sha(raw["restored_state_sha256"])
        ):
            _fail("recurrent_checkpoint_halt_revert_case_invalid")
        seen.add(case_id)
        exact = raw["best_state_sha256"] == raw["restored_state_sha256"]
        normalized.append(
            {
                **dict(raw),
                "passed": raw["halt_observed"] and raw["reverted_to_best"] and exact,
            }
        )
    body = {"schema": HALT_REVERT_REPORT_SCHEMA, "cases": normalized}
    return {**body, "report_sha256": _digest(body)}


def _optional_report(
    value: Mapping[str, Any] | None,
    *,
    schema: str,
    role: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    return _verified_document(
        value,
        schema=schema,
        digest_key="report_sha256",
        role=role,
    )


def _unmeasured(gate: str, *, producer_sha256: str) -> dict[str, Any]:
    return evidence_gate(
        gate=gate,
        status=UNMEASURED,
        probes_graded=0,
        probes_passed=0,
        evidence_sha256=_digest({"gate": gate, "status": UNMEASURED}),
        verifier_sha256=producer_sha256,
    )


def build_recurrent_promotion_gates(
    *,
    pilot_result: Mapping[str, Any],
    paired_grade: Mapping[str, Any],
    independent_verdict: Mapping[str, Any],
    plan_artifact_sha256: str,
    producer_sha256: str,
    branch_report: Mapping[str, Any] | None = None,
    halt_report: Mapping[str, Any] | None = None,
    permanent_retention_report: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Derive all ten checkpoint gates from their native evidence surfaces."""

    if not _is_sha(producer_sha256) or not _is_sha(plan_artifact_sha256):
        _fail("recurrent_checkpoint_evidence_producer_invalid")
    pilot = _verified_document(
        pilot_result,
        schema=PILOT_RESULT_SCHEMA,
        digest_key="verdict_sha256",
        role="pilot_result",
    )
    grade = _verified_document(
        paired_grade,
        schema=PAIRED_GRADE_SCHEMA,
        digest_key="grade_sha256",
        role="paired_grade",
    )
    independent = _verified_document(
        independent_verdict,
        schema=INDEPENDENT_VERDICT_SCHEMA,
        digest_key=None,
        role="independent_verdict",
    )
    if (
        pilot.get("grade_sha256") != grade["grade_sha256"]
        or pilot.get("plan_sha256") != grade.get("plan_sha256")
        # The independent verifier intentionally hashes the complete canonical
        # plan artifact (including its semantic plan_sha256 field).  Keep both
        # identities explicit rather than treating them as interchangeable.
        or independent.get("plan_sha256") != plan_artifact_sha256
    ):
        _fail("recurrent_checkpoint_campaign_identity_mismatch")

    arms = pilot.get("arm_results")
    task_count = grade.get("observed_task_count")
    if (
        not isinstance(arms, Mapping)
        or set(arms) != set(_ALL_ARMS)
        or type(task_count) is not int
        or task_count <= 0
        or any(
            not isinstance(arms[arm], Mapping) or arms[arm].get("total") != task_count
            for arm in _ALL_ARMS
        )
    ):
        _fail("recurrent_checkpoint_arm_summary_invalid")
    rules = pilot.get("advance_rules")
    mechanics = pilot.get("mechanics")
    comparisons = grade.get("comparisons")
    if not all(isinstance(value, Mapping) for value in (rules, mechanics, comparisons)):
        _fail("recurrent_checkpoint_campaign_summary_invalid")

    campaign_evidence_sha256 = _digest(
        {
            "pilot_result_sha256": pilot["verdict_sha256"],
            "paired_grade_sha256": grade["grade_sha256"],
        }
    )
    four_arm_pass = (
        pilot.get("evidence_valid") is True
        and pilot.get("passed") is True
        and grade.get("observed_cell_count") == task_count * len(_ALL_ARMS)
        and rules.get("all_model_adapter_source_reset_scorer_and_detached_receipts_validate")
        is True
        and mechanics.get("ordinary_generation_exact_match") is True
    )
    rows = [
        evidence_gate(
            gate="four_arm_semantics",
            status=PASS if four_arm_pass else FAIL,
            probes_graded=task_count,
            probes_passed=task_count if four_arm_pass else 0,
            evidence_sha256=campaign_evidence_sha256,
            verifier_sha256=producer_sha256,
        )
    ]

    base_vanilla = arms["base_vanilla"]
    adapter_vanilla = arms["adapter_vanilla"]
    vanilla_pass = (
        rules.get("adapter_vanilla_output_is_byte_identical_to_base_vanilla_per_task") is True
        and rules.get("adapter_vanilla_total_correct_is_not_below_base_vanilla") is True
        and _count(adapter_vanilla.get("correct"), role="adapter_vanilla_correct")
        >= _count(base_vanilla.get("correct"), role="base_vanilla_correct")
    )
    rows.append(
        evidence_gate(
            gate="vanilla_no_regression",
            status=PASS if vanilla_pass else FAIL,
            probes_graded=task_count,
            probes_passed=task_count if vanilla_pass else 0,
            evidence_sha256=campaign_evidence_sha256,
            verifier_sha256=producer_sha256,
        )
    )

    adapter_rlc = arms["adapter_rlc"]
    base_rlc = arms["base_rlc"]
    required_comparisons = ("adapter_rlc_gain", "adapter_effect_under_rlc")
    recurrent_gain_pass = (
        rules.get("adapter_rlc_total_correct_strictly_exceeds_adapter_vanilla") is True
        and rules.get("adapter_rlc_total_correct_strictly_exceeds_base_rlc") is True
        and _count(adapter_rlc.get("correct"), role="adapter_rlc_correct")
        > _count(adapter_vanilla.get("correct"), role="adapter_vanilla_correct")
        and _count(adapter_rlc.get("correct"), role="adapter_rlc_correct")
        > _count(base_rlc.get("correct"), role="base_rlc_correct")
        and all(
            isinstance(comparisons.get(name), Mapping)
            and comparisons[name].get("tier") == "PROVEN"
            for name in required_comparisons
        )
    )
    rows.append(
        evidence_gate(
            gate="recurrent_gain",
            status=PASS if recurrent_gain_pass else FAIL,
            probes_graded=task_count,
            probes_passed=task_count if recurrent_gain_pass else 0,
            evidence_sha256=grade["grade_sha256"],
            verifier_sha256=producer_sha256,
        )
    )

    try:
        interaction_pass = exact_interaction_proven(grade.get("interaction", {}))
    except ValueError as exc:
        raise RecurrentCheckpointEvidenceError(
            "recurrent_checkpoint_interaction_invalid"
        ) from exc
    rows.append(
        evidence_gate(
            gate="positive_interaction",
            status=PASS if interaction_pass else FAIL,
            probes_graded=task_count,
            probes_passed=task_count if interaction_pass else 0,
            evidence_sha256=grade["grade_sha256"],
            verifier_sha256=producer_sha256,
        )
    )

    family_comparison = comparisons.get("adapter_effect_under_vanilla")
    family_evidence = (
        family_comparison.get("evidence") if isinstance(family_comparison, Mapping) else None
    )
    domain_counts = grade.get("domain_counts")
    if not isinstance(domain_counts, Mapping) or not domain_counts:
        _fail("recurrent_checkpoint_domain_counts_invalid")
    family_count = len(domain_counts)
    family_pass = (
        isinstance(family_evidence, Mapping)
        and family_evidence.get("all_families_noninferior") is True
        and not family_evidence.get("regressed_families")
    )
    rows.append(
        evidence_gate(
            gate="family_retention",
            status=PASS if family_pass else FAIL,
            probes_graded=family_count,
            probes_passed=family_count if family_pass else 0,
            evidence_sha256=grade["grade_sha256"],
            verifier_sha256=producer_sha256,
        )
    )

    branch = _optional_report(
        branch_report, schema=BRANCH_REPORT_SCHEMA, role="branch_report"
    )
    if branch is None:
        rows.append(_unmeasured("branch_specialization", producer_sha256=producer_sha256))
    else:
        cases = branch.get("cases")
        if not isinstance(cases, list) or not cases:
            _fail("recurrent_checkpoint_branch_report_cases_invalid")
        replayed_branch = branch_specialization_report(
            [
                {key: value for key, value in case.items() if key != "specialized"}
                for case in cases
                if isinstance(case, Mapping)
            ]
        )
        if branch != replayed_branch:
            _fail("recurrent_checkpoint_branch_report_replay_mismatch")
        passed = sum(1 for case in cases if isinstance(case, Mapping) and case.get("specialized"))
        rows.append(
            evidence_gate(
                gate="branch_specialization",
                status=PASS if passed == len(cases) else FAIL,
                probes_graded=len(cases),
                probes_passed=passed,
                evidence_sha256=branch["report_sha256"],
                verifier_sha256=producer_sha256,
            )
        )

    contract_total = 0
    contract_passed = 0
    for arm_name in _RLC_ARMS:
        summary = arms[arm_name]
        reasons = summary.get("score_reasons")
        terminations = summary.get("decode_terminations")
        if not isinstance(reasons, Mapping) or not isinstance(terminations, Mapping):
            _fail("recurrent_checkpoint_contract_summary_invalid")
        reason_total = sum(
            _count(count, role=f"{arm_name}_{reason}") for reason, count in reasons.items()
        )
        malformed = sum(
            _count(count, role=f"{arm_name}_{reason}")
            for reason, count in reasons.items()
            if reason not in _VALID_SCORE_REASONS
        )
        complete = _count(terminations.get("contract_complete", 0), role="contract_complete")
        contract_total += task_count
        contract_passed += (
            complete
            if reason_total == task_count and malformed == 0
            else max(0, complete - malformed)
        )
    contract_ok = contract_passed == contract_total
    rows.append(
        evidence_gate(
            gate="contract_integrity",
            status=PASS if contract_ok else FAIL,
            probes_graded=contract_total,
            probes_passed=contract_passed,
            evidence_sha256=pilot["verdict_sha256"],
            verifier_sha256=producer_sha256,
        )
    )

    halt = _optional_report(
        halt_report, schema=HALT_REVERT_REPORT_SCHEMA, role="halt_revert_report"
    )
    if halt is None:
        rows.append(_unmeasured("halt_revert", producer_sha256=producer_sha256))
    else:
        cases = halt.get("cases")
        if not isinstance(cases, list) or not cases:
            _fail("recurrent_checkpoint_halt_revert_report_cases_invalid")
        replayed_halt = halt_revert_report(
            [
                {key: value for key, value in case.items() if key != "passed"}
                for case in cases
                if isinstance(case, Mapping)
            ]
        )
        if halt != replayed_halt:
            _fail("recurrent_checkpoint_halt_revert_report_replay_mismatch")
        passed = sum(1 for case in cases if isinstance(case, Mapping) and case.get("passed"))
        rows.append(
            evidence_gate(
                gate="halt_revert",
                status=PASS if passed == len(cases) else FAIL,
                probes_graded=len(cases),
                probes_passed=passed,
                evidence_sha256=halt["report_sha256"],
                verifier_sha256=producer_sha256,
            )
        )

    if permanent_retention_report is None:
        rows.append(_unmeasured("broad_retention", producer_sha256=producer_sha256))
    else:
        if not isinstance(permanent_retention_report, Mapping):
            _fail("recurrent_checkpoint_broad_retention_report_invalid")
        try:
            normalized_retention = gate_report(permanent_retention_report.get("gates", []))
        except ValueError as exc:
            raise RecurrentCheckpointEvidenceError(
                "recurrent_checkpoint_broad_retention_report_invalid"
            ) from exc
        if dict(permanent_retention_report) != normalized_retention:
            _fail("recurrent_checkpoint_broad_retention_report_invalid")
        retention_gates = normalized_retention["gates"]
        retention_pass = all(row["verdict"] == PASS for row in retention_gates)
        rows.append(
            evidence_gate(
                gate="broad_retention",
                status=PASS if retention_pass else FAIL,
                probes_graded=len(retention_gates),
                probes_passed=sum(row["verdict"] == PASS for row in retention_gates),
                evidence_sha256=normalized_retention["gate_report_sha256"],
                verifier_sha256=producer_sha256,
            )
        )

    independent_pass = (
        independent.get("passed") is True
        and independent.get("failures") == []
        and independent.get("committed_records") == task_count * len(_ALL_ARMS)
        and independent.get("published_verdict") == grade.get("verdict")
        and independent.get("recomputed_verdict") == grade.get("verdict")
        and independent.get("verified_verdict") == grade.get("verdict")
        and independent.get("production_semantic_grade_sha256")
        == independent.get("independent_semantic_grade_sha256")
        == pilot.get("independent_semantic_grade_sha256")
    )
    rows.append(
        evidence_gate(
            gate="independent_replay",
            status=PASS if independent_pass else FAIL,
            probes_graded=task_count,
            probes_passed=task_count if independent_pass else 0,
            evidence_sha256=_digest(independent),
            verifier_sha256=producer_sha256,
        )
    )
    return rows


__all__ = [
    "BRANCH_REPORT_SCHEMA",
    "HALT_REVERT_REPORT_SCHEMA",
    "INDEPENDENT_VERDICT_SCHEMA",
    "PAIRED_GRADE_SCHEMA",
    "PILOT_RESULT_SCHEMA",
    "RecurrentCheckpointEvidenceError",
    "branch_specialization_report",
    "build_recurrent_promotion_gates",
    "halt_revert_report",
]
