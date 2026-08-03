"""Native-artifact adapters for recurrent checkpoint promotion evidence."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.learning.permanent_distillation import (
    PASS as DISTILLATION_PASS,
)
from core.learning.permanent_distillation import (
    REQUIRED_GATES as DISTILLATION_GATES,
)
from core.learning.permanent_distillation import (
    gate_report,
    gate_result,
)
from core.learning.recurrent_checkpoint_evidence import (
    INDEPENDENT_VERDICT_SCHEMA,
    PAIRED_GRADE_SCHEMA,
    PILOT_RESULT_SCHEMA,
    RecurrentCheckpointEvidenceError,
    branch_specialization_report,
    build_recurrent_promotion_gates,
    halt_revert_report,
)
from core.learning.recurrent_checkpoint_promotion import (
    FAIL,
    PASS,
    PROMOTE,
    UNMEASURED,
    checkpoint_candidate,
    evaluate_checkpoint_candidate,
)

PRODUCER = "a" * 64
PLAN_ARTIFACT = "9" * 64


def _digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _seal(value: dict[str, object], key: str) -> dict[str, object]:
    value[key] = _digest(value)
    return value


def _grade() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": PAIRED_GRADE_SCHEMA,
        "plan_sha256": "1" * 64,
        "observed_task_count": 40,
        "observed_cell_count": 160,
        "verdict": "gain_preverified",
        "domain_counts": {f"family-{index}": 6 for index in range(7)},
        "comparisons": {
            "adapter_rlc_gain": {"tier": "PROVEN", "evidence": {}},
            "adapter_effect_under_rlc": {"tier": "PROVEN", "evidence": {}},
            "adapter_effect_under_vanilla": {
                "tier": "PROVEN",
                "evidence": {
                    "all_families_noninferior": True,
                    "regressed_families": [],
                },
            },
        },
        "interaction": {
            "lower": {"numerator": 1, "denominator": 10},
            "one_sided_exact_sign_flip_p": {"numerator": 1, "denominator": 100},
        },
    }
    return _seal(body, "grade_sha256")


def _arm(*, correct: int, rlc: bool) -> dict[str, object]:
    return {
        "correct": correct,
        "total": 40,
        "score_reasons": {"correct": correct, "incorrect_or_schema_mismatch": 40 - correct},
        "decode_terminations": ({"contract_complete": 40} if rlc else {"ordinary_generation": 40}),
    }


def _pilot(grade: dict[str, object]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": PILOT_RESULT_SCHEMA,
        "grade_sha256": grade["grade_sha256"],
        "independent_semantic_grade_sha256": "2" * 64,
        "plan_sha256": grade["plan_sha256"],
        "evidence_valid": True,
        "passed": True,
        "advance_rules": {
            "all_56_cells_commit_and_replay": True,
            "all_model_adapter_source_reset_scorer_and_detached_receipts_validate": True,
            "adapter_vanilla_output_is_byte_identical_to_base_vanilla_per_task": True,
            "adapter_vanilla_total_correct_is_not_below_base_vanilla": True,
            "adapter_rlc_total_correct_strictly_exceeds_adapter_vanilla": True,
            "adapter_rlc_total_correct_strictly_exceeds_base_rlc": True,
        },
        "mechanics": {"ordinary_generation_exact_match": True},
        "arm_results": {
            "base_vanilla": _arm(correct=25, rlc=False),
            "base_rlc": _arm(correct=24, rlc=True),
            "adapter_vanilla": _arm(correct=25, rlc=False),
            "adapter_rlc": _arm(correct=32, rlc=True),
        },
    }
    return _seal(body, "verdict_sha256")


def _independent(grade: dict[str, object]) -> dict[str, object]:
    return {
        "schema": INDEPENDENT_VERDICT_SCHEMA,
        "plan_sha256": PLAN_ARTIFACT,
        "passed": True,
        "failures": [],
        "committed_records": 160,
        "published_verdict": grade["verdict"],
        "recomputed_verdict": grade["verdict"],
        "verified_verdict": grade["verdict"],
        "production_semantic_grade_sha256": "2" * 64,
        "independent_semantic_grade_sha256": "2" * 64,
    }


def _branch_report(*, cosine: float = 0.7) -> dict[str, object]:
    return branch_specialization_report(
        [
            {
                "task_id": f"task-{index}",
                "branch_count": 2,
                "distinct_role_count": 2,
                "max_pairwise_cosine": cosine,
                "selection_margin_positive": True,
            }
            for index in range(40)
        ]
    )


def _halt_report(*, exact: bool = True) -> dict[str, object]:
    return halt_revert_report(
        [
            {
                "case_id": f"lesion-{index}",
                "trigger": "non_finite_state",
                "halt_observed": True,
                "reverted_to_best": True,
                "best_state_sha256": "3" * 64,
                "restored_state_sha256": "3" * 64 if exact else "4" * 64,
            }
            for index in range(20)
        ]
    )


def _retention_report() -> dict[str, object]:
    return gate_report(
        [
            gate_result(
                gate=gate,
                battery_schema=f"test.{gate}.v1",
                probes_graded=20,
                probes_passed=20,
                verdict=DISTILLATION_PASS,
                evidence_sha256=f"{index + 20:064x}",
            )
            for index, gate in enumerate(DISTILLATION_GATES)
        ]
    )


def _all_evidence() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    grade = _grade()
    return _pilot(grade), grade, _independent(grade)


def test_complete_native_evidence_can_produce_promotable_candidate() -> None:
    pilot, grade, independent = _all_evidence()
    gates = build_recurrent_promotion_gates(
        pilot_result=pilot,
        paired_grade=grade,
        independent_verdict=independent,
        plan_artifact_sha256=PLAN_ARTIFACT,
        producer_sha256=PRODUCER,
        branch_report=_branch_report(),
        halt_report=_halt_report(),
        permanent_retention_report=_retention_report(),
    )
    candidate = checkpoint_candidate(
        candidate_id="candidate-2",
        parent_id="parent-1",
        candidate_artifact_sha256="5" * 64,
        parent_artifact_sha256="6" * 64,
        candidate_active=False,
        source_commit="7" * 40,
        source_closure_sha256="8" * 64,
        campaign_grade_sha256=grade["grade_sha256"],
        independent_verdict_sha256=_digest(independent),
        gates=gates,
    )
    decision = evaluate_checkpoint_candidate(
        candidate, expected_verifiers={gate["gate"]: PRODUCER for gate in gates}
    )

    assert {gate["status"] for gate in gates} == {PASS}
    assert decision["outcome"] == PROMOTE


def test_absent_later_batteries_are_unmeasured_not_passing() -> None:
    pilot, grade, independent = _all_evidence()
    gates = build_recurrent_promotion_gates(
        pilot_result=pilot,
        paired_grade=grade,
        independent_verdict=independent,
        plan_artifact_sha256=PLAN_ARTIFACT,
        producer_sha256=PRODUCER,
    )

    statuses = {gate["gate"]: gate["status"] for gate in gates}
    assert statuses["branch_specialization"] == UNMEASURED
    assert statuses["halt_revert"] == UNMEASURED
    assert statuses["broad_retention"] == UNMEASURED


def test_negative_directional_gain_and_malformed_decode_fail_their_gates() -> None:
    pilot, grade, independent = _all_evidence()
    pilot["advance_rules"]["adapter_rlc_total_correct_strictly_exceeds_adapter_vanilla"] = False
    pilot["arm_results"]["adapter_rlc"]["score_reasons"] = {
        "correct": 32,
        "incorrect_or_schema_mismatch": 7,
        "final_answer_marker_count_invalid": 1,
    }
    pilot["arm_results"]["adapter_rlc"]["decode_terminations"] = {
        "contract_complete": 39,
        "token_limit_contract_incomplete": 1,
    }
    pilot.pop("verdict_sha256")
    _seal(pilot, "verdict_sha256")

    gates = build_recurrent_promotion_gates(
        pilot_result=pilot,
        paired_grade=grade,
        independent_verdict=independent,
        plan_artifact_sha256=PLAN_ARTIFACT,
        producer_sha256=PRODUCER,
        branch_report=_branch_report(),
        halt_report=_halt_report(),
        permanent_retention_report=_retention_report(),
    )
    statuses = {gate["gate"]: gate["status"] for gate in gates}
    assert statuses["recurrent_gain"] == FAIL
    assert statuses["contract_integrity"] == FAIL


def test_branch_collapse_and_inexact_revert_are_measured_failures() -> None:
    pilot, grade, independent = _all_evidence()
    gates = build_recurrent_promotion_gates(
        pilot_result=pilot,
        paired_grade=grade,
        independent_verdict=independent,
        plan_artifact_sha256=PLAN_ARTIFACT,
        producer_sha256=PRODUCER,
        branch_report=_branch_report(cosine=0.999),
        halt_report=_halt_report(exact=False),
        permanent_retention_report=_retention_report(),
    )
    statuses = {gate["gate"]: gate["status"] for gate in gates}
    assert statuses["branch_specialization"] == FAIL
    assert statuses["halt_revert"] == FAIL


def test_rehashed_derived_branch_and_revert_claims_fail_semantic_replay() -> None:
    pilot, grade, independent = _all_evidence()
    branch = _branch_report(cosine=0.999)
    branch["cases"][0]["specialized"] = True
    branch_material = dict(branch)
    branch_material.pop("report_sha256")
    branch["report_sha256"] = _digest(branch_material)
    with pytest.raises(RecurrentCheckpointEvidenceError, match="branch_report_replay"):
        build_recurrent_promotion_gates(
            pilot_result=pilot,
            paired_grade=grade,
            independent_verdict=independent,
            plan_artifact_sha256=PLAN_ARTIFACT,
            producer_sha256=PRODUCER,
            branch_report=branch,
        )

    halt = _halt_report(exact=False)
    halt["cases"][0]["passed"] = True
    halt_material = dict(halt)
    halt_material.pop("report_sha256")
    halt["report_sha256"] = _digest(halt_material)
    with pytest.raises(RecurrentCheckpointEvidenceError, match="halt_revert_report_replay"):
        build_recurrent_promotion_gates(
            pilot_result=pilot,
            paired_grade=grade,
            independent_verdict=independent,
            plan_artifact_sha256=PLAN_ARTIFACT,
            producer_sha256=PRODUCER,
            halt_report=halt,
        )


def test_tampered_native_digest_and_cross_campaign_plan_are_rejected() -> None:
    pilot, grade, independent = _all_evidence()
    tampered = copy.deepcopy(pilot)
    tampered["passed"] = False
    with pytest.raises(RecurrentCheckpointEvidenceError, match="pilot_result_digest_invalid"):
        build_recurrent_promotion_gates(
            pilot_result=tampered,
            paired_grade=grade,
            independent_verdict=independent,
            plan_artifact_sha256=PLAN_ARTIFACT,
            producer_sha256=PRODUCER,
        )

    pilot, grade, independent = _all_evidence()
    independent["plan_sha256"] = "f" * 64
    with pytest.raises(RecurrentCheckpointEvidenceError, match="campaign_identity_mismatch"):
        build_recurrent_promotion_gates(
            pilot_result=pilot,
            paired_grade=grade,
            independent_verdict=independent,
            plan_artifact_sha256=PLAN_ARTIFACT,
            producer_sha256=PRODUCER,
        )
