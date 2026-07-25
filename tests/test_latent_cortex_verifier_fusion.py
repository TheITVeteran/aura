from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.verifier_fusion import (
    CHECKED_OUTCOME_SCHEMA,
    VERIFIER_IDS,
    VerifierFusionLedger,
    build_verifier_fusion_evidence,
    build_verifier_fusion_receipt,
    checked_signals_from_receipt,
    validate_verifier_fusion_evidence,
    validate_verifier_fusion_receipt,
)
from core.brain.llm.latent_cortex.worker_handler import config_from_job


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _signal(probability: float, source: str) -> dict[str, object]:
    return {
        "probability_correct": probability,
        "source_receipt_sha256": _hash(source),
    }


def _outcomes(
    *,
    bucket: str,
    verifier_ids: tuple[str, ...],
    count: int = 24,
    shared_error_ordinals: frozenset[int] = frozenset(),
) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        correct = index % 2 == 0
        signals = {}
        for verifier_id in verifier_ids:
            prediction = (
                not correct if index in shared_error_ordinals else correct
            )
            signals[verifier_id] = _signal(
                0.9 if prediction else 0.1,
                f"{bucket}:{index}:{verifier_id}",
            )
        rows.append(
            {
                "schema": CHECKED_OUTCOME_SCHEMA,
                "bucket": bucket,
                "task_sha256": _hash(f"{bucket}:task:{index}"),
                "grade_receipt_sha256": _hash(f"{bucket}:grade:{index}"),
                "checked": True,
                "outcome_correct": correct,
                "signals": signals,
            }
        )
    return rows


def _current_receipts(
    *,
    blind_score: float = 0.9,
    neural_score: float | None = 0.9,
) -> dict[str, object]:
    return {
        "blind_review": {
            "schema": "aura.rlc.blind_branch_review.v1",
            "rows": [{"branch": 0, "score": blind_score}],
            "receipt_sha256": _hash("blind"),
        },
        "decoy_verification": {"selection_admitted": True},
        "generative_verifier": {
            "schema": "aura.rlc.generative_verifier.v1",
            "causal_refutation": False,
            "receipt_sha256": _hash("generative"),
        },
        "counterfactual_verifier": {},
        "prefix_stability": {},
        "neural_uncertainty": (
            {
                "schema": "aura.rlc.neural_uncertainty_receipt.v1",
                "selected_branch": 0,
                "latest_supported_scores": {"0": neural_score},
                "receipt_sha256": _hash("uncertainty"),
            }
            if neural_score is not None
            else {}
        ),
        "mistake_locator": {},
    }


def _build(
    evidence: dict[str, object],
    *,
    blind_score: float = 0.9,
    neural_score: float | None = 0.9,
) -> dict[str, object]:
    return build_verifier_fusion_receipt(
        **_current_receipts(
            blind_score=blind_score,
            neural_score=neural_score,
        ),
        selected_branch=0,
        evidence=evidence,
    )


def test_bootstrap_evidence_is_canonical_and_cannot_fuse() -> None:
    evidence = build_verifier_fusion_evidence(
        bucket="logic|normal",
        checked_outcomes=[],
    )

    assert validate_verifier_fusion_evidence(evidence) == evidence
    assert evidence["evidence_state"] == "bootstrap_unmeasured"
    receipt = _build(evidence)
    assert receipt["probabilistic_sources_observed"] == 2
    assert receipt["probabilistic_sources_admitted"] == 0
    assert receipt["fusion_measurement_admitted"] is False
    assert receipt["fused_probability_correct"] is None
    assert receipt["selection_authority_admitted"] is False
    assert receipt["correctness_authority_admitted"] is False


def test_two_domain_calibrated_sources_fuse_with_a_hard_weight_cap() -> None:
    evidence = build_verifier_fusion_evidence(
        bucket="logic|normal",
        checked_outcomes=_outcomes(
            bucket="logic|normal",
            verifier_ids=("blind_task_verifier", "neural_uncertainty"),
        ),
    )

    receipt = _build(evidence)

    assert evidence["evidence_state"] == "domain_measured"
    assert receipt["fusion_measurement_admitted"] is True
    assert receipt["verdict"] == "historically_supported"
    assert receipt["fused_probability_correct"] == 1.0
    assert receipt["source_weights"] == {
        "blind_task_verifier": 0.5,
        "neural_uncertainty": 0.5,
    }
    assert max(receipt["source_weights"].values()) <= 0.5
    assert receipt["effective_independent_sources"] == 2.0
    assert all(
        signal["calibration_scope"] == "domain"
        for signal in receipt["signals"]
        if signal["admitted_to_fusion"]
    )


def test_one_calibrated_probabilistic_source_never_becomes_fusion_authority() -> None:
    evidence = build_verifier_fusion_evidence(
        bucket="logic|normal",
        checked_outcomes=_outcomes(
            bucket="logic|normal",
            verifier_ids=("blind_task_verifier",),
        ),
    )

    receipt = _build(evidence, neural_score=None)

    assert receipt["probabilistic_sources_admitted"] == 1
    assert receipt["source_weights"] == {}
    assert receipt["fusion_measurement_admitted"] is False
    assert receipt["verdict"] == "insufficient_independent_evidence"


def test_global_calibration_fallback_is_labeled_and_not_domain_washed() -> None:
    evidence = build_verifier_fusion_evidence(
        bucket="unseen-domain|normal",
        checked_outcomes=_outcomes(
            bucket="logic|normal",
            verifier_ids=("blind_task_verifier", "neural_uncertainty"),
        ),
    )

    receipt = _build(evidence)

    assert evidence["evidence_state"] == "global_measured"
    assert evidence["scopes"]["domain"]["checked_tasks"] == 0
    assert receipt["fusion_measurement_admitted"] is True
    assert {
        signal["calibration_scope"]
        for signal in receipt["signals"]
        if signal["admitted_to_fusion"]
    } == {"global"}


def test_shared_historical_errors_reduce_effective_support_and_force_abstention() -> None:
    evidence = build_verifier_fusion_evidence(
        bucket="logic|normal",
        checked_outcomes=_outcomes(
            bucket="logic|normal",
            verifier_ids=("blind_task_verifier", "neural_uncertainty"),
            shared_error_ordinals=frozenset({0, 1, 2, 3}),
        ),
    )

    receipt = _build(evidence)
    dependence = receipt["pairwise_dependence"][0]

    assert dependence["measured"] is True
    assert dependence["dependence"] > 0.0
    assert receipt["effective_independent_sources"] < 1.5
    assert receipt["fusion_measurement_admitted"] is False
    assert receipt["fused_probability_correct"] is None


def test_unpaired_reliability_history_is_not_mistaken_for_independence() -> None:
    blind_rows = _outcomes(
        bucket="logic|normal",
        verifier_ids=("blind_task_verifier",),
    )
    neural_rows = _outcomes(
        bucket="logic|normal",
        verifier_ids=("neural_uncertainty",),
    )
    for index, row in enumerate(neural_rows):
        row["task_sha256"] = _hash(f"neural-only:{index}")
    evidence = build_verifier_fusion_evidence(
        bucket="logic|normal",
        checked_outcomes=[*blind_rows, *neural_rows],
    )

    receipt = _build(evidence)
    dependence = receipt["pairwise_dependence"][0]

    assert receipt["probabilistic_sources_admitted"] == 2
    assert dependence == {
        "pair": "blind_task_verifier|neural_uncertainty",
        "scope": "none",
        "n": 0,
        "dependence": None,
        "conservative_dependence_upper_bound": 1.0,
        "measured": False,
    }
    assert receipt["dependence_coverage_complete"] is False
    assert receipt["fusion_measurement_admitted"] is False
    assert receipt["effective_independent_sources"] == 1.0


def test_receipt_and_evidence_tampering_fail_reconstruction() -> None:
    evidence = build_verifier_fusion_evidence(
        bucket="logic|normal",
        checked_outcomes=_outcomes(
            bucket="logic|normal",
            verifier_ids=("blind_task_verifier", "neural_uncertainty"),
        ),
    )
    receipt = _build(evidence)
    sources = _current_receipts()

    assert (
        validate_verifier_fusion_receipt(
            receipt,
            **sources,
            selected_branch=0,
            evidence=evidence,
        )
        == receipt
    )
    altered_receipt = dict(receipt)
    altered_receipt["fused_probability_correct"] = 0.99
    with pytest.raises(ValueError, match="differs from reconstruction"):
        validate_verifier_fusion_receipt(
            altered_receipt,
            **sources,
            selected_branch=0,
            evidence=evidence,
        )
    altered_evidence = dict(evidence)
    altered_evidence["checked_tasks_total"] += 1
    with pytest.raises(ValueError, match="identity is invalid"):
        validate_verifier_fusion_evidence(altered_evidence)
    unanchored = _outcomes(
        bucket="logic|normal",
        verifier_ids=("blind_task_verifier",),
        count=1,
    )[0]
    unanchored.pop("grade_receipt_sha256")
    with pytest.raises(ValueError, match="fields are invalid"):
        build_verifier_fusion_evidence(
            bucket="logic|normal",
            checked_outcomes=[unanchored],
        )


def test_worker_config_rejects_changed_evidence_snapshot() -> None:
    evidence = build_verifier_fusion_evidence(
        bucket="logic|normal",
        checked_outcomes=[],
    )
    assert config_from_job({"verifier_fusion_evidence": evidence}).verifier_fusion_evidence == evidence

    altered = dict(evidence)
    altered["evidence_state"] = "domain_measured"
    with pytest.raises(ValueError, match="verifier_fusion_evidence invalid"):
        config_from_job({"verifier_fusion_evidence": altered})


def test_ledger_persists_checked_outcomes_and_rejects_duplicate_tasks(
    tmp_path: Path,
) -> None:
    ledger = VerifierFusionLedger(tmp_path / "checked.jsonl")
    task = _hash("ledger-task")
    signals = {
        verifier_id: _signal(0.9, f"ledger:{verifier_id}")
        for verifier_id in VERIFIER_IDS[:2]
    }

    assert ledger.record_checked(
        bucket="logic|normal",
        task_sha256=task,
        grade_receipt_sha256=_hash("ledger-grade"),
        outcome_correct=True,
        signals=signals,
    )
    restored = VerifierFusionLedger(tmp_path / "checked.jsonl")
    assert restored.status()["checked_outcomes"] == 1
    assert restored.evidence(bucket="logic|normal")["checked_tasks_total"] == 1
    with pytest.raises(ValueError, match="already recorded"):
        restored.record_checked(
            bucket="logic|normal",
            task_sha256=task,
            grade_receipt_sha256=_hash("ledger-grade"),
            outcome_correct=True,
            signals=signals,
        )


def test_verified_receipt_extraction_binds_final_branch_and_excludes_prior_refutation() -> None:
    receipt = {
        **_current_receipts(),
        "selected_branch": 0,
    }

    signals = checked_signals_from_receipt(receipt)

    assert set(signals) == {"blind_task_verifier", "neural_uncertainty"}
    assert signals["blind_task_verifier"]["source_receipt_sha256"] == _hash(
        "blind"
    )
    assert "generative_refutation" not in signals
