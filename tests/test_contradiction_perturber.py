"""Causal, bounded, and reversible SPARK-032 perturbation contracts."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from core.brain.llm.latent_cortex.contradiction_perturber import (
    ContradictionPerturberConfig,
    PerturbationArmResult,
    run_contradiction_perturbation,
    validate_contradiction_perturbation_receipt,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.verified_best import (
    VERIFIER_OBSERVATION_SCHEMA,
    tensor_sha256,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _rehash(receipt):
    forged = copy.deepcopy(receipt)
    payload = dict(forged)
    payload.pop("receipt_sha256")
    forged["receipt_sha256"] = canonical_sha256(payload)
    return forged


def _tensor(*, learned: bool = True, position: int = 2):
    payload = {
        "mode": "learned" if learned else "unavailable",
        "selected_branch": 0,
        "selected_branch_candidate": (
            {"transition_index": 4, "position_index": position}
            if learned
            else None
        ),
        "branches": [
            {
                "candidate_probability": 0.91 if learned else None,
            }
        ],
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _observation(score: float, name: str, *, authoritative: bool = True):
    if not authoritative:
        return {
            "schema": VERIFIER_OBSERVATION_SCHEMA,
            "score": score,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "sample_count": 0,
            "basis": "uncalibrated_scalar",
            "independent": False,
            "evidence_sha256": "",
        }
    return {
        "schema": VERIFIER_OBSERVATION_SCHEMA,
        "score": score,
        "lower_bound": score,
        "upper_bound": score,
        "sample_count": 1,
        "basis": "deterministic_exact",
        "independent": True,
        "evidence_sha256": _digest(name),
    }


def _states():
    baseline = np.ones((1, 4, 16), dtype=np.float32)
    anchor = np.array(baseline, copy=True)
    anchor[:, 2, :] = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    return baseline, anchor


def _evaluator(
    scores: dict[str, float],
    *,
    authoritative: bool = True,
    unstable: bool = False,
    unequal_compute: bool = False,
):
    def evaluate(name: str, state, replicate: int):
        score = scores[name] + (0.02 * replicate if unstable else 0.0)
        return PerturbationArmResult(
            probe_tokens_sha256=_digest(
                f"{name}:{replicate}:{tensor_sha256(state)}"
            ),
            probe_token_count=12,
            observation=_observation(
                score,
                f"{name}:{replicate if unstable else 'stable'}",
                authoritative=authoritative,
            ),
            layer_apps=101 + (1 if unequal_compute and name == "matched_random" else 0),
        )

    return evaluate


def _run(evaluate, *, protected=()):
    baseline, anchor = _states()
    tensor = _tensor()
    config = ContradictionPerturberConfig()
    result, receipt = run_contradiction_perturbation(
        baseline=baseline,
        anchor=anchor,
        protected_positions=protected,
        contradiction_tensor=tensor,
        selected_branch=0,
        config=config,
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=evaluate,
    )
    return baseline, result, receipt, tensor, config


def test_guided_state_is_retained_only_after_repeated_equal_compute_win():
    baseline, result, receipt, tensor, config = _run(
        _evaluator(
            {
                "no_op": 0.50,
                "matched_random": 0.55,
                "contradiction_guided": 0.80,
            }
        )
    )
    assert receipt["status"] == "retained"
    assert receipt["state_mutation_applied"] is True
    assert receipt["guided_beats_controls"] is True
    assert receipt["repeat_stable"] is True
    assert receipt["all_arms_equal_compute"] is True
    assert receipt["answer_text_stored"] is False
    assert tensor_sha256(result) != tensor_sha256(baseline)
    assert np.array_equal(result[:, :2, :], baseline[:, :2, :])
    assert np.array_equal(result[:, 3:, :], baseline[:, 3:, :])
    assert max(
        arm["relative_target_delta_rms"] for arm in receipt["arms"]
    ) <= config.max_relative_delta_rms + 1e-6
    validate_contradiction_perturbation_receipt(
        receipt,
        expected_config=config,
        contradiction_tensor=tensor,
        expected_selected_branch=0,
        expected_protected_positions=(),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
    )


@pytest.mark.parametrize(
    ("evaluate", "reason"),
    [
        (
            _evaluator(
                {
                    "no_op": 0.70,
                    "matched_random": 0.60,
                    "contradiction_guided": 0.65,
                }
            ),
            "guided_candidate_did_not_beat_controls",
        ),
        (
            _evaluator(
                {
                    "no_op": 0.50,
                    "matched_random": 0.40,
                    "contradiction_guided": 0.90,
                },
                authoritative=False,
            ),
            "non_authoritative_verifier_observation",
        ),
        (
            _evaluator(
                {
                    "no_op": 0.50,
                    "matched_random": 0.40,
                    "contradiction_guided": 0.90,
                },
                unstable=True,
            ),
            "verifier_repeat_instability",
        ),
        (
            _evaluator(
                {
                    "no_op": 0.50,
                    "matched_random": 0.40,
                    "contradiction_guided": 0.90,
                },
                unequal_compute=True,
            ),
            "control_compute_mismatch",
        ),
    ],
)
def test_failed_admission_restores_exact_baseline(evaluate, reason):
    baseline, result, receipt, _tensor_receipt, _config = _run(evaluate)
    assert receipt["status"] == "restored"
    assert receipt["reason"] == reason
    assert receipt["state_mutation_applied"] is False
    assert receipt["rollback_proven"] is True
    assert tensor_sha256(result) == tensor_sha256(baseline)


def test_protected_evidence_coordinate_is_never_evaluated():
    calls = []

    def evaluate(*args):
        calls.append(args)
        raise AssertionError("protected coordinate reached evaluator")

    baseline, result, receipt, _tensor_receipt, _config = _run(
        evaluate,
        protected=(2,),
    )
    assert calls == []
    assert receipt["status"] == "skipped"
    assert receipt["reason"] == "candidate_targets_immutable_evidence"
    assert tensor_sha256(result) == tensor_sha256(baseline)


def test_unavailable_tensor_and_verifier_remain_inert():
    baseline, anchor = _states()
    tensor = _tensor(learned=False)
    result, receipt = run_contradiction_perturbation(
        baseline=baseline,
        anchor=anchor,
        protected_positions=(),
        contradiction_tensor=tensor,
        selected_branch=0,
        config=ContradictionPerturberConfig(),
        verifier_policy_sha256="",
        decoy_review_sha256="",
        evaluate=None,
    )
    assert receipt["status"] == "skipped"
    assert receipt["reason"] == "learned_localized_candidate_unavailable"
    assert tensor_sha256(result) == tensor_sha256(baseline)


def test_evaluator_failure_rolls_back_without_partial_arm_authority():
    def fail(_name, _state, _replicate):
        raise RuntimeError("probe failed")

    baseline, result, receipt, _tensor_receipt, _config = _run(fail)
    assert receipt["status"] == "restored"
    assert receipt["reason"] == "evaluation_failed:RuntimeError"
    assert receipt["arms"] == []
    assert receipt["rollback_proven"] is True
    assert tensor_sha256(result) == tensor_sha256(baseline)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("guided_beats_controls", True),
        ("all_observations_authoritative", True),
        ("authority_scope", "selected_branch_target_position_only"),
        ("baseline_state_sha256", "0" * 64),
    ],
)
def test_rehashed_evaluator_failure_cannot_claim_partial_authority(field, value):
    def fail(_name, _state, _replicate):
        raise RuntimeError("probe failed")

    _baseline, _result, receipt, tensor, config = _run(fail)
    receipt[field] = value
    forged = _rehash(receipt)
    with pytest.raises(ValueError, match="did not roll back"):
        validate_contradiction_perturbation_receipt(
            forged,
            expected_config=config,
            contradiction_tensor=tensor,
            expected_selected_branch=0,
            expected_protected_positions=(),
            verifier_policy_sha256=_digest("verifier"),
            decoy_review_sha256=_digest("decoy"),
        )


def test_rehashed_inactive_receipt_cannot_claim_evidence_or_rollback():
    baseline, anchor = _states()
    tensor = _tensor(learned=False)
    _result, receipt = run_contradiction_perturbation(
        baseline=baseline,
        anchor=anchor,
        protected_positions=(),
        contradiction_tensor=tensor,
        selected_branch=0,
        config=ContradictionPerturberConfig(),
        verifier_policy_sha256="",
        decoy_review_sha256="",
        evaluate=None,
    )
    receipt["repeat_stable"] = True
    forged = _rehash(receipt)
    with pytest.raises(ValueError, match="inactive.*claims authority"):
        validate_contradiction_perturbation_receipt(
            forged,
            expected_config=ContradictionPerturberConfig(),
            contradiction_tensor=tensor,
            expected_selected_branch=0,
            expected_protected_positions=(),
            verifier_policy_sha256="",
            decoy_review_sha256="",
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("arms", 0, "replicates", 0, "layer_apps"), 0, "replicate is invalid"),
        (("arms", 0, "replicates", 0, "probe_token_count"), 0, "replicate is invalid"),
        (("arms", 0, "target_delta_rms"), -0.1, "arm geometry is invalid"),
        (("reason",), "unsupported_reason", "decision reconstruction"),
    ],
)
def test_rehashed_evaluated_receipt_must_reconstruct_all_bounds(
    path,
    value,
    message,
):
    _baseline, _result, receipt, tensor, config = _run(
        _evaluator(
            {
                "no_op": 0.50,
                "matched_random": 0.55,
                "contradiction_guided": 0.80,
            }
        )
    )
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    forged = _rehash(receipt)
    with pytest.raises(ValueError, match=message):
        validate_contradiction_perturbation_receipt(
            forged,
            expected_config=config,
            contradiction_tensor=tensor,
            expected_selected_branch=0,
            expected_protected_positions=(),
            verifier_policy_sha256=_digest("verifier"),
            decoy_review_sha256=_digest("decoy"),
        )


def test_rehashed_authority_lie_is_rejected():
    _baseline, _result, receipt, tensor, config = _run(
        _evaluator(
            {
                "no_op": 0.50,
                "matched_random": 0.55,
                "contradiction_guided": 0.80,
            }
        )
    )
    forged = copy.deepcopy(receipt)
    forged["guided_beats_controls"] = False
    forged = _rehash(forged)
    with pytest.raises(ValueError, match="decision reconstruction"):
        validate_contradiction_perturbation_receipt(
            forged,
            expected_config=config,
            contradiction_tensor=tensor,
            expected_selected_branch=0,
            expected_protected_positions=(),
            verifier_policy_sha256=_digest("verifier"),
            decoy_review_sha256=_digest("decoy"),
        )


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "unsafe"},
        {"max_relative_delta_rms": 0.5},
        {"min_verifier_margin": -0.1},
        {"replicates": 1},
        {"seed": True},
        {"unknown": 1},
    ],
)
def test_config_rejects_unbounded_or_unknown_values(value):
    with pytest.raises(ValueError):
        ContradictionPerturberConfig.from_value(value)
