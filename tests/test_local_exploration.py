"""SPARK-033 local stochastic exploration and control contracts."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from core.brain.llm.latent_cortex.counterfactual_probe import (
    CounterfactualProbeResult,
)
from core.brain.llm.latent_cortex.local_exploration import (
    LocalExplorationConfig,
    run_local_exploration,
    validate_local_exploration_receipt,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.verified_best import (
    VERIFIER_OBSERVATION_SCHEMA,
    tensor_sha256,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _rehash(value):
    forged = copy.deepcopy(value)
    payload = dict(forged)
    payload.pop("receipt_sha256")
    forged["receipt_sha256"] = canonical_sha256(payload)
    return forged


def _tensor(
    *,
    learned: bool = True,
    stable_score: float = 0.05,
    target: int = 2,
):
    cells = []
    for _transition in range(2):
        cells.append(
            [
                {
                    "position_index": position,
                    "contradiction_probability": (
                        0.91 if position == target else stable_score if position == 1 else 0.45
                    ),
                }
                for position in range(4)
            ]
        )
    candidate = {"transition_index": 1, "position_index": target} if learned else None
    payload = {
        "mode": "learned" if learned else "unavailable",
        "selected_branch": 0,
        "selected_branch_candidate": candidate,
        "branches": [
            {
                "candidate": candidate,
                "candidate_probability": 0.91 if learned else None,
                "tensor": cells if learned else [],
            }
        ],
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _uncertainty(*, learned: bool = True, entropy: float = 0.8):
    payload = {
        "mode": "learned" if learned else "unavailable",
        "selected_branch": 0,
        "branches": [
            {
                "observations": (
                    [
                        {
                            "estimate": {
                                "supported": True,
                                "predictive_entropy": entropy,
                            }
                        }
                    ]
                    if learned
                    else []
                )
            }
        ],
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _perturbation():
    payload = {
        "status": "restored",
        "state_mutation_applied": False,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _observation(
    score: float,
    label: str,
    *,
    authoritative: bool = True,
):
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
        "evidence_sha256": _digest(label),
    }


def _evaluator(
    *,
    conditioned=(0.65, 0.85, 0.70),
    baseline: float = 0.50,
    sham: float = 0.55,
    authoritative: bool = True,
    nondeterministic: bool = False,
    unequal_compute: bool = False,
    collapse_outputs: bool = False,
    baseline_order_leak: bool = False,
):
    def evaluate(label: str, state, replicate: int):
        family, ordinal_text = label.split(":")
        ordinal = int(ordinal_text)
        score = (
            baseline
            if family == "baseline"
            else sham
            if family == "stable_sham"
            else conditioned[ordinal]
        )
        probe_identity = (
            "collapsed"
            if collapse_outputs and family == "conditioned_target"
            else label
            if baseline_order_leak and family == "baseline"
            else tensor_sha256(state)
        )
        if nondeterministic and replicate:
            probe_identity += f":replicate:{replicate}"
        observation_identity = (
            f"{label}:replicate:{replicate}"
            if nondeterministic
            else label
            if baseline_order_leak and family == "baseline"
            else tensor_sha256(state)
        )
        return CounterfactualProbeResult(
            probe_tokens_sha256=_digest(probe_identity),
            probe_token_count=12,
            observation=_observation(
                score + (0.01 * replicate if nondeterministic else 0.0),
                observation_identity,
                authoritative=authoritative,
            ),
            layer_apps=101 + (1 if unequal_compute and label == "stable_sham:0" else 0),
        )

    return evaluate


def _states():
    baseline = np.ones((1, 4, 16), dtype=np.float32)
    baseline[:, 1, :] *= 0.75
    baseline[:, 2, :] *= 1.25
    return baseline


def _run(evaluate, *, protected=(0,), tensor=None, uncertainty=None):
    baseline = _states()
    tensor = tensor or _tensor()
    uncertainty = uncertainty or _uncertainty()
    perturbation = _perturbation()
    config = LocalExplorationConfig()
    result, receipt = run_local_exploration(
        baseline=baseline,
        protected_positions=protected,
        contradiction_tensor=tensor,
        contradiction_perturbation=perturbation,
        neural_uncertainty=uncertainty,
        selected_branch=0,
        config=config,
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=evaluate,
    )
    return (
        baseline,
        result,
        receipt,
        tensor,
        perturbation,
        uncertainty,
        config,
    )


def _validate(receipt, tensor, perturbation, uncertainty, config):
    return validate_local_exploration_receipt(
        receipt,
        expected_config=config,
        contradiction_tensor=tensor,
        contradiction_perturbation=perturbation,
        neural_uncertainty=uncertainty,
        expected_selected_branch=0,
        expected_protected_positions=(0,),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
    )


def test_conditioned_candidate_retained_only_after_controlled_win():
    (
        baseline,
        result,
        receipt,
        tensor,
        perturbation,
        uncertainty,
        config,
    ) = _run(_evaluator())
    assert receipt["status"] == "retained"
    assert receipt["selected_candidate"] == 1
    assert receipt["conditioned_beats_controls"] is True
    assert receipt["state_mutation_applied"] is True
    assert receipt["all_candidates_equal_compute"] is True
    assert receipt["repeat_deterministic"] is True
    assert receipt["conditioned_unique_probe_count"] == 3
    assert receipt["conditioned_probe_entropy_bits"] > 1.0
    assert receipt["baseline_probe_entropy_bits"] == 0.0
    assert receipt["target_position"] == 2
    assert receipt["sham_position"] == 1
    assert tensor_sha256(result) != tensor_sha256(baseline)
    assert np.array_equal(result[:, :2, :], baseline[:, :2, :])
    assert np.array_equal(result[:, 3:, :], baseline[:, 3:, :])
    _validate(receipt, tensor, perturbation, uncertainty, config)


@pytest.mark.parametrize(
    ("evaluate", "reason"),
    [
        (
            _evaluator(conditioned=(0.40, 0.45, 0.50)),
            "conditioned_candidate_did_not_beat_controls",
        ),
        (
            _evaluator(authoritative=False),
            "non_authoritative_verifier_observation",
        ),
        (
            _evaluator(nondeterministic=True),
            "counterfactual_repeat_nondeterminism",
        ),
        (
            _evaluator(unequal_compute=True),
            "control_compute_mismatch",
        ),
        (
            _evaluator(baseline_order_leak=True),
            "baseline_control_order_leakage",
        ),
        (
            _evaluator(collapse_outputs=True),
            "conditioned_exploration_did_not_increase_output_diversity",
        ),
    ],
)
def test_nonwinning_or_unproven_search_restores_exact_baseline(
    evaluate,
    reason,
):
    baseline, result, receipt, *_rest = _run(evaluate)
    assert receipt["status"] == "restored"
    assert receipt["reason"] == reason
    assert receipt["state_mutation_applied"] is False
    assert receipt["rollback_proven"] is True
    assert tensor_sha256(result) == tensor_sha256(baseline)


def test_stable_sham_is_evaluated_but_never_persisted():
    baseline, _result, receipt, *_rest = _run(_evaluator())
    sham_rows = [row for row in receipt["candidates"] if row["family"] == "stable_sham"]
    assert sham_rows
    assert all(row["changed_positions"] == [1] for row in sham_rows)
    assert all(row["state_sha256"] != tensor_sha256(baseline) for row in sham_rows)
    assert receipt["authority_scope"] == ("selected_branch_target_position_only")


@pytest.mark.parametrize(
    ("tensor", "uncertainty", "protected", "reason"),
    [
        (
            _tensor(learned=False),
            _uncertainty(),
            (0,),
            "learned_sources_are_unavailable",
        ),
        (
            _tensor(),
            _uncertainty(entropy=0.1),
            (0,),
            "entropy_is_below_admission_floor",
        ),
        (
            _tensor(stable_score=0.8),
            _uncertainty(),
            (0, 3),
            "stable_sham_position_is_unavailable",
        ),
        (
            _tensor(target=2),
            _uncertainty(),
            (0, 2),
            "target_is_not_writable",
        ),
        (
            _tensor(),
            {
                **_uncertainty(),
                "branches": [None],
            },
            (0,),
            "uncertainty_branch_is_malformed",
        ),
    ],
)
def test_unadmitted_sources_remain_compute_inert(
    tensor,
    uncertainty,
    protected,
    reason,
):
    calls = []

    def evaluate(*args):
        calls.append(args)
        raise AssertionError("unadmitted source reached evaluator")

    baseline = _states()
    result, receipt = run_local_exploration(
        baseline=baseline,
        protected_positions=protected,
        contradiction_tensor=tensor,
        contradiction_perturbation=_perturbation(),
        neural_uncertainty=uncertainty,
        selected_branch=0,
        config=LocalExplorationConfig(),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=evaluate,
    )
    assert calls == []
    assert receipt["status"] == "skipped"
    assert receipt["reason"] == reason
    assert tensor_sha256(result) == tensor_sha256(baseline)


def test_radius_below_state_dtype_resolution_skips_without_evaluation():
    calls = []

    def evaluate(*args):
        calls.append(args)
        raise AssertionError("collapsed candidates reached evaluator")

    baseline = np.full((1, 4, 16), 1_000.0, dtype=np.float16)
    result, receipt = run_local_exploration(
        baseline=baseline,
        protected_positions=(0,),
        contradiction_tensor=_tensor(),
        contradiction_perturbation=_perturbation(),
        neural_uncertainty=_uncertainty(),
        selected_branch=0,
        config=LocalExplorationConfig(
            max_relative_delta_rms=1e-8,
        ),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=evaluate,
    )
    assert calls == []
    assert receipt["status"] == "skipped"
    assert receipt["reason"] == ("candidate_diversity_below_dtype_resolution")
    assert tensor_sha256(result) == tensor_sha256(baseline)


def test_evaluator_failure_restores_without_partial_candidate_authority():
    def fail(_label, _state, _replicate):
        raise RuntimeError("probe failed")

    baseline, result, receipt, *_rest = _run(fail)
    assert receipt["status"] == "restored"
    assert receipt["reason"] == "evaluation_failed:RuntimeError"
    assert receipt["candidates"] == []
    assert receipt["rollback_proven"] is True
    assert tensor_sha256(result) == tensor_sha256(baseline)


def test_retained_prior_perturbation_prevents_stale_uncertainty_stacking():
    calls = []

    def evaluate(*args):
        calls.append(args)
        raise AssertionError("stale uncertainty reached evaluator")

    baseline = _states()
    perturbation = _perturbation()
    perturbation["state_mutation_applied"] = True
    perturbation = _rehash(perturbation)
    result, receipt = run_local_exploration(
        baseline=baseline,
        protected_positions=(0,),
        contradiction_tensor=_tensor(),
        contradiction_perturbation=perturbation,
        neural_uncertainty=_uncertainty(),
        selected_branch=0,
        config=LocalExplorationConfig(),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=evaluate,
    )
    assert calls == []
    assert receipt["status"] == "skipped"
    assert receipt["reason"] == ("uncertainty_source_precedes_retained_perturbation")
    assert tensor_sha256(result) == tensor_sha256(baseline)


def test_generator_is_source_bound_and_exactly_replayable():
    first = _run(_evaluator())[2]
    second = _run(_evaluator())[2]
    assert first["seed_sha256"] == second["seed_sha256"]
    assert first["generator"] == second["generator"]
    assert first["evaluation_order"] == second["evaluation_order"]
    assert [(row["label"], row["state_sha256"]) for row in first["candidates"]] == [
        (row["label"], row["state_sha256"]) for row in second["candidates"]
    ]
    assert first["generator_replay_proven"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("conditioned_beats_controls", False, "decision reconstruction"),
        ("selected_candidate", 0, "decision reconstruction"),
        ("authority_scope", "none", "retained wrong state"),
        ("resulting_state_sha256", "0" * 64, "retained wrong state"),
    ],
)
def test_rehashed_authority_or_selection_lie_is_rejected(
    field,
    value,
    message,
):
    (
        _baseline,
        _result,
        receipt,
        tensor,
        perturbation,
        uncertainty,
        config,
    ) = _run(_evaluator())
    receipt[field] = value
    with pytest.raises(ValueError, match=message):
        _validate(
            _rehash(receipt),
            tensor,
            perturbation,
            uncertainty,
            config,
        )


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "unsafe"},
        {"candidates": 1},
        {"replicates": 1},
        {"max_relative_delta_rms": 0.5},
        {"min_predictive_entropy": -0.1},
        {"max_stable_contradiction_probability": 1.1},
        {"min_verifier_margin": -0.1},
        {"min_unique_conditioned_probes": 4},
        {"seed": True},
        {"unknown": 1},
    ],
)
def test_config_rejects_unbounded_or_unknown_values(value):
    with pytest.raises(ValueError):
        LocalExplorationConfig.from_value(value)
