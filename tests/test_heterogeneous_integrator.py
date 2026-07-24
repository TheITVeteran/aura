from __future__ import annotations

import hashlib

import numpy as np
import pytest

from core.brain.llm.latent_cortex.counterfactual_probe import (
    CounterfactualProbeResult,
)
from core.brain.llm.latent_cortex.heterogeneous_integrator import (
    HeterogeneousIntegrationConfig,
    IntegrationPolicyResult,
    build_heterogeneous_decode_receipt,
    run_heterogeneous_integration,
    validate_heterogeneous_decode_receipt,
    validate_heterogeneous_integration_receipt,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.verified_best import (
    VerifierObservation,
    tensor_sha256,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rehash(value):
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _observation(
    score: float,
    identity: str,
    *,
    authoritative: bool = True,
):
    if authoritative:
        return VerifierObservation(
            score=score,
            lower_bound=score,
            upper_bound=score,
            sample_count=1,
            basis="deterministic_exact",
            independent=True,
            evidence_sha256=_digest(identity),
        ).to_dict()
    return VerifierObservation.from_value(score).to_dict()


def _states():
    old = np.arange(64, dtype=np.float32).reshape(1, 4, 16) / 10.0
    new = np.array(old, copy=True)
    new[:, 2, :] += 0.125
    return old, new


def _source(
    old,
    new,
    *,
    old_score: float = 0.50,
    new_score: float = 0.80,
    kind: str = "contradiction_perturbation",
):
    old_result = {
        "observation": _observation(old_score, "source-old"),
    }
    new_result = {
        "observation": _observation(new_score, "source-new"),
    }
    if kind == "contradiction_perturbation":
        source = {
            "status": "retained",
            "state_mutation_applied": True,
            "baseline_state_sha256": tensor_sha256(old),
            "resulting_state_sha256": tensor_sha256(new),
            "arms": [
                {
                    "name": "no_op",
                    "replicates": [old_result, old_result],
                },
                {
                    "name": "matched_random",
                    "replicates": [old_result, old_result],
                },
                {
                    "name": "contradiction_guided",
                    "replicates": [new_result, new_result],
                },
            ],
        }
        return _rehash(source), _inactive("local")
    source = {
        "status": "retained",
        "state_mutation_applied": True,
        "baseline_state_sha256": tensor_sha256(old),
        "resulting_state_sha256": tensor_sha256(new),
        "selected_candidate": 1,
        "candidates": [
            {
                "label": "baseline:0",
                "replicates": [old_result, old_result],
            },
            {
                "label": "conditioned_target:1",
                "replicates": [new_result, new_result],
            },
        ],
    }
    return _inactive("perturbation"), _rehash(source)


def _inactive(identity: str):
    return _rehash(
        {
            "status": "skipped",
            "state_mutation_applied": False,
            "identity": identity,
        }
    )


def _evaluator(
    *,
    scores=None,
    authoritative: bool = True,
    nondeterministic: bool = False,
    total_compute_mismatch: bool = False,
    lane_compute_mismatch: bool = False,
    lane_evidence_mismatch: bool = False,
    js_divergence: float = 0.20,
):
    values = scores or {
        "select_old": 0.50,
        "select_new": 0.70,
        "probability_fusion": 0.90,
    }

    def evaluate(policy, old, new, gamma, replicate):
        old_apps = 100
        new_apps = 100 + (1 if lane_compute_mismatch and policy == "probability_fusion" else 0)
        total = old_apps + new_apps
        if total_compute_mismatch and policy == "select_new":
            old_apps += 1
            new_apps += 1
            total += 2
        repeat_suffix = f":{replicate}" if nondeterministic else ""
        lane_suffix = f":{policy}" if lane_evidence_mismatch else ""
        probe = CounterfactualProbeResult(
            probe_tokens_sha256=_digest(f"probe:{policy}{repeat_suffix}"),
            probe_token_count=12,
            observation=_observation(
                values[policy],
                f"observation:{policy}{repeat_suffix}",
                authoritative=authoritative,
            ),
            layer_apps=total,
        )
        return IntegrationPolicyResult(
            probe=probe,
            incumbent_state_sha256=tensor_sha256(old),
            corrected_state_sha256=tensor_sha256(new),
            fusion_weight=gamma,
            old_lane_layer_apps=old_apps,
            new_lane_layer_apps=new_apps,
            old_initial_logits_sha256=_digest(f"old-initial{lane_suffix}"),
            new_initial_logits_sha256=_digest(f"new-initial{lane_suffix}"),
            old_logits_trace_sha256=_digest(f"old-lane:{policy}"),
            new_logits_trace_sha256=_digest(f"new-lane:{policy}"),
            policy_logits_trace_sha256=_digest(f"policy:{policy}{repeat_suffix}"),
            mean_js_divergence_bits=js_divergence,
            max_js_divergence_bits=js_divergence,
            divergence_samples=12,
        )

    return evaluate


def _run(evaluate, *, source_kind="contradiction_perturbation", config=None):
    old, new = _states()
    perturbation, exploration = _source(
        old,
        new,
        kind=source_kind,
    )
    result, policy, receipt = run_heterogeneous_integration(
        incumbent_state=old,
        corrected_state=new,
        contradiction_perturbation=perturbation,
        local_exploration=exploration,
        config=config or HeterogeneousIntegrationConfig(),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=evaluate,
    )
    return old, new, result, policy, receipt, perturbation, exploration


def test_fusion_wins_only_after_beating_both_equal_compute_selections():
    old, new, result, policy, receipt, *_ = _run(_evaluator())
    assert policy == "probability_fusion"
    assert receipt["status"] == "selected"
    assert receipt["fusion_beats_selection"] is True
    assert receipt["new_beats_old"] is True
    assert receipt["fusion_weight"] == pytest.approx(0.8 / 1.3)
    assert receipt["all_policies_equal_compute"] is True
    assert receipt["all_lanes_equal_compute"] is True
    assert receipt["shared_lane_evidence"] is True
    assert tensor_sha256(result) == tensor_sha256(new)
    assert tensor_sha256(result) != tensor_sha256(old)


def test_corrected_selection_wins_when_fusion_does_not():
    old, new, result, policy, receipt, *_ = _run(
        _evaluator(
            scores={
                "select_old": 0.50,
                "select_new": 0.80,
                "probability_fusion": 0.70,
            }
        )
    )
    assert policy == "select_new"
    assert receipt["status"] == "selected"
    assert receipt["fusion_beats_selection"] is False
    assert receipt["new_beats_old"] is True
    assert tensor_sha256(result) == tensor_sha256(new)
    assert tensor_sha256(result) != tensor_sha256(old)


@pytest.mark.parametrize(
    ("evaluate", "reason"),
    [
        (
            _evaluator(
                scores={
                    "select_old": 0.50,
                    "select_new": 0.50,
                    "probability_fusion": 0.50,
                }
            ),
            "no_policy_earned_separated_bounds",
        ),
        (
            _evaluator(authoritative=False),
            "non_authoritative_verifier_observation",
        ),
        (
            _evaluator(nondeterministic=True),
            "policy_repeat_nondeterminism",
        ),
        (
            _evaluator(total_compute_mismatch=True),
            "policy_compute_mismatch",
        ),
        (
            _evaluator(lane_evidence_mismatch=True),
            "policy_lane_evidence_mismatch",
        ),
        (
            _evaluator(js_divergence=0.0),
            "candidate_distributions_not_distinct",
        ),
    ],
)
def test_unproven_policy_comparison_abstains_to_exact_incumbent(
    evaluate,
    reason,
):
    old, _new, result, policy, receipt, *_ = _run(evaluate)
    assert policy == "select_old"
    assert receipt["status"] == "abstained"
    assert receipt["reason"] == reason
    assert receipt["decode_policy_applied"] is False
    assert receipt["rollback_proven"] is True
    assert tensor_sha256(result) == tensor_sha256(old)


def test_unequal_lanes_are_rejected_into_failure_abstention():
    old, _new, result, policy, receipt, *_ = _run(_evaluator(lane_compute_mismatch=True))
    assert policy == "select_old"
    assert receipt["status"] == "abstained"
    assert receipt["reason"] == "evaluation_failed:ValueError"
    assert receipt["rollback_proven"] is True
    assert tensor_sha256(result) == tensor_sha256(old)


def test_policy_result_directly_rejects_unequal_lane_compute():
    evaluator = _evaluator(lane_compute_mismatch=True)
    old, new = _states()
    result = evaluator(
        "probability_fusion",
        old,
        new,
        0.6,
        0,
    )
    with pytest.raises(
        ValueError,
        match="heterogeneous integration policy result is invalid",
    ):
        result.normalized()


def test_local_exploration_source_is_supported_without_free_weight():
    _old, _new, _result, policy, receipt, *_ = _run(
        _evaluator(),
        source_kind="local_exploration",
    )
    assert policy == "probability_fusion"
    assert receipt["source_kind"] == "local_exploration"
    assert receipt["fusion_weight"] == pytest.approx(0.8 / 1.3)


def test_final_fusion_decode_binds_dual_lane_execution_without_text():
    *_, integration, _perturbation, _exploration = _run(_evaluator())
    initial = _digest("fused-initial")
    audit = {
        "incumbent_state_sha256": integration["incumbent_state_sha256"],
        "corrected_state_sha256": integration["corrected_state_sha256"],
        "fusion_weight": integration["fusion_weight"],
        "old_lane_layer_apps": 200,
        "new_lane_layer_apps": 200,
        "old_initial_logits_sha256": _digest("old-initial"),
        "new_initial_logits_sha256": _digest("new-initial"),
        "policy_initial_logits_sha256": initial,
        "old_logits_trace_sha256": _digest("old-trace"),
        "new_logits_trace_sha256": _digest("new-trace"),
        "policy_logits_trace_sha256": _digest("fused-trace"),
        "mean_js_divergence_bits": 0.2,
        "max_js_divergence_bits": 0.3,
        "divergence_samples": 12,
    }
    receipt = build_heterogeneous_decode_receipt(
        integration=integration,
        output_tokens=[1, 2, 3],
        termination="token_limit",
        first_logits_sha256=initial,
        fusion_audit=audit,
    )
    assert receipt["execution_kind"] == ("dual_lane_probability_fusion")
    assert receipt["answer_text_stored"] is False
    assert (
        validate_heterogeneous_decode_receipt(
            receipt,
            integration=integration,
            expected_output_tokens=[1, 2, 3],
        )
        == receipt
    )
    with pytest.raises(ValueError, match="output tokens differ"):
        validate_heterogeneous_decode_receipt(
            receipt,
            integration=integration,
            expected_output_tokens=[1, 2, 4],
        )


def test_final_decode_rejects_rehashed_policy_or_lane_lie():
    *_, integration, _perturbation, _exploration = _run(_evaluator())
    initial = _digest("fused-initial")
    audit = {
        "incumbent_state_sha256": integration["incumbent_state_sha256"],
        "corrected_state_sha256": integration["corrected_state_sha256"],
        "fusion_weight": integration["fusion_weight"],
        "old_lane_layer_apps": 200,
        "new_lane_layer_apps": 200,
        "old_initial_logits_sha256": _digest("old-initial"),
        "new_initial_logits_sha256": _digest("new-initial"),
        "policy_initial_logits_sha256": initial,
        "old_logits_trace_sha256": _digest("old-trace"),
        "new_logits_trace_sha256": _digest("new-trace"),
        "policy_logits_trace_sha256": _digest("fused-trace"),
        "mean_js_divergence_bits": 0.2,
        "max_js_divergence_bits": 0.3,
        "divergence_samples": 12,
    }
    receipt = build_heterogeneous_decode_receipt(
        integration=integration,
        output_tokens=[1, 2, 3],
        termination="token_limit",
        first_logits_sha256=initial,
        fusion_audit=audit,
    )
    forged = dict(receipt)
    forged["selected_policy"] = "select_new"
    forged = _rehash(forged)
    with pytest.raises(ValueError):
        validate_heterogeneous_decode_receipt(
            forged,
            integration=integration,
        )
    bad_audit = dict(audit)
    bad_audit["new_lane_layer_apps"] = 199
    with pytest.raises(ValueError, match="fusion execution audit"):
        build_heterogeneous_decode_receipt(
            integration=integration,
            output_tokens=[1],
            termination="token_limit",
            first_logits_sha256=initial,
            fusion_audit=bad_audit,
        )
    for field, value in (
        ("incumbent_state_sha256", _digest("substituted-incumbent")),
        ("corrected_state_sha256", _digest("substituted-correction")),
        ("fusion_weight", 0.123),
    ):
        forged = {
            **receipt,
            "fusion_audit": {
                **receipt["fusion_audit"],
                field: value,
            },
        }
        forged = _rehash(forged)
        with pytest.raises(ValueError, match="fusion execution audit"):
            validate_heterogeneous_decode_receipt(
                forged,
                integration=integration,
            )


def test_no_retained_source_is_compute_inert():
    old, new = _states()
    calls = []

    def evaluate(*args):
        calls.append(args)
        raise AssertionError("inactive integration reached evaluator")

    result, policy, receipt = run_heterogeneous_integration(
        incumbent_state=old,
        corrected_state=new,
        contradiction_perturbation=_inactive("perturbation"),
        local_exploration=_inactive("exploration"),
        config=HeterogeneousIntegrationConfig(),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=evaluate,
    )
    assert calls == []
    assert policy == "select_old"
    assert receipt["status"] == "skipped"
    assert receipt["reason"] == ("requires_exactly_one_retained_source")
    assert tensor_sha256(result) == tensor_sha256(old)


def test_two_retained_sources_are_rejected_without_policy_compute():
    old, new = _states()
    perturbation, _ = _source(old, new)
    _, exploration = _source(
        old,
        new,
        kind="local_exploration",
    )
    calls = []

    def evaluate(*args):
        calls.append(args)
        raise AssertionError("ambiguous source reached evaluator")

    result, policy, receipt = run_heterogeneous_integration(
        incumbent_state=old,
        corrected_state=new,
        contradiction_perturbation=perturbation,
        local_exploration=exploration,
        config=HeterogeneousIntegrationConfig(),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=evaluate,
    )

    assert calls == []
    assert policy == "select_old"
    assert receipt["status"] == "skipped"
    assert receipt["reason"] == "requires_exactly_one_retained_source"
    assert tensor_sha256(result) == tensor_sha256(old)


def test_evaluator_failure_abstains_without_partial_policy_authority():
    def fail(*_args):
        raise RuntimeError("probe failed")

    old, _new, result, policy, receipt, *_ = _run(fail)
    assert policy == "select_old"
    assert receipt["status"] == "abstained"
    assert receipt["reason"] == "evaluation_failed:RuntimeError"
    assert receipt["policies"] == []
    assert receipt["rollback_proven"] is True
    assert tensor_sha256(result) == tensor_sha256(old)


def test_validator_rejects_rehashed_failure_evidence_lies():
    def fail(*_args):
        raise RuntimeError("probe failed")

    *_, receipt, perturbation, exploration = _run(fail)
    for field, value in (
        ("all_policies_equal_compute", True),
        ("all_lanes_equal_compute", True),
        ("mean_js_divergence_bits", 0.2),
        ("new_beats_old", True),
        ("decode_policy_applied", True),
    ):
        forged = dict(receipt)
        forged[field] = value
        forged = _rehash(forged)
        with pytest.raises(
            ValueError,
            match="did not abstain",
        ):
            validate_heterogeneous_integration_receipt(
                forged,
                expected_config=HeterogeneousIntegrationConfig(),
                contradiction_perturbation=perturbation,
                local_exploration=exploration,
                verifier_policy_sha256=_digest("verifier"),
                decoy_review_sha256=_digest("decoy"),
            )


def test_validator_rejects_rehashed_inactive_reason_lie():
    old, new = _states()
    perturbation = _inactive("perturbation")
    exploration = _inactive("exploration")
    _result, _policy, receipt = run_heterogeneous_integration(
        incumbent_state=old,
        corrected_state=new,
        contradiction_perturbation=perturbation,
        local_exploration=exploration,
        config=HeterogeneousIntegrationConfig(),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=None,
    )
    forged = dict(receipt)
    forged["reason"] = "trust_me"
    forged = _rehash(forged)
    with pytest.raises(ValueError, match="skip reason is invalid"):
        validate_heterogeneous_integration_receipt(
            forged,
            expected_config=HeterogeneousIntegrationConfig(),
            contradiction_perturbation=perturbation,
            local_exploration=exploration,
            verifier_policy_sha256=_digest("verifier"),
            decoy_review_sha256=_digest("decoy"),
        )


def test_malformed_extra_source_replicate_cannot_be_ignored():
    old, new = _states()
    perturbation, exploration = _source(old, new)
    perturbation["arms"][0]["replicates"].append(None)
    perturbation = _rehash(perturbation)
    calls = []

    def evaluate(*args):
        calls.append(args)
        raise AssertionError("malformed source reached evaluator")

    result, policy, receipt = run_heterogeneous_integration(
        incumbent_state=old,
        corrected_state=new,
        contradiction_perturbation=perturbation,
        local_exploration=exploration,
        config=HeterogeneousIntegrationConfig(),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=evaluate,
    )
    assert calls == []
    assert policy == "select_old"
    assert receipt["status"] == "skipped"
    assert receipt["reason"] == "source_observations_are_invalid"
    assert tensor_sha256(result) == tensor_sha256(old)


@pytest.mark.parametrize(
    "source_kind",
    ["contradiction_perturbation", "local_exploration"],
)
def test_duplicate_source_labels_cannot_hide_ambiguous_evidence(source_kind):
    old, new = _states()
    perturbation, exploration = _source(
        old,
        new,
        kind=source_kind,
    )
    source = perturbation if source_kind == "contradiction_perturbation" else exploration
    rows_key = "arms" if source_kind == "contradiction_perturbation" else "candidates"
    source[rows_key].append(dict(source[rows_key][0]))
    source = _rehash(source)
    if source_kind == "contradiction_perturbation":
        perturbation = source
    else:
        exploration = source

    result, policy, receipt = run_heterogeneous_integration(
        incumbent_state=old,
        corrected_state=new,
        contradiction_perturbation=perturbation,
        local_exploration=exploration,
        config=HeterogeneousIntegrationConfig(),
        verifier_policy_sha256=_digest("verifier"),
        decoy_review_sha256=_digest("decoy"),
        evaluate=_evaluator(),
    )

    assert policy == "select_old"
    assert receipt["status"] == "skipped"
    assert receipt["reason"] in {
        "perturbation_arms_are_invalid",
        "exploration_candidates_are_invalid",
    }
    assert tensor_sha256(result) == tensor_sha256(old)


def test_validator_rejects_rehashed_policy_authority_lies():
    *_, receipt, perturbation, exploration = _run(_evaluator())
    for field, value in (
        ("selected_policy", "select_old"),
        ("fusion_beats_selection", False),
        ("decode_policy_applied", False),
        ("authority_scope", "none"),
    ):
        forged = dict(receipt)
        forged[field] = value
        forged = _rehash(forged)
        with pytest.raises(
            ValueError,
            match="decision reconstruction failed",
        ):
            validate_heterogeneous_integration_receipt(
                forged,
                expected_config=HeterogeneousIntegrationConfig(),
                contradiction_perturbation=perturbation,
                local_exploration=exploration,
                verifier_policy_sha256=_digest("verifier"),
                decoy_review_sha256=_digest("decoy"),
            )


def test_validator_rejects_rehashed_lane_and_observation_lies():
    *_, receipt, perturbation, exploration = _run(_evaluator())
    for field in (
        "all_policies_equal_compute",
        "all_lanes_equal_compute",
        "all_observations_authoritative",
        "repeat_deterministic",
        "shared_lane_evidence",
    ):
        forged = dict(receipt)
        forged[field] = not forged[field]
        forged = _rehash(forged)
        with pytest.raises(ValueError):
            validate_heterogeneous_integration_receipt(
                forged,
                expected_config=HeterogeneousIntegrationConfig(),
                contradiction_perturbation=perturbation,
                local_exploration=exploration,
                verifier_policy_sha256=_digest("verifier"),
                decoy_review_sha256=_digest("decoy"),
            )


def test_validator_rejects_rehashed_policy_state_lineage_lies():
    *_, receipt, perturbation, exploration = _run(_evaluator())
    for field, value in (
        ("incumbent_state_sha256", _digest("substituted-incumbent")),
        ("corrected_state_sha256", _digest("substituted-correction")),
        ("fusion_weight", 0.123),
    ):
        forged = {
            **receipt,
            "policies": [
                {
                    **receipt["policies"][0],
                    "replicates": [
                        {
                            **receipt["policies"][0]["replicates"][0],
                            field: value,
                        },
                        *receipt["policies"][0]["replicates"][1:],
                    ],
                },
                *receipt["policies"][1:],
            ],
        }
        forged = _rehash(forged)
        with pytest.raises(ValueError, match="policy state lineage differs"):
            validate_heterogeneous_integration_receipt(
                forged,
                expected_config=HeterogeneousIntegrationConfig(),
                contradiction_perturbation=perturbation,
                local_exploration=exploration,
                verifier_policy_sha256=_digest("verifier"),
                decoy_review_sha256=_digest("decoy"),
            )


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "invented"},
        {"replicates": 1},
        {"replicates": 4},
        {"min_verifier_margin": -0.1},
        {"min_js_divergence_bits": 1.1},
        {"extra": True},
    ],
)
def test_config_rejects_unbounded_or_unknown_values(config):
    with pytest.raises(ValueError):
        HeterogeneousIntegrationConfig.from_value(config)
