"""Contracts for the per-recurrence value-of-computation policy."""

import hashlib
from copy import deepcopy

import pytest

from core.brain.llm.latent_cortex.action_calibration import (
    ACTION_RESOURCE_DIMENSIONS,
    GLOBAL_BOUND_FAMILY_COUNT,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.value_of_computation import (
    ACTION_TRANSITION_SCHEMA,
    ACTION_VOCABULARY,
    MIN_ACTION_TRIALS,
    ActionEvidence,
    CertifiedActionEvidence,
    CognitiveStateSignal,
    ValueOfComputationPolicy,
    build_evidence_snapshot,
    feasible_actions,
    transition_reward,
    validate_action_decision,
    validate_action_trace_row,
    validate_action_transition,
    validate_evidence_snapshot,
)
from core.brain.llm.latent_cortex.verified_best import VerifierObservation
from tests.fixtures.action_calibration import certified_action_snapshot


def _state(**overrides):
    values = {
        "step_index": 1,
        "max_steps": 8,
        "neural_steps": 1,
        "min_neural_steps": 1,
        "active_branches": 2,
        "total_branches": 2,
        "residual": 0.4,
        "residual_delta": 0.1,
        "verifier_score": 0.5,
        "verifier_delta": 0.05,
        "disagreement": 0.4,
        "uncertainty": 0.6,
        "budget_remaining_fraction": 0.8,
        "has_memory": True,
        "has_evidence": True,
        "has_verifier": True,
        "has_savepoint": True,
        "can_execute": False,
        "answer_verified": False,
        "irreducible_uncertainty": False,
        "previously_selected": (),
    }
    values.update(overrides)
    if "previously_selected" not in overrides:
        values["previously_selected"] = tuple(
            OperationKind.DECOMPOSE for _ in range(values["step_index"])
        )
    return CognitiveStateSignal(**values)


def _policy(cells=None):
    return ValueOfComputationPolicy(
        build_evidence_snapshot(bucket="general|none|short|s:mid|u:mid", cells=cells or {})
    )


def test_vocabulary_is_exactly_the_sixteen_typed_epistemic_actions():
    assert ACTION_VOCABULARY == tuple(OperationKind)
    assert len(ACTION_VOCABULARY) == 16
    assert {action.value for action in ACTION_VOCABULARY} == {
        "decompose",
        "blind_resolve",
        "branch",
        "search_memory",
        "retrieve_evidence",
        "execute",
        "simulate",
        "falsify",
        "check_assumption",
        "regenerate_from_prefix",
        "formalize",
        "compare",
        "backtrack",
        "compress_state",
        "answer",
        "abstain",
    }


def test_executor_inventory_and_state_preconditions_are_fail_closed():
    state = _state(can_execute=False, has_memory=False, has_savepoint=False)
    actions = feasible_actions(state, executors=ACTION_VOCABULARY)
    assert OperationKind.EXECUTE not in actions
    assert OperationKind.SEARCH_MEMORY not in actions
    assert OperationKind.BACKTRACK not in actions
    assert OperationKind.REGENERATE_FROM_PREFIX not in actions
    assert OperationKind.BRANCH in actions
    assert OperationKind.FALSIFY in actions

    only_declared = feasible_actions(
        _state(
            step_index=2,
            active_branches=1,
            total_branches=1,
            can_execute=True,
        ),
        executors=(OperationKind.BLIND_RESOLVE, OperationKind.ANSWER),
    )
    assert only_declared == (OperationKind.BLIND_RESOLVE, OperationKind.ANSWER)

    floor_pending = feasible_actions(
        _state(neural_steps=0, min_neural_steps=2),
        executors=(OperationKind.BRANCH, OperationKind.COMPARE, OperationKind.ABSTAIN),
    )
    assert floor_pending == (OperationKind.BRANCH,)


def test_unavailable_execute_cannot_be_selected_even_with_strong_evidence():
    evidence = ActionEvidence()
    for _ in range(MIN_ACTION_TRIALS):
        evidence = evidence.append(gain=1.0, cost=0.01)
    policy = _policy({OperationKind.EXECUTE: evidence})
    decision = policy.choose(
        _state(active_branches=1, total_branches=1, can_execute=False),
        executors=(OperationKind.EXECUTE, OperationKind.BLIND_RESOLVE),
    )
    assert decision["action"] == OperationKind.BLIND_RESOLVE.value
    assert OperationKind.EXECUTE.value not in decision["feasible_actions"]


def test_legacy_online_moments_never_claim_independent_measurement():
    weak = ActionEvidence()
    strong = ActionEvidence()
    for _ in range(MIN_ACTION_TRIALS):
        weak = weak.append(gain=-0.1, cost=0.2)
        strong = strong.append(gain=0.6, cost=0.1)
    policy = _policy(
        {
            OperationKind.BLIND_RESOLVE: weak,
            OperationKind.FORMALIZE: strong,
        }
    )
    decision = policy.choose(
        _state(),
        executors=(OperationKind.BLIND_RESOLVE, OperationKind.FORMALIZE),
    )
    assert decision["action"] == OperationKind.FORMALIZE.value
    assert decision["mode"] == "bootstrap"
    assert decision["evidence"]["basis"] == "bootstrap_prior"
    assert decision["evidence"]["n"] == MIN_ACTION_TRIALS


def _certified_snapshot(cells):
    return certified_action_snapshot(
        bucket="general|none|short|s:mid|u:mid",
        cells={
            action: (
                cell.gain_lcb,
                cell.gain_mean,
                cell.gain_ucb,
                cell.cost_ucb,
            )
            for action, cell in cells.items()
        },
    )


def _certified_cell(*, gain_lcb, gain_mean, gain_ucb, cost):
    lower = {"numerator": int(gain_lcb * 10), "denominator": 10}
    upper = {"numerator": int(gain_ucb * 10), "denominator": 10}
    return CertifiedActionEvidence.from_dict(
        {
            "n": 20,
            "unique_task_count": 20,
            "measured": True,
            "gain_mean": gain_mean,
            "gain_lcb": gain_lcb,
            "gain_ucb": gain_ucb,
            "cost_mean": cost,
            "cost_ucb": cost,
            "gain_bounds": {
                "method": ("simultaneous rational Clopper-Pearson contrast bounds"),
                "family_count": GLOBAL_BOUND_FAMILY_COUNT,
                "family_alpha": {"numerator": 1, "denominator": 20},
                "component_alpha": {"numerator": 1, "denominator": 680},
                "simultaneous_coverage_lower": {
                    "numerator": 19,
                    "denominator": 20,
                },
                "lower": lower,
                "upper": upper,
                "certified": True,
            },
            "cost_bounds": {
                "method": "simultaneous Hoeffding upper bound",
                "family_count": GLOBAL_BOUND_FAMILY_COUNT,
                "family_alpha": {"numerator": 1, "denominator": 20},
                "bounded_interval": [0.0, 1.0],
                "normalization": ("max fraction of preregistered action-resource caps"),
                "dimensions": list(ACTION_RESOURCE_DIMENSIONS),
            },
            "calibration_candidate_sha256": "a" * 64,
            "policy_sha256": "b" * 64,
        }
    )


def test_certified_positive_bounds_control_measured_selection(
    tmp_path,
    monkeypatch,
):
    weak = _certified_cell(
        gain_lcb=0.1,
        gain_mean=0.2,
        gain_ucb=0.3,
        cost=0.2,
    )
    strong = _certified_cell(
        gain_lcb=0.4,
        gain_mean=0.5,
        gain_ucb=0.6,
        cost=0.1,
    )
    snapshot, root_pem = _certified_snapshot(
        {
            OperationKind.BLIND_RESOLVE: weak,
            OperationKind.FORMALIZE: strong,
        }
    )
    root_path = tmp_path / "action-calibration-root.pem"
    root_path.write_bytes(root_pem)
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT",
        str(root_path),
    )
    policy = ValueOfComputationPolicy(snapshot)
    decision = policy.choose(
        _state(),
        executors=(OperationKind.BLIND_RESOLVE, OperationKind.FORMALIZE),
    )
    assert decision["action"] == OperationKind.FORMALIZE.value
    assert decision["mode"] == "measured"
    assert decision["evidence"]["basis"] == "measured_lcb_per_cost_ucb"


def test_sparse_exploration_is_named_and_never_claimed_as_measured():
    decision = _policy().choose(
        _state(step_index=3),
        executors=(OperationKind.BLIND_RESOLVE, OperationKind.FORMALIZE),
    )
    assert decision["mode"] == "bounded_explore"
    assert decision["evidence"]["basis"] == "bootstrap_prior"
    assert decision["evidence"]["measured"] is False
    assert validate_action_decision(decision) == decision

    tampered = deepcopy(decision)
    tampered["action"] = "answer"
    with pytest.raises(ValueError, match="infeasible|digest"):
        validate_action_decision(tampered)


def test_terminal_rules_distinguish_verified_answer_budget_and_abstention():
    policy = _policy()
    verified = policy.choose(
        _state(step_index=2, answer_verified=True),
        executors=(OperationKind.BLIND_RESOLVE, OperationKind.ANSWER, OperationKind.ABSTAIN),
    )
    assert (verified["action"], verified["mode"]) == ("answer", "verified_stop")

    execute = policy.choose(
        _state(step_index=2, can_execute=True, answer_verified=True),
        executors=(
            OperationKind.EXECUTE,
            OperationKind.ANSWER,
            OperationKind.ABSTAIN,
        ),
    )
    assert (execute["action"], execute["mode"]) == ("execute", "verified_execute")
    assert validate_action_decision(execute) == execute

    budget_answer = policy.choose(
        _state(step_index=2, budget_remaining_fraction=0.01, uncertainty=0.2),
        executors=(OperationKind.ANSWER, OperationKind.ABSTAIN),
    )
    assert (budget_answer["action"], budget_answer["mode"]) == (
        "answer",
        "budget_stop",
    )

    abstain = policy.choose(
        _state(step_index=2, irreducible_uncertainty=True),
        executors=(OperationKind.BLIND_RESOLVE, OperationKind.ABSTAIN),
    )
    assert (abstain["action"], abstain["mode"]) == (
        "abstain",
        "irreducible_abstain",
    )


def test_evidence_snapshot_is_exact_bounded_and_content_addressed():
    snapshot = build_evidence_snapshot(
        bucket="b",
        cells={OperationKind.COMPARE: ActionEvidence().append(gain=0.2, cost=0.1)},
    )
    assert validate_evidence_snapshot(snapshot) == snapshot
    tampered = deepcopy(snapshot)
    tampered["cells"]["compare"]["gain_sum"] = 0.9
    with pytest.raises(ValueError, match="digest|mathematically inconsistent"):
        validate_evidence_snapshot(tampered)
    smuggled = deepcopy(snapshot)
    smuggled["cells"]["compare"]["extra"] = True
    with pytest.raises(ValueError, match="fields differ"):
        validate_evidence_snapshot(smuggled)

    with pytest.raises(ValueError, match="mathematically inconsistent"):
        ActionEvidence(
            n=2,
            gain_sum=2.0,
            gain_sq_sum=0.1,
            cost_sum=0.2,
            cost_sq_sum=0.02,
        )


def test_transition_reward_recomputes_and_rejects_fabricated_checked_rows():
    metrics = transition_reward(
        verified_delta=0.4,
        information_gain=0.2,
        diversity_gain=0.1,
        unsupported_confidence=0.0,
        cost=0.1,
    )
    transition = {
        "schema": ACTION_TRANSITION_SCHEMA,
        "bucket": "b",
        "snapshot_sha256": "a" * 64,
        "decision_sha256": "b" * 64,
        "step_index": 2,
        "action": "falsify",
        "mode": "measured",
        "outcome": "completed",
        "checked": True,
        "metrics": metrics,
    }
    assert validate_action_transition(transition) == transition

    unchecked = {**transition, "checked": False}
    with pytest.raises(ValueError, match="not independently checked"):
        validate_action_transition(unchecked)
    fabricated = deepcopy(transition)
    fabricated["metrics"]["reward"] += 0.5
    with pytest.raises(ValueError, match="does not match"):
        validate_action_transition(fabricated)


def test_action_trace_recomputes_policy_and_public_transition_metrics():
    state = _state(
        step_index=0,
        active_branches=2,
        total_branches=2,
        residual=0.8,
        disagreement=0.3,
        verifier_score=None,
        verifier_delta=None,
        has_verifier=False,
    )
    snapshot = build_evidence_snapshot(bucket="b", cells={})
    executors = (OperationKind.DECOMPOSE,)
    decision = ValueOfComputationPolicy(snapshot).choose(state, executors=executors)
    transition = {
        "schema": ACTION_TRANSITION_SCHEMA,
        "bucket": "b",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "step_index": 0,
        "action": decision["action"],
        "mode": decision["mode"],
        "outcome": "completed",
        "checked": False,
        "metrics": transition_reward(
            verified_delta=0.0,
            information_gain=0.0,
            diversity_gain=0.1,
            unsupported_confidence=0.0,
            cost=0.01,
        ),
    }
    row = {
        "decision": decision,
        "transition": transition,
        "state_signal": state.to_dict(),
        "state_before": {
            "residual": 0.8,
            "disagreement": 0.3,
            "verifier_score": None,
            "budget_remaining_fraction": 0.8,
        },
        "state_after": {
            "residual": 0.7,
            "disagreement": 0.4,
            "verifier_score": None,
            "observed_verifier_score": None,
        },
        "affected_branches": 2,
        "verification": {
            "target_branch": None,
            "observation": {},
            "decision": "not_run",
            "restored": False,
        },
    }
    assert (
        validate_action_trace_row(
            row,
            evidence_snapshot=snapshot,
            executors=executors,
        )["transition"]
        == transition
    )

    tampered = deepcopy(row)
    tampered["state_after"]["disagreement"] = 0.2
    with pytest.raises(ValueError, match="metrics differ"):
        validate_action_trace_row(
            tampered,
            evidence_snapshot=snapshot,
            executors=executors,
        )


def test_reverted_verifier_probe_does_not_replace_accepted_score():
    state = _state(
        step_index=2,
        residual=0.4,
        disagreement=0.4,
        verifier_score=0.8,
        verifier_delta=0.0,
        has_verifier=True,
    )
    snapshot = build_evidence_snapshot(bucket="b", cells={})
    executors = (OperationKind.FALSIFY,)
    decision = ValueOfComputationPolicy(snapshot).choose(state, executors=executors)
    transition = {
        "schema": ACTION_TRANSITION_SCHEMA,
        "bucket": "b",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "step_index": 2,
        "action": "falsify",
        "mode": decision["mode"],
        "outcome": "verifier_regression_reverted_2",
        "checked": True,
        "metrics": transition_reward(
            verified_delta=-0.5,
            information_gain=-0.3,
            diversity_gain=0.0,
            unsupported_confidence=0.5,
            cost=0.1,
        ),
    }
    row = {
        "decision": decision,
        "transition": transition,
        "state_signal": state.to_dict(),
        "state_before": {
            "residual": 0.4,
            "disagreement": 0.4,
            "verifier_score": 0.8,
            "budget_remaining_fraction": 0.8,
        },
        "state_after": {
            "residual": 0.4,
            "disagreement": 0.4,
            "verifier_score": 0.8,
            "observed_verifier_score": 0.3,
        },
        "affected_branches": 2,
        "verification": {
            "target_branch": 0,
            "observation": VerifierObservation.from_value(0.3).to_dict(),
            "decision": "ranking_only",
            "restored": False,
        },
    }
    validate_action_trace_row(
        row,
        evidence_snapshot=snapshot,
        executors=executors,
    )

    tampered = deepcopy(row)
    tampered["state_after"]["verifier_score"] = 0.3
    with pytest.raises(ValueError, match="accepted verifier state"):
        validate_action_trace_row(
            tampered,
            evidence_snapshot=snapshot,
            executors=executors,
        )


def test_overlapping_confidence_interval_preserves_incumbent_point_score():
    state = _state(
        step_index=2,
        residual=0.4,
        disagreement=0.4,
        verifier_score=0.8,
        verifier_delta=0.0,
        has_verifier=True,
    )
    snapshot = build_evidence_snapshot(bucket="b", cells={})
    executors = (OperationKind.FALSIFY,)
    decision = ValueOfComputationPolicy(snapshot).choose(
        state,
        executors=executors,
    )
    transition = {
        "schema": ACTION_TRANSITION_SCHEMA,
        "bucket": "b",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "step_index": 2,
        "action": "falsify",
        "mode": decision["mode"],
        "outcome": "verified_best_preserved",
        "checked": True,
        "metrics": transition_reward(
            verified_delta=0.04,
            information_gain=0.0,
            diversity_gain=0.0,
            unsupported_confidence=0.0,
            cost=0.1,
        ),
    }
    observation = VerifierObservation(
        score=0.84,
        lower_bound=0.78,
        upper_bound=0.90,
        sample_count=32,
        basis="calibrated_interval",
        independent=True,
        evidence_sha256=hashlib.sha256(b"overlap").hexdigest(),
    )
    row = {
        "decision": decision,
        "transition": transition,
        "state_signal": state.to_dict(),
        "state_before": {
            "residual": 0.4,
            "disagreement": 0.4,
            "verifier_score": 0.8,
            "budget_remaining_fraction": 0.8,
        },
        "state_after": {
            "residual": 0.4,
            "disagreement": 0.4,
            "verifier_score": 0.8,
            "observed_verifier_score": 0.84,
        },
        "affected_branches": 1,
        "verification": {
            "target_branch": 0,
            "observation": observation.to_dict(),
            "decision": "preserve_verified",
            "restored": True,
        },
    }
    validated = validate_action_trace_row(
        row,
        evidence_snapshot=snapshot,
        executors=executors,
    )
    assert validated["state_after"]["verifier_score"] == 0.8


def test_invalid_state_and_empty_executor_inventory_fail_before_selection():
    with pytest.raises(ValueError, match="active branch"):
        _state(active_branches=3, total_branches=2)
    with pytest.raises(ValueError, match="no executable"):
        _policy().choose(_state(), executors=())
