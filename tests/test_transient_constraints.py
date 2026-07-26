from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain.llm.latent_cortex.blind_review import (  # noqa: E402
    _control_texts,
    run_decoy_preflight,
)
from core.brain.llm.latent_cortex.counterfactual_probe import (  # noqa: E402
    CounterfactualProbeResult,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256  # noqa: E402
from core.brain.llm.latent_cortex.resource_accounting import (  # noqa: E402
    ModelComputeProfile,
    ResourceLedger,
    build_information_receipt,
)
from core.brain.llm.latent_cortex.transient_constraints import (  # noqa: E402
    TransientConstraintConfig,
    TransientConstraintLedger,
    build_empty_transient_constraint_receipt,
    validate_transient_constraint_receipt,
)
from core.brain.llm.latent_cortex.types import ComputeBudget  # noqa: E402
from core.brain.llm.latent_cortex.verified_best import (  # noqa: E402
    VerifierObservation,
    tensor_sha256,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _exact(score: float, label: str) -> dict:
    return VerifierObservation(
        score=score,
        lower_bound=score,
        upper_bound=score,
        sample_count=1,
        basis="deterministic_exact",
        independent=True,
        evidence_sha256=_digest(label),
    ).to_dict()


def _interval(score: float, lower: float, upper: float, label: str) -> dict:
    return VerifierObservation(
        score=score,
        lower_bound=lower,
        upper_bound=upper,
        sample_count=8,
        basis="calibrated_interval",
        independent=True,
        evidence_sha256=_digest(label),
    ).to_dict()


class _Evaluator:
    def __init__(self, *, guided_score: float = 1.0, sham_score: float = 0.0):
        self.guided_score = guided_score
        self.sham_score = sham_score
        self.calls: list[tuple[str, int]] = []
        self.budget: ComputeBudget | None = None

    def __call__(self, label: str, state, replicate: int) -> CounterfactualProbeResult:
        self.calls.append((label, replicate))
        if self.budget is not None:
            self.budget.charge(
                8,
                8,
                operation="test_constraint_probe",
                attention_pairs=64,
                output_head_tokens=8,
            )
            self.budget.charge_verifier(
                "test_constraint_verifier",
                input_bytes=64,
                output_bytes=64,
                host_scalar_ops=16,
            )
        score = self.guided_score if label == "negative_direction" else self.sham_score
        return CounterfactualProbeResult(
            probe_tokens_sha256=_digest(f"{label}:{replicate}:{np.asarray(state).sum()}"),
            probe_token_count=8,
            observation=_exact(score, f"{label}:{replicate}"),
            layer_apps=64,
        )


def _ledger(
    *,
    ttl: int = 3,
    branches: int = 2,
) -> TransientConstraintLedger:
    return TransientConstraintLedger(
        episode_id="episode-test",
        objective_sha256="a" * 64,
        n_branches=branches,
        protected_positions={index: (0,) for index in range(branches)},
        config=TransientConstraintConfig(ttl_action_steps=ttl),
    )


def _states() -> tuple[np.ndarray, np.ndarray]:
    parent = np.full((1, 4, 8), 2.0, dtype=np.float32)
    failed = np.array(parent, copy=True)
    failed[:, 1:, :] = 3.0
    return parent, failed


def _admit(
    ledger: TransientConstraintLedger,
    *,
    branch: int = 0,
    action: str = "falsify",
    step: int = 0,
    evaluator: _Evaluator | None = None,
    observation: dict | None = None,
    incumbent: dict | None = None,
    verifier_preflight_sha256: str = "c" * 64,
    source_kv_boundary_sha256: str = "d" * 64,
) -> dict:
    parent, failed = _states()
    budget = ComputeBudget()
    active_evaluator = evaluator or _Evaluator()
    active_evaluator.budget = budget
    return ledger.consider_verified_failure(
        parent_state=parent,
        failed_state=failed,
        branch_index=branch,
        source_action=action,
        action_step=step,
        source_kv_boundary_sha256=source_kv_boundary_sha256,
        observation=observation or _exact(0.0, "source-failure"),
        incumbent_observation=incumbent,
        verifier_policy_sha256="b" * 64,
        verifier_preflight_sha256=verifier_preflight_sha256,
        evaluate=active_evaluator,
        budget=budget,
    )


def _rehash_receipt(receipt: dict) -> None:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(payload)


def _rehash_attempt(attempt: dict) -> None:
    payload = dict(attempt)
    payload.pop("attempt_sha256", None)
    attempt["attempt_sha256"] = canonical_sha256(payload)


def _valid_preflight() -> dict:
    preflight_seed = _digest("decoy-preflight:episode-test:" + "a" * 64)
    control_scores = {
        text: score
        for text, score in zip(
            _control_texts(preflight_seed).values(),
            (1.0, 0.0, 0.5, 0.5),
            strict=True,
        )
    }
    return run_decoy_preflight(
        control_scores.__getitem__,
        episode_id="episode-test",
        objective_sha256="a" * 64,
    )


def _external_evidence(receipt: dict) -> dict:
    preflight = _valid_preflight()
    information = build_information_receipt(
        sources=[],
        policies={"verifier": "b" * 64},
    )
    profile = ModelComputeProfile(
        model_type="constraint-test",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        head_dim=4,
    )
    resources = ResourceLedger(profile)
    totals = {name: 0 for name in resources.totals()}
    for attempt in receipt["attempts"]:
        for arm in attempt["arms"]:
            for replicate in arm["replicates"]:
                for name, amount in replicate["resource_delta"].items():
                    totals[name] += amount
                for name, amount in replicate["resource_after"].items():
                    totals[name] = max(totals[name], amount)
    resources.charge("constraint_trials", **totals)
    action_trace = []
    for attempt in receipt["attempts"]:
        action_trace.append(
            {
                "transition": {
                    "step_index": attempt["created_action_step"],
                    "action": attempt["source_action"],
                },
                "verification": {
                    "target_branch": attempt["branch_index"],
                    "observation": attempt["source_observation"],
                    "decision": "reject_verified_failure",
                    "restored": True,
                    "attempt_parent_state_sha256": attempt["parent_state_sha256"],
                    "constraint_input_state_sha256": attempt["parent_state_sha256"],
                    "candidate_state_sha256": attempt["failed_state_sha256"],
                    "restore_target_state_sha256": attempt["parent_state_sha256"],
                    "kv_boundary_before_sha256": attempt["source_kv_boundary_sha256"],
                    "kv_boundary_after_sha256": attempt["source_kv_boundary_sha256"],
                    "branch_step_before": 0,
                    "branch_step_after": 1,
                },
                "transient_constraint_attempt": attempt,
                "transient_constraint": {},
            }
        )
    for application in receipt["applications"]:
        while len(action_trace) <= application["applied_action_step"]:
            action_trace.append(
                {
                    "transition": {
                        "step_index": len(action_trace),
                        "action": "decompose",
                    },
                    "verification": {
                        "target_branch": None,
                        "observation": {},
                        "decision": "not_run",
                    },
                    "transient_constraint_attempt": {},
                    "transient_constraint": {},
                }
            )
        row = action_trace[application["applied_action_step"]]
        row["transition"]["action"] = application["applied_action"]
        row["transient_constraint"] = application
        row["verification"] = {
            "target_branch": application["branch_index"],
            "observation": application["followup_observation"],
            "decision": "reject_verified_failure",
            "restored": True,
            "attempt_parent_state_sha256": application["pre_state_sha256"],
            "constraint_input_state_sha256": application["pre_state_sha256"],
            "candidate_state_sha256": application["post_recurrence_state_sha256"],
            "restore_target_state_sha256": application["pre_state_sha256"],
            "kv_boundary_before_sha256": application["kv_boundary_before_sha256"],
            "kv_boundary_after_sha256": application["kv_boundary_after_sha256"],
            "branch_step_before": application["branch_step_before"],
            "branch_step_after": application["branch_step_after"],
        }
    return {
        "cognitive_action_trace": action_trace,
        "verifier_preflight": preflight,
        "information_accounting": information,
        "resource_accounting": resources.to_receipt(),
        "kv_state_tree": {
            "nodes": [
                {
                    "node_sha256": "d" * 64,
                    "branch_index": 0,
                }
            ]
        },
        "require_external_bindings": True,
    }


def test_verified_failure_admits_one_use_branch_local_constraint_and_reduces_repeat():
    ledger = _ledger()
    evaluator = _Evaluator()
    attempt = _admit(ledger, evaluator=evaluator)
    assert attempt["status"] == "admitted"
    assert attempt["guided_beats_controls"] is True
    assert attempt["controls_repeat_failure"] is True
    assert len(evaluator.calls) == 6
    assert (
        ledger.pending_action(
            branch_index=0,
            action_step=1,
            kv_boundary_sha256="d" * 64,
            state_sha256=tensor_sha256(_states()[0]),
        )
        == "falsify"
    )

    current = np.full((1, 4, 8), 2.0, dtype=np.float32)
    changed, application = ledger.apply_next(
        current,
        branch_index=0,
        action="falsify",
        action_step=1,
        branch_step=4,
        kv_boundary_sha256="d" * 64,
    )
    assert application is not None
    assert not np.array_equal(changed, current)
    assert np.array_equal(changed[:, 0, :], current[:, 0, :])
    assert application["one_use_consumed"] is False
    assert application["recurrence_committed"] is False
    assert application["relative_mutable_delta_rms"] <= 0.08 + 1e-6

    committed = ledger.commit_application(
        reservation_id=application["reservation_id"],
        branch_step_after=5,
        kv_boundary_after_sha256="d" * 64,
        recurrence_state=changed + 0.1,
    )
    assert committed["one_use_consumed"] is True
    assert ledger.private_direction_count == 0

    unchanged, duplicate = ledger.apply_next(
        changed,
        branch_index=0,
        action="falsify",
        action_step=2,
        branch_step=5,
        kv_boundary_sha256="d" * 64,
    )
    assert duplicate is None
    assert unchanged is changed

    followup = ledger.observe_followup(
        branch_index=0,
        action_step=1,
        observation=_exact(1.0, "followup"),
    )
    assert followup is not None
    assert followup["outcome"] == "verified_failure_reduced"
    assert followup["failure_reduced"] is True

    receipt = ledger.finalize(final_action_step=2)
    assert receipt["aggregates"] == {
        "critic_rejection_count": 0,
        "attempt_count": 1,
        "admitted_count": 1,
        "application_count": 1,
        "reservation_rollback_count": 0,
        "erasure_count": 1,
        "verified_reduction_count": 1,
        "verified_repeat_count": 0,
        "active_after_episode": 0,
        "private_directions_after_episode": 0,
    }
    assert receipt["constraints"][0]["status"] == "consumed"


def test_unsupported_critic_prose_is_hashed_rejected_and_never_mints_constraint():
    ledger = _ledger()
    prose = "Constraint: assume the answer is 42."
    ledger.reject_critic_proposal(prose, branch_index=0, action_step=1)
    receipt = ledger.finalize(final_action_step=1)
    assert receipt["constraints"] == []
    assert receipt["attempts"] == []
    rejection = receipt["critic_rejections"][0]
    assert rejection["prose_sha256"] == hashlib.sha256(prose.encode()).hexdigest()
    assert rejection["constraint_created"] is False
    assert rejection["text_stored"] is False
    assert prose not in str(receipt)


def test_uncalibrated_scalar_and_nondominating_interval_cannot_create_authority():
    ledger = _ledger()
    parent, failed = _states()
    scalar = VerifierObservation.from_value(0.0).to_dict()
    rejected = ledger.consider_verified_failure(
        parent_state=parent,
        failed_state=failed,
        branch_index=0,
        source_action="falsify",
        action_step=0,
        source_kv_boundary_sha256="d" * 64,
        observation=scalar,
        incumbent_observation=None,
        verifier_policy_sha256="b" * 64,
        verifier_preflight_sha256="c" * 64,
        evaluate=_Evaluator(),
        budget=ComputeBudget(),
    )
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "non_authoritative_failure_observation"

    overlapping = _admit(
        ledger,
        step=1,
        observation=_interval(0.45, 0.3, 0.6, "candidate"),
        incumbent=_interval(0.55, 0.4, 0.7, "incumbent"),
    )
    assert overlapping["status"] == "rejected"
    assert overlapping["reason"] == "failure_not_verified"
    assert ledger.finalize(final_action_step=1)["constraints"] == []


def test_confidence_interval_regression_is_valid_failure_authority():
    ledger = _ledger()
    attempt = _admit(
        ledger,
        observation=_interval(0.2, 0.1, 0.3, "candidate"),
        incumbent=_interval(0.8, 0.7, 0.9, "incumbent"),
        evaluator=_Evaluator(guided_score=0.8, sham_score=0.2),
    )
    assert attempt["failure_kind"] == "confidence_interval_regression"
    assert attempt["status"] == "admitted"


def test_calibrated_zero_interval_is_consistent_failure_authority():
    ledger = _ledger()
    attempt = _admit(
        ledger,
        observation=_interval(0.0, 0.0, 0.0, "calibrated-zero"),
    )
    assert attempt["failure_kind"] == "calibrated_zero_rejection"
    assert attempt["status"] == "admitted"


def test_constraint_scope_requires_same_branch_action_and_unexpired_step():
    ledger = _ledger(ttl=2)
    _admit(ledger, action="check_assumption", step=2)
    state = np.full((1, 4, 8), 2.0, dtype=np.float32)
    for branch, action, step in (
        (1, "check_assumption", 3),
        (0, "falsify", 3),
    ):
        output, application = ledger.apply_next(
            state,
            branch_index=branch,
            action=action,
            action_step=step,
            branch_step=4,
            kv_boundary_sha256="d" * 64,
        )
        assert output is state
        assert application is None

    expired, application = ledger.apply_next(
        state,
        branch_index=0,
        action="check_assumption",
        action_step=5,
        branch_step=4,
        kv_boundary_sha256="d" * 64,
    )
    assert expired is state
    assert application is None
    receipt = ledger.finalize(final_action_step=5)
    assert receipt["constraints"][0]["status"] == "expired_ttl"


def test_pending_constraint_with_stale_kv_is_erased_before_policy_selection():
    ledger = _ledger()
    _admit(ledger)

    assert (
        ledger.pending_action(
            branch_index=0,
            action_step=1,
            kv_boundary_sha256="e" * 64,
            state_sha256=tensor_sha256(_states()[0]),
        )
        is None
    )
    assert ledger.private_direction_count == 0
    receipt = ledger.finalize(final_action_step=1)
    assert receipt["constraints"][0]["status"] == "expired_stale_kv"
    assert receipt["erasures"][0]["reason"] == "stale_kv_boundary"


def test_pending_constraint_with_stale_parent_state_is_erased_before_policy_selection():
    ledger = _ledger()
    _admit(ledger)
    stale_state = np.full((1, 4, 8), 2.5, dtype=np.float32)

    assert (
        ledger.pending_action(
            branch_index=0,
            action_step=1,
            kv_boundary_sha256="d" * 64,
            state_sha256=tensor_sha256(stale_state),
        )
        is None
    )
    assert ledger.private_direction_count == 0
    receipt = ledger.finalize(final_action_step=1)
    assert receipt["constraints"][0]["status"] == "expired_stale_state"
    assert receipt["erasures"][0]["reason"] == "stale_parent_state"


def test_constraint_application_rejects_a_different_state_with_the_same_kv_boundary():
    ledger = _ledger()
    _admit(ledger)
    stale_state = np.full((1, 4, 8), 2.5, dtype=np.float32)

    unchanged, application = ledger.apply_next(
        stale_state,
        branch_index=0,
        action="falsify",
        action_step=1,
        branch_step=4,
        kv_boundary_sha256="d" * 64,
    )

    assert unchanged is stale_state
    assert application is None
    receipt = ledger.finalize(final_action_step=1)
    assert receipt["constraints"][0]["status"] == "expired_stale_state"
    assert receipt["erasures"][0]["reason"] == "stale_parent_state"


def test_relative_rms_bound_is_computed_only_over_mutable_slots():
    ledger = TransientConstraintLedger(
        episode_id="episode-test",
        objective_sha256="a" * 64,
        n_branches=1,
        protected_positions={0: (0, 1, 2)},
        config=TransientConstraintConfig(max_relative_delta_rms=0.08),
    )
    attempt = _admit(ledger)
    assert attempt["status"] == "admitted"
    guided = next(arm for arm in attempt["arms"] if arm["name"] == "negative_direction")
    assert guided["relative_mutable_delta_rms"] <= 0.08 + 1e-6

    state = np.full((1, 4, 8), 2.0, dtype=np.float32)
    changed, reservation = ledger.apply_next(
        state,
        branch_index=0,
        action="falsify",
        action_step=1,
        branch_step=4,
        kv_boundary_sha256="d" * 64,
    )
    assert reservation is not None
    mutable_delta_rms = float(np.sqrt(np.mean(np.square(changed[:, 3:, :] - state[:, 3:, :]))))
    mutable_state_rms = float(np.sqrt(np.mean(np.square(state[:, 3:, :]))))
    assert mutable_delta_rms <= 0.08 * mutable_state_rms + 1e-7
    assert np.array_equal(changed[:, :3, :], state[:, :3, :])
    ledger.rollback_application(
        reservation_id=reservation["reservation_id"],
        restored_state=state,
        branch_step_after=4,
        kv_boundary_after_sha256="d" * 64,
        reason="cancelled",
    )
    ledger.abort_all()
    assert ledger.private_direction_count == 0


def test_no_constraint_when_guided_arm_does_not_beat_repeating_controls():
    ledger = _ledger()
    attempt = _admit(
        ledger,
        evaluator=_Evaluator(guided_score=0.0, sham_score=0.0),
    )
    assert attempt["status"] == "restored"
    assert attempt["reason"] == "guided_candidate_did_not_beat_controls"
    assert attempt["guided_beats_controls"] is False
    receipt = ledger.finalize(final_action_step=0)
    assert receipt["constraints"] == []
    assert receipt["aggregates"]["admitted_count"] == 0


def test_unequal_verifier_output_work_prevents_resource_parity():
    class OutcomeSizedEvaluator(_Evaluator):
        def __call__(self, label, state, replicate):
            self.calls.append((label, replicate))
            assert self.budget is not None
            self.budget.charge(
                8,
                8,
                operation="test_constraint_probe",
                attention_pairs=64,
                output_head_tokens=8,
            )
            self.budget.charge_verifier(
                "test_constraint_verifier",
                input_bytes=64,
                output_bytes=96 if label == "negative_direction" else 64,
                host_scalar_ops=16,
            )
            score = 1.0 if label == "negative_direction" else 0.0
            return CounterfactualProbeResult(
                probe_tokens_sha256=_digest(f"{label}:{replicate}"),
                probe_token_count=8,
                observation=_exact(score, f"{label}:{replicate}"),
                layer_apps=64,
            )

    ledger = _ledger()
    attempt = _admit(ledger, evaluator=OutcomeSizedEvaluator())
    assert attempt["status"] == "restored"
    assert attempt["all_arms_equal_allocated_resources"] is False
    assert attempt["reason"] == "control_resource_mismatch"
    output_sizes = {
        replicate["resource_delta"]["verifier_output_bytes"]
        for arm in attempt["arms"]
        for replicate in arm["replicates"]
    }
    assert output_sizes == {64, 96}


def test_equal_but_unmetered_probe_claims_cannot_mint_constraint_authority():
    class UnmeteredEvaluator(_Evaluator):
        def __call__(self, label, state, replicate):
            score = 1.0 if label == "negative_direction" else 0.0
            return CounterfactualProbeResult(
                probe_tokens_sha256=_digest(f"unmetered:{label}:{replicate}"),
                probe_token_count=8,
                observation=_exact(score, f"unmetered:{label}:{replicate}"),
                layer_apps=64,
            )

    ledger = _ledger()
    attempt = _admit(ledger, evaluator=UnmeteredEvaluator())
    assert attempt["status"] == "restored"
    assert attempt["all_arms_equal_allocated_resources"] is True
    assert attempt["all_arms_fully_metered"] is False
    assert attempt["reason"] == "control_resource_accounting_incomplete"


def test_non_repeating_controls_cannot_support_repeated_error_reduction_claim():
    ledger = _ledger()
    attempt = _admit(
        ledger,
        evaluator=_Evaluator(guided_score=1.0, sham_score=0.5),
    )
    assert attempt["status"] == "restored"
    assert attempt["controls_repeat_failure"] is False
    assert attempt["reason"] == "controls_did_not_repeat_verified_failure"


def test_evaluator_failure_restores_without_constraint_authority():
    ledger = _ledger()

    def broken(_label, _state, _replicate):
        raise RuntimeError("probe failed")

    parent, failed = _states()
    attempt = ledger.consider_verified_failure(
        parent_state=parent,
        failed_state=failed,
        branch_index=0,
        source_action="falsify",
        action_step=0,
        source_kv_boundary_sha256="d" * 64,
        observation=_exact(0.0, "source"),
        incumbent_observation=None,
        verifier_policy_sha256="b" * 64,
        verifier_preflight_sha256="c" * 64,
        evaluate=broken,
    )
    assert attempt["status"] == "restored"
    assert attempt["reason"] == "evaluation_failed:RuntimeError"
    assert ledger.finalize(final_action_step=0)["constraints"] == []


def test_service_validator_rejects_fully_rehashed_guided_success_lie():
    ledger = _ledger()
    _admit(ledger)
    receipt = ledger.finalize(final_action_step=0)
    forged = copy.deepcopy(receipt)
    guided = next(
        arm for arm in forged["attempts"][0]["arms"] if arm["name"] == "negative_direction"
    )
    for replicate in guided["replicates"]:
        replicate["observation"] = _exact(0.0, "forged-guided")
    _rehash_attempt(forged["attempts"][0])
    _rehash_receipt(forged)
    with pytest.raises(ValueError, match="trial decision|authority identity"):
        validate_transient_constraint_receipt(
            forged,
            episode_id="episode-test",
            objective_sha256="a" * 64,
            n_branches=2,
            protected_positions={0: (0,), 1: (0,)},
            expected_config=TransientConstraintConfig(),
        )


def test_service_validator_rejects_second_application_of_one_use_constraint():
    ledger = _ledger()
    _admit(ledger)
    state = np.full((1, 4, 8), 2.0, dtype=np.float32)
    changed, reservation = ledger.apply_next(
        state,
        branch_index=0,
        action="falsify",
        action_step=1,
        branch_step=4,
        kv_boundary_sha256="d" * 64,
    )
    assert reservation is not None
    ledger.commit_application(
        reservation_id=reservation["reservation_id"],
        branch_step_after=5,
        kv_boundary_after_sha256="d" * 64,
        recurrence_state=changed + 0.1,
    )
    receipt = ledger.finalize(final_action_step=1)
    forged = copy.deepcopy(receipt)
    duplicate = copy.deepcopy(forged["applications"][0])
    duplicate["ordinal"] = 1
    duplicate["applied_action_step"] = 2
    payload = dict(duplicate)
    payload.pop("application_sha256")
    duplicate["application_sha256"] = canonical_sha256(payload)
    forged["applications"].append(duplicate)
    forged["aggregates"]["application_count"] = 2
    _rehash_receipt(forged)
    with pytest.raises(ValueError, match="application is invalid"):
        validate_transient_constraint_receipt(
            forged,
            episode_id="episode-test",
            objective_sha256="a" * 64,
            n_branches=2,
            protected_positions={0: (0,), 1: (0,)},
        )


def test_reserved_constraint_rolls_back_exactly_without_consuming_authority():
    ledger = _ledger()
    _admit(ledger)
    state = np.full((1, 4, 8), 2.0, dtype=np.float32)
    changed, reservation = ledger.apply_next(
        state,
        branch_index=0,
        action="falsify",
        action_step=1,
        branch_step=4,
        kv_boundary_sha256="d" * 64,
    )
    assert reservation is not None
    rollback = ledger.rollback_application(
        reservation_id=reservation["reservation_id"],
        restored_state=state,
        branch_step_after=4,
        kv_boundary_after_sha256="d" * 64,
        reason="budget_refused",
    )
    assert rollback["restored_state_sha256"] == rollback["pre_state_sha256"]
    assert rollback["authority_consumed"] is False
    assert ledger.private_direction_count == 1

    changed_again, second_reservation = ledger.apply_next(
        state,
        branch_index=0,
        action="falsify",
        action_step=2,
        branch_step=4,
        kv_boundary_sha256="d" * 64,
    )
    assert second_reservation is not None
    assert np.array_equal(changed_again, changed)
    ledger.commit_application(
        reservation_id=second_reservation["reservation_id"],
        branch_step_after=5,
        kv_boundary_after_sha256="d" * 64,
        recurrence_state=changed_again + 0.1,
    )
    receipt = ledger.finalize(final_action_step=2)
    assert receipt["aggregates"]["reservation_rollback_count"] == 1
    assert receipt["aggregates"]["application_count"] == 1
    assert receipt["constraints"][0]["status"] == "consumed"
    assert receipt["erasures"][0]["all_zero_before_release"] is True


def test_stale_kv_boundary_expires_and_zeroizes_private_direction():
    ledger = _ledger()
    _admit(ledger)
    state = np.full((1, 4, 8), 2.0, dtype=np.float32)
    unchanged, application = ledger.apply_next(
        state,
        branch_index=0,
        action="falsify",
        action_step=1,
        branch_step=4,
        kv_boundary_sha256="e" * 64,
    )
    assert unchanged is state
    assert application is None
    assert ledger.private_direction_count == 0
    receipt = ledger.finalize(final_action_step=1)
    assert receipt["constraints"][0]["status"] == "expired_stale_kv"
    assert receipt["erasures"][0]["reason"] == "stale_kv_boundary"


def test_stale_oldest_constraint_does_not_suppress_newer_matching_constraint():
    ledger = _ledger()
    stale = _admit(ledger, step=0, source_kv_boundary_sha256="d" * 64)
    current = _admit(ledger, step=1, source_kv_boundary_sha256="e" * 64)
    assert stale["status"] == "admitted"
    assert current["status"] == "admitted"

    state = np.full((1, 4, 8), 2.0, dtype=np.float32)
    changed, reservation = ledger.apply_next(
        state,
        branch_index=0,
        action="falsify",
        action_step=2,
        branch_step=4,
        kv_boundary_sha256="e" * 64,
    )

    assert reservation is not None
    assert reservation["constraint_id"] == current["constraint_id"]
    assert not np.array_equal(changed, state)
    ledger.rollback_application(
        reservation_id=reservation["reservation_id"],
        restored_state=state,
        branch_step_after=4,
        kv_boundary_after_sha256="e" * 64,
        reason="cancelled",
    )
    ledger.abort_all()
    assert ledger.private_direction_count == 0


def test_episode_abort_zeroizes_private_direction_and_finalizes_diagnostic_receipt():
    ledger = _ledger()
    attempt = _admit(ledger)
    assert attempt["status"] == "admitted"

    ledger.abort_all()
    assert ledger.private_direction_count == 0
    receipt = ledger.finalize(final_action_step=1)

    assert receipt["constraints"][0]["status"] == "aborted_episode_failure"
    assert receipt["constraints"][0]["private_direction_erased"] is True
    assert receipt["erasures"][0]["reason"] == "episode_aborted"
    assert receipt["aggregates"]["active_after_episode"] == 0
    assert receipt["aggregates"]["private_directions_after_episode"] == 0


def test_external_binding_accepts_exact_sources_and_rejects_rehashed_lies():
    ledger = _ledger()
    preflight = _valid_preflight()
    _admit(
        ledger,
        verifier_preflight_sha256=preflight["receipt_sha256"],
    )
    state = np.full((1, 4, 8), 2.0, dtype=np.float32)
    changed, reservation = ledger.apply_next(
        state,
        branch_index=0,
        action="falsify",
        action_step=1,
        branch_step=4,
        kv_boundary_sha256="d" * 64,
    )
    assert reservation is not None
    ledger.commit_application(
        reservation_id=reservation["reservation_id"],
        branch_step_after=5,
        kv_boundary_after_sha256="d" * 64,
        recurrence_state=changed + 0.1,
    )
    receipt = ledger.finalize(final_action_step=1)
    evidence = _external_evidence(receipt)
    assert (
        validate_transient_constraint_receipt(
            receipt,
            episode_id="episode-test",
            objective_sha256="a" * 64,
            n_branches=2,
            protected_positions={0: (0,), 1: (0,)},
            **evidence,
        )["receipt_sha256"]
        == receipt["receipt_sha256"]
    )

    attacked_action = copy.deepcopy(evidence)
    attacked_action["cognitive_action_trace"][0]["transient_constraint_attempt"]["reason"] = (
        "forged"
    )
    with pytest.raises(ValueError, match="source binding"):
        validate_transient_constraint_receipt(
            receipt,
            episode_id="episode-test",
            objective_sha256="a" * 64,
            n_branches=2,
            protected_positions={0: (0,), 1: (0,)},
            **attacked_action,
        )

    attacked_policy = copy.deepcopy(evidence)
    attacked_policy["information_accounting"]["policies"]["verifier"] = "f" * 64
    attacked_policy["information_accounting"] = build_information_receipt(
        sources=attacked_policy["information_accounting"]["sources"],
        policies=attacked_policy["information_accounting"]["policies"],
    )
    with pytest.raises(ValueError, match="source binding"):
        validate_transient_constraint_receipt(
            receipt,
            episode_id="episode-test",
            objective_sha256="a" * 64,
            n_branches=2,
            protected_positions={0: (0,), 1: (0,)},
            **attacked_policy,
        )

    attacked_kv = copy.deepcopy(evidence)
    attacked_kv["kv_state_tree"]["nodes"][0]["node_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source binding"):
        validate_transient_constraint_receipt(
            receipt,
            episode_id="episode-test",
            objective_sha256="a" * 64,
            n_branches=2,
            protected_positions={0: (0,), 1: (0,)},
            **attacked_kv,
        )


def test_empty_receipt_is_valid_and_explicitly_has_no_authority():
    receipt = build_empty_transient_constraint_receipt(
        episode_id="empty",
        objective_sha256="d" * 64,
        n_branches=1,
        protected_positions={0: ()},
    )
    assert receipt["attempts"] == []
    assert receipt["constraints"] == []
    assert receipt["applications"] == []
    assert receipt["critic_prose_authority"] is False
