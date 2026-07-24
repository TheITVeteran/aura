from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from core.brain.llm.latent_cortex.counterfactual_probe import (
    CounterfactualProbeResult,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.types import ComputeBudget
from core.brain.llm.latent_cortex.verified_best import VerifierObservation
from core.brain.llm.latent_cortex.virtual_quanta import (
    ARM_NAMES,
    DISABLED,
    VIRTUAL_QUANTA_RECEIPT_SCHEMA,
    VirtualQuantaConfig,
    build_empty_virtual_quanta_receipt,
    run_virtual_quanta,
    validate_virtual_quanta_receipt,
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


def _states():
    baseline = np.full((1, 4, 8), 2.0, dtype=np.float32)
    baseline[:, 0, :] = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    anchor = np.array(baseline, copy=True)
    anchor[:, 1:, :] += np.linspace(-0.4, 0.4, 8, dtype=np.float32)
    return baseline, anchor


class _Evaluator:
    def __init__(
        self,
        budget: ComputeBudget,
        *,
        guided: float = 1.0,
        control: float = 0.0,
        output_bytes: dict[str, int] | None = None,
    ):
        self.budget = budget
        self.guided = guided
        self.control = control
        self.output_bytes = output_bytes or {}
        self.calls: list[tuple[str, int]] = []

    def __call__(self, label: str, state, replicate: int):
        self.calls.append((label, replicate))
        self.budget.charge(
            8,
            8,
            operation="test_virtual_quantum_probe",
            attention_pairs=64,
            output_head_tokens=8,
        )
        self.budget.charge_verifier(
            "test_virtual_quantum_verifier",
            input_bytes=64,
            output_bytes=self.output_bytes.get(label, 64),
            host_scalar_ops=64,
        )
        score = self.guided if label == "guided_quantum" else self.control
        return CounterfactualProbeResult(
            probe_tokens_sha256=_digest(f"{label}:{replicate}:{float(np.asarray(state).sum())}"),
            probe_token_count=8,
            observation=_exact(score, f"{label}:{replicate}:{score}"),
            layer_apps=64,
        )


def _run(
    *,
    evaluator: _Evaluator | None = None,
    config: VirtualQuantaConfig | None = None,
    created_step: int = 0,
    fail_apply: bool = False,
):
    baseline, anchor = _states()
    budget = evaluator.budget if evaluator is not None else ComputeBudget()
    applied: list[np.ndarray] = []
    restored: list[np.ndarray] = []

    def apply(state):
        if fail_apply:
            raise RuntimeError("forced application failure")
        value = np.array(state, copy=True)
        applied.append(value)
        return value

    def restore(state):
        value = np.array(state, copy=True)
        restored.append(value)
        return value

    receipt = run_virtual_quanta(
        baseline_state=baseline,
        anchor_state=anchor,
        branch_index=0,
        protected_positions=(0,),
        source_positions=(0,),
        episode_id="episode-quanta",
        objective_sha256="a" * 64,
        subject_sha256="b" * 64,
        source_kv_boundary_sha256="c" * 64,
        verifier_policy_sha256="d" * 64,
        verifier_preflight_sha256="e" * 64,
        created_step=created_step,
        config=config or VirtualQuantaConfig(),
        evaluate=evaluator,
        apply_state=apply if evaluator is not None else None,
        restore_state=restore if evaluator is not None else None,
        budget=budget,
        unavailable_reason="test_no_evaluator",
    )
    return receipt, baseline, applied, restored


def _validate(receipt: dict) -> dict:
    return validate_virtual_quanta_receipt(
        receipt,
        episode_id="episode-quanta",
        objective_sha256="a" * 64,
        n_branches=2,
        expected_config=VirtualQuantaConfig.from_value(receipt["config"]),
    )


def _rehash(receipt: dict) -> None:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(payload)


def test_guided_quantum_requires_measured_win_applies_once_and_erases():
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    evaluator = _Evaluator(budget)
    receipt, baseline, applied, restored = _run(evaluator=evaluator)

    assert receipt["schema"] == VIRTUAL_QUANTA_RECEIPT_SCHEMA
    assert receipt["status"] == "applied"
    assert receipt["guided_beats_controls"] is True
    assert receipt["all_arms_stable"] is True
    assert receipt["all_arms_equal_resources"] is True
    assert receipt["all_arms_fully_metered"] is True
    assert receipt["contribution"]["lower_bound"] == 1.0
    assert receipt["contribution"]["measured_before_credit"] is True
    assert receipt["application"]["uses"] == 1
    assert receipt["application"]["ttl_valid"] is True
    assert receipt["application"]["one_use"] is True
    assert len(applied) == 1
    assert restored == []
    assert not np.array_equal(applied[0], baseline)
    assert np.array_equal(applied[0][:, 0, :], baseline[:, 0, :])
    assert receipt["erasure"]["all_zero_before_release"] is True
    assert receipt["erasure"]["private_reference_released"] is True
    assert receipt["critic_prose_authority"] is False
    assert receipt["caller_vector_authority"] is False
    assert receipt["durable_weight_change"] is False
    assert receipt["answer_text_stored"] is False
    assert len(evaluator.calls) == 3 * VirtualQuantaConfig().replicates
    assert _validate(receipt)["receipt_sha256"] == receipt["receipt_sha256"]


def test_nonzero_creation_step_binds_quantum_identity_and_ttl():
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    receipt, _, _, _ = _run(
        evaluator=_Evaluator(budget),
        created_step=17,
    )

    assert receipt["created_step"] == 17
    assert receipt["expires_step"] == 18
    assert receipt["application"]["ttl_valid"] is True
    _validate(receipt)


def test_no_quantum_and_random_controls_are_distinct_bounded_and_counterbalanced():
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    receipt, _, _, _ = _run(evaluator=_Evaluator(budget))
    arms = {row["name"]: row for row in receipt["arms"]}

    assert tuple(arms) == ARM_NAMES
    assert arms["no_quantum"]["delta_rms"] == 0.0
    assert arms["matched_random"]["state_sha256"] != arms["guided_quantum"]["state_sha256"]
    assert arms["matched_random"]["delta_rms"] == pytest.approx(
        arms["guided_quantum"]["delta_rms"],
        rel=1e-5,
        abs=1e-8,
    )
    assert arms["guided_quantum"]["relative_mutable_delta_rms"] <= 0.05 + 1e-6
    assert all(row["protected_positions_unchanged"] for row in arms.values())
    assert receipt["execution_order"][0]["arms"] != receipt["execution_order"][1]["arms"]
    for arm_name in ARM_NAMES:
        positions = [row["arms"].index(arm_name) for row in receipt["execution_order"]]
        assert len(set(positions)) == len(positions)


def test_tied_controls_restore_and_never_credit_quantum():
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    evaluator = _Evaluator(budget, guided=0.5, control=0.5)
    receipt, baseline, applied, restored = _run(evaluator=evaluator)

    assert receipt["status"] == "restored"
    assert receipt["reason"] == "guided_quantum_did_not_beat_controls"
    assert receipt["guided_beats_controls"] is False
    assert applied == []
    assert len(restored) == 1
    assert np.array_equal(restored[0], baseline)
    assert receipt["application"]["uses"] == 0
    assert receipt["application"]["post_state_sha256"] == receipt["baseline_state_sha256"]
    assert receipt["erasure"]["reason"] == "trial_restored_or_failed"
    _validate(receipt)


def test_resource_mismatch_refuses_even_when_guided_score_wins():
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    evaluator = _Evaluator(
        budget,
        output_bytes={"guided_quantum": 96},
    )
    receipt, _, applied, restored = _run(evaluator=evaluator)

    assert receipt["status"] == "restored"
    assert receipt["reason"] == "counterfactual_resource_mismatch"
    assert receipt["all_arms_equal_resources"] is False
    assert receipt["guided_beats_controls"] is False
    assert applied == []
    assert len(restored) == 1
    _validate(receipt)


def test_unmetered_equal_claims_refuse_authority():
    class Unmetered:
        def __call__(self, label, _state, replicate):
            score = 1.0 if label == "guided_quantum" else 0.0
            return CounterfactualProbeResult(
                probe_tokens_sha256=_digest(f"{label}:{replicate}"),
                probe_token_count=8,
                observation=_exact(score, f"{label}:{replicate}"),
                layer_apps=64,
            )

    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    baseline, anchor = _states()
    restored = []
    receipt = run_virtual_quanta(
        baseline_state=baseline,
        anchor_state=anchor,
        branch_index=0,
        protected_positions=(0,),
        source_positions=(0,),
        episode_id="episode-quanta",
        objective_sha256="a" * 64,
        subject_sha256="b" * 64,
        source_kv_boundary_sha256="c" * 64,
        verifier_policy_sha256="d" * 64,
        verifier_preflight_sha256="e" * 64,
        created_step=0,
        config=VirtualQuantaConfig(),
        evaluate=Unmetered(),
        apply_state=lambda state: state,
        restore_state=lambda state: restored.append(np.array(state, copy=True)) or state,
        budget=budget,
    )

    assert receipt["status"] == "restored"
    assert receipt["reason"] == "counterfactual_resource_accounting_incomplete"
    assert receipt["all_arms_equal_resources"] is True
    assert receipt["all_arms_fully_metered"] is False
    assert len(restored) == 1
    _validate(receipt)


def test_evaluator_exception_restores_baseline_and_zeroizes_direction():
    class Broken(_Evaluator):
        def __call__(self, label, state, replicate):
            if label == "guided_quantum":
                raise RuntimeError("forced evaluator failure")
            return super().__call__(label, state, replicate)

    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    receipt, baseline, applied, restored = _run(evaluator=Broken(budget))

    assert receipt["status"] == "restored"
    assert receipt["reason"] == "counterfactual_failed:RuntimeError"
    assert applied == []
    assert len(restored) == 1
    assert np.array_equal(restored[0], baseline)
    assert receipt["erasure"]["all_zero_before_release"] is True
    assert receipt["erasure"]["private_reference_released"] is True
    _validate(receipt)


def test_application_exception_preserves_measured_win_but_restores_without_credit():
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    receipt, baseline, applied, restored = _run(
        evaluator=_Evaluator(budget),
        fail_apply=True,
    )

    assert receipt["status"] == "restored"
    assert receipt["reason"] == "counterfactual_failed:RuntimeError"
    assert receipt["guided_beats_controls"] is True
    assert receipt["application"]["attempted"] is True
    assert receipt["application"]["applied"] is False
    assert receipt["application"]["uses"] == 1
    assert len(restored) == 1
    assert applied == []
    assert np.array_equal(restored[0], baseline)
    _validate(receipt)


def test_degenerate_prompt_latent_returns_valid_no_authority_receipt():
    baseline = np.ones((1, 2, 4), dtype=np.float32)
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    receipt = run_virtual_quanta(
        baseline_state=baseline,
        anchor_state=baseline,
        branch_index=0,
        protected_positions=(),
        source_positions=(),
        episode_id="episode-quanta",
        objective_sha256="a" * 64,
        subject_sha256="b" * 64,
        source_kv_boundary_sha256="c" * 64,
        verifier_policy_sha256="d" * 64,
        verifier_preflight_sha256="e" * 64,
        created_step=0,
        config=VirtualQuantaConfig(),
        evaluate=_Evaluator(budget),
        apply_state=lambda state: state,
        restore_state=lambda state: state,
        budget=budget,
    )

    assert receipt["status"] == "unavailable"
    assert receipt["reason"] == "counterfactual_failed:ValueError"
    assert receipt["quantum_id"] == ""
    assert receipt["arms"] == []
    _validate(receipt)


def test_missing_verifier_is_explicit_and_cannot_mint_authority():
    receipt, _, applied, restored = _run()

    assert receipt["status"] == "unavailable"
    assert receipt["reason"] == "test_no_evaluator"
    assert receipt["arms"] == []
    assert receipt["quantum_id"] == ""
    assert receipt["erasure"] == {}
    assert applied == []
    assert restored == []
    _validate(receipt)


def test_validator_rejects_fully_rehashed_contribution_and_erasure_lies():
    budget = ComputeBudget(max_layer_apps=10_000, wall_clock_s=30.0)
    receipt, _, _, _ = _run(evaluator=_Evaluator(budget))

    contribution_lie = copy.deepcopy(receipt)
    contribution_lie["contribution"]["lower_bound"] = 0.5
    _rehash(contribution_lie)
    with pytest.raises(ValueError, match="decision evidence"):
        _validate(contribution_lie)

    erasure_lie = copy.deepcopy(receipt)
    erasure_lie["erasure"]["all_zero_before_release"] = False
    erasure_payload = dict(erasure_lie["erasure"])
    erasure_payload.pop("erasure_sha256")
    erasure_lie["erasure"]["erasure_sha256"] = canonical_sha256(erasure_payload)
    _rehash(erasure_lie)
    with pytest.raises(ValueError, match="erasure"):
        _validate(erasure_lie)

    order_lie = copy.deepcopy(receipt)
    order_lie["execution_order"][0]["arms"].reverse()
    _rehash(order_lie)
    with pytest.raises(ValueError, match="execution order"):
        _validate(order_lie)

    quantum_id_lie = copy.deepcopy(receipt)
    quantum_id_lie["quantum_id"] = "vq-" + "0" * 24
    quantum_id_lie["erasure"]["quantum_id"] = quantum_id_lie["quantum_id"]
    erasure_payload = dict(quantum_id_lie["erasure"])
    erasure_payload.pop("erasure_sha256")
    quantum_id_lie["erasure"]["erasure_sha256"] = canonical_sha256(erasure_payload)
    _rehash(quantum_id_lie)
    with pytest.raises(ValueError, match="identity differs"):
        _validate(quantum_id_lie)

    malformed_positions = copy.deepcopy(receipt)
    malformed_positions["protected_positions"] = None
    _rehash(malformed_positions)
    with pytest.raises(ValueError, match="positions"):
        _validate(malformed_positions)


def test_config_rejects_unbounded_or_unknown_controls():
    with pytest.raises(ValueError, match="unknown keys"):
        VirtualQuantaConfig.from_value({"payload": "free-form text"})
    with pytest.raises(ValueError, match="delta bound"):
        VirtualQuantaConfig(max_relative_delta_rms=0.5)
    with pytest.raises(ValueError, match="replicates"):
        VirtualQuantaConfig(replicates=1)
    with pytest.raises(ValueError, match="TTL"):
        VirtualQuantaConfig(ttl_steps=0)


def test_inactive_receipts_are_valid_by_construction():
    disabled = VirtualQuantaConfig(mode=DISABLED)
    receipt = build_empty_virtual_quanta_receipt(
        episode_id="episode-disabled",
        objective_sha256="a" * 64,
        subject_sha256="b" * 64,
        branch_index=0,
        source_kv_boundary_sha256="c" * 64,
        protected_positions=(0,),
        source_positions=(0,),
        config=disabled,
        status="disabled",
        reason="configured_disabled",
    )
    assert (
        validate_virtual_quanta_receipt(
            receipt,
            episode_id="episode-disabled",
            objective_sha256="a" * 64,
            n_branches=1,
            expected_config=disabled,
        )["status"]
        == "disabled"
    )

    with pytest.raises(ValueError, match="status"):
        build_empty_virtual_quanta_receipt(
            episode_id="episode-disabled",
            objective_sha256="a" * 64,
            subject_sha256="b" * 64,
            branch_index=0,
            source_kv_boundary_sha256="c" * 64,
            protected_positions=(),
            source_positions=(),
            config=VirtualQuantaConfig(),
            status="disabled",
        )
    with pytest.raises(ValueError, match="source positions"):
        build_empty_virtual_quanta_receipt(
            episode_id="episode-disabled",
            objective_sha256="a" * 64,
            subject_sha256="b" * 64,
            branch_index=0,
            source_kv_boundary_sha256="c" * 64,
            protected_positions=(),
            source_positions=(0,),
        )

    baseline, anchor = _states()
    with pytest.raises(ValueError, match="episode identity"):
        run_virtual_quanta(
            baseline_state=baseline,
            anchor_state=anchor,
            branch_index=0,
            protected_positions=(0,),
            source_positions=(0,),
            episode_id="",
            objective_sha256="a" * 64,
            subject_sha256="b" * 64,
            source_kv_boundary_sha256="c" * 64,
            verifier_policy_sha256="",
            verifier_preflight_sha256="",
            created_step=0,
            config=disabled,
            evaluate=None,
            apply_state=None,
            restore_state=None,
            budget=ComputeBudget(),
        )
