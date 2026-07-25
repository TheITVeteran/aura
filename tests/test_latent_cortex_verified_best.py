"""Confidence-bound best-state authority and branch-local rollback."""

from __future__ import annotations

import copy
import hashlib

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.latent_cortex_service import LatentCortexService  # noqa: E402
from core.brain.llm.latent_cortex.branches import BranchEnsemble, BranchState  # noqa: E402
from core.brain.llm.latent_cortex.counterfactual_probe import (  # noqa: E402
    CounterfactualProbeResult,
)
from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.epistemic_state import OperationKind  # noqa: E402
from core.brain.llm.latent_cortex.loop_core import canonical_sha256  # noqa: E402
from core.brain.llm.latent_cortex.recurrence import (  # noqa: E402
    HaltingController,
    WindowRunner,
)
from core.brain.llm.latent_cortex.task_verifiers import (  # noqa: E402
    check_arithmetic_claims,
)
from core.brain.llm.latent_cortex.transient_constraints import (  # noqa: E402
    TransientConstraintConfig,
    TransientConstraintLedger,
    validate_transient_constraint_receipt,
)
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.brain.llm.latent_cortex.value_of_computation import (  # noqa: E402
    ActionEvidence,
    build_evidence_snapshot,
    validate_action_trace_row,
)
from core.brain.llm.latent_cortex.verified_best import (  # noqa: E402
    VERIFIER_OBSERVATION_SCHEMA,
    VerifierObservation,
    build_verified_best_receipt,
    tensor_sha256,
    validate_verified_best_receipt,
)
from core.brain.llm.latent_cortex.virtual_quanta import (  # noqa: E402
    VirtualQuantaConfig,
    validate_virtual_quanta_receipt,
)


class _Workspace:
    def __init__(self, z):
        self.z = z

    def update(self, z):
        self.z = z


def _branch(index: int, value: float) -> BranchState:
    state = mx.full((1, 2, 4), value)
    return BranchState(
        index=index,
        role=f"role-{index}",
        workspace=_Workspace(state),
        halting=HaltingController(
            config=RecurrenceConfig(min_steps=1, max_steps=6),
            best_state=state,
        ),
        z=state,
        anchor=state,
    )


def _observation(
    *,
    score: float,
    lower: float,
    upper: float,
    name: str,
    basis: str = "calibrated_interval",
    samples: int = 32,
):
    return {
        "schema": VERIFIER_OBSERVATION_SCHEMA,
        "score": score,
        "lower_bound": lower,
        "upper_bound": upper,
        "sample_count": samples,
        "basis": basis,
        "independent": True,
        "evidence_sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
    }


def _action(
    step: int,
    branch: int,
    observation,
    decision: str,
    restored: bool,
    *,
    candidate_state_sha256: str | None = None,
    restore_target_state_sha256: str | None = None,
):
    verification = {
        "target_branch": branch,
        "observation": observation.to_dict(),
        "decision": decision,
        "restored": restored,
    }
    if candidate_state_sha256 is not None:
        verification["candidate_state_sha256"] = candidate_state_sha256
        verification["restore_target_state_sha256"] = restore_target_state_sha256 or ""
    return {
        "decision": {"step_index": step},
        "verification": verification,
    }


def test_scalar_verifier_can_rank_but_cannot_certify_state():
    branch = _branch(0, 1.0)
    ensemble = BranchEnsemble(
        [branch],
        BranchConfig(n_branches=1),
        RecurrenceConfig(min_steps=1, max_steps=6),
    )
    observation, decision, restored = ensemble.observe_verified_best(
        branch,
        0.95,
        action_step=0,
    )
    assert observation.authoritative is False
    assert decision == "ranking_only"
    assert restored is False
    assert branch.verified_best_state is None
    with pytest.raises(ValueError, match="underpowered"):
        VerifierObservation.from_value(
            _observation(
                score=0.8,
                lower=0.7,
                upper=0.9,
                name="underpowered",
                samples=7,
            )
        )


def test_confidence_dominance_promotes_and_overlap_preserves_branch_locally():
    first = _branch(0, 1.0)
    peer = _branch(1, 10.0)
    ensemble = BranchEnsemble(
        [first, peer],
        BranchConfig(n_branches=2),
        RecurrenceConfig(min_steps=1, max_steps=6),
    )
    first_observation, decision, restored = ensemble.observe_verified_best(
        first,
        _observation(score=0.80, lower=0.75, upper=0.85, name="first"),
        action_step=0,
    )
    assert decision == "promote" and restored is False
    certified_state = first.z
    peer_before = peer.z

    first.z = mx.full((1, 2, 4), 2.0)
    first.workspace.update(first.z)
    overlap, decision, restored = ensemble.observe_verified_best(
        first,
        _observation(score=0.84, lower=0.80, upper=0.88, name="overlap"),
        action_step=1,
    )
    assert decision == "preserve_verified" and restored is True
    assert tensor_sha256(first.z) == tensor_sha256(certified_state)
    assert peer.z is peer_before

    first.z = mx.full((1, 2, 4), 3.0)
    first.workspace.update(first.z)
    better, decision, restored = ensemble.observe_verified_best(
        first,
        _observation(score=0.94, lower=0.90, upper=0.97, name="better"),
        action_step=2,
    )
    assert decision == "promote" and restored is False
    branch_best = first.verified_best_observation
    assert branch_best
    assert branch_best["lower_bound"] == 0.9

    actions = [
        _action(0, 0, first_observation, "promote", False),
        _action(1, 0, overlap, "preserve_verified", True),
        _action(2, 0, better, "promote", False),
    ]
    ensemble.final_state(first)
    ensemble.final_state(peer)
    receipt = build_verified_best_receipt(
        branches=[first, peer],
        cognitive_action_trace=actions,
        loop_stability={"receipt_sha256": "d" * 64},
    )
    validated = validate_verified_best_receipt(
        receipt,
        cognitive_action_trace=actions,
        loop_stability={"receipt_sha256": "d" * 64},
        expected_n_branches=2,
    )
    assert validated["authoritative_promotions"] == 2
    assert validated["verified_preservations"] == 1

    forged = copy.deepcopy(receipt)
    forged["branches"][0]["decisions"][1]["decision"] = "promote"
    with pytest.raises(ValueError):
        validate_verified_best_receipt(
            forged,
            cognitive_action_trace=actions,
            loop_stability={"receipt_sha256": "d" * 64},
            expected_n_branches=2,
        )
    with pytest.raises(ValueError, match="loop-stability"):
        validate_verified_best_receipt(
            receipt,
            cognitive_action_trace=actions,
            loop_stability={},
            expected_n_branches=2,
        )


def test_deterministic_exact_observation_and_final_reversion_are_explicit():
    branch = _branch(0, 1.0)
    recurrence = RecurrenceConfig(min_steps=1, max_steps=6)
    ensemble = BranchEnsemble([branch], BranchConfig(n_branches=1), recurrence)
    exact = _observation(
        score=1.0,
        lower=1.0,
        upper=1.0,
        name="theorem",
        basis="deterministic_exact",
        samples=1,
    )
    ensemble.observe_verified_best(branch, exact, action_step=0)
    branch.z = mx.full((1, 2, 4), 9.0)
    branch.workspace.update(branch.z)
    final, reverted, source = ensemble.final_state(branch)
    assert reverted is True
    assert source == "verified"
    assert tensor_sha256(final) == branch.verified_best_state_sha256


def test_deterministic_exact_zero_is_rejected_and_never_becomes_verified_best():
    branch = _branch(0, 1.0)
    restore_target = branch.z
    restore_target_sha256 = tensor_sha256(restore_target)
    recurrence = RecurrenceConfig(min_steps=1, max_steps=6)
    ensemble = BranchEnsemble([branch], BranchConfig(n_branches=1), recurrence)
    rejected = _observation(
        score=0.0,
        lower=0.0,
        upper=0.0,
        samples=1,
        basis="deterministic_exact",
        name="exact-rejection",
    )
    observation, decision, restored = ensemble.observe_verified_best(
        branch,
        rejected,
        action_step=0,
        restore_target_state_sha256=restore_target_sha256,
    )
    assert decision == "reject_verified_failure"
    assert restored is False
    assert branch.verified_best_state is None
    assert branch.verified_best_observation == {}

    branch.z = mx.full((1, 2, 4), 7.0)
    branch.workspace.update(branch.z)
    with pytest.raises(RuntimeError, match="committed parent"):
        ensemble.commit_verified_failure_restore(branch, action_step=0)
    branch.z = restore_target
    branch.workspace.update(branch.z)
    ensemble.commit_verified_failure_restore(branch, action_step=0)
    ensemble.final_state(branch)
    actions = [
        _action(
            0,
            0,
            observation,
            "reject_verified_failure",
            True,
            candidate_state_sha256=branch.verified_best_trace[0]["candidate_state_sha256"],
            restore_target_state_sha256=restore_target_sha256,
        )
    ]
    receipt = build_verified_best_receipt(
        branches=[branch],
        cognitive_action_trace=actions,
        loop_stability={"receipt_sha256": "d" * 64},
    )
    validated = validate_verified_best_receipt(
        receipt,
        cognitive_action_trace=actions,
        loop_stability={"receipt_sha256": "d" * 64},
        expected_n_branches=1,
    )
    assert validated["authoritative_promotions"] == 0
    assert validated["branches"][0]["final_best_state_sha256"] == ""


def test_one_branch_savepoint_does_not_mutate_peer_transaction_state():
    first = _branch(0, 1.0)
    peer = _branch(1, 2.0)
    ensemble = BranchEnsemble(
        [first, peer],
        BranchConfig(n_branches=2),
        RecurrenceConfig(min_steps=1, max_steps=6),
    )
    assert ensemble.savepoint_branch(first) is True
    assert first.savepoint is not None
    assert peer.savepoint is None


def test_fixed_depth_keeps_scheduled_state_even_with_verified_incumbent():
    branch = _branch(0, 1.0)
    recurrence = RecurrenceConfig(min_steps=1, max_steps=2, fixed_depth=True)
    branch.halting.config = recurrence
    ensemble = BranchEnsemble([branch], BranchConfig(n_branches=1), recurrence)
    ensemble.observe_verified_best(
        branch,
        _observation(
            score=1.0,
            lower=1.0,
            upper=1.0,
            name="fixed",
            basis="deterministic_exact",
            samples=1,
        ),
        action_step=0,
    )
    scheduled = mx.full((1, 2, 4), 4.0)
    branch.z = scheduled
    branch.workspace.update(scheduled)
    final, reverted, source = ensemble.final_state(branch)
    assert final is scheduled
    assert reverted is False
    assert source == "current"


class _BoundedVerifier:
    def __init__(self):
        self.bound_calls = 0

    def __call__(self, text: str) -> float:
        if text.startswith("Independent consistency check:"):
            return float(check_arithmetic_claims(text)["score"])
        return 0.8

    def observe_with_bounds(self, _text: str):
        self.bound_calls += 1
        score = min(0.95, 0.78 + 0.02 * self.bound_calls)
        return _observation(
            score=score,
            lower=score,
            upper=score,
            name=f"live-{self.bound_calls}",
            basis="deterministic_exact",
            samples=1,
        )


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 128 for character in text][:16]

    def decode(self, ids):
        return " ".join(str(item) for item in ids)


def _tiny_engine(*, seed: int, allow_vanilla_fallback: bool = True):
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=8,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    cells = {}
    for action in OperationKind:
        cell = ActionEvidence()
        for _ in range(8):
            cell = cell.append(
                gain=1.0 if action is OperationKind.FALSIFY else -1.0,
                cost=0.1,
            )
        cells[action] = cell
    evidence = build_evidence_snapshot(
        bucket="verified-best|none|short|s:mid|u:mid",
        cells=cells,
    )
    engine = LatentCortexEngine(
        model,
        _Tokenizer(),
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=seed),
            recurrence=RecurrenceConfig(
                min_steps=1,
                max_steps=4,
                convergence_eps=1e-9,
            ),
            branches=BranchConfig(n_branches=1),
            latent_opt=LatentOptConfig(enabled=False),
            decode_max_tokens=4,
            verifier_probe_max_tokens=16,
            allow_vanilla_fallback=allow_vanilla_fallback,
        ),
    )
    return engine, evidence


def _matched_constraint_evaluator(**kwargs):
    budget = kwargs["budget"]

    def evaluate(label, _state, replicate):
        budget.charge(
            8,
            8,
            operation="test_matched_constraint_probe",
            attention_pairs=64,
            output_head_tokens=8,
        )
        budget.charge_verifier(
            "test_matched_constraint_verifier",
            input_bytes=64,
            output_bytes=64,
            host_scalar_ops=64,
        )
        score = 1.0 if label == "negative_direction" else 0.0
        return CounterfactualProbeResult(
            probe_tokens_sha256=hashlib.sha256(f"{label}:{replicate}".encode()).hexdigest(),
            probe_token_count=8,
            observation=_observation(
                score=score,
                lower=score,
                upper=score,
                name=f"{label}:{replicate}",
                basis="deterministic_exact",
                samples=1,
            ),
            layer_apps=64,
        )

    return evaluate


def _matched_virtual_and_constraint_evaluator(**kwargs):
    budget = kwargs["budget"]

    def evaluate(label, _state, replicate):
        budget.charge(
            8,
            8,
            operation="test_matched_virtual_probe",
            attention_pairs=64,
            output_head_tokens=8,
        )
        budget.charge_verifier(
            "test_matched_virtual_verifier",
            input_bytes=64,
            output_bytes=64,
            host_scalar_ops=64,
        )
        score = 1.0 if label in {"guided_quantum", "negative_direction"} else 0.0
        return CounterfactualProbeResult(
            probe_tokens_sha256=hashlib.sha256(f"{label}:{replicate}".encode()).hexdigest(),
            probe_token_count=8,
            observation=_observation(
                score=score,
                lower=score,
                upper=score,
                name=f"virtual:{label}:{replicate}",
                basis="deterministic_exact",
                samples=1,
            ),
            layer_apps=64,
        )

    return evaluate


def test_real_tiny_qwen_applies_virtual_quantum_before_recurrence_and_proves_receipt(
    monkeypatch,
):
    engine, evidence = _tiny_engine(seed=17, allow_vanilla_fallback=False)
    verifier = _BoundedVerifier()
    monkeypatch.setattr(
        engine,
        "_counterfactual_probe_evaluator",
        _matched_virtual_and_constraint_evaluator,
    )
    original_savepoint = BranchEnsemble.savepoint_all
    initialization_states: dict[int, str] = {}

    def capture_initialization(self, *, authority):
        if authority == "episode_initialization":
            initialization_states.update(
                {branch.index: tensor_sha256(branch.z) for branch in self.branches}
            )
        return original_savepoint(self, authority=authority)

    monkeypatch.setattr(BranchEnsemble, "savepoint_all", capture_initialization)
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
        verifier=verifier,
        action_policy_evidence=evidence,
    )

    assert result.ok is True
    receipt = result.receipt.to_dict()
    virtual = receipt["virtual_quanta"]
    target = virtual["branch_index"]
    assert virtual["status"] == "applied"
    assert virtual["reason"] == "guided_quantum_verified"
    assert virtual["guided_beats_controls"] is True
    assert virtual["all_arms_equal_resources"] is True
    assert virtual["all_arms_fully_metered"] is True
    assert virtual["application"]["uses"] == 1
    assert virtual["application"]["post_state_sha256"] == initialization_states[target]
    assert virtual["application"]["post_state_sha256"] != virtual["baseline_state_sha256"]
    assert virtual["erasure"]["all_zero_before_release"] is True
    assert virtual["erasure"]["private_reference_released"] is True
    validated = validate_virtual_quanta_receipt(
        virtual,
        episode_id=receipt["episode_id"],
        objective_sha256=receipt["input_tokens_sha256"],
        n_branches=1,
        expected_config=VirtualQuantaConfig(),
        cognitive_slots=receipt["cognitive_slots"],
        verifier_preflight=receipt["verifier_preflight"],
        information_accounting=receipt["budget"]["information_accounting"],
        resource_accounting=receipt["budget"]["resource_accounting"],
        kv_state_tree=receipt["kv_state_tree"],
        require_external_bindings=True,
    )
    assert validated["receipt_sha256"] == virtual["receipt_sha256"]
    contract_errors = LatentCortexService._receipt_contract_errors(
        receipt,
        {
            "n_slots": 4,
            "n_branches": 1,
            "min_steps": 1,
            "max_steps": 4,
            "verifier_probe_max_tokens": 16,
        },
    )
    assert "virtual_quanta_unproven" not in contract_errors

    forged = copy.deepcopy(virtual)
    forged["authority_scope"] = "cross_episode_durable"
    payload = dict(forged)
    payload.pop("receipt_sha256")
    forged["receipt_sha256"] = canonical_sha256(payload)
    with pytest.raises(ValueError, match="identity"):
        validate_virtual_quanta_receipt(
            forged,
            episode_id=receipt["episode_id"],
            objective_sha256=receipt["input_tokens_sha256"],
            n_branches=1,
            expected_config=VirtualQuantaConfig(),
        )

    policy_forged = copy.deepcopy(virtual)
    policy_forged["verifier_policy_sha256"] = "f" * 64
    policy_payload = dict(policy_forged)
    policy_payload.pop("receipt_sha256")
    policy_forged["receipt_sha256"] = canonical_sha256(policy_payload)
    with pytest.raises(ValueError, match="external source binding"):
        validate_virtual_quanta_receipt(
            policy_forged,
            episode_id=receipt["episode_id"],
            objective_sha256=receipt["input_tokens_sha256"],
            n_branches=1,
            expected_config=VirtualQuantaConfig(),
            cognitive_slots=receipt["cognitive_slots"],
            verifier_preflight=receipt["verifier_preflight"],
            information_accounting=receipt["budget"]["information_accounting"],
            resource_accounting=receipt["budget"]["resource_accounting"],
            kv_state_tree=receipt["kv_state_tree"],
            require_external_bindings=True,
        )


def test_real_tiny_qwen_branch_action_runs_verified_latent_tree_and_service_accepts():
    from core.brain.llm.latent_cortex.latent_tree_search import (
        LatentTreeSearchConfig,
        validate_latent_tree_receipt,
    )

    engine, _ = _tiny_engine(seed=18, allow_vanilla_fallback=False)
    engine.config.branches = BranchConfig(n_branches=2, exchange_interval=1)
    engine.config.virtual_quanta = {"mode": "disabled"}
    engine.config.latent_tree_search = {
        "strategy": "uct",
        "max_nodes": 3,
        "max_depth": 1,
        "branching_factor": 2,
        "min_verifier_margin": 0.0,
    }
    cells = {}
    for action in OperationKind:
        cell = ActionEvidence()
        for _ in range(8):
            cell = cell.append(
                gain=1.0 if action is OperationKind.BRANCH else -1.0,
                cost=0.1,
            )
        cells[action] = cell
    evidence = build_evidence_snapshot(
        bucket="verified-tree|none|short|s:mid|u:mid",
        cells=cells,
    )
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(max_layer_apps=1_000_000, wall_clock_s=30.0),
        verifier=_BoundedVerifier(),
        action_policy_evidence=evidence,
    )

    assert result.ok is True
    assert result.answer_replacement_private == {}
    receipt = result.receipt.to_dict()
    tree = receipt["latent_tree_search"]
    assert tree["status"] == "executed"
    assert tree["transactions"]
    assert any(row["status"] == "committed" for row in tree["transactions"])
    discarded_ordinals = sorted(
        {
            ordinal
            for transaction in tree["transactions"]
            for ordinal in transaction["discarded_recurrent_kv_call_ordinals"]
        }
    )
    committed_ordinals = sorted(
        {
            ordinal
            for transaction in tree["transactions"]
            for ordinal in transaction["committed_recurrent_kv_call_ordinals"]
        }
    )
    assert discarded_ordinals
    assert committed_ordinals
    assert receipt["loop_stability"]["excluded_speculative_kv_call_ordinals"] == discarded_ordinals
    kv_ordinals = {row["ordinal"] for row in receipt["loop_stability"]["kv_bound"]["calls"]}
    assert set(discarded_ordinals) <= kv_ordinals
    validate_latent_tree_receipt(
        tree,
        episode_id=receipt["episode_id"],
        objective_sha256=receipt["input_tokens_sha256"],
        expected_config=LatentTreeSearchConfig.from_value(engine.config.latent_tree_search),
        kv_state_tree=receipt["kv_state_tree"],
        cognitive_action_trace=receipt["cognitive_action_trace"],
        resource_accounting=receipt["budget"]["resource_accounting"],
        loop_stability=receipt["loop_stability"],
        require_external_bindings=True,
    )
    contract_errors = LatentCortexService._receipt_contract_errors(
        receipt,
        {
            "n_slots": 4,
            "n_branches": 2,
            "min_steps": 1,
            "max_steps": 4,
            "verifier_probe_max_tokens": 16,
            "virtual_quanta": {"mode": "disabled"},
            "latent_tree_search": engine.config.latent_tree_search,
        },
    )
    assert "latent_tree_search_unproven" not in contract_errors

    mismatched = copy.deepcopy(receipt)
    mismatched_loop = mismatched["loop_stability"]
    loop_core = mismatched_loop["loop_core"]
    replacement_ordinal = next(
        row["ordinal"]
        for row in mismatched_loop["kv_bound"]["calls"]
        if row["persist"] is False
        and row["start"] == loop_core["prelude_end"]
        and row["end"] == loop_core["coda_start"]
        and row["ordinal"] not in discarded_ordinals
    )
    mismatched_loop["excluded_speculative_kv_call_ordinals"] = sorted(
        [replacement_ordinal, *discarded_ordinals[1:]]
    )
    mismatched_loop["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in mismatched_loop.items() if key != "receipt_sha256"}
    )
    from core.brain.llm.latent_cortex.loop_stability import (
        validate_loop_stability_receipt,
    )

    validate_loop_stability_receipt(
        mismatched_loop,
        recurrent_grounding=mismatched["recurrent_grounding"],
        expected_loop_core=loop_core,
    )
    mismatched_errors = LatentCortexService._receipt_contract_errors(
        mismatched,
        {
            "n_slots": 4,
            "n_branches": 2,
            "min_steps": 1,
            "max_steps": 4,
            "verifier_probe_max_tokens": 16,
            "virtual_quanta": {"mode": "disabled"},
            "latent_tree_search": engine.config.latent_tree_search,
        },
    )
    assert "latent_tree_search_unproven" in mismatched_errors


def test_real_tiny_qwen_engine_meters_and_receipts_bounded_verifier_authority():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=8,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    cells = {}
    for action in OperationKind:
        cell = ActionEvidence()
        for _ in range(8):
            cell = cell.append(
                gain=1.0 if action is OperationKind.FALSIFY else -1.0,
                cost=0.1,
            )
        cells[action] = cell
    evidence = build_evidence_snapshot(
        bucket="verified-best|none|short|s:mid|u:mid",
        cells=cells,
    )
    verifier = _BoundedVerifier()
    engine = LatentCortexEngine(
        model,
        _Tokenizer(),
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=19),
            recurrence=RecurrenceConfig(
                min_steps=1,
                max_steps=4,
                convergence_eps=1e-9,
            ),
            branches=BranchConfig(n_branches=1),
            latent_opt=LatentOptConfig(enabled=False),
            decode_max_tokens=4,
        ),
    )
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
        verifier=verifier,
        action_policy_evidence=evidence,
    )
    assert result.ok is True
    verified = result.receipt.verified_best_state
    assert verifier.bound_calls >= 1
    assert verified["authoritative_promotions"] >= 1
    winner = verified["branches"][result.receipt.selected_branch]
    assert result.receipt.best_step == winner["final_best_step"]
    assert any(
        row["verification"]["observation"].get("authoritative") is True
        for row in result.receipt.cognitive_action_trace
    )


class _FailureThenRecoveryVerifier(_BoundedVerifier):
    def observe_with_bounds(self, _text: str):
        self.bound_calls += 1
        score = 0.0 if self.bound_calls == 1 else 1.0
        return _observation(
            score=score,
            lower=score,
            upper=score,
            name=f"failure-recovery-{self.bound_calls}",
            basis="deterministic_exact",
            samples=1,
        )


def test_real_tiny_qwen_engine_applies_verified_constraint_on_live_recurrence(
    monkeypatch,
):
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=8,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    cells = {}
    for action in OperationKind:
        cell = ActionEvidence()
        for _ in range(8):
            cell = cell.append(
                gain=1.0 if action is OperationKind.FALSIFY else -1.0,
                cost=0.1,
            )
        cells[action] = cell
    evidence = build_evidence_snapshot(
        bucket="verified-best|none|short|s:mid|u:mid",
        cells=cells,
    )
    verifier = _FailureThenRecoveryVerifier()
    config = CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=23),
        recurrence=RecurrenceConfig(
            min_steps=1,
            max_steps=4,
            convergence_eps=1e-9,
        ),
        branches=BranchConfig(n_branches=1),
        latent_opt=LatentOptConfig(enabled=False),
        decode_max_tokens=4,
    )
    engine = LatentCortexEngine(
        model,
        _Tokenizer(),
        config=config,
    )

    monkeypatch.setattr(
        engine,
        "_counterfactual_probe_evaluator",
        _matched_constraint_evaluator,
    )
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
        verifier=verifier,
        action_policy_evidence=evidence,
    )
    assert result.ok is True
    transient = result.receipt.transient_negative_constraints
    assert transient["aggregates"]["admitted_count"] == 1
    assert transient["aggregates"]["application_count"] == 1
    assert transient["aggregates"]["verified_reduction_count"] == 1
    assert transient["constraints"][0]["status"] == "consumed"
    application = transient["applications"][0]
    assert application["recurrence_committed"] is True
    assert application["protected_positions_unchanged"] is True
    assert application["kv_boundary_before_sha256"] == application["kv_boundary_after_sha256"]
    receipt = result.receipt.to_dict()
    executors = tuple(OperationKind(item) for item in receipt["value_of_computation"]["executors"])
    for row in receipt["cognitive_action_trace"]:
        validate_action_trace_row(
            row,
            evidence_snapshot=evidence,
            executors=executors,
        )
    validated = validate_transient_constraint_receipt(
        transient,
        episode_id=receipt["episode_id"],
        objective_sha256=receipt["input_tokens_sha256"],
        n_branches=1,
        protected_positions={0: ()},
        expected_config=TransientConstraintConfig(),
        cognitive_action_trace=receipt["cognitive_action_trace"],
        verifier_preflight=receipt["verifier_preflight"],
        information_accounting=receipt["budget"]["information_accounting"],
        resource_accounting=receipt["budget"]["resource_accounting"],
        kv_state_tree=receipt["kv_state_tree"],
        verified_best_state=receipt["verified_best_state"],
        loop_stability=receipt["loop_stability"],
        require_verified_best_binding=True,
        require_external_bindings=True,
    )
    assert validated["receipt_sha256"] == transient["receipt_sha256"]
    contract_config = {
        "n_slots": 4,
        "n_branches": 1,
        "min_steps": 1,
        "max_steps": 4,
        "verifier_probe_max_tokens": 48,
    }
    contract_errors = LatentCortexService._receipt_contract_errors(
        receipt,
        contract_config,
    )
    assert "transient_negative_constraints_unproven" not in contract_errors

    forged_receipt = copy.deepcopy(receipt)
    forged_verified = forged_receipt["verified_best_state"]
    failed_decision = next(
        decision
        for branch in forged_verified["branches"]
        for decision in branch["decisions"]
        if decision["decision"] == "reject_verified_failure"
    )
    failed_decision["resulting_state_sha256"] = "f" * 64
    forged_verified["receipt_sha256"] = hashlib.sha256(b"invalid-on-purpose").hexdigest()
    forged_errors = LatentCortexService._receipt_contract_errors(
        forged_receipt,
        contract_config,
    )
    assert "transient_negative_constraints_unproven" in forged_errors


def test_real_counterfactual_evaluator_is_fixed_compute_metered_and_restoring():
    engine, _ = _tiny_engine(seed=31)
    budget = ComputeBudget(max_layer_apps=500_000, wall_clock_s=30.0)
    budget.bind_model(engine.model)
    cache = engine._fresh_cache()
    embeddings, _ = engine._prefill([5, 9, 17, 3, 42, 7], cache, budget)
    runner = WindowRunner(engine.model.model, budget)
    ensemble = BranchEnsemble.seed(
        embeddings,
        engine.config.workspace,
        engine.config.branches,
        engine.config.recurrence,
        runner,
        cache,
        engine.prelude_end,
    )
    branch = ensemble.branches[0]
    baseline_state_sha256 = tensor_sha256(branch.z)
    baseline_offsets = tuple(layer.offset for layer in cache)

    class FixedExactVerifier:
        def __init__(self):
            self.input_bytes: list[int] = []

        def observe_with_bounds(self, text: str):
            self.input_bytes.append(len(text.encode("utf-8")))
            return _observation(
                score=0.5,
                lower=0.5,
                upper=0.5,
                name="fixed-counterfactual-observation",
                basis="deterministic_exact",
                samples=1,
            )

    verifier = FixedExactVerifier()
    evaluator = engine._counterfactual_probe_evaluator(
        branch=branch,
        cache=cache,
        runner=runner,
        budget=budget,
        bridge_tokens=[],
        verifier=engine._meter_verifier(verifier, budget),
    )
    assert evaluator is not None

    resource_deltas = []
    for replicate, scale in enumerate((0.99, 1.0, 1.01)):
        before = budget.resource_ledger.totals()
        result = evaluator("mechanics_probe", branch.z * scale, replicate)
        after = budget.resource_ledger.totals()
        delta = {name: after[name] - before[name] for name in before}
        resource_deltas.append(delta)

        assert result.probe_token_count == engine.config.verifier_probe_max_tokens
        assert result.layer_apps == delta["transformer_layer_apps"]
        assert delta["output_head_tokens"] == result.probe_token_count
        assert delta["attention_query_key_pairs"] > 0
        assert delta["verifier_calls"] == 1
        assert delta["verifier_input_bytes"] == 1024
        assert delta["verifier_output_bytes"] > 0
        assert tensor_sha256(branch.z) == baseline_state_sha256
        assert tuple(layer.offset for layer in cache) == baseline_offsets

    assert resource_deltas[0] == resource_deltas[1] == resource_deltas[2]
    assert verifier.input_bytes == [1024, 1024, 1024]


def test_exception_after_constraint_admission_zeroizes_private_direction(
    monkeypatch,
):
    engine, evidence = _tiny_engine(seed=37, allow_vanilla_fallback=False)
    verifier = _FailureThenRecoveryVerifier()
    monkeypatch.setattr(
        engine,
        "_counterfactual_probe_evaluator",
        _matched_constraint_evaluator,
    )
    original_consider = TransientConstraintLedger.consider_verified_failure
    observed = {}

    def admit_then_raise(self, *args, **kwargs):
        attempt = original_consider(self, *args, **kwargs)
        if attempt["status"] == "admitted":
            observed["ledger"] = self
            observed["private_before_exception"] = self.private_direction_count
            raise RuntimeError("forced failure after transient admission")
        return attempt

    monkeypatch.setattr(
        TransientConstraintLedger,
        "consider_verified_failure",
        admit_then_raise,
    )
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
        verifier=verifier,
        action_policy_evidence=evidence,
    )

    assert result.ok is False
    assert observed["private_before_exception"] == 1
    ledger = observed["ledger"]
    assert ledger.private_direction_count == 0
    diagnostic = ledger.finalize(final_action_step=0)
    assert diagnostic["constraints"][0]["status"] == "aborted_episode_failure"
    assert diagnostic["erasures"][0]["reason"] == "episode_aborted"
