"""Confidence-bound best-state authority and branch-local rollback."""

from __future__ import annotations

import copy
import hashlib

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.branches import BranchEnsemble, BranchState  # noqa: E402
from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.epistemic_state import OperationKind  # noqa: E402
from core.brain.llm.latent_cortex.recurrence import HaltingController  # noqa: E402
from core.brain.llm.latent_cortex.task_verifiers import (  # noqa: E402
    check_arithmetic_claims,
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
)
from core.brain.llm.latent_cortex.verified_best import (  # noqa: E402
    VERIFIER_OBSERVATION_SCHEMA,
    VerifierObservation,
    build_verified_best_receipt,
    tensor_sha256,
    validate_verified_best_receipt,
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


def _action(step: int, branch: int, observation, decision: str, restored: bool):
    return {
        "decision": {"step_index": step},
        "verification": {
            "target_branch": branch,
            "observation": observation.to_dict(),
            "decision": decision,
            "restored": restored,
        },
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
