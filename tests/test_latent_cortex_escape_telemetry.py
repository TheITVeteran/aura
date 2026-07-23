"""Contract tests: attractor escape ladder + latent safety telemetry.

Escape contract:
- divergence gets a second life through ordered rungs, never raw noise first;
- every rung starts a probation that must BEAT the pre-escape best or the
  branch reverts and halts honestly with escape_failed_*;
- legitimate halts (converged / max_steps / budget) are never escaped;
- attempts are bounded and fully receipted.

Telemetry contract:
- per-slot trajectories, drift, exchange divergence, selection disagreement,
  and anomalies are recorded bounded and receipt-borne;
- detectors name themselves; nothing claims semantic meaning.
"""
from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.escape import (  # noqa: E402
    ESCAPE_RUNGS,
    BranchEscapeLadder,
    EscapeConfig,
)
from core.brain.llm.latent_cortex.recurrence import HaltingController  # noqa: E402
from core.brain.llm.latent_cortex.telemetry import (  # noqa: E402
    LATENT_TELEMETRY_SCHEMA,
    MAX_RECORDED_STEPS,
    LatentTelemetry,
)
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.brain.llm.latent_cortex.workspace import LatentWorkspace  # noqa: E402

N_LAYERS = 8
PROMPT_TOKENS = [5, 9, 17, 3, 42, 7, 11, 23, 2, 88]


@pytest.fixture(scope="module")
def tiny_model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=N_LAYERS,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


class _FakeBranch:
    """A minimal BranchState stand-in the ladder can drive."""

    def __init__(self, recurrence: RecurrenceConfig | None = None):
        self.index = 0
        self.role = "constructive_solution"
        self.steps = 0
        self.z = mx.ones((1, 4, 8))
        self.anchor = mx.ones((1, 4, 8))
        self.halting = HaltingController(
            config=recurrence or RecurrenceConfig(max_steps=16, min_steps=1),
            baseline_rms=1.0,
            best_state=self.z,
        )
        self.halting.best_score = -0.5
        self.halting.best_step = 0

        class _WS:
            def __init__(self, owner):
                self.owner = owner

            def update(self, z):
                self.owner.z = z

            def restore_context_evidence(self, z):
                return z

        self.workspace = _WS(self)


# ── Ladder mechanics ────────────────────────────────────────────────────


def test_divergence_gets_ordered_rungs_then_honest_halt():
    branch = _FakeBranch()
    ladder = BranchEscapeLadder(
        EscapeConfig(max_attempts=4, probation_steps=1), branch.index
    )
    rungs_seen = []
    for _ in range(len(ESCAPE_RUNGS)):
        action = ladder.on_divergence(branch, "diverged_norm")
        if action == "escaped":
            rungs_seen.append(ladder.attempts[-1].rung)
            # Probation fails: next divergence reverts and halts.
            action = ladder.on_divergence(branch, "diverged_norm")
            assert action == "halt:escape_failed_diverged_norm"
            assert ladder.attempts[-1].outcome == "failed"
        else:
            break
    assert rungs_seen == list(ESCAPE_RUNGS[: len(rungs_seen)])
    assert rungs_seen, "at least one rung must be attempted"
    # Ladder exhausted ⇒ divergence halts plainly.
    final = ladder.on_divergence(branch, "diverged_norm")
    assert final == "halt:diverged_norm"


def test_probation_success_retains_the_escape():
    branch = _FakeBranch()
    ladder = BranchEscapeLadder(
        EscapeConfig(max_attempts=2, probation_steps=3), branch.index
    )
    assert ladder.on_divergence(branch, "diverged_norm") == "escaped"
    # The branch improves past its pre-escape best.
    branch.steps += 1
    branch.halting.best_score = 0.5
    assert ladder.on_step(branch) == "retained"
    assert ladder.attempts[0].outcome == "retained"


def test_probation_deadline_reverts_to_pre_escape_best():
    branch = _FakeBranch()
    pre_best = branch.halting.best_state
    ladder = BranchEscapeLadder(
        EscapeConfig(max_attempts=1, probation_steps=1), branch.index
    )
    assert ladder.on_divergence(branch, "diverged_nonfinite") == "escaped"
    branch.steps += 2  # deadline passed without improvement
    action = ladder.on_step(branch)
    assert action.startswith("halt:escape_failed_")
    assert ladder.attempts[0].outcome == "failed"
    assert bool(mx.all(branch.z == pre_best))


def test_stall_detection_triggers_escape_with_revert_first():
    recurrence = RecurrenceConfig(max_steps=32, min_steps=1, convergence_eps=0.001)
    branch = _FakeBranch(recurrence)
    ladder = BranchEscapeLadder(
        EscapeConfig(stall_patience=2, max_attempts=1), branch.index
    )
    # Best step IS the latest step ⇒ not stalled, no escape.
    branch.steps = 3
    branch.halting.best_step = 2
    branch.halting.residual_trail.extend([0.5, 0.5, 0.5])
    assert ladder.on_step(branch) == ""
    # Best far behind + residuals above convergence ⇒ stalled. A stall means
    # we are NOT at the best state, so reverting to it is the honest rung 1.
    branch.steps = 6
    branch.halting.best_step = 2
    branch.halting.residual_trail.extend([0.5, 0.5, 0.5])
    action = ladder.on_step(branch)
    assert action == "escaped"
    assert ladder.attempts[0].rung == "revert_best"
    assert ladder.attempts[0].trigger == "stalled"


def test_matched_perturbation_is_small_and_matched():
    branch = _FakeBranch()
    config = EscapeConfig(perturbation_scale=0.02, max_attempts=4)
    ladder = BranchEscapeLadder(config, branch.index)
    before = branch.halting.best_state
    ladder._apply_rung(branch, "matched_perturbation")
    delta = branch.z - before
    delta_rms = float(mx.mean(mx.sqrt(mx.mean(mx.square(delta), axis=-1))))
    base_rms = float(mx.mean(mx.sqrt(mx.mean(mx.square(before), axis=-1))))
    assert delta_rms == pytest.approx(0.02 * base_rms, rel=0.05)


def test_escape_perturbation_preserves_sealed_evidence_rows():
    embeddings = mx.random.normal((1, 6, 8))
    evidence = mx.random.normal((1, 1, 8))
    workspace = LatentWorkspace.from_prompt_embeddings(
        embeddings,
        WorkspaceConfig(n_slots=6, seed=11),
        context_seeds=[("reference", evidence)],
    )
    workspace.seal_context_evidence()
    branch = _FakeBranch()
    branch.workspace = workspace
    branch.z = workspace.z
    branch.anchor = workspace.seed_z
    branch.halting.best_state = workspace.z
    before = workspace.z

    BranchEscapeLadder(
        EscapeConfig(perturbation_scale=0.02), branch.index
    )._apply_rung(branch, "matched_perturbation")

    assert bool(mx.array_equal(branch.z[:, 1:2, :], before[:, 1:2, :]))
    assert not bool(mx.array_equal(branch.z[:, 2:, :], before[:, 2:, :]))


def test_role_shift_changes_the_branch_role():
    branch = _FakeBranch()
    ladder = BranchEscapeLadder(EscapeConfig(), branch.index)
    original_role = branch.role
    ladder._apply_rung(branch, "role_shift")
    assert branch.role != original_role


def test_unresolved_probation_is_finalized_honestly():
    branch = _FakeBranch()
    ladder = BranchEscapeLadder(EscapeConfig(probation_steps=8), branch.index)
    assert ladder.on_divergence(branch, "diverged_norm") == "escaped"
    ladder.finalize()
    assert ladder.attempts[0].outcome == "unresolved"
    receipt = ladder.to_receipt()
    assert receipt["attempts"][0]["outcome"] == "unresolved"
    assert receipt["on_probation"] is False


# ── Telemetry mechanics ─────────────────────────────────────────────────


def test_telemetry_records_slots_drift_and_caps():
    telemetry = LatentTelemetry()
    z = mx.ones((1, 4, 8))
    anchor = mx.ones((1, 4, 8))
    for _ in range(MAX_RECORDED_STEPS + 10):
        telemetry.record_step(0, z, anchor, residual=0.1)
    receipt = telemetry.to_receipt()
    assert receipt["schema"] == LATENT_TELEMETRY_SCHEMA
    assert len(receipt["slot_rms_trails"]["0"]) == MAX_RECORDED_STEPS
    assert len(receipt["slot_rms_trails"]["0"][0]) == 4
    assert receipt["drift_trails"]["0"][0] == pytest.approx(0.0, abs=1e-6)
    assert receipt["recorded_steps"]["0"] == MAX_RECORDED_STEPS + 10


def test_telemetry_flags_residual_spikes():
    telemetry = LatentTelemetry()
    z = mx.ones((1, 4, 8))
    for residual in (0.1, 0.1, 0.1, 0.1):
        telemetry.record_step(0, z, z, residual=residual)
    telemetry.record_step(0, z, z, residual=5.0)
    spikes = [a for a in telemetry.anomalies if a["kind"] == "residual_spike"]
    assert len(spikes) == 1
    assert spikes[0]["value"] == pytest.approx(5.0)


def test_telemetry_flags_dormant_and_dominant_slots():
    telemetry = LatentTelemetry()
    z = mx.ones((1, 4, 8))
    scale = mx.array([1.0, 1.0, 1e-6, 100.0]).reshape(1, 4, 1)
    telemetry.record_step(0, z * scale, z, residual=0.1)
    kinds = {a["kind"] for a in telemetry.anomalies}
    assert "slot_dormant" in kinds
    assert "slot_dominant" in kinds


def test_telemetry_records_exchange_divergence_and_selection():
    telemetry = LatentTelemetry()
    a = mx.ones((1, 1, 8))
    b = -mx.ones((1, 1, 8))
    telemetry.record_exchange([a, b])
    assert telemetry.exchange_snapshots[0]["min_cos"] == pytest.approx(-1.0, abs=1e-4)
    telemetry.record_selection([0.9, 0.1], selected=0)
    assert telemetry.selection["disagreement_spread"] == pytest.approx(0.8)
    telemetry.record_fast_weights({"decision": "erased", "max_drop": 3.2, "rescales": 2})
    receipt = telemetry.to_receipt()
    assert receipt["fast_weight_functional_delta"]["decision"] == "erased"


def test_disabled_telemetry_records_nothing():
    telemetry = LatentTelemetry(enabled=False)
    telemetry.record_step(0, mx.ones((1, 2, 4)), mx.ones((1, 2, 4)), residual=9.0)
    telemetry.record_selection([1.0], selected=0)
    assert telemetry.to_receipt() == {}


# ── Engine integration ──────────────────────────────────────────────────


def _config(**overrides) -> CortexConfig:
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=7),
        recurrence=overrides.pop(
            "recurrence", RecurrenceConfig(max_steps=4, min_steps=1)
        ),
        branches=overrides.pop("branches", BranchConfig(n_branches=2)),
        latent_opt=LatentOptConfig(enabled=False),
        decode_max_tokens=4,
        **overrides,
    )


def test_episode_ships_telemetry_receipt(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config())
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    telemetry = result.receipt.latent_telemetry
    assert telemetry["schema"] == LATENT_TELEMETRY_SCHEMA
    assert telemetry["slot_rms_trails"], "per-slot trajectories must be recorded"
    assert telemetry["selection"]["scores"]
    assert "latent_telemetry" in result.receipt.to_dict()


def test_telemetry_can_be_disabled_per_episode(tiny_model):
    engine = LatentCortexEngine(tiny_model, config=_config(telemetry_enabled=False))
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    assert result.receipt.latent_telemetry == {}


def test_diverging_episode_attempts_escape_before_halting(tiny_model, monkeypatch):
    """Force divergence and verify the ladder runs inside a real episode."""
    import core.brain.llm.latent_cortex.branches as branches_mod

    original = branches_mod.recurrence_step
    calls = {"count": 0}

    def exploding_step(z, runner, cache, start, end, config, step, **kwargs):
        result = original(z, runner, cache, start, end, config, step, **kwargs)
        calls["count"] += 1
        if calls["count"] >= 2:
            return result * 1e6  # blow past the divergence ratio
        return result

    monkeypatch.setattr(branches_mod, "recurrence_step", exploding_step)
    engine = LatentCortexEngine(
        tiny_model,
        config=_config(
            recurrence=RecurrenceConfig(
                max_steps=8, min_steps=1, divergence_ratio=2.0
            ),
            branches=BranchConfig(n_branches=1),
            escape={"max_attempts": 2, "probation_steps": 1},
        ),
    )
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    receipt = result.receipt
    assert receipt.escape, "the ladder must have been consulted"
    branch_receipt = receipt.escape["0"]
    assert branch_receipt["attempts"]
    assert any(
        flag in receipt.honest_flags
        for flag in ("attractor_escape_retained", "attractor_escape_failed")
    ), receipt.honest_flags
    # The answer still decodes and the invariant still holds.
    assert receipt.params_unchanged is True
    anomalies = receipt.latent_telemetry.get("anomalies", [])
    assert isinstance(anomalies, list)


def test_escape_can_be_disabled(tiny_model):
    engine = LatentCortexEngine(
        tiny_model, config=_config(escape={"enabled": False})
    )
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    assert result.receipt.escape == {}


def test_config_validation_bounds_escape_settings():
    bad = _config(escape={"stall_patience": 0})
    assert any("stall_patience" in problem for problem in bad.validate())
    unknown = _config(escape={"warp_factor": 9})
    assert any("unknown keys" in problem for problem in unknown.validate())
    fine = _config(escape={"max_attempts": 2, "perturbation_scale": 0.05})
    assert fine.validate() == []


def test_escape_config_rejects_nonsense():
    with pytest.raises(TypeError):
        EscapeConfig(unknown_field=1)  # type: ignore[call-arg]
    ladder = BranchEscapeLadder(EscapeConfig(max_attempts=0), 0)
    assert ladder.can_attempt() is False
    branch = _FakeBranch()
    assert ladder.on_divergence(branch, "diverged_norm") == "halt:diverged_norm"
    assert math.isfinite(branch.halting.best_score)
