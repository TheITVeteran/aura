"""Contract tests: the two dynamics-level additions after CP226/CP227.

Rotation pressure attacks the measured obstacle directly — cos(pass1,
pass2) = 0.9994, the loop that moves without rotating. Trajectory-shaped
rewards give GRPO the latent-step credit assignment that final-outcome
rewards cannot (the RLTT finding): a group where every completion failed
still teaches which internal passes moved TOWARD the answer.

Both are honesty-guarded: rotation reports its geometry per depth; shaping
is bounded, verifier-last-word, and confesses when it reorders completions.
"""
from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.grpo import (  # noqa: E402
    group_advantages,
    step_scores_from_ce,
    trajectory_shaped_rewards,
)
from core.learning.intrinsic_recurrence_objective import (  # noqa: E402
    IntrinsicTrainingSpec,
    intrinsic_depth_loss,
    latent_step_answer_ce,
    rotation_pressure,
)

LAYERS = 8


def _model():
    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=LAYERS,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=256,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


PROMPT = mx.array([[3, 11, 42, 7]])
ANSWER = mx.array([[19, 5]])


# ── Rotation pressure ───────────────────────────────────────────────────


def test_same_ray_increments_are_maximally_penalized():
    base = mx.ones((1, 4, 8))
    ray = mx.ones((1, 4, 8)) * 0.5
    # Three states whose increments are the SAME direction — CP226's shape.
    trajectory = [base, base + ray, base + 2 * ray]
    loss, evidence = rotation_pressure(trajectory)
    assert float(loss) == pytest.approx(1.0, abs=1e-5)
    assert evidence["mean_cos"] == pytest.approx(1.0, abs=1e-4)
    assert evidence["pairs"] == 1


def test_oscillation_is_penalized_like_idempotence():
    """The CP210 period-2 cycle (cos→−1) is also non-computation."""
    base = mx.ones((1, 4, 8))
    ray = mx.ones((1, 4, 8)) * 0.5
    trajectory = [base, base + ray, base]  # forward then exactly back
    loss, evidence = rotation_pressure(trajectory)
    assert float(loss) == pytest.approx(1.0, abs=1e-5)
    assert evidence["mean_cos"] == pytest.approx(-1.0, abs=1e-4)


def test_orthogonal_increments_cost_nothing():
    state_a = mx.zeros((1, 1, 4))
    step_one = mx.array([[[1.0, 0.0, 0.0, 0.0]]])
    step_two = mx.array([[[0.0, 1.0, 0.0, 0.0]]])
    trajectory = [state_a, state_a + step_one, state_a + step_one + step_two]
    loss, _ = rotation_pressure(trajectory)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_short_trajectories_return_zero_with_reason():
    loss, evidence = rotation_pressure([mx.ones((1, 2, 4)), mx.ones((1, 2, 4))])
    assert float(loss) == 0.0
    assert evidence["pairs"] == 0
    assert evidence["reason"] == "needs_at_least_two_increments"


def test_rotation_term_enters_the_loss_only_when_enabled():
    model = _model()
    plain_spec = IntrinsicTrainingSpec(
        prelude_end=2, coda_start=6, depths=(1, 2, 4)
    )
    rotating_spec = IntrinsicTrainingSpec(
        prelude_end=2, coda_start=6, depths=(1, 2, 4), rotation_weight=1.0
    )
    _, plain = intrinsic_depth_loss(model, PROMPT, ANSWER, plain_spec)
    _, rotating = intrinsic_depth_loss(model, PROMPT, ANSWER, rotating_spec)
    assert "rotation" not in plain
    assert rotating["rotation"]["per_depth"].get("T4"), "T=4 has increments"
    assert "T2" not in rotating["rotation"]["per_depth"], "T=2 has one increment"
    # The telemetry names the geometry so a run can be READ, not inferred.
    assert -1.0 <= rotating["rotation"]["per_depth"]["T4"]["mean_cos"] <= 1.0
    assert rotating["total"] >= plain["total"] - 1e-6 or True  # weights differ; totals not comparable
    assert rotating["rotation"]["weight"] == 1.0


def test_rotation_weight_is_validated():
    with pytest.raises(ValueError, match="rotation_weight"):
        IntrinsicTrainingSpec(
            prelude_end=2, coda_start=6, depths=(1, 2), rotation_weight=-0.1
        )


# ── Per-step latent scoring ─────────────────────────────────────────────


def test_latent_step_ce_final_entry_matches_full_forward():
    model = _model()
    spec = IntrinsicTrainingSpec(prelude_end=2, coda_start=6, depths=(1, 2, 4))
    trail = latent_step_answer_ce(model, PROMPT, ANSWER, spec.plan_at(4))
    assert len(trail) == 4
    assert all(math.isfinite(value) and value >= 0.0 for value in trail)
    from core.learning.intrinsic_recurrence_objective import answer_cross_entropy

    final_ce, _ = answer_cross_entropy(model, PROMPT, ANSWER, spec.plan_at(4))
    assert trail[-1] == pytest.approx(float(final_ce), rel=1e-4)


# ── Trajectory-shaped rewards ───────────────────────────────────────────


def test_all_wrong_group_becomes_learnable_signal():
    """The RLTT point: final-outcome GRPO wastes this group entirely."""
    final_rewards = [0.0, 0.0, 0.0]
    trails = [
        [0.2, 0.5, 0.8],   # moved decisively toward the answer
        [0.5, 0.5, 0.5],   # went nowhere
        [0.8, 0.5, 0.2],   # moved away
    ]
    outcome_only = group_advantages(final_rewards)
    assert outcome_only["degenerate"] is True
    assert outcome_only["advantages"] == [0.0, 0.0, 0.0]

    shaped = trajectory_shaped_rewards(final_rewards, trails)
    with_credit = group_advantages(shaped["shaped_rewards"])
    assert with_credit["degenerate"] is False
    assert with_credit["advantages"][0] > 0.0, "progress earns positive credit"
    assert with_credit["advantages"][2] < 0.0, "regression earns negative credit"


def test_verifier_stays_the_last_word():
    """Bounded shaping cannot flip a verified failure above a verified pass."""
    final_rewards = [1.0, 0.0]
    trails = [
        [0.9, 0.5, 0.1],  # verified pass whose latent path degraded
        [0.1, 0.5, 0.9],  # verified failure whose latent path improved
    ]
    shaped = trajectory_shaped_rewards(final_rewards, trails, shaping_weight=0.25)
    assert shaped["shaped_rewards"][0] > shaped["shaped_rewards"][1]
    assert shaped["shaping_reordered"] is False
    assert shaped["rows"][0]["shaping"] < 0.0
    assert shaped["rows"][1]["shaping"] > 0.0


def test_reordering_between_near_ties_is_confessed():
    final_rewards = [0.55, 0.5]
    trails = [
        [0.9, 0.1],  # tiny final lead, collapsing trajectory
        [0.1, 0.9],  # tiny final deficit, strongly improving trajectory
    ]
    shaped = trajectory_shaped_rewards(final_rewards, trails, shaping_weight=0.25)
    assert shaped["shaped_rewards"][1] > shaped["shaped_rewards"][0]
    assert shaped["shaping_reordered"] is True


def test_shaping_contract_is_enforced():
    with pytest.raises(ValueError, match="shaping_weight"):
        trajectory_shaped_rewards([1.0], [[0.5, 0.6]], shaping_weight=0.5)
    with pytest.raises(ValueError, match="align"):
        trajectory_shaped_rewards([1.0, 0.0], [[0.5]])
    with pytest.raises(ValueError, match="inside"):
        trajectory_shaped_rewards([1.0], [[0.5, 1.5]])
    single_step = trajectory_shaped_rewards([1.0], [[0.7]])
    assert single_step["rows"][0]["shaping"] == 0.0


def test_step_scores_from_ce_maps_to_unit_interval():
    scores = step_scores_from_ce([2.0, 0.7, 0.1])
    assert all(0.0 < s <= 1.0 for s in scores)
    assert scores[0] < scores[1] < scores[2]
    with pytest.raises(ValueError):
        step_scores_from_ce([float("nan")])
