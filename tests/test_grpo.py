"""Verifier-driven RL (CP229).

The ways RL manufactures fake progress are specific and each has a test
here: degenerate groups that produce a smooth loss curve over zero signal,
reward that drifts from the grader to model confidence, and unconstrained
optimization that trades general ability for the one graded metric.
"""
from __future__ import annotations

import pytest

from core.learning.grpo import (
    GRPOConfig,
    GRPOTelemetry,
    group_advantages,
    reward_from_verdict,
)

# ── Group-relative advantage ────────────────────────────────────────────


def test_advantages_are_centred_on_the_group():
    report = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert report["degenerate"] is False
    assert sum(report["advantages"]) == pytest.approx(0.0, abs=1e-6)
    # Correct completions are reinforced, wrong ones suppressed.
    assert report["advantages"][0] > 0 and report["advantages"][1] < 0


def test_all_correct_group_yields_no_signal_and_says_so():
    """This is the failure that hides: zero advantages average into a loss
    curve that looks like smooth convergence over nothing."""
    report = group_advantages([1.0, 1.0, 1.0, 1.0])
    assert report["degenerate"] is True
    assert report["all_correct"] is True
    assert report["all_wrong"] is False
    assert report["advantages"] == [0.0] * 4


def test_all_wrong_group_is_distinguished_from_all_correct():
    """Identical in the loss curve; opposite fixes (easier vs harder tasks)."""
    report = group_advantages([0.0, 0.0, 0.0])
    assert report["degenerate"] is True
    assert report["all_wrong"] is True
    assert report["all_correct"] is False


def test_uniform_partial_reward_is_not_mislabeled_as_success_or_failure():
    report = group_advantages([0.05, 0.05, 0.05, 0.05])
    assert report["degenerate"] is True
    assert report["all_wrong"] is False
    assert report["all_correct"] is False
    assert report["uniform_partial"] is True

    telemetry = GRPOTelemetry()
    telemetry.observe(report)
    verdict = telemetry.verdict(GRPOConfig(max_degenerate_fraction=0.5))
    assert verdict["learning_signal"] is False
    assert verdict["uniform_partial_groups"] == 1
    assert "uniform_partial_reward" in verdict["diagnosis"]


def test_a_lucky_outlier_cannot_dominate_the_group():
    report = group_advantages([0.0, 0.0, 0.0, 0.0, 0.0, 100.0], clip=2.0)
    assert max(report["advantages"]) <= 2.0
    assert min(report["advantages"]) >= -2.0


def test_degenerate_rewards_and_bad_graders_are_refused():
    with pytest.raises(ValueError, match="at least two"):
        group_advantages([1.0])
    with pytest.raises(ValueError, match="finite"):
        group_advantages([1.0, float("nan")])
    with pytest.raises(ValueError, match="finite"):
        group_advantages([1.0, float("inf")])


# ── Reward comes from the grader, and format credit stays small ─────────


def test_reward_is_correctness():
    assert reward_from_verdict({"correct": True}) == 1.0
    assert reward_from_verdict({"correct": False, "parsed": None}) == 0.0


def test_format_credit_is_bounded_because_format_is_easier_to_learn():
    """A large format term produces a model that is beautifully formatted
    and wrong."""
    partial = reward_from_verdict(
        {"correct": False, "parsed": 41}, format_credit=0.1
    )
    assert partial == 0.1
    assert partial < reward_from_verdict({"correct": True})
    with pytest.raises(ValueError, match="format_credit"):
        reward_from_verdict({"correct": False, "parsed": 1}, format_credit=0.5)


def test_unanswered_earns_nothing_even_with_format_credit():
    assert reward_from_verdict(
        {"correct": False, "parsed": None}, format_credit=0.2
    ) == 0.0


# ── Configuration guards ────────────────────────────────────────────────


def test_group_of_one_is_refused():
    """A group of one has no baseline, which is the entire mechanism."""
    with pytest.raises(ValueError, match="at least 2"):
        GRPOConfig(group_size=1)
    with pytest.raises(ValueError, match="advantage_clip"):
        GRPOConfig(advantage_clip=0.0)
    with pytest.raises(ValueError, match="max_degenerate_fraction"):
        GRPOConfig(max_degenerate_fraction=1.1)


# ── Telemetry decides whether the run is learning at all ────────────────


def test_telemetry_flags_a_run_with_no_signal():
    config = GRPOConfig(group_size=4, max_degenerate_fraction=0.5)
    telemetry = GRPOTelemetry()
    for _ in range(9):
        telemetry.observe(group_advantages([0.0, 0.0, 0.0, 0.0]))
    telemetry.observe(group_advantages([1.0, 0.0, 1.0, 0.0]))
    verdict = telemetry.verdict(config)
    assert verdict["learning_signal"] is False
    assert verdict["degenerate_fraction"] == 0.9
    assert "too_hard" in verdict["diagnosis"]


def test_telemetry_distinguishes_too_easy_from_too_hard():
    config = GRPOConfig(group_size=4, max_degenerate_fraction=0.5)
    telemetry = GRPOTelemetry()
    for _ in range(10):
        telemetry.observe(group_advantages([1.0, 1.0, 1.0, 1.0]))
    verdict = telemetry.verdict(config)
    assert verdict["learning_signal"] is False
    assert "too_easy" in verdict["diagnosis"]
    assert verdict["mean_reward"] == 1.0


def test_healthy_run_reports_learning_signal():
    config = GRPOConfig(group_size=4)
    telemetry = GRPOTelemetry()
    for _ in range(10):
        telemetry.observe(group_advantages([1.0, 0.0, 1.0, 0.0]))
    verdict = telemetry.verdict(config)
    assert verdict["learning_signal"] is True
    assert verdict["usable_groups"] == 10
    assert verdict["diagnosis"] == "healthy"


def test_empty_telemetry_does_not_claim_success():
    assert GRPOTelemetry().verdict(GRPOConfig())["learning_signal"] is False


def test_telemetry_state_round_trips_exactly_for_resume():
    telemetry = GRPOTelemetry()
    telemetry.observe(group_advantages([1.0, 0.0, 1.0, 0.0]))
    telemetry.observe(group_advantages([0.0, 0.0, 0.0, 0.0]))

    restored = GRPOTelemetry.from_state(telemetry.state())

    assert restored == telemetry
    assert restored.verdict(GRPOConfig()) == telemetry.verdict(GRPOConfig())


def test_invalid_telemetry_resume_state_is_refused():
    state = GRPOTelemetry().state()
    state["groups"] = 1
    state["degenerate"] = 2
    with pytest.raises(ValueError, match="degenerate exceeds groups"):
        GRPOTelemetry.from_state(state)

    state = GRPOTelemetry().state()
    state["groups"] = 1
    state["reward_sum"] = 1.1
    with pytest.raises(ValueError, match="reward_sum exceeds groups"):
        GRPOTelemetry.from_state(state)


# ── Loss: MLX-dependent, so guarded ─────────────────────────────────────


def test_grpo_loss_and_kl_leash():
    mx = pytest.importorskip("mlx.core")
    from core.learning.grpo import grpo_loss

    policy = [mx.array(-2.0), mx.array(-3.0)]
    advantages = [1.0, -1.0]
    loss, telemetry = grpo_loss(policy, advantages)
    assert telemetry["kl"] == 0.0
    # Reinforcing a likely completion and suppressing an unlikely one.
    assert float(loss) == pytest.approx((2.0 - 3.0) / 2, rel=1e-5)

    # A policy that has drifted from the reference pays a KL cost.
    drifted, drift_telemetry = grpo_loss(
        policy, advantages, reference_logprobs=[mx.array(-2.0), mx.array(-3.0)],
        kl_coefficient=1.0,
    )
    assert drift_telemetry["kl"] == pytest.approx(0.0, abs=1e-6)
    _, far = grpo_loss(
        policy, advantages, reference_logprobs=[mx.array(-1.0), mx.array(-1.0)],
        kl_coefficient=1.0,
    )
    assert far["kl"] > 0.0, "drift from the reference must be penalized"


def test_grpo_loss_is_token_normalized_and_kl_is_token_level():
    mx = pytest.importorskip("mlx.core")
    from core.learning.grpo import grpo_loss

    policy = [mx.array([-2.0, -2.0, -2.0]), mx.array([-3.0])]
    advantages = [1.0, -1.0]
    loss, _ = grpo_loss(policy, advantages)
    assert float(loss) == pytest.approx((2.0 - 3.0) / 2, rel=1e-5)

    _, telemetry = grpo_loss(
        policy,
        advantages,
        reference_logprobs=[mx.array([-1.0, -1.0, -1.0]), mx.array([-3.0])],
        kl_coefficient=1.0,
    )
    expected_first = pytest.approx((2.718281828 - 1.0 - 1.0) / 2, rel=1e-5)
    assert telemetry["kl"] == expected_first


def test_misaligned_inputs_are_refused():
    mx = pytest.importorskip("mlx.core")
    from core.learning.grpo import grpo_loss

    with pytest.raises(ValueError, match="align"):
        grpo_loss([mx.array(-1.0)], [1.0, 2.0])
    with pytest.raises(ValueError, match="nothing to optimize"):
        grpo_loss([], [])
