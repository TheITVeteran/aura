"""Optimizing thoughts instead of weights (CP231).

Two Anima Rationis warnings are enforced here as code, not discipline:
line 220 (optimizing confidence strengthens confident mistakes) and line
453 (the matched random control is essential).
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.learning.latent_optimization import (  # noqa: E402
    LatentObjective,
    consistency_score,
    manifold_distance,
    matched_random_control,
    optimization_verdict,
    optimize_latent,
)

TARGET = mx.array([[1.0, 2.0, 3.0, 4.0]])


def _verifier_score(state):
    """Stands in for a real verifier: closer to target is better."""
    return -mx.sum(mx.square(state - TARGET))


# ── The objective refuses to be a confidence maximizer ──────────────────


def test_objective_without_verifier_or_consistency_is_refused():
    """Optimizing confidence makes wrong answers more confident and calls
    it progress."""
    with pytest.raises(ValueError, match="strengthens confident mistakes"):
        LatentObjective(verifier_weight=0.0, consistency_weight=0.0)


def test_manifold_term_is_mandatory():
    """Without it the optimizer generates adversarial examples against the
    model's own decoder."""
    with pytest.raises(ValueError, match="adversarial-example"):
        LatentObjective(manifold_weight=0.0)


# ── Optimization actually optimizes ─────────────────────────────────────


def test_latent_optimization_improves_a_verifier_score():
    start = mx.array([[0.4, 0.9, 1.6, 2.1]])
    result = optimize_latent(
        start, _verifier_score,
        objective=LatentObjective(manifold_weight=0.01),
        steps=20, learning_rate=0.1, max_drift=5.0,
    )
    assert result["improved"] is True
    assert result["final_score"] > result["initial_score"]
    # The trajectory is returned because an optimization nobody can inspect
    # is indistinguishable from noise with a good story.
    assert len(result["score_trajectory"]) == result["steps_taken"] + 1


def test_manifold_penalty_restrains_drift():
    start = mx.array([[0.4, 0.9, 1.6, 2.1]])
    loose = optimize_latent(
        start, _verifier_score, objective=LatentObjective(manifold_weight=0.01),
        steps=20, learning_rate=0.1, max_drift=5.0,
    )
    tight = optimize_latent(
        start, _verifier_score, objective=LatentObjective(manifold_weight=50.0),
        steps=20, learning_rate=0.1, max_drift=5.0,
    )
    assert tight["drift"] < loose["drift"]


def test_hitting_the_drift_wall_is_reported_not_hidden():
    """A run that stopped early did not spend its budget, and the receipt
    must say so."""
    start = mx.array([[0.4, 0.9, 1.6, 2.1]])
    result = optimize_latent(
        start, _verifier_score, objective=LatentObjective(manifold_weight=0.001),
        steps=50, learning_rate=0.5, max_drift=0.05,
    )
    assert result["stopped_early"] is True
    assert result["steps_taken"] < 50


# ── The control that decides whether any of it meant anything ───────────


def test_directed_optimization_beats_matched_random():
    start = mx.array([[0.4, 0.9, 1.6, 2.1]])
    result = optimize_latent(
        start, _verifier_score, objective=LatentObjective(manifold_weight=0.01),
        steps=30, learning_rate=0.1, max_drift=5.0,
    )
    control = matched_random_control(
        start, _verifier_score, drift=result["drift"], trials=16
    )
    verdict = optimization_verdict(result, control)
    assert verdict["beats_random"] is True
    assert verdict["verdict"] == "directed latent optimization"


def test_undirected_movement_is_called_what_it_is():
    """The whole point of the control: movement is not achievement."""
    start = mx.array([[0.4, 0.9, 1.6, 2.1]])
    noise_result = {
        "initial_score": -30.0,
        "final_score": -29.0,   # barely moved
        "drift": 1.0,
    }
    control = matched_random_control(
        start, _verifier_score, drift=1.0, trials=16
    )
    verdict = optimization_verdict(noise_result, control)
    assert verdict["beats_random"] is False
    assert "indistinguishable" in verdict["verdict"]


def test_control_matches_the_optimizer_displacement():
    """Comparing against a different magnitude proves nothing."""
    start = mx.ones((1, 4))
    control = matched_random_control(start, _verifier_score, drift=0.25, trials=4)
    assert control["matched_drift"] == pytest.approx(0.25)
    assert control["trials"] == 4


def test_verdict_compares_against_the_best_control_not_the_mean():
    """Beating the average of random noise is a bar a lucky direction
    clears."""
    optimized = {"initial_score": 0.0, "final_score": 1.0, "drift": 1.0}
    control = {"mean_score": 0.1, "best_score": 0.99}
    verdict = optimization_verdict(optimized, control, margin=0.05)
    assert verdict["beats_random"] is False, "must use best, not mean"


# ── Supporting measures ─────────────────────────────────────────────────


def test_gradient_is_finite_at_the_anchor():
    """Every optimization's first step sits exactly at the anchor, where
    d/dx sqrt(x) is infinite -- which produced a NaN gradient and a run
    that silently made no progress."""
    anchor = mx.array([[0.4, 0.9, 1.6, 2.1]])
    grad = mx.grad(lambda z: manifold_distance(z, anchor))(anchor)
    assert bool(mx.all(mx.isfinite(grad))), "NaN gradient at the starting point"


def test_zero_anchor_does_not_explode_the_scale():
    zero = mx.zeros((1, 4))
    moved = mx.ones((1, 4)) * 0.01
    assert float(manifold_distance(moved, zero)) < 100.0


def test_manifold_distance_is_scale_relative():
    anchor = mx.ones((1, 8)) * 100.0
    near = anchor + 1.0
    assert float(manifold_distance(anchor, anchor)) == pytest.approx(0.0, abs=1e-5)
    assert float(manifold_distance(near, anchor)) < 0.05


def test_consistency_is_agreement_not_confidence():
    assert consistency_score(["a", "a", "a"]) == 1.0
    assert consistency_score(["a", "b", "a", "c"]) == 0.5
    with pytest.raises(ValueError, match="at least two"):
        consistency_score(["a"])


def test_invalid_settings_are_refused():
    start = mx.array([[0.4, 0.9, 1.6, 2.1]])
    objective = LatentObjective()
    with pytest.raises(ValueError, match="steps"):
        optimize_latent(start, _verifier_score, objective=objective, steps=0)
    with pytest.raises(ValueError, match="learning_rate"):
        optimize_latent(
            start, _verifier_score, objective=objective, learning_rate=0.0
        )
    with pytest.raises(ValueError, match="max_drift"):
        optimize_latent(start, _verifier_score, objective=objective, max_drift=0.0)
    with pytest.raises(ValueError, match="share a shape"):
        manifold_distance(mx.ones((1, 4)), mx.ones((1, 5)))
