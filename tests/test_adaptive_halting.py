"""Spending thought where it is needed (CP230).

Anima Rationis makes the compute-scaling curve the success criterion and
names latent overthinking as the failure mode. A fixed T is not an
allocation policy, and a flat curve measured at fixed T cannot tell you
whether depth would have helped.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.learning.adaptive_halting import (  # noqa: E402
    HaltingHead,
    HaltingPolicy,
    allocation_report,
    decide_steps,
    overthinking_report,
    ponder_loss,
    validate_verified_stopping_teacher_receipt,
    verified_stopping_teacher,
)

HIDDEN = 16


def _states(count):
    return [mx.ones((1, 4, HIDDEN)) * (i + 1) for i in range(count)]


# ── Identity at attach ──────────────────────────────────────────────────


def test_untrained_head_never_changes_behaviour():
    """Allocation is earned, not imposed at attach time."""
    head = HaltingHead(HIDDEN)
    assert head.is_identity()
    # p = 0.5 everywhere, exactly on the default threshold. Identity is an
    # explicit state, so equality at the threshold cannot activate it.
    assert float(head.halt_probability(mx.ones((1, 4, HIDDEN)))) == pytest.approx(0.5)
    report = decide_steps(head, _states(8), HaltingPolicy(min_steps=1, max_steps=8))
    assert report["steps"] == 8
    assert report["halted_early"] is False


def test_head_stays_small():
    head = HaltingHead(5120)
    assert head.parameter_count() == 5121, "a large halting net is a second model"


# ── Halting decisions are inspectable ───────────────────────────────────


def test_minimum_steps_are_respected():
    """Even a head that always wants to stop cannot skip the floor."""
    head = HaltingHead(HIDDEN, threshold=0.1)
    policy = HaltingPolicy(min_steps=3, max_steps=8)
    report = decide_steps(head, _states(8), policy)
    assert report["steps"] >= 3


def test_maximum_steps_bound_the_spend():
    head = HaltingHead(HIDDEN, threshold=0.99)
    report = decide_steps(head, _states(8), HaltingPolicy(min_steps=1, max_steps=4))
    assert report["steps"] == 4
    assert report["halted_early"] is False


def test_a_trained_head_can_stop_early_and_shows_its_work():
    head = HaltingHead(HIDDEN, threshold=0.5)
    head.bias = mx.array([5.0])  # strongly wants to halt
    report = decide_steps(head, _states(8), HaltingPolicy(min_steps=2, max_steps=8))
    assert report["steps"] == 2
    assert report["halted_early"] is True
    # A halting decision nobody can inspect is indistinguishable from a
    # constant.
    assert len(report["halt_probabilities"]) == 2


# ── The ponder objective must not be dominated by the compute term ──────


def test_accuracy_beats_efficiency_in_the_objective():
    """An objective dominated by compute produces a model that always halts
    immediately and reports excellent efficiency."""
    policy = HaltingPolicy(min_steps=1, max_steps=3, ponder_cost=0.01)
    losses = [mx.array(5.0), mx.array(1.0), mx.array(0.5)]
    halt_now = [mx.array(0.99), mx.array(0.5), mx.array(0.5)]
    think_more = [mx.array(0.01), mx.array(0.01), mx.array(0.99)]
    early, _ = ponder_loss(losses, halt_now, policy)
    late, _ = ponder_loss(losses, think_more, policy)
    assert float(late) < float(early), "thinking longer must win when it is much more accurate"


def test_ponder_cost_breaks_ties_toward_less_compute():
    policy = HaltingPolicy(min_steps=1, max_steps=3, ponder_cost=0.1)
    flat = [mx.array(1.0), mx.array(1.0), mx.array(1.0)]
    early, early_telemetry = ponder_loss(
        flat, [mx.array(0.99), mx.array(0.5), mx.array(0.5)], policy
    )
    late, late_telemetry = ponder_loss(
        flat, [mx.array(0.01), mx.array(0.01), mx.array(0.99)], policy
    )
    assert float(early) < float(late), "equal accuracy should prefer fewer steps"
    assert early_telemetry["expected_steps"] < late_telemetry["expected_steps"]


def test_halting_distribution_is_a_distribution():
    """The last step absorbs all remaining mass, so weights sum to one."""
    policy = HaltingPolicy(min_steps=1, max_steps=3)
    losses = [mx.array(1.0)] * 3
    _, telemetry = ponder_loss(losses, [mx.array(0.3), mx.array(0.3), mx.array(0.3)], policy)
    assert 1.0 <= telemetry["expected_steps"] <= 3.0
    assert telemetry["expected_loss"] == pytest.approx(1.0, rel=1e-5)


def test_verified_stopping_teacher_prefers_early_equal_quality_and_replays():
    teacher = verified_stopping_teacher(
        [0.4, 0.4, 0.4],
        [1, 2, 4],
        ponder_cost=0.05,
        temperature=0.1,
    )
    receipt = teacher.receipt()

    assert teacher.selected_step == 1
    assert teacher.probabilities[0] > teacher.probabilities[1]
    assert teacher.probabilities[1] > teacher.probabilities[2]
    assert sum(teacher.probabilities) == pytest.approx(1.0)
    assert validate_verified_stopping_teacher_receipt(receipt) == receipt


def test_verified_stopping_teacher_rejects_resealed_false_atoms():
    import copy
    import hashlib
    import json

    receipt = verified_stopping_teacher(
        [0.9, 0.3],
        [1, 2],
        ponder_cost=0.01,
        temperature=0.2,
    ).receipt()
    attacked = copy.deepcopy(receipt)
    attacked["probabilities"] = list(reversed(attacked["probabilities"]))
    payload = {key: value for key, value in attacked.items() if key != "receipt_sha256"}
    attacked["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    with pytest.raises(ValueError, match="arithmetic"):
        validate_verified_stopping_teacher_receipt(attacked)


@pytest.mark.parametrize("invalid_loss", [True, "0.4", None])
def test_verified_stopping_teacher_rejects_scalar_type_confusion(invalid_loss):
    with pytest.raises(ValueError, match="steps and losses"):
        verified_stopping_teacher(
            [invalid_loss, 0.3],
            [1, 2],
            ponder_cost=0.01,
            temperature=0.2,
        )


# ── The falsifiable claim ───────────────────────────────────────────────


def test_allocation_tracking_difficulty_is_recognized():
    report = allocation_report([(1, 1), (2, 2), (4, 4), (8, 7), (8, 8)])
    assert report["correlation"] > 0.9
    assert report["allocates_by_difficulty"] is True


def test_constant_allocation_is_not_a_policy():
    """A fixed spend is what we already had; it must not read as success."""
    report = allocation_report([(1, 4), (2, 4), (4, 4), (8, 4)])
    assert report["allocates_by_difficulty"] is False
    assert "not a policy" in report["reason"]


def test_weak_correlation_is_not_called_allocation():
    """Noise with a sign is not thought allocation."""
    report = allocation_report([(1, 3), (8, 4), (2, 2), (4, 3), (8, 3), (1, 4)])
    assert report["correlation"] < 0.3
    assert report["allocates_by_difficulty"] is False


def test_backwards_allocation_is_caught():
    report = allocation_report([(1, 8), (2, 6), (4, 4), (8, 1)])
    assert report["correlation"] < 0
    assert report["allocates_by_difficulty"] is False


# ── Overthinking ────────────────────────────────────────────────────────


def test_overthinking_is_detected():
    """CP226's shape: the loop kept moving while accuracy fell."""
    report = overthinking_report([2.0, 1.0, 1.4, 1.9])
    assert report["overthinks"] is True
    assert report["best_step"] == 2
    assert report["wasted_steps"] == 2


def test_monotone_improvement_is_not_called_overthinking():
    report = overthinking_report([2.0, 1.5, 1.0, 0.8])
    assert report["overthinks"] is False
    assert report["best_step"] == 4
    assert report["wasted_steps"] == 0


# ── Fail closed ─────────────────────────────────────────────────────────


def test_invalid_configuration_is_refused():
    with pytest.raises(ValueError, match="threshold"):
        HaltingHead(HIDDEN, threshold=1.0)
    with pytest.raises(ValueError, match="hidden_size"):
        HaltingHead(0)
    with pytest.raises(ValueError, match="cannot exceed"):
        HaltingPolicy(min_steps=5, max_steps=2)
    with pytest.raises(ValueError, match="ponder_cost"):
        HaltingPolicy(ponder_cost=2.0)
    with pytest.raises(ValueError, match="align"):
        ponder_loss([mx.array(1.0)], [mx.array(0.5), mx.array(0.5)], HaltingPolicy())
    with pytest.raises(ValueError, match="at least three"):
        allocation_report([(1, 1), (2, 2)])
    with pytest.raises(ValueError, match="at least two"):
        overthinking_report([1.0])
    with pytest.raises(ValueError, match="state width"):
        HaltingHead(HIDDEN).halt_probability(mx.ones((1, 4, HIDDEN + 1)))
