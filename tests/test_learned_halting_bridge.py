"""Attaching learned halting to the live engine (CP234).

The engine halts on residual convergence -- fixed-point detection, not
thought allocation. CP226 measured where they come apart: a loop that kept
moving (0.55, 0.50, 0.32) while accuracy fell to zero.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.learned_halting_bridge import (  # noqa: E402
    LEARNED,
    RESIDUAL,
    HaltingBridgeConfig,
    bridge_receipt,
    should_halt,
)
from core.learning.adaptive_halting import HaltingHead  # noqa: E402

HIDDEN = 16
STATE = mx.ones((1, 4, HIDDEN))


# ── Default is the current engine, unchanged ────────────────────────────


def test_residual_mode_is_the_existing_policy():
    """A live cortex is not the place to discover a new policy is worse."""
    config = HaltingBridgeConfig(mode=RESIDUAL, min_steps=1, max_steps=8)
    moving = should_halt(step=2, residual_trail=[0.5, 0.3], config=config)
    assert moving["halt"] is False
    assert moving["reason"] == "still_moving"

    settled = should_halt(step=2, residual_trail=[0.5, 0.001], config=config)
    assert settled["halt"] is True
    assert settled["reason"] == "converged"


def test_residual_mode_needs_no_head():
    config = HaltingBridgeConfig(mode=RESIDUAL)
    assert should_halt(step=1, residual_trail=[0.4], config=config)["halt"] is False


# ── Bounds hold in both modes ───────────────────────────────────────────


def test_min_steps_beats_every_other_signal():
    config = HaltingBridgeConfig(mode=RESIDUAL, min_steps=3, max_steps=8)
    verdict = should_halt(step=1, residual_trail=[0.0001], config=config)
    assert verdict["halt"] is False
    assert verdict["reason"] == "below_min_steps"


def test_max_steps_caps_an_eager_head():
    head = HaltingHead(HIDDEN)
    head.bias = mx.array([-50.0])  # never wants to stop
    config = HaltingBridgeConfig(mode=LEARNED, min_steps=1, max_steps=4)
    verdict = should_halt(
        step=4, residual_trail=[0.9], config=config, head=head, state=STATE
    )
    assert verdict["halt"] is True
    assert verdict["reason"] == "max_steps_reached"


def test_convergence_is_a_floor_in_learned_mode_too():
    """A loop at its fixed point has stopped computing, whatever the head
    wants -- that is a fact about the dynamics, not a preference."""
    head = HaltingHead(HIDDEN)
    head.bias = mx.array([-50.0])
    config = HaltingBridgeConfig(mode=LEARNED, min_steps=1, max_steps=8)
    verdict = should_halt(
        step=2, residual_trail=[0.3, 0.001], config=config, head=head, state=STATE
    )
    assert verdict["halt"] is True
    assert verdict["reason"] == "converged"


# ── The learned head can actually decide ────────────────────────────────


def test_a_trained_head_stops_a_still_moving_loop():
    """The CP226 case: motion is not progress, and residual halting cannot
    see the difference."""
    head = HaltingHead(HIDDEN)
    head.bias = mx.array([10.0])
    config = HaltingBridgeConfig(mode=LEARNED, min_steps=1, max_steps=8)
    verdict = should_halt(
        step=2, residual_trail=[0.55, 0.50], config=config, head=head, state=STATE
    )
    assert verdict["halt"] is True
    assert verdict["reason"] == "head_satisfied"
    assert verdict["halt_probability"] > 0.5


def test_an_untrained_head_reproduces_the_residual_policy():
    """Attaching the mechanism must grant nothing on its own."""
    head = HaltingHead(HIDDEN, threshold=0.75)  # zero weights => p = 0.5
    config = HaltingBridgeConfig(mode=LEARNED, min_steps=1, max_steps=8)
    learned = should_halt(
        step=2, residual_trail=[0.55, 0.50], config=config, head=head, state=STATE
    )
    residual = should_halt(
        step=2, residual_trail=[0.55, 0.50],
        config=HaltingBridgeConfig(mode=RESIDUAL, min_steps=1, max_steps=8),
    )
    assert learned["halt"] == residual["halt"] is False


# ── Silent fallback is refused ──────────────────────────────────────────


def test_learned_mode_without_a_head_is_refused():
    """Falling back silently would report learned allocation while running
    the residual rule -- the exact defect class this codebase keeps hitting."""
    config = HaltingBridgeConfig(mode=LEARNED)
    with pytest.raises(ValueError, match="requires a halting head"):
        should_halt(step=1, residual_trail=[0.5], config=config)
    with pytest.raises(ValueError, match="requires the current latent state"):
        should_halt(
            step=1, residual_trail=[0.5], config=config, head=HaltingHead(HIDDEN)
        )


# ── The receipt says whether the head did anything ──────────────────────


def test_receipt_reports_a_learned_run_that_never_used_its_head():
    """Running the old policy under a new name must be visible."""
    config = HaltingBridgeConfig(mode=LEARNED, min_steps=1, max_steps=4)
    verdicts = [
        {"halt": False, "reason": "head_wants_more", "step": 1, "halt_probability": 0.1},
        {"halt": False, "reason": "head_wants_more", "step": 2, "halt_probability": 0.2},
        {"halt": True, "reason": "max_steps_reached", "step": 4, "halt_probability": None},
    ]
    receipt = bridge_receipt(verdicts, config)
    assert receipt["stopped_by_head"] == 0
    assert receipt["head_was_causal"] is False
    assert receipt["steps_taken"] == 4


def test_receipt_confirms_a_causal_head():
    config = HaltingBridgeConfig(mode=LEARNED, min_steps=1, max_steps=8)
    verdicts = [
        {"halt": False, "reason": "head_wants_more", "step": 1, "halt_probability": 0.2},
        {"halt": True, "reason": "head_satisfied", "step": 2, "halt_probability": 0.9},
    ]
    receipt = bridge_receipt(verdicts, config)
    assert receipt["head_was_causal"] is True
    assert receipt["reasons"]["head_satisfied"] == 1


def test_invalid_configuration_fails_closed():
    with pytest.raises(ValueError, match="mode must be"):
        HaltingBridgeConfig(mode="vibes")
    with pytest.raises(ValueError, match="cannot exceed"):
        HaltingBridgeConfig(min_steps=9, max_steps=2)
    with pytest.raises(ValueError, match="convergence_residual"):
        HaltingBridgeConfig(convergence_residual=1.0)
    with pytest.raises(ValueError, match="1-based"):
        should_halt(step=0, residual_trail=[], config=HaltingBridgeConfig())
    with pytest.raises(ValueError, match="no halting verdicts"):
        bridge_receipt([], HaltingBridgeConfig())


# ── The head is actually wired into the live controller ─────────────────


def _controller(**overrides):
    from core.brain.llm.latent_cortex.recurrence import (
        HaltingController,
        RecurrenceConfig,
    )

    defaults = dict(max_steps=8, min_steps=1, convergence_eps=0.01, fixed_depth=False)
    defaults.update(overrides)
    return HaltingController(config=RecurrenceConfig(**defaults))


def test_controller_without_a_head_is_the_old_policy():
    controller = _controller()
    assert controller.halting_head is None
    decision = controller.observe(0, mx.ones((1, 4, HIDDEN)), residual=0.5)
    assert decision.should_halt is False
    assert controller.head_halts == 0


def test_an_attached_untrained_head_changes_nothing():
    """Attaching the mechanism must grant nothing -- the head is zero-init."""
    controller = _controller()
    controller.halting_head = HaltingHead(HIDDEN, threshold=0.5)
    decision = controller.observe(0, mx.ones((1, 4, HIDDEN)), residual=0.5)
    assert decision.should_halt is False
    assert controller.head_halts == 0


def test_a_trained_head_halts_a_still_moving_loop_in_the_controller():
    """The CP226 case, now inside the live halting path."""
    controller = _controller()
    head = HaltingHead(HIDDEN, threshold=0.5)
    head.bias = mx.array([10.0])
    controller.halting_head = head
    decision = controller.observe(0, mx.ones((1, 4, HIDDEN)), residual=0.5)
    assert decision.should_halt is True
    assert decision.reason == "head_satisfied"
    assert controller.head_halts == 1


def test_convergence_still_wins_over_an_eager_head_in_the_controller():
    controller = _controller()
    head = HaltingHead(HIDDEN, threshold=0.5)
    head.bias = mx.array([-50.0])   # never wants to stop
    controller.halting_head = head
    decision = controller.observe(0, mx.ones((1, 4, HIDDEN)), residual=0.0001)
    assert decision.should_halt is True
    assert decision.reason == "converged", "the fixed-point floor must hold"


def test_fixed_depth_training_ignores_the_head():
    """v2 training requires fixed-depth recurrence; the head must not
    silently reintroduce variable depth under it."""
    controller = _controller(fixed_depth=True)
    head = HaltingHead(HIDDEN, threshold=0.5)
    head.bias = mx.array([10.0])
    controller.halting_head = head
    decision = controller.observe(0, mx.ones((1, 4, HIDDEN)), residual=0.5)
    assert decision.should_halt is False
    assert controller.head_halts == 0


def test_divergence_still_preempts_the_head():
    controller = _controller()
    head = HaltingHead(HIDDEN, threshold=0.5)
    head.bias = mx.array([10.0])
    controller.halting_head = head
    bad = mx.full((1, 4, HIDDEN), float("nan"))
    decision = controller.observe(0, bad, residual=0.5)
    assert decision.reason == "diverged_nonfinite"
