"""Protected-memory contracts (CP213): control state must survive depth.

The measured failure is that the recurrent update is a contraction
(residual 0.302 -> 0.026, asymptoting), so every direction in the tensor is
pulled into one attractor -- including the ones holding counters, bindings
and subgoals. These tests pin the fix: the protected lane persists across
arbitrarily many steps while the semantic lane keeps converging.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.learning.protected_memory import (  # noqa: E402
    MemoryLayout,
    apply_protected_transition,
    memory_retention,
    semantic_convergence,
    write_gates,
)


def _layout() -> MemoryLayout:
    return MemoryLayout(n_slots=8, memory_slots=(4, 5), control_slots=(6,))


def _state(value: float, slots: int = 8, hidden: int = 16):
    return mx.full((1, slots, hidden), value)


# ── Layout ──────────────────────────────────────────────────────────────


def test_layout_partitions_slots_and_rejects_overlap():
    layout = _layout()
    assert layout.protected == (4, 5, 6)
    assert layout.semantic == (0, 1, 2, 3, 7)
    assert set(layout.protected) & set(layout.semantic) == set()
    with pytest.raises(ValueError, match="both memory and control"):
        MemoryLayout(n_slots=8, memory_slots=(4,), control_slots=(4,))
    with pytest.raises(ValueError, match="semantic slot must remain"):
        MemoryLayout(n_slots=2, memory_slots=(0, 1))
    with pytest.raises(ValueError, match="inside"):
        MemoryLayout(n_slots=4, memory_slots=(9,))


# ── The load-bearing property: survival across depth ────────────────────


def test_protected_lane_survives_a_contraction_that_erases_semantics():
    """Drive 32 steps of a hard contraction toward zero. Semantic slots
    collapse (correctly); protected slots must still hold their value."""
    layout = _layout()
    state = _state(1.0)
    initial = mx.array(state)
    trail = [mx.array(state)]
    for _ in range(32):
        contracted = 0.5 * state  # a brutal contraction toward zero
        state, _gate = apply_protected_transition(state, contracted, layout)
        trail.append(mx.array(state))

    retention = memory_retention(initial, state, layout)
    assert retention["cosine"] > 0.999
    assert retention["relative_drift"] < 0.01, "protected lane must persist"

    semantic_index = mx.array(list(layout.semantic))
    semantic_final = mx.take(state, semantic_index, axis=1)
    assert float(mx.max(mx.abs(semantic_final))) < 1e-6, (
        "semantic lane should still converge — the point is separation, "
        "not blocking convergence everywhere"
    )


def test_semantic_convergence_is_measured_separately_from_memory():
    layout = _layout()
    state = _state(1.0)
    trail = [mx.array(state)]
    for _ in range(8):
        state, _gate = apply_protected_transition(state, 0.5 * state, layout)
        trail.append(mx.array(state))
    residuals = semantic_convergence(trail, layout)
    assert len(residuals) == 8
    # Semantic residual stays at the contraction's own rate; it is not
    # polluted by the protected lane sitting still.
    assert all(value > 0.5 for value in residuals)


def test_counter_value_remains_recoverable_at_depth_32():
    """The concrete question: is a stored 'counter' still readable after 32
    steps of recurrence?"""
    layout = MemoryLayout(n_slots=6, memory_slots=(5,))
    hidden = 16
    state = mx.zeros((1, 6, hidden))
    counter = mx.reshape(mx.arange(hidden, dtype=mx.float32) / hidden, (1, 1, hidden))
    state = mx.concatenate([state[:, :5, :], counter], axis=1)
    stored = mx.array(state[:, 5:6, :])
    for step in range(32):
        contracted = 0.3 * state + 0.1 * mx.sin(mx.array(float(step)))
        state, _gate = apply_protected_transition(state, contracted, layout)
    recovered = state[:, 5:6, :]
    error = float(
        mx.linalg.norm(mx.reshape(recovered - stored, (-1,)))
        / mx.maximum(mx.linalg.norm(mx.reshape(stored, (-1,))), 1e-9)
    )
    assert error < 0.02, f"counter drifted {100*error:.2f}% over 32 steps"


# ── Write gating ────────────────────────────────────────────────────────


def test_default_is_preserve_not_overwrite():
    """A candidate that merely restates memory must not consume a write."""
    layout = _layout()
    state = _state(1.0)
    gate = write_gates(state, state, layout)
    assert float(mx.max(gate)) < 0.2, "agreeing candidate should not overwrite"


def test_strong_disagreement_opens_the_gate():
    layout = _layout()
    previous = _state(1.0)
    agreeing = write_gates(previous, previous, layout)
    disagreeing = write_gates(previous, _state(-4.0), layout)
    assert float(mx.max(disagreeing)) > float(mx.max(agreeing))


def test_gate_is_zero_on_semantic_slots():
    layout = _layout()
    gate = write_gates(_state(1.0), _state(-4.0), layout)
    for index in layout.semantic:
        assert float(gate[0, index, 0]) == 0.0
    for index in layout.protected:
        assert float(gate[0, index, 0]) > 0.0


def test_semantic_slots_receive_the_contracted_update_unchanged():
    layout = _layout()
    previous = _state(1.0)
    contracted = _state(0.25)
    result, _gate = apply_protected_transition(previous, contracted, layout)
    for index in layout.semantic:
        assert float(result[0, index, 0]) == pytest.approx(0.25, abs=1e-6)


def test_shape_and_layout_mismatches_fail_closed():
    layout = _layout()
    with pytest.raises(ValueError, match="share a shape"):
        apply_protected_transition(_state(1.0), _state(1.0, slots=4), layout)
    with pytest.raises(ValueError, match="slot count"):
        apply_protected_transition(
            _state(1.0, slots=4), _state(1.0, slots=4), layout
        )


def test_optional_orthogonal_carry_preserves_norm():
    """Identity is the default carry; an orthogonal transport is allowed
    and must not inflate or shrink the protected lane."""
    layout = MemoryLayout(n_slots=4, memory_slots=(3,))
    state = mx.random.normal((1, 4, 8), key=mx.random.key(0))
    before = float(mx.linalg.norm(mx.reshape(state[:, 3:4, :], (-1,))))

    def rotate(value):
        return mx.concatenate([value[..., 1:], value[..., :1]], axis=-1)

    result, _gate = apply_protected_transition(
        state, state, layout, carry=rotate
    )
    after = float(mx.linalg.norm(mx.reshape(result[:, 3:4, :], (-1,))))
    assert after == pytest.approx(before, rel=0.02)


def test_explicit_write_updates_only_the_addressed_slot():
    """A write is a decision. Gate 1.0 on one slot overwrites exactly it;
    every other protected slot is untouched."""
    layout = MemoryLayout(n_slots=5, memory_slots=(3, 4))
    previous = _state(1.0, slots=5)
    contracted = _state(-1.0, slots=5)
    gate = mx.reshape(
        mx.array([0.0, 0.0, 0.0, 1.0, 0.0]), (1, 5, 1)
    )
    result, applied = apply_protected_transition(
        previous, contracted, layout, write_gate=gate
    )
    assert float(result[0, 3, 0]) == pytest.approx(-1.0, abs=1e-6)
    assert float(result[0, 4, 0]) == pytest.approx(1.0, abs=1e-6)
    for index in layout.semantic:
        assert float(result[0, index, 0]) == pytest.approx(-1.0, abs=1e-6)
    assert float(applied[0, 3, 0]) == pytest.approx(1.0, abs=1e-6)


def test_write_gate_cannot_reach_semantic_slots():
    """Even a gate of all ones only affects the protected lane."""
    layout = MemoryLayout(n_slots=4, memory_slots=(3,))
    _result, applied = apply_protected_transition(
        _state(1.0, slots=4),
        _state(-1.0, slots=4),
        layout,
        write_gate=mx.ones((1, 4, 1)),
    )
    for index in layout.semantic:
        assert float(applied[0, index, 0]) == 0.0
    assert float(applied[0, 3, 0]) == pytest.approx(1.0, abs=1e-6)
