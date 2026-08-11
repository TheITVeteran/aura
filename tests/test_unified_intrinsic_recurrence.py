"""Contracts for the single-path intrinsic recurrent controller."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.intrinsic_recurrence import RecurrentDepthPlan  # noqa: E402
from core.learning.protected_memory import MemoryLayout  # noqa: E402
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
    unified_recurrent_hidden_states,
    unified_recurrent_logits,
)


def _model() -> Model:
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=8,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=256,
            rope_theta=10_000.0,
        )
    )
    mx.eval(model.parameters())
    return model


TOKENS = mx.array([[3, 11, 42, 7, 19, 23]])


def _controller() -> UnifiedRecurrentController:
    return UnifiedRecurrentController(
        UnifiedRecurrenceConfig(hidden_size=64, correction_rank=8)
    )


def test_identity_controller_preserves_base_forward_at_one_iteration() -> None:
    model = _model()
    controller = _controller()
    logits, telemetry = unified_recurrent_logits(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=1),
        controller,
    )
    assert controller.identity_initialized()
    assert bool(mx.allclose(logits, model(TOKENS), atol=1e-5))
    assert telemetry.executed_iterations == 1
    assert telemetry.halted is False


def test_continuous_depth_basis_is_defined_and_distinct_beyond_train_depth() -> None:
    controller = _controller()
    at_four = controller.depth_features(4)
    at_sixteen = controller.depth_features(16)
    at_thousand = controller.depth_features(1_000)
    assert not bool(mx.array_equal(at_four, at_sixteen))
    assert not bool(mx.array_equal(at_sixteen, at_thousand))
    assert bool(mx.all(at_thousand <= 1.0))
    assert bool(mx.all(at_thousand >= 0.0))


def test_protected_memory_survives_while_semantic_lane_keeps_moving() -> None:
    model = _model()
    controller = _controller()
    _final, trajectory, telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=5, renormalize=True),
        controller,
        memory_layout=MemoryLayout(
            n_slots=TOKENS.shape[1],
            memory_slots=(0, 1),
            control_slots=(2,),
        ),
    )
    assert len(trajectory) == 5
    assert telemetry.memory_retention is not None
    assert telemetry.memory_retention["cosine"] > 0.99999
    assert telemetry.memory_retention["relative_drift"] < 1e-6
    assert telemetry.memory_retention["slots"] == 3
    assert telemetry.semantic_residuals
    assert any(value > 0.01 for value in telemetry.semantic_residuals)
    assert telemetry.memory_write_means == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_learned_halt_is_causal_but_off_until_explicitly_enabled() -> None:
    model = _model()
    controller = _controller()
    controller.halt_state_weight = mx.zeros_like(controller.halt_state_weight)
    controller.halt_motion_weight = mx.array(0.0)
    controller.halt_bias = mx.array(20.0)
    plan = RecurrentDepthPlan(2, 6, iterations=8)
    _final, full, full_telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        plan,
        controller,
        adaptive_halt=False,
    )
    _final, halted, halted_telemetry = unified_recurrent_hidden_states(
        model,
        TOKENS,
        plan,
        controller,
        adaptive_halt=True,
    )
    assert len(full) == 8
    assert full_telemetry.halted is False
    assert len(halted) == controller.config.minimum_iterations
    assert halted_telemetry.halted is True
    assert halted_telemetry.halt_reason == "learned_threshold"


def test_trained_correction_changes_the_real_answer_path() -> None:
    model = _model()
    controller = _controller()
    baseline, _ = unified_recurrent_logits(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
    )
    controller.correction_b = mx.ones_like(controller.correction_b) * 0.01
    changed, telemetry = unified_recurrent_logits(
        model,
        TOKENS,
        RecurrentDepthPlan(2, 6, iterations=3),
        controller,
    )
    assert not bool(mx.allclose(baseline, changed, atol=1e-5))
    assert telemetry.receipt()["teacher_available"] is False
    assert telemetry.receipt()["solver_available"] is False


def test_invalid_unified_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="correction rank"):
        UnifiedRecurrenceConfig(hidden_size=4, correction_rank=8)
    with pytest.raises(ValueError, match="minimum iterations"):
        unified_recurrent_hidden_states(
            _model(),
            TOKENS,
            RecurrentDepthPlan(2, 6, iterations=1),
            _controller(),
            adaptive_halt=True,
        )
    with pytest.raises(ValueError, match="token positions"):
        unified_recurrent_hidden_states(
            _model(),
            TOKENS,
            RecurrentDepthPlan(2, 6, iterations=3),
            _controller(),
            memory_layout=MemoryLayout(n_slots=5, memory_slots=(0,)),
        )
