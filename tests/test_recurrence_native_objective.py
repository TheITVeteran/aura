"""Contract tests: the recurrence-native training objective.

Three things must be TRUE, not aspirational, on a real (tiny random) model:
1. the recurrent forward is differentiable and the recurrence is causal in
   the training graph (T changes the loss);
2. gradients reach the WINDOW layers through every recurrent application;
3. a few descent steps on a window layer reduce the loss — the objective
   is learnable, which is what 'teach the weights to think in loops' means.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.recurrence_native_objective import (  # noqa: E402
    objective_receipt,
    recurrence_native_loss,
    recurrent_forward_logits,
)

TOKENS = [5, 9, 17, 3, 42, 7, 11, 23, 2, 88, 14, 60]
ANSWER_START = 8


def _model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=8,
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


def test_recurrent_forward_shapes_and_step_causality():
    model = _model()
    logits_1 = recurrent_forward_logits(model, TOKENS, recurrent_steps=1)
    logits_4 = recurrent_forward_logits(model, TOKENS, recurrent_steps=4)
    assert logits_1.shape == (1, len(TOKENS), 128)
    assert bool(mx.all(mx.isfinite(logits_1)))
    # Recurrence must be causal in the graph: more steps ⇒ different logits.
    assert not bool(mx.allclose(logits_1, logits_4, atol=1e-5))
    loss_1 = float(recurrence_native_loss(model, TOKENS, ANSWER_START, recurrent_steps=1))
    loss_4 = float(recurrence_native_loss(model, TOKENS, ANSWER_START, recurrent_steps=4))
    assert loss_1 != loss_4


def test_gradients_reach_window_layers_through_recurrence():
    model = _model()
    inner = model.model
    window_layer = inner.layers[3]  # inside the [2, 6) recurrent region

    def loss_fn(weight):
        original = window_layer.mlp.down_proj.weight
        window_layer.mlp.down_proj.weight = weight
        try:
            return recurrence_native_loss(
                model, TOKENS, ANSWER_START, recurrent_steps=2
            )
        finally:
            window_layer.mlp.down_proj.weight = original

    weight = window_layer.mlp.down_proj.weight
    value, grad = mx.value_and_grad(loss_fn)(weight)
    mx.eval(value, grad)
    assert bool(mx.isfinite(value))
    assert float(mx.linalg.norm(mx.reshape(grad, (-1,)))) > 0.0


def test_descent_on_window_layer_reduces_recurrent_loss():
    model = _model()
    inner = model.model
    window_layer = inner.layers[4]

    def loss_fn(weight):
        original = window_layer.mlp.down_proj.weight
        window_layer.mlp.down_proj.weight = weight
        try:
            return recurrence_native_loss(
                model, TOKENS, ANSWER_START, recurrent_steps=2
            )
        finally:
            window_layer.mlp.down_proj.weight = original

    weight = window_layer.mlp.down_proj.weight
    grad_fn = mx.value_and_grad(loss_fn)
    first_loss, _ = grad_fn(weight)
    losses = [float(first_loss)]
    for _ in range(6):
        value, grad = grad_fn(weight)
        weight = weight - 0.05 * grad / mx.maximum(
            mx.linalg.norm(mx.reshape(grad, (-1,))), 1e-8
        )
        mx.eval(weight)
        losses.append(float(value))
    final_loss = float(grad_fn(weight)[0])
    assert final_loss < losses[0], (
        f"descent must reduce the recurrent loss: {losses[0]} → {final_loss}"
    )


def test_only_answer_tokens_carry_loss():
    model = _model()
    full = recurrence_native_loss(model, TOKENS, 1, recurrent_steps=1)
    tail = recurrence_native_loss(
        model, TOKENS, len(TOKENS) - 1, recurrent_steps=1
    )
    assert float(full) != float(tail)


def test_input_validation():
    model = _model()
    with pytest.raises(ValueError, match="recurrent_steps"):
        recurrent_forward_logits(model, TOKENS, recurrent_steps=0)
    with pytest.raises(ValueError, match="alpha"):
        recurrent_forward_logits(model, TOKENS, alpha=0.0)
    with pytest.raises(ValueError, match="answer_start"):
        recurrence_native_loss(model, TOKENS, 0)
    with pytest.raises(ValueError, match="two tokens"):
        recurrence_native_loss(model, [5], 1)
    receipt = objective_receipt(
        recurrent_steps=2, alpha=0.5, batch_count=4, mean_loss=1.25
    )
    assert receipt["objective"] == "answer_span_ce_under_recurrent_forward"


def test_depth_curriculum_penalizes_late_step_degradation():
    """The hinge fires exactly when deeper recurrence is WORSE — the
    trainable form of S(x, T+1) >= S(x, T)."""
    from core.learning.recurrence_native_objective import depth_curriculum_loss

    model = _model()
    curriculum = depth_curriculum_loss(model, TOKENS, ANSWER_START, depths=(1, 2, 4))
    assert bool(mx.isfinite(curriculum))

    losses = {
        depth: float(
            recurrence_native_loss(model, TOKENS, ANSWER_START, recurrent_steps=depth)
        )
        for depth in (1, 2, 4)
    }
    mean_loss = sum(losses.values()) / 3
    hinge = max(losses[2] - losses[1], 0.0) + max(losses[4] - losses[2], 0.0)
    assert float(curriculum) == pytest.approx(mean_loss + 0.5 * hinge, rel=1e-4)
    # Weight 0 reduces the curriculum to the pure depth-ladder mean.
    plain = depth_curriculum_loss(
        model, TOKENS, ANSWER_START, depths=(1, 2, 4), monotonicity_weight=0.0
    )
    assert float(plain) == pytest.approx(mean_loss, rel=1e-4)


def test_depth_curriculum_is_learnable():
    from core.learning.recurrence_native_objective import depth_curriculum_loss

    model = _model()
    window_layer = model.model.layers[4]

    def loss_fn(weight):
        original = window_layer.mlp.down_proj.weight
        window_layer.mlp.down_proj.weight = weight
        try:
            return depth_curriculum_loss(model, TOKENS, ANSWER_START, depths=(1, 2))
        finally:
            window_layer.mlp.down_proj.weight = original

    weight = window_layer.mlp.down_proj.weight
    grad_fn = mx.value_and_grad(loss_fn)
    first, _ = grad_fn(weight)
    for _ in range(6):
        value, grad = grad_fn(weight)
        weight = weight - 0.05 * grad / mx.maximum(
            mx.linalg.norm(mx.reshape(grad, (-1,))), 1e-8
        )
        mx.eval(weight)
    final = float(grad_fn(weight)[0])
    assert final < float(first), f"curriculum must be learnable: {float(first)} → {final}"


def test_depth_curriculum_input_validation():
    from core.learning.recurrence_native_objective import depth_curriculum_loss

    model = _model()
    with pytest.raises(ValueError, match="depths"):
        depth_curriculum_loss(model, TOKENS, ANSWER_START, depths=(2,))
    with pytest.raises(ValueError, match="depths"):
        depth_curriculum_loss(model, TOKENS, ANSWER_START, depths=(4, 2))
    with pytest.raises(ValueError, match="monotonicity_weight"):
        depth_curriculum_loss(
            model, TOKENS, ANSWER_START, monotonicity_weight=-1.0
        )
