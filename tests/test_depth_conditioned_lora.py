"""Depth-conditioned operator contracts (CP219).

Measured here: phase injected into the recurrence INPUT improves CE but
does not break the contraction (residual ratio 0.1142 -> 0.1048). To make
step t compute something different from step t+1, the OPERATOR must
differ. Think-at-Hard reports gains from exactly that (depth-aware LoRA),
so these tests pin the mechanism for Aura's adapter stack.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
)
from core.learning.depth_conditioned_lora import (  # noqa: E402
    DepthConditionedLoRA,
    current_depth_index,
    recurrent_depth_index,
    wrap_depth_conditioned,
)


def _model() -> Model:
    args = ModelArgs(
        model_type="qwen2", hidden_size=32, num_hidden_layers=4,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=64, num_key_value_heads=2, max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    model.freeze()
    for index in (1, 2):
        attention = model.model.layers[index].self_attn
        attention.o_proj = ScopedLoRALinear.from_base(attention.o_proj, r=2)
    mx.eval(model.parameters())
    return model


def _scoped():
    return _model().model.layers[1].self_attn.o_proj


# ── Identity at init: cannot break a working operator on day one ────────


def test_zero_deltas_make_init_bit_identical_to_the_shared_adapter():
    scoped = _scoped()
    conditioned = DepthConditionedLoRA(scoped, depths=4)
    for step in range(4):
        a, b = conditioned.factors_for(step)
        assert bool(mx.all(a == scoped.lora_a))
        assert bool(mx.all(b == scoped.lora_b))
        assert conditioned.is_identity_at(step)
    assert conditioned.to_receipt()["identity_at_init"] is True
    assert conditioned.differentiation() == [0.0, 0.0, 0.0, 0.0]


def test_training_a_delta_differentiates_only_that_depth():
    scoped = _scoped()
    conditioned = DepthConditionedLoRA(scoped, depths=4)
    # BOTH factors must move: dW = A @ B vanishes if either is zero.
    conditioned.depth_a[2] = mx.ones_like(scoped.lora_a) * 0.5
    conditioned.depth_b[2] = mx.ones_like(scoped.lora_b) * 0.5

    assert not conditioned.is_identity_at(2)
    assert conditioned.is_identity_at(0)
    a2, _ = conditioned.factors_for(2)
    a0, _ = conditioned.factors_for(0)
    assert not bool(mx.all(a2 == a0)), "step 2 must transform differently"
    spread = conditioned.differentiation()
    assert spread[2] > 0.0
    assert spread[0] == spread[1] == spread[3] == 0.0


def test_depth_beyond_the_bank_reuses_the_last_operator():
    """Depth extrapolation must RUN so it can be measured. A model trained
    to depth 8 has to be answerable at depth 32."""
    conditioned = DepthConditionedLoRA(_scoped(), depths=4)
    conditioned.depth_a[3] = mx.ones_like(conditioned.scoped.lora_a) * 0.25
    far, _ = conditioned.factors_for(99)
    last, _ = conditioned.factors_for(3)
    assert bool(mx.all(far == last))


def test_delta_scale_modulates_differentiation():
    scoped = _scoped()
    strong = DepthConditionedLoRA(scoped, depths=2, delta_scale=1.0)
    weak = DepthConditionedLoRA(scoped, depths=2, delta_scale=0.1)
    for conditioned in (strong, weak):
        conditioned.depth_a[1] = mx.ones_like(scoped.lora_a) * 0.5
        conditioned.depth_b[1] = mx.ones_like(scoped.lora_b) * 0.5
    assert strong.differentiation()[1] > weak.differentiation()[1]


# ── Plumbing ────────────────────────────────────────────────────────────


def test_depth_index_context_is_published_and_restored():
    assert current_depth_index() == 0
    with recurrent_depth_index(5):
        assert current_depth_index() == 5
        with recurrent_depth_index(2):
            assert current_depth_index() == 2
        assert current_depth_index() == 5
    assert current_depth_index() == 0
    with pytest.raises(ValueError, match="non-negative"):
        with recurrent_depth_index(-1):
            pass


def test_wrapping_finds_every_scoped_projection():
    model = _model()
    wrapped = wrap_depth_conditioned(model, depths=3)
    assert set(wrapped) == {
        "model.layers.1.self_attn.o_proj",
        "model.layers.2.self_attn.o_proj",
    }
    for conditioned in wrapped.values():
        assert conditioned.depths == 3
        assert conditioned.to_receipt()["identity_at_init"] is True


def test_wrapping_an_unprepared_model_fails_closed():
    args = ModelArgs(
        model_type="qwen2", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=64, num_key_value_heads=2, max_position_embeddings=128,
        rope_theta=10000.0,
    )
    with pytest.raises(RuntimeError, match="wrap the recurrent window"):
        wrap_depth_conditioned(Model(args), depths=2)


def test_trainable_parameters_are_exposed_per_depth():
    conditioned = DepthConditionedLoRA(_scoped(), depths=3)
    parameters = conditioned.trainable()
    assert len(parameters) == 6
    assert "depth_2.lora_b" in parameters


def test_invalid_configuration_is_refused():
    scoped = _scoped()
    with pytest.raises(ValueError, match="depths"):
        DepthConditionedLoRA(scoped, depths=0)
    with pytest.raises(ValueError, match="delta_scale"):
        DepthConditionedLoRA(scoped, depths=2, delta_scale=99.0)
    with pytest.raises(ValueError, match="non-negative"):
        DepthConditionedLoRA(scoped, depths=2).factors_for(-1)


def test_a_half_trained_delta_is_still_exactly_identity():
    """dW = A @ B, so a nonzero A with a zero B contributes nothing. The
    receipt must say identity rather than claiming a live delta."""
    scoped = _scoped()
    conditioned = DepthConditionedLoRA(scoped, depths=2)
    conditioned.depth_a[1] = mx.ones_like(scoped.lora_a) * 0.5
    assert conditioned.is_identity_at(1)
    assert conditioned.differentiation()[1] == 0.0
    a, b = conditioned.factors_for(1)
    delta = (mx.ones((1, 4, a.shape[0])) @ a) @ b
    assert float(mx.max(mx.abs(delta))) == 0.0
