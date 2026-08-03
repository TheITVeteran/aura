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
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
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
    strong = DepthConditionedLoRA(_scoped(), depths=2, delta_scale=1.0)
    weak = DepthConditionedLoRA(_scoped(), depths=2, delta_scale=0.1)
    for conditioned in (strong, weak):
        conditioned.depth_a[1] = mx.ones_like(conditioned.scoped.lora_a) * 0.5
        conditioned.depth_b[1] = mx.ones_like(conditioned.scoped.lora_b) * 0.5
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


def test_wrapped_depth_tensors_are_optimizer_and_checkpoint_visible():
    model = _model()
    wrap_depth_conditioned(model, depths=3)

    names = {name for name, _value in tree_flatten(model.trainable_parameters())}

    for layer_index in (1, 2):
        prefix = f"model.layers.{layer_index}.self_attn.o_proj"
        assert f"{prefix}.lora_a" in names
        assert f"{prefix}.lora_b" in names
        for depth in range(3):
            assert f"{prefix}.depth_a.{depth}" in names
            assert f"{prefix}.depth_b.{depth}" in names


def test_optimizer_updates_only_the_executed_depth_bank():
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    model = _model()
    wrap_depth_conditioned(model, depths=3)
    projection = model.model.layers[1].self_attn.o_proj
    # Make both factor gradients observable on the first measured update.
    projection.lora_b = mx.ones_like(projection.lora_b) * 0.1
    before_a = [value + 0 for value in projection.depth_a]
    before_b = [value + 0 for value in projection.depth_b]
    mx.eval(before_a, before_b)
    inputs = mx.ones((1, 4, projection.lora_a.shape[0]))

    def loss_fn(active_model):
        active = active_model.model.layers[1].self_attn.o_proj
        with recurrence_adapter_scope(start=0, stop=4):
            with recurrent_depth_index(1):
                output = active(inputs)
        return mx.mean(output * output)

    value, gradients = nn.value_and_grad(model, loss_fn)(model)
    flat_gradients = dict(tree_flatten(gradients))
    mx.eval(value, gradients)
    prefix = "model.layers.1.self_attn.o_proj"
    assert float(mx.max(mx.abs(flat_gradients[f"{prefix}.depth_a.1"]))) > 0.0
    assert float(mx.max(mx.abs(flat_gradients[f"{prefix}.depth_b.1"]))) > 0.0
    for depth in (0, 2):
        assert float(mx.max(mx.abs(flat_gradients[f"{prefix}.depth_a.{depth}"]))) == 0.0
        assert float(mx.max(mx.abs(flat_gradients[f"{prefix}.depth_b.{depth}"]))) == 0.0

    optimizer = optim.SGD(learning_rate=0.01)
    optimizer.update(model, gradients)
    mx.eval(model.trainable_parameters())
    assert not bool(mx.array_equal(projection.depth_a[1], before_a[1]))
    assert not bool(mx.array_equal(projection.depth_b[1], before_b[1]))
    for depth in (0, 2):
        assert bool(mx.array_equal(projection.depth_a[depth], before_a[depth]))
        assert bool(mx.array_equal(projection.depth_b[depth], before_b[depth]))


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


# ── Wired end-to-end, not present in name only ──────────────────────────


def test_forward_actually_consults_the_bank_at_each_depth():
    """CP211's lesson: a mechanism nothing reads is decoration. The
    projection's OUTPUT must change with the recurrent step."""
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    model = _model()
    banks = wrap_depth_conditioned(model, depths=3)
    bank = banks["model.layers.1.self_attn.o_proj"]
    projection = model.model.layers[1].self_attn.o_proj
    # Give depth 2 a live delta (BOTH factors, or dW = A@B is zero).
    bank.depth_a[2] = mx.ones_like(projection.lora_a) * 0.4
    bank.depth_b[2] = mx.ones_like(projection.lora_b) * 0.4

    x = mx.ones((1, 4, projection.lora_a.shape[0]))
    with recurrence_adapter_scope(start=0, stop=4):
        with recurrent_depth_index(0):
            at_zero = projection(x)
        with recurrent_depth_index(2):
            at_two = projection(x)
    assert not bool(mx.allclose(at_zero, at_two)), (
        "the effective operator must differ by depth, or depth "
        "conditioning is present in name only"
    )


def test_untrained_bank_leaves_the_forward_bit_identical():
    """Zero-initialized deltas must not perturb a working adapter."""
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    model = _model()
    projection = model.model.layers[1].self_attn.o_proj
    x = mx.ones((1, 4, projection.lora_a.shape[0]))
    with recurrence_adapter_scope(start=0, stop=4):
        before = projection(x)
    wrap_depth_conditioned(model, depths=4)
    with recurrence_adapter_scope(start=0, stop=4):
        with recurrent_depth_index(3):
            after = projection(x)
    assert bool(mx.allclose(before, after))
