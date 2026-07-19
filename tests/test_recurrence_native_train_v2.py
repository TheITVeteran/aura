"""Trainer-v2 determinism, composition, and exact-resume contracts."""

from __future__ import annotations

from argparse import Namespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402
from mlx_lm.tuner.lora import LoRALinear  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
    recurrence_adapter_scope,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    depth_curriculum_loss_v2,
)
from core.learning.recurrence_training_state import (  # noqa: E402
    load_recurrence_checkpoint,
    save_recurrence_checkpoint,
)
from tools.recurrence_native_train_v2 import (  # noqa: E402
    _deterministic_order,
    _run,
    _streamed_depth_value_and_grad,
    _wrap_window_layers,
)


def _model(seed: int = 123) -> Model:
    mx.random.seed(seed)
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=16,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _prepared(seed: int = 123):
    model = _model(seed)
    wrapped = _wrap_window_layers(
        model,
        rank=2,
        targets=("o_proj",),
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    optimizer = optim.AdamW(learning_rate=1e-3)
    optimizer.init(model.trainable_parameters())
    return model, optimizer, wrapped


def _step(model, optimizer):
    hidden = mx.ones((1, 3, 16))

    def loss_fn(mdl):
        total = mx.zeros(())
        with recurrence_adapter_scope():
            for index in (1, 2):
                projected = mdl.model.layers[index].self_attn.o_proj(hidden)
                total = total + mx.mean(mx.square(projected))
        return total

    value, gradients = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, gradients)
    mx.eval(value, model.trainable_parameters(), optimizer.state)
    return float(value)


def test_epoch_order_is_stateless_deterministic_and_complete():
    first = _deterministic_order(23, 1777, 4)
    assert first == _deterministic_order(23, 1777, 4)
    assert sorted(first) == list(range(23))
    assert first != _deterministic_order(23, 1777, 5)


def test_window_wrapper_is_scoped_and_only_lora_is_trainable():
    model, _optimizer, wrapped = _prepared()
    assert wrapped == [
        "model.layers.1.self_attn.o_proj",
        "model.layers.2.self_attn.o_proj",
    ]
    assert all(
        isinstance(model.model.layers[index].self_attn.o_proj, ScopedLoRALinear)
        for index in (1, 2)
    )
    keys = [key for key, _value in tree_flatten(model.trainable_parameters())]
    assert keys
    assert all(key.endswith((".lora_a", ".lora_b")) for key in keys)


def test_fresh_adapter_initialization_is_reproducible_from_training_seed():
    first = _model()
    second = _model()
    mx.random.seed(1777)
    _wrap_window_layers(
        first,
        rank=2,
        targets=("o_proj",),
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    mx.random.seed(1777)
    _wrap_window_layers(
        second,
        rank=2,
        targets=("o_proj",),
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    first_params = dict(tree_flatten(first.trainable_parameters()))
    second_params = dict(tree_flatten(second.trainable_parameters()))
    assert set(first_params) == set(second_params)
    assert all(
        bool(mx.array_equal(first_params[key], second_params[key]))
        for key in first_params
    )


def test_scoped_adapter_composes_over_frozen_personality_lora():
    model = _model()
    parent = model.model.layers[1].self_attn
    personality = LoRALinear.from_base(parent.o_proj, r=2)
    personality.lora_b = mx.ones_like(personality.lora_b) * 0.01
    parent.o_proj = personality
    wrapped = _wrap_window_layers(
        model,
        rank=2,
        targets=("o_proj",),
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    scoped = model.model.layers[1].self_attn.o_proj
    assert wrapped
    assert isinstance(scoped, ScopedLoRALinear)
    assert isinstance(scoped.linear, LoRALinear)
    keys = [key for key, _value in tree_flatten(model.trainable_parameters())]
    assert all(".linear.lora_" not in key for key in keys)


def test_resume_matches_uninterrupted_adapter_and_optimizer_state(tmp_path):
    continuous, continuous_optimizer, _ = _prepared()
    resumed_source, resumed_optimizer, _ = _prepared()
    _step(continuous, continuous_optimizer)
    _step(continuous, continuous_optimizer)

    _step(resumed_source, resumed_optimizer)
    identities = {
        "config_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "execution_spec_sha256": "c" * 64,
    }
    save_recurrence_checkpoint(
        tmp_path / "run",
        adapter_tensors=dict(tree_flatten(resumed_source.trainable_parameters())),
        optimizer_tensors=dict(tree_flatten(resumed_optimizer.state)),
        state={**identities, "step": 1, "epoch": 0, "cursor": 1, "order": [0]},
    )
    loaded = load_recurrence_checkpoint(
        tmp_path / "run",
        **{f"expected_{key}": value for key, value in identities.items()},
    )
    resumed, optimizer_after_resume, _ = _prepared()
    resumed.load_weights(list(loaded.adapter_tensors.items()), strict=False)
    optimizer_after_resume.state = loaded.optimizer_state
    optimizer_after_resume.init(resumed.trainable_parameters())
    _step(resumed, optimizer_after_resume)

    continuous_params = dict(tree_flatten(continuous.trainable_parameters()))
    resumed_params = dict(tree_flatten(resumed.trainable_parameters()))
    assert set(continuous_params) == set(resumed_params)
    assert all(
        bool(mx.array_equal(continuous_params[key], resumed_params[key]))
        for key in continuous_params
    )
    continuous_state = dict(tree_flatten(continuous_optimizer.state))
    resumed_state = dict(tree_flatten(optimizer_after_resume.state))
    assert set(continuous_state) == set(resumed_state)
    assert all(
        bool(mx.array_equal(continuous_state[key], resumed_state[key]))
        for key in continuous_state
    )


def test_streamed_depth_gradient_matches_monolithic_objective():
    monolithic, _monolithic_optimizer, _ = _prepared(seed=811)
    streamed, _streamed_optimizer, _ = _prepared(seed=811)
    prompt = [5, 9, 17]
    answer = [7, 11]
    spec = RLCExecutionSpec(
        n_slots=2,
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
        exchange_interval=1,
    )
    depths = (1, 2)
    weight = 0.5

    def loss_fn(model):
        return depth_curriculum_loss_v2(
            model,
            prompt,
            answer,
            spec=spec,
            depths=depths,
            monotonicity_weight=weight,
        )

    expected_value, expected_gradients = nn.value_and_grad(
        monolithic, loss_fn
    )(monolithic)
    actual_value, actual_gradients = _streamed_depth_value_and_grad(
        streamed,
        prompt,
        answer,
        spec=spec,
        depths=depths,
        monotonicity_weight=weight,
    )
    mx.eval(expected_value, expected_gradients, actual_gradients)
    assert actual_value == pytest.approx(float(expected_value), rel=1e-5, abs=1e-5)
    expected = dict(tree_flatten(expected_gradients))
    actual = dict(tree_flatten(actual_gradients))
    assert set(actual) == set(expected)
    for key in expected:
        assert bool(
            mx.allclose(actual[key], expected[key], rtol=1e-4, atol=1e-5)
        ), key


def test_run_requires_model_lane_before_any_load():
    with pytest.raises(RuntimeError, match="model-lane lease"):
        _run(Namespace(), model_lane_lease=object())
