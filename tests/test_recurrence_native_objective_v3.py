"""v3 objective contracts (CP181): positive depth advantage + real width.

The v2 hinge tolerated equality — recurrence that does nothing was
optimal; v3's margin hinge must demand a positive deep-over-shallow
advantage while still refusing to reward shallow damage. The v2
mean-answer CE collapsed virtual width — v3's diversity penalty must fire
on identical branch states, vanish on decorrelated ones, expose per-pair
cosines as telemetry, and remain differentiable end to end on the live
execution graph.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    LivePathForward,
    detached_monotonicity_penalty,
)
from core.learning.recurrence_native_objective_v3 import (  # noqa: E402
    branch_diversity_penalty,
    depth_curriculum_loss_v3,
    depth_margin_penalty,
)

PROMPT = [5, 9, 17, 3, 42]
ANSWER = [7, 11, 23]


def _model() -> Model:
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=4,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    model.freeze()
    for index in (1, 2):
        parent = model.model.layers[index].self_attn
        wrapped = ScopedLoRALinear.from_base(parent.o_proj, r=2, scale=1.0)
        wrapped.lora_a = mx.ones_like(wrapped.lora_a) * 0.03
        wrapped.lora_b = mx.ones_like(wrapped.lora_b) * 0.03
        parent.o_proj = wrapped
    mx.eval(model.parameters())
    return model


def _spec(**changes) -> RLCExecutionSpec:
    values = RLCExecutionSpec(
        n_slots=3,
        branch_roles=("constructive_solution", "counterexample_search"),
        recurrent_steps=2,
        exchange_interval=1,
    ).to_dict()
    values.update(changes)
    return RLCExecutionSpec.from_dict(values)


# ── Depth margin hinge ──────────────────────────────────────────────────


def test_equal_depths_now_pay_the_margin_v2_tolerated():
    equal = [mx.array(1.0), mx.array(1.0)]
    assert float(detached_monotonicity_penalty(equal)) == 0.0  # v2: free
    assert float(depth_margin_penalty(equal, margin=0.05)) == pytest.approx(
        0.05
    )  # v3: equality costs the full margin


def test_margin_releases_only_at_positive_advantage():
    clearing = [mx.array(1.0), mx.array(0.90)]
    assert float(depth_margin_penalty(clearing, margin=0.05)) == 0.0
    short = [mx.array(1.0), mx.array(0.97)]
    assert float(depth_margin_penalty(short, margin=0.05)) == pytest.approx(
        0.02, abs=1e-6
    )


def test_margin_hinge_cannot_reward_shallow_damage():
    def penalty_for_shallow(shallow):
        return depth_margin_penalty([shallow, mx.array(1.0)], margin=0.05)

    def penalty_for_deep(deep):
        return depth_margin_penalty([mx.array(1.0), deep], margin=0.05)

    shallow_grad = mx.grad(penalty_for_shallow)(mx.array(1.0))
    deep_grad = mx.grad(penalty_for_deep)(mx.array(1.0))
    assert float(shallow_grad) == 0.0  # damaging shallow buys nothing
    assert float(deep_grad) == 1.0  # improving deep releases the hinge


def test_margin_ladder_penalizes_each_flat_transition():
    ladder = [mx.array(1.0), mx.array(1.0), mx.array(1.0)]
    assert float(depth_margin_penalty(ladder, margin=0.1)) == pytest.approx(0.2)


# ── Branch diversity ────────────────────────────────────────────────────


def _forward_with_states(*states) -> LivePathForward:
    return LivePathForward(
        branch_logits=tuple(mx.zeros((1, 3, 8)) for _ in states),
        branch_states=tuple(states),
        exchanges=1,
        prompt_tokens=5,
        answer_tokens=3,
        bridge_tokens=0,
    )


def test_identical_states_pay_and_orthogonal_states_ride_free():
    base = mx.ones((1, 3, 8))
    collapsed = _forward_with_states(base, base)
    penalty, cosines = branch_diversity_penalty(collapsed, target_cos=0.98)
    assert float(penalty) > 0.0
    assert cosines == [pytest.approx(1.0)]

    orthogonal_a = mx.concatenate([mx.ones((1, 3, 4)), mx.zeros((1, 3, 4))], axis=-1)
    orthogonal_b = mx.concatenate([mx.zeros((1, 3, 4)), mx.ones((1, 3, 4))], axis=-1)
    decorrelated = _forward_with_states(orthogonal_a, orthogonal_b)
    penalty, cosines = branch_diversity_penalty(decorrelated, target_cos=0.98)
    assert float(penalty) == 0.0
    assert cosines == [pytest.approx(0.0, abs=1e-6)]


def test_single_branch_makes_no_diversity_demand():
    lone = _forward_with_states(mx.ones((1, 3, 8)))
    penalty, cosines = branch_diversity_penalty(lone)
    assert float(penalty) == 0.0 and cosines == []


def test_diversity_penalty_is_differentiable_toward_decorrelation():
    base = mx.ones((1, 3, 8))

    def penalty_for(state):
        forward = _forward_with_states(base, state)
        penalty, _ = branch_diversity_penalty(forward, target_cos=0.5)
        return penalty

    # Exactly parallel states sit at cosine's stationary point; probe from a
    # slightly non-parallel state, where decorrelation pressure must exist.
    probe = mx.ones((1, 3, 8))
    probe = mx.concatenate([probe[..., :7], probe[..., 7:] * 0.5], axis=-1)
    gradient = mx.grad(penalty_for)(probe)
    assert float(mx.linalg.norm(mx.reshape(gradient, (-1,)))) > 0.0


# ── End-to-end composite on the live execution graph ────────────────────


def test_depth_curriculum_v3_is_finite_learnable_and_telemetered():
    model = _model()
    spec = _spec()

    def loss_for_b(lora_b):
        wrapped = model.model.layers[1].self_attn.o_proj
        wrapped.lora_b = lora_b
        loss, _telemetry = depth_curriculum_loss_v3(
            model,
            PROMPT,
            ANSWER,
            spec=spec,
            depths=(1, 2),
            monotonicity_weight=0.5,
            depth_margin=0.05,
            diversity_weight=0.25,
        )
        return loss

    wrapped = model.model.layers[1].self_attn.o_proj
    value, gradient = mx.value_and_grad(loss_for_b)(wrapped.lora_b)
    assert bool(mx.isfinite(value))
    assert float(mx.linalg.norm(mx.reshape(gradient, (-1,)))) > 0.0

    loss, telemetry = depth_curriculum_loss_v3(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        depths=(1, 2),
        depth_margin=0.05,
        diversity_weight=0.25,
    )
    assert bool(mx.isfinite(loss))
    assert telemetry["schema"].endswith(".v3")
    assert len(telemetry["depth_losses"]) == 2
    assert set(telemetry["post_exchange_pairwise_cos"]) == {"1", "2"}
    for cosines in telemetry["post_exchange_pairwise_cos"].values():
        assert len(cosines) == 1  # exactly one pair for two branches
        assert -1.0 <= cosines[0] <= 1.0
    assert telemetry["margin_penalty"] >= 0.0
    assert telemetry["diversity_penalty"] >= 0.0


def test_v3_validation_rejects_bad_inputs():
    model = _model()
    with pytest.raises(ValueError, match="strictly increasing"):
        depth_curriculum_loss_v3(
            model, PROMPT, ANSWER, spec=_spec(), depths=(2, 1)
        )
    with pytest.raises(ValueError, match="depth_margin"):
        depth_margin_penalty([mx.array(1.0), mx.array(1.0)], margin=-0.1)
    with pytest.raises(ValueError, match="diversity_target_cos"):
        branch_diversity_penalty(
            _forward_with_states(mx.ones((1, 3, 8)), mx.ones((1, 3, 8))),
            target_cos=1.5,
        )
