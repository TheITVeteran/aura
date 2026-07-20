"""Training the model to survive and use its own recurrence (CP227).

CP226 measured the untrained retrofit collapsing the 32B to 0% answered by
T=8 -- not by numerical blowup but by output failure, localized to the coda
receiving states it was never trained to accept. These tests pin the three
design decisions that follow from that finding.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.intrinsic_recurrence import checkpointed_window  # noqa: E402
from core.learning.intrinsic_recurrence_objective import (  # noqa: E402
    IntrinsicTrainingSpec,
    adapted_layer_indices,
    answer_cross_entropy,
    depth_tolerance,
    intrinsic_depth_loss,
)

LAYERS = 8


def _model():
    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=LAYERS,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=256,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


PROMPT = mx.array([[3, 11, 42, 7]])
ANSWER = mx.array([[19, 5]])
SPEC = IntrinsicTrainingSpec(prelude_end=2, coda_start=6, depths=(1, 2, 4))


# ── The coda is what broke, so the coda gets adapted ────────────────────


def test_adaptation_covers_the_coda_not_just_the_window():
    """Training only the window would leave the component that actually
    collapsed in CP226 completely untouched."""
    indices = adapted_layer_indices(SPEC, LAYERS)
    assert set(range(2, 6)).issubset(indices), "window must be adapted"
    assert set(range(6, LAYERS)).issubset(indices), "coda must be adapted"
    assert 0 not in indices and 1 not in indices, "prelude runs once, unadapted"


def test_a_spec_with_no_coda_is_refused():
    spec = IntrinsicTrainingSpec(prelude_end=2, coda_start=8, depths=(1, 2))
    with pytest.raises(ValueError, match="no coda layers"):
        adapted_layer_indices(spec, LAYERS)


# ── T=1 anchors base ability ────────────────────────────────────────────


def test_depth_one_anchor_is_mandatory():
    """Without it the model can buy depth tolerance by getting worse, and
    the ladder would show that as progress."""
    with pytest.raises(ValueError, match="depth 1 must be present"):
        IntrinsicTrainingSpec(prelude_end=2, coda_start=6, depths=(2, 4))


def test_the_anchor_sits_outside_the_priced_selection():
    """Inside the softmin a model could win by making every depth equally
    bad; outside, base ability has to hold."""
    model = _model()
    loss, telemetry = intrinsic_depth_loss(model, PROMPT, ANSWER, SPEC)
    assert telemetry["anchor_ce"] > 0
    assert float(loss) == pytest.approx(
        telemetry["priced_ce"] + SPEC.anchor_weight * telemetry["anchor_ce"],
        rel=1e-5,
    )


# ── The loss flows through the answer's own deepened computation ────────


def test_loss_is_depth_sensitive():
    """The previous objectives scored an answer that traversed the window
    once regardless of depth, so no gradient could ever teach depth."""
    model = _model()
    shallow, _ = answer_cross_entropy(model, PROMPT, ANSWER, SPEC.plan_at(1))
    deep, _ = answer_cross_entropy(model, PROMPT, ANSWER, SPEC.plan_at(4))
    assert abs(float(shallow) - float(deep)) > 1e-4


def test_gradients_reach_window_and_coda():
    model = _model()

    def loss_fn():
        loss, _ = answer_cross_entropy(model, PROMPT, ANSWER, SPEC.plan_at(2))
        return loss

    import mlx.nn as nn

    grads = nn.value_and_grad(model, lambda m: loss_fn())(model)[1]
    layer_grads = grads["model"]["layers"]
    window_grad = float(mx.sum(mx.abs(layer_grads[3]["mlp"]["down_proj"]["weight"])))
    coda_grad = float(mx.sum(mx.abs(layer_grads[7]["mlp"]["down_proj"]["weight"])))
    assert window_grad > 0, "no gradient reached the recurrent window"
    assert coda_grad > 0, "no gradient reached the coda -- it cannot learn"


def test_telemetry_reports_per_depth_ce_not_just_a_scalar():
    """Outcome-only scoring is how a period-2 cycle and a fixed-point
    collapse both went unnoticed."""
    model = _model()
    _, telemetry = intrinsic_depth_loss(model, PROMPT, ANSWER, SPEC)
    assert set(telemetry["per_depth_ce"]) == {"T1", "T2", "T4"}
    assert telemetry["selected_depth"] in SPEC.depths
    assert len(telemetry["priced_costs"]) == 3


def test_selected_depth_is_the_depth_not_an_index():
    """adaptive_depth_loss returns the selected DEPTH. Treating it as an
    index mislabels the selection whenever it lands in range and only
    raises when it does not -- which reads as test-order flakiness."""
    from core.learning.recurrence_native_objective_v4 import adaptive_depth_loss

    losses = [mx.array(5.0), mx.array(1.0), mx.array(4.0)]
    _, _, selected = adaptive_depth_loss(losses, [1, 2, 4], compute_price=0.01)
    assert selected == 2, "the cheapest priced cost is at depth 2"

    model = _model()
    for _ in range(6):
        _, telemetry = intrinsic_depth_loss(_model(), PROMPT, ANSWER, SPEC)
        assert telemetry["selected_depth"] in SPEC.depths
    del model


# ── Reading whether the collapse is repaired ────────────────────────────


def test_collapse_is_detected_and_repair_is_recognized():
    collapsed = depth_tolerance({"T1": 1.0, "T2": 2.4, "T4": 9.0})
    assert collapsed["collapse_repaired"] is False
    assert collapsed["worst_relative_ce"] == pytest.approx(9.0)

    repaired = depth_tolerance({"T1": 1.0, "T2": 0.95, "T4": 0.90})
    assert repaired["collapse_repaired"] is True
    assert repaired["depth_helps"] is True


def test_flat_ladder_is_not_reported_as_depth_helping():
    flat = depth_tolerance({"T1": 1.0, "T2": 1.0, "T4": 1.0})
    assert flat["collapse_repaired"] is True
    assert flat["depth_helps"] is False, (
        "surviving depth is not the same as benefiting from it"
    )


# ── Checkpointing is real, not nominal ──────────────────────────────────


def test_checkpointing_preserves_the_loss_and_the_gradient():
    """A checkpoint that changed the answer would be a bug, not a memory
    saving."""
    import mlx.nn as nn

    model = _model()

    def compute():
        loss, _ = answer_cross_entropy(model, PROMPT, ANSWER, SPEC.plan_at(2))
        return loss

    plain_loss, plain_grads = nn.value_and_grad(model, lambda m: compute())(model)
    with checkpointed_window(model, group_size=2):
        ckpt_loss, ckpt_grads = nn.value_and_grad(model, lambda m: compute())(model)
    assert float(ckpt_loss) == pytest.approx(float(plain_loss), rel=1e-4)
    a = plain_grads["model"]["layers"][3]["mlp"]["down_proj"]["weight"]
    b = ckpt_grads["model"]["layers"][3]["mlp"]["down_proj"]["weight"]
    assert bool(mx.allclose(a, b, atol=1e-4)), "checkpointing changed the gradient"


def test_checkpointing_refuses_to_run_with_kv_caches():
    """Recomputing a checkpointed group would append its keys a second time
    and silently corrupt the attention history."""
    from core.learning.intrinsic_recurrence import (
        make_recurrent_caches,
        recurrent_hidden_states,
    )

    model = _model()
    plan = SPEC.plan_at(2)
    caches = make_recurrent_caches(model, plan)
    with checkpointed_window(model):
        with pytest.raises(ValueError, match="cannot be combined with KV caches"):
            recurrent_hidden_states(model, PROMPT, plan, caches=caches)
