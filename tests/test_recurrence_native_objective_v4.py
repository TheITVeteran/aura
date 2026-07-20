"""v4 objective contracts (CP209): make virtual width survive training.

The live CP195 run collapsed branch width under v3 (cosine 0.9998). These
tests pin the three fixes: a separation metric that is conditioned where
cosine is not and excludes the deliberately-shared comm slot, a linear
hinge whose gradient does not vanish at collapse, and a softmin answer
loss that matches inference-time branch SELECTION instead of training
every branch into the same generalist.
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
)
from core.learning.recurrence_native_objective_v3 import (  # noqa: E402
    branch_diversity_penalty,
)
from core.learning.recurrence_native_objective_v4 import (  # noqa: E402
    adaptive_depth_loss,
    branch_decorrelation_penalty,
    depth_curriculum_loss_v4,
    load_balance_penalty,
    monotone_improvement_penalty,
    oscillation_penalty,
    pairwise_separations,
    softmin_answer_loss,
    trajectory_loss_v4,
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
        n_slots=4,
        branch_roles=("constructive_solution", "counterexample_search"),
        recurrent_steps=2,
        exchange_interval=1,
    ).to_dict()
    values.update(changes)
    return RLCExecutionSpec.from_dict(values)


def _forward(*states, logits=None) -> LivePathForward:
    return LivePathForward(
        branch_logits=tuple(
            logits[index] if logits else mx.zeros((1, 3, 8))
            for index in range(len(states))
        ),
        branch_states=tuple(states),
        exchanges=1,
        prompt_tokens=5,
        answer_tokens=3,
        bridge_tokens=0,
    )


# ── The v3 defect, pinned as evidence ───────────────────────────────────


def test_v3_penalty_was_negligible_at_the_observed_operating_point():
    """The live run sat at cosine 0.9998 with answer CE 0.263; v3's
    quadratic contributed 0.04% of the loss. v4's hinge must dominate it
    by orders of magnitude at the SAME geometry."""
    near_identical = mx.ones((1, 4, 16))
    perturbed = mx.array(near_identical)
    perturbed = perturbed + 0.01 * mx.concatenate(
        [mx.zeros((1, 4, 15)), mx.ones((1, 4, 1))], axis=-1
    )
    forward = _forward(near_identical, perturbed)

    v3_penalty, v3_cosines = branch_diversity_penalty(forward, target_cos=0.98)
    v4_penalty, v4_separations = branch_decorrelation_penalty(
        forward, target_separation=0.30, comm_slot=0
    )

    assert v3_cosines[0] > 0.99, "fixture must reproduce near-collapse"
    # v3's quadratic is worth essentially nothing against a ~0.26 CE.
    assert float(v3_penalty) < 1e-3
    # v4's hinge is a real fraction of the loss at the same geometry.
    assert float(v4_penalty) > 0.2
    assert float(v4_penalty) > 100 * float(v3_penalty)
    assert v4_separations[0] < 0.05  # collapsed, and legible as such


def test_separation_is_better_conditioned_than_cosine_near_collapse():
    """Cosine compresses the entire collapse region into [0.98, 1.0];
    separation spreads it, which is what gives the optimizer signal."""

    def pair(scale: float):
        base = mx.ones((1, 4, 16))
        other = base + scale * mx.random.normal((1, 4, 16), key=mx.random.key(0))
        return _forward(base, other)

    tight = pairwise_separations(pair(0.01), comm_slot=0)[0]
    loose = pairwise_separations(pair(0.30), comm_slot=0)[0]
    assert float(tight) < float(loose)
    # The metric resolves differences cosine would round away.
    assert float(loose) / max(float(tight), 1e-9) > 5.0


def test_hinge_gradient_does_not_vanish_at_collapse():
    """v3's quadratic gradient decays to zero exactly at collapse; v4's
    linear hinge holds constant pressure until the target is met."""

    def v4_penalty_for(state):
        penalty, _ = branch_decorrelation_penalty(
            _forward(mx.ones((1, 4, 16)), state),
            target_separation=0.30,
            comm_slot=0,
        )
        return penalty

    def v3_penalty_for(state):
        penalty, _ = branch_diversity_penalty(
            _forward(mx.ones((1, 4, 16)), state), target_cos=0.98
        )
        return penalty

    collapsed = mx.ones((1, 4, 16)) * 1.001
    v4_grad = mx.grad(v4_penalty_for)(collapsed)
    v3_grad = mx.grad(v3_penalty_for)(collapsed)
    v4_norm = float(mx.linalg.norm(mx.reshape(v4_grad, (-1,))))
    v3_norm = float(mx.linalg.norm(mx.reshape(v3_grad, (-1,))))
    assert v4_norm > 0.0
    assert v4_norm > 10 * v3_norm


def test_comm_slot_is_excluded_because_exchange_shares_it_by_design():
    """Slot 0 is dragged to consensus by _exchange_and_decorrelate.
    Counting it as collapse inflated every v3 reading."""
    left = mx.concatenate(
        [mx.ones((1, 1, 8)), mx.zeros((1, 1, 8))], axis=1
    )  # comm slot shared, thought slot distinct
    right = mx.concatenate([mx.ones((1, 1, 8)), mx.ones((1, 1, 8))], axis=1)

    with_comm = pairwise_separations(_forward(left, right), comm_slot=99)[0]
    without_comm = pairwise_separations(_forward(left, right), comm_slot=0)[0]
    assert float(without_comm) > float(with_comm)


# ── Selection-matched training ──────────────────────────────────────────


def _logits_favoring(branch_index: int, answer):
    """Logits where the chosen branch predicts the answer confidently."""
    logits = []
    for index in range(2):
        base = mx.zeros((1, len(answer), 8))
        if index == branch_index:
            hot = mx.zeros((1, len(answer), 8))
            for position, token in enumerate(answer):
                column = mx.zeros((8,))
                column = mx.where(
                    mx.arange(8) == (token % 8), mx.full((8,), 10.0), column
                )
                hot = mx.concatenate(
                    [
                        hot[:, :position, :],
                        column[None, None, :],
                        hot[:, position + 1 :, :],
                    ],
                    axis=1,
                )
            base = hot
        logits.append(base)
    return logits


def test_softmin_concentrates_gradient_on_the_best_branch():
    answer = [1, 2, 3]
    forward = _forward(
        mx.ones((1, 4, 16)),
        mx.zeros((1, 4, 16)),
        logits=_logits_favoring(1, answer),
    )
    loss, branch_losses, weights = softmin_answer_loss(
        forward, answer, temperature=0.5
    )
    assert branch_losses[1] < branch_losses[0], "branch 1 must be the better arm"
    assert weights[1] > weights[0], "selection weight must follow quality"
    assert weights[1] > 0.9
    # The composite tracks the WINNER, not the average of the two.
    assert float(loss) < sum(branch_losses) / 2


def test_softmin_temperature_spans_best_of_and_mean():
    answer = [1, 2, 3]
    forward = _forward(
        mx.ones((1, 4, 16)),
        mx.zeros((1, 4, 16)),
        logits=_logits_favoring(1, answer),
    )
    _hard, losses, hard_weights = softmin_answer_loss(
        forward, answer, temperature=0.01
    )
    _soft, _losses, soft_weights = softmin_answer_loss(
        forward, answer, temperature=10.0
    )
    assert max(hard_weights) > 0.99  # approaches hard best-of
    assert abs(soft_weights[0] - soft_weights[1]) < 0.2  # approaches the mean
    assert len(losses) == 2


def test_single_branch_makes_no_diversity_or_balance_demand():
    lone = _forward(mx.ones((1, 4, 16)))
    penalty, separations = branch_decorrelation_penalty(lone)
    assert float(penalty) == 0.0 and separations == []
    assert float(load_balance_penalty([1.0], 1)) == 0.0


def test_load_balance_penalizes_starved_branches():
    balanced = load_balance_penalty([0.5, 0.5], 2)
    starved = load_balance_penalty([0.99, 0.01], 2)
    assert float(starved) > float(balanced)
    assert float(balanced) == pytest.approx(0.0, abs=1e-9)


# ── End-to-end on the live execution graph ──────────────────────────────


def test_v4_composite_is_finite_learnable_and_telemetered():
    model = _model()
    spec = _spec()

    def loss_for_b(lora_b):
        model.model.layers[1].self_attn.o_proj.lora_b = lora_b
        loss, _telemetry = depth_curriculum_loss_v4(
            model, PROMPT, ANSWER, spec=spec, depths=(1, 2)
        )
        return loss

    wrapped = model.model.layers[1].self_attn.o_proj
    value, gradient = mx.value_and_grad(loss_for_b)(wrapped.lora_b)
    assert bool(mx.isfinite(value))
    assert float(mx.linalg.norm(mx.reshape(gradient, (-1,)))) > 0.0

    loss, telemetry = depth_curriculum_loss_v4(
        model, PROMPT, ANSWER, spec=spec, depths=(1, 2)
    )
    assert bool(mx.isfinite(loss))
    assert telemetry["schema"].endswith(".v4")
    assert set(telemetry["pairwise_separation"]) == {"1", "2"}
    for depth in ("1", "2"):
        assert len(telemetry["pairwise_separation"][depth]) == 1
        assert len(telemetry["branch_losses"][depth]) == 2
        assert sum(telemetry["branch_weights"][depth]) == pytest.approx(
            1.0, abs=1e-5
        )
    assert telemetry["diversity_penalty"] >= 0.0
    assert telemetry["load_balance_penalty"] >= 0.0


def test_v4_rejects_bad_inputs():
    model = _model()
    with pytest.raises(ValueError, match="strictly increasing"):
        depth_curriculum_loss_v4(
            model, PROMPT, ANSWER, spec=_spec(), depths=(2, 1)
        )
    with pytest.raises(ValueError, match="target_separation"):
        branch_decorrelation_penalty(
            _forward(mx.ones((1, 4, 8)), mx.ones((1, 4, 8))),
            target_separation=99.0,
        )
    with pytest.raises(ValueError, match="temperature"):
        softmin_answer_loss(
            _forward(mx.ones((1, 4, 8)), mx.ones((1, 4, 8))),
            ANSWER,
            temperature=0.0,
        )


# ── Adaptive depth: the fix for the family-conflict collapse ────────────


def test_adaptive_depth_selects_deep_for_khop_shaped_losses():
    """khop measured -33% CE from depth 1->8. Depth must be selected."""
    losses = [mx.array(2.745), mx.array(2.30), mx.array(2.05), mx.array(1.838)]
    loss, priced, selected = adaptive_depth_loss(
        losses, (1, 2, 4, 8), compute_price=0.01, temperature=0.15
    )
    assert selected == 8
    # Softmin is a SOFT minimum: it sits just above min by construction.
    # The property that matters is that it tracks the winner, not the mean.
    assert min(priced) <= float(loss) < min(priced) + 0.15
    assert float(loss) < sum(priced) / len(priced) - 0.2


def test_adaptive_depth_selects_shallow_for_modular_shaped_losses():
    """modular measured +15% CE from depth 1->8 — anti-monotone in every
    sample. The old hinge demanded improvement here and could never get
    it; adaptive selection simply stays shallow."""
    losses = [mx.array(2.702), mx.array(2.88), mx.array(2.95), mx.array(3.110)]
    loss, priced, selected = adaptive_depth_loss(
        losses, (1, 2, 4, 8), compute_price=0.01, temperature=0.15
    )
    assert selected == 1
    assert min(priced) <= float(loss) < min(priced) + 0.15
    assert float(loss) < sum(priced) / len(priced)


def test_compute_price_breaks_ties_toward_less_computation():
    flat = [mx.array(2.0), mx.array(2.0), mx.array(2.0)]
    _loss, _priced, selected = adaptive_depth_loss(
        flat, (1, 2, 4), compute_price=0.05
    )
    assert selected == 1, "equal accuracy must not buy extra compute"


def test_adaptive_depth_gradient_reaches_the_selected_depth():
    def loss_for(deep_value):
        losses = [mx.array(3.0), deep_value]
        loss, _priced, _selected = adaptive_depth_loss(
            losses, (1, 8), compute_price=0.0, temperature=0.05
        )
        return loss

    gradient = mx.grad(loss_for)(mx.array(1.0))
    assert float(gradient) > 0.9, "winning depth must carry the gradient"


def test_adaptive_depth_validates_inputs():
    with pytest.raises(ValueError, match="align"):
        adaptive_depth_loss([mx.array(1.0)], (1, 2))
    with pytest.raises(ValueError, match="compute_price"):
        adaptive_depth_loss(
            [mx.array(1.0), mx.array(1.0)], (1, 2), compute_price=5.0
        )


# ── Trajectory objective: thinking vs spinning ──────────────────────────


def test_monotone_improvement_penalizes_a_degrading_trajectory():
    """The measured failure: CE peaks at step 2 then degrades monotonically."""
    degrading = [mx.array(v) for v in (1.92, 2.04, 2.08, 2.12, 2.15)]
    improving = [mx.array(v) for v in (2.15, 2.02, 1.94, 1.88, 1.80)]
    # mean over 4 transitions of relu(delta + margin): (0.14+0.06+0.06+0.05)/4
    assert float(monotone_improvement_penalty(degrading, margin=0.02)) == pytest.approx(
        0.0775, abs=1e-4
    )
    # A trajectory that actually improves clears the hinge entirely.
    assert float(monotone_improvement_penalty(improving, margin=0.02)) == 0.0


def test_improvement_cannot_be_won_by_damaging_earlier_steps():
    """Previous step is gradient-detached: the only way through the hinge
    is to make LATER steps genuinely better."""

    def penalty_for_earlier(earlier):
        return monotone_improvement_penalty(
            [earlier, mx.array(2.0)], margin=0.02
        )

    def penalty_for_later(later):
        return monotone_improvement_penalty(
            [mx.array(2.0), later], margin=0.02
        )

    assert float(mx.grad(penalty_for_earlier)(mx.array(2.0))) == 0.0
    assert float(mx.grad(penalty_for_later)(mx.array(2.0))) == 1.0


def test_oscillation_penalty_targets_the_measured_period_2_cycle():
    """Ping-pong (anti-correlated deltas) is penalized; consistent motion
    in one direction rides free."""
    base = mx.ones((1, 4, 8))
    step = 0.1 * mx.ones((1, 4, 8))
    ping_pong = [base, base + step, base, base + step, base]
    advancing = [base + float(index) * step for index in range(5)]

    osc_penalty, osc_cos = oscillation_penalty(ping_pong)
    adv_penalty, adv_cos = oscillation_penalty(advancing)

    assert float(osc_penalty) > 0.9  # deltas are anti-parallel
    assert all(value < -0.9 for value in osc_cos)
    assert float(adv_penalty) == 0.0
    assert all(value > 0.9 for value in adv_cos)


def test_oscillation_needs_three_states():
    penalty, cosines = oscillation_penalty([mx.ones((1, 2, 4))])
    assert float(penalty) == 0.0 and cosines == []


def test_trajectory_loss_scores_every_probed_step_and_reports_the_curve():
    model = _model()
    spec = _spec(branch_roles=("constructive_solution",))
    loss, telemetry = trajectory_loss_v4(
        model, PROMPT, ANSWER, spec=spec, depth=4
    )
    assert bool(mx.isfinite(loss))
    assert telemetry["probed_steps"] == 4
    assert len(telemetry["step_losses"]) == 4
    assert len(telemetry["delta_cosines"]) == 2
    assert 0 <= telemetry["best_step_index"] < 4
    assert telemetry["improvement_penalty"] >= 0.0
    assert telemetry["oscillation_penalty"] >= 0.0


def test_trajectory_loss_is_learnable_through_the_recurrent_adapter():
    model = _model()
    spec = _spec(branch_roles=("constructive_solution",))

    def loss_for_b(lora_b):
        model.model.layers[1].self_attn.o_proj.lora_b = lora_b
        loss, _telemetry = trajectory_loss_v4(
            model, PROMPT, ANSWER, spec=spec, depth=3
        )
        return loss

    wrapped = model.model.layers[1].self_attn.o_proj
    gradient = mx.grad(loss_for_b)(wrapped.lora_b)
    assert float(mx.linalg.norm(mx.reshape(gradient, (-1,)))) > 0.0


def test_probe_steps_subset_bounds_cost_on_large_models():
    model = _model()
    spec = _spec(branch_roles=("constructive_solution",))
    _loss, telemetry = trajectory_loss_v4(
        model, PROMPT, ANSWER, spec=spec, depth=8, probe_steps=(1, 4, 8)
    )
    assert telemetry["probed_steps"] == 3
    with pytest.raises(ValueError, match="probe_steps"):
        trajectory_loss_v4(
            model, PROMPT, ANSWER, spec=spec, depth=4, probe_steps=(1, 9)
        )
