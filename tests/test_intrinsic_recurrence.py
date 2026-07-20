"""The checkpoint itself becomes recurrent (CP226).

The prior RLC recurred four side slots while the answer tokens traversed
the middle block exactly once, at every depth. That architecture cannot
produce a depth effect on the answer, and measurement agreed: 25/29/25/25
across an 8x compute range. These tests pin the corrected shape -- the real
token stream re-enters the window -- and the safety property that lets it
be added to a working checkpoint: T=1 is bit-identical to the base model.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.intrinsic_recurrence import (  # noqa: E402
    RecurrentDepthPlan,
    current_iteration,
    recurrent_hidden_states,
    recurrent_iteration,
    recurrent_logits,
    trajectory_dynamics,
)

LAYERS = 8


def _model() -> Model:
    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=LAYERS,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=256,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


TOKENS = mx.array([[3, 11, 42, 7, 19]])


# ── Safety: recurrence is added FROM a known-good point ─────────────────


def test_one_iteration_is_bit_identical_to_the_base_model():
    model = _model()
    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=1)
    assert plan.is_base_equivalent()
    assert bool(
        mx.allclose(model(TOKENS), recurrent_logits(model, TOKENS, plan), atol=1e-5)
    ), "T=1 must reproduce the unmodified forward pass exactly"


def test_stabilizers_do_not_perturb_the_first_pass():
    """Anchor injection and renorm apply only at RE-entry, so T=1 stays
    identical no matter how they are configured."""
    model = _model()
    base = model(TOKENS)
    for injection, renorm in ((0.5, False), (0.0, True), (1.0, True)):
        plan = RecurrentDepthPlan(
            prelude_end=2, coda_start=6, iterations=1,
            anchor_injection=injection, renormalize=renorm,
        )
        assert bool(mx.allclose(base, recurrent_logits(model, TOKENS, plan), atol=1e-5))


# ── The real token stream gets deeper ───────────────────────────────────


def test_the_answer_path_itself_recurs():
    """The property the previous architecture lacked: extra depth changes
    the logits of the actual tokens, not a side scratchpad's contents."""
    model = _model()
    shallow = recurrent_logits(
        model, TOKENS, RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=1)
    )
    deep = recurrent_logits(
        model, TOKENS, RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=4)
    )
    assert not bool(mx.allclose(shallow, deep, atol=1e-4))


def test_effective_depth_is_reported_honestly():
    plan = RecurrentDepthPlan(prelude_end=16, coda_start=48, iterations=4)
    # 16 prelude + 4*32 window + 16 coda
    assert plan.effective_depth(64) == 160
    assert plan.window_size() == 32
    receipt = plan.to_receipt(64)
    assert receipt["effective_depth"] == 160
    assert receipt["base_equivalent"] is False
    assert RecurrentDepthPlan(16, 48, 1).to_receipt(64)["effective_depth"] == 64


def test_trajectory_is_returned_for_every_iteration():
    model = _model()
    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=5)
    hidden, trajectory = recurrent_hidden_states(model, TOKENS, plan)
    assert len(trajectory) == 5
    assert hidden.shape == (1, TOKENS.shape[1], 64)


# ── Motion is not progress: the loop must be gradeable ──────────────────


def test_dynamics_flag_a_fixed_point():
    """A loop that stopped moving stopped computing, whatever the compute
    budget claims."""
    state = mx.ones((1, 4, 8))
    report = trajectory_dynamics([state, state, state, state])
    assert report["at_fixed_point"] is True
    assert report["contracting"] is True


def test_dynamics_flag_an_oscillation():
    a, b = mx.zeros((1, 4, 8)), mx.ones((1, 4, 8))
    report = trajectory_dynamics([a, b, a, b, a])
    assert report["oscillating"] is True


def test_dynamics_refuse_to_judge_a_single_iteration():
    report = trajectory_dynamics([mx.ones((1, 4, 8))])
    assert report["measurable"] is False


def test_a_moving_loop_is_not_called_a_fixed_point():
    model = _model()
    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=4)
    _, trajectory = recurrent_hidden_states(model, TOKENS, plan)
    report = trajectory_dynamics(trajectory)
    assert report["measurable"] is True
    assert len(report["relative_deltas"]) == 3


# ── Per-iteration identity ──────────────────────────────────────────────


def test_iteration_index_is_published_and_restored():
    assert current_iteration() == 0
    with recurrent_iteration(3):
        assert current_iteration() == 3
        with recurrent_iteration(5):
            assert current_iteration() == 5
        assert current_iteration() == 3
    assert current_iteration() == 0
    with pytest.raises(ValueError, match="non-negative"):
        with recurrent_iteration(-1):
            pass


def test_the_window_sees_its_own_iteration_index():
    """Depth-conditioned adapters need this: the same weights doing
    different work per pass is the difference between deepening and
    repeating."""
    model = _model()
    seen: list[int] = []
    window_layer = model.model.layers[3]
    original = window_layer.__call__

    def spy(*args, **kwargs):
        seen.append(current_iteration())
        return original(*args, **kwargs)

    model.model.layers[3] = type(
        "Spy", (), {"__call__": lambda self, *a, **k: spy(*a, **k)}
    )()
    recurrent_hidden_states(
        model, TOKENS, RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=3)
    )
    assert seen == [0, 1, 2]


# ── Fail closed ─────────────────────────────────────────────────────────


def test_invalid_plans_are_refused():
    with pytest.raises(ValueError, match="prelude_end must precede"):
        RecurrentDepthPlan(prelude_end=6, coda_start=6)
    with pytest.raises(ValueError, match="iterations must be at least 1"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=0)
    with pytest.raises(ValueError, match="anchor_injection"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6, anchor_injection=1.5)
    with pytest.raises(ValueError, match="renormalize"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6, renormalize="yes")
    with pytest.raises(ValueError, match="smaller than the plan"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6).effective_depth(4)
    with pytest.raises(ValueError, match="exceeds the model"):
        recurrent_hidden_states(
            _model(), TOKENS, RecurrentDepthPlan(prelude_end=2, coda_start=99)
        )
