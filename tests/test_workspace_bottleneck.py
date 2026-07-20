"""Force reliance on the workspace during training (CP224).

Measured: slots are causal (6/6) yet depth is flat. The model READS the
workspace without ROUTING reasoning through it, because solving from the
fully-visible prompt was never made costly. This masks prompt positions
during training only, so some examples are answerable only from what the
recurrence deposited.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.learning.workspace_bottleneck import (  # noqa: E402
    BottleneckSchedule,
    apply_bottleneck,
    prompt_visibility_mask,
    reliance_score,
)


def test_schedule_starts_transparent_and_ramps():
    """Asking a model to rely on an untrained workspace from step one
    teaches only that the task is impossible."""
    schedule = BottleneckSchedule(start_fraction=0.0, end_fraction=0.5, warmup_steps=100)
    assert schedule.fraction_at(0) == 0.0
    assert schedule.fraction_at(50) == pytest.approx(0.25)
    assert schedule.fraction_at(100) == pytest.approx(0.5)
    assert schedule.fraction_at(10_000) == pytest.approx(0.5), "must clamp"


def test_the_question_itself_is_never_hidden():
    """Masking the tail turns a harder task into an impossible one, and a
    model cannot learn to route through the workspace by guessing."""
    mask = prompt_visibility_mask(100, fraction=0.9, seed=7, protected_tail=24)
    assert float(mx.sum(mask[0, -24:, 0])) == 24.0
    assert float(mx.sum(mask)) < 100.0


def test_masking_is_deterministic_in_the_seed():
    a = prompt_visibility_mask(80, fraction=0.5, seed=3)
    b = prompt_visibility_mask(80, fraction=0.5, seed=3)
    c = prompt_visibility_mask(80, fraction=0.5, seed=4)
    assert bool(mx.array_equal(a, b)), "a resumed step must mask identically"
    assert not bool(mx.array_equal(a, c))


def test_geometry_is_preserved_so_only_information_is_removed():
    """Positions are zeroed, not removed: same shapes, same RoPE offsets,
    less information -- which is the intended pressure."""
    hidden = mx.ones((1, 64, 16))
    masked, receipt = apply_bottleneck(hidden, fraction=0.5, seed=1, protected_tail=8)
    assert masked.shape == hidden.shape
    assert receipt["hidden_positions"] > 0
    assert receipt["kept_positions"] + receipt["hidden_positions"] == 64
    assert 0.0 < receipt["effective_fraction"] < 1.0


def test_zero_fraction_is_the_unmodified_objective():
    hidden = mx.random.normal((1, 40, 16), key=mx.random.key(2))
    masked, receipt = apply_bottleneck(hidden, fraction=0.0, seed=1)
    assert bool(mx.array_equal(masked, hidden))
    assert receipt["hidden_positions"] == 0


def test_reliance_is_a_magnitude_not_a_yes_or_no():
    """'The slots are causal' is binary and was already true while depth
    stayed flat. This is the number the bottleneck exists to raise."""
    weak = reliance_score(intact_loss=1.0, ablated_loss=1.02)
    strong = reliance_score(intact_loss=1.0, ablated_loss=1.9)
    assert weak["workspace_load_bearing"] is False
    assert strong["workspace_load_bearing"] is True
    assert strong["reliance"] > weak["reliance"]


def test_invalid_configuration_fails_closed():
    with pytest.raises(ValueError, match="end_fraction"):
        BottleneckSchedule(start_fraction=0.5, end_fraction=0.1)
    with pytest.raises(ValueError, match="start_fraction"):
        BottleneckSchedule(start_fraction=1.0)
    with pytest.raises(ValueError, match="warmup_steps"):
        BottleneckSchedule(warmup_steps=-1)
    with pytest.raises(ValueError, match="fraction"):
        prompt_visibility_mask(10, fraction=1.0, seed=1)
    with pytest.raises(ValueError, match="prompt_length"):
        prompt_visibility_mask(0, fraction=0.5, seed=1)
    with pytest.raises(ValueError, match="intact_loss"):
        reliance_score(intact_loss=0.0, ablated_loss=1.0)
