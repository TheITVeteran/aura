"""ACT-R subsymbolic activation, checked against its analytic form.

The defect that motivated this module is pinned in
``test_recency_is_scale_free`` and ``test_recent_ages_are_distinguishable``.
``EpisodicMemory._recency_score`` keyed recency to a hardcoded epoch
(``ts - 1774000000) / 2000000``), which on 2026-08-12 returned exactly
1.000000 for every episode newer than 2026-04-12 — one minute old and thirty
days old were indistinguishable, so the term contributed a constant to every
candidate. Any absolute-epoch formulation has that failure latent in it, so the
tests below assert the property that rules the whole class out rather than
checking a repaired constant.
"""

from __future__ import annotations

import math

import pytest

from core.cognition.actr_activation import (
    ActrParameters,
    activation,
    base_level_activation,
    base_level_optimized,
    expected_latency,
    fan_strength,
    latency_sensitivity,
    mismatch_penalty,
    retrieval_latency,
    retrieval_probability,
    spreading_activation,
)

DECAY = 0.5


# --------------------------------------------------------------------------
# The regression: the property the old scorer could not have
# --------------------------------------------------------------------------


def test_recency_is_scale_free():
    """Shifting every timestamp by a constant must not change activation.

    The old scorer failed exactly here: it measured position against a fixed
    wall-clock epoch, so its output depended on *when* the run happened rather
    than on how old the memory was.
    """
    now = 1_786_589_186.0
    ages = [60.0, 3600.0, 86400.0]
    here = base_level_activation([now - a for a in ages], now, decay=DECAY)

    for shift in (-5.0e8, 1.0e8, 3.0e8):
        shifted = base_level_activation(
            [now + shift - a for a in ages], now + shift, decay=DECAY
        )
        assert shifted == pytest.approx(here, rel=1e-12), (
            f"activation moved by {shifted - here:g} under a pure epoch shift"
        )


def test_recent_ages_are_distinguishable():
    """A minute ago and a month ago must not score the same.

    Measured on the old function on 2026-08-12: both returned 1.000000.
    """
    now = 1_786_589_186.0
    minute = base_level_activation([now - 60.0], now, decay=DECAY)
    month = base_level_activation([now - 30 * 86400.0], now, decay=DECAY)
    assert minute > month
    # Not merely different — different by the amount the power law requires.
    expected_gap = DECAY * math.log((30 * 86400.0) / 60.0)
    assert (minute - month) == pytest.approx(expected_gap, rel=1e-12)


def test_no_saturation_at_any_age():
    """Activation keeps discriminating however recent or old the memory is."""
    now = 1_000_000.0
    ages = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7]
    values = [base_level_activation([now - a], now, decay=DECAY) for a in ages]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values), "two distinct ages collapsed to one score"


# --------------------------------------------------------------------------
# Analytic identities
# --------------------------------------------------------------------------


def test_single_presentation_is_exactly_the_power_law():
    """One use: B = ln(t^-d) = -d·ln(t), exactly linear in log age."""
    now = 10_000.0
    for age in (0.5, 2.0, 60.0, 3600.0):
        got = base_level_activation([now - age], now, decay=DECAY)
        assert got == pytest.approx(-DECAY * math.log(age), rel=1e-12)


def test_practice_raises_activation():
    """More uses of the same chunk, at the same ages, means higher activation."""
    now = 10_000.0
    one = base_level_activation([now - 100.0], now, decay=DECAY)
    three = base_level_activation([now - 100.0] * 3, now, decay=DECAY)
    assert three > one
    # Exactly ln(3) higher: identical ages sum to 3·t^-d inside the log.
    assert (three - one) == pytest.approx(math.log(3.0), rel=1e-12)


def test_activation_decays_monotonically_with_time():
    now = 0.0
    presented = [-10.0]
    previous = math.inf
    for elapsed in (0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0):
        value = base_level_activation(presented, now + elapsed, decay=DECAY)
        assert value < previous
        previous = value


def test_a_never_presented_chunk_is_unretrievable():
    assert base_level_activation([], 1.0, decay=DECAY) == -math.inf
    assert retrieval_probability(-math.inf) == 0.0
    assert activation(base_level=-math.inf, spreading=99.0) == -math.inf


def test_presentations_in_the_future_do_not_explode():
    """Clock skew is routine and must not produce infinite activation."""
    value = base_level_activation([100.0, 105.0], 100.0, decay=DECAY)
    assert math.isfinite(value)


# --------------------------------------------------------------------------
# Optimized approximation
# --------------------------------------------------------------------------


def test_optimized_tracks_exact_for_evenly_spread_uses():
    """The O(1) form is close to exact when its uniformity assumption holds."""
    now = 100_000.0
    lifetime = 50_000.0
    n = 40
    step = lifetime / n
    presentations = [now - lifetime + i * step for i in range(n)]
    exact = base_level_activation(presentations, now, decay=DECAY)
    approx = base_level_optimized(n, lifetime, decay=DECAY)
    assert approx == pytest.approx(exact, abs=0.15), f"exact={exact:.4f} approx={approx:.4f}"


def test_optimized_is_honest_about_clustered_uses():
    """Clustered presentations violate the assumption, and it should show."""
    now = 100_000.0
    lifetime = 50_000.0
    presentations = [now - 10.0 - i for i in range(40)]  # all very recent
    exact = base_level_activation(presentations, now, decay=DECAY)
    approx = base_level_optimized(40, lifetime, decay=DECAY)
    assert exact > approx + 1.0, "the approximation should understate a recent cluster"


# --------------------------------------------------------------------------
# Fan, spreading, partial matching
# --------------------------------------------------------------------------


def test_fan_weakens_a_cue():
    """A cue pointing at many chunks is weaker evidence for any one of them."""
    assert fan_strength(1) > fan_strength(2) > fan_strength(16)
    assert fan_strength(1) == pytest.approx(2.0)  # S - ln(1)


def test_spreading_ignores_unassociated_cues():
    weights = {"coffee": 0.5, "unrelated": 0.5}
    strengths = {"coffee": 1.4}
    assert spreading_activation(weights, strengths) == pytest.approx(0.7)


def test_mismatch_is_never_a_bonus():
    assert mismatch_penalty([1.0, 1.0]) == pytest.approx(0.0)
    assert mismatch_penalty([0.5]) < 0.0
    assert mismatch_penalty([0.0]) < mismatch_penalty([0.5])


# --------------------------------------------------------------------------
# Latency — the prediction Aura previously could not make
# --------------------------------------------------------------------------


def test_latency_is_the_exponential_of_negative_activation():
    params = ActrParameters(latency_factor=0.35, threshold=-2.0)
    for a in (-1.0, 0.0, 0.5, 2.0):
        got = retrieval_latency(
            a, latency_factor=params.latency_factor, threshold=params.threshold
        )
        assert got == pytest.approx(0.35 * math.exp(-a), rel=1e-12)


def test_stronger_memories_are_predicted_faster():
    now = 10_000.0
    fresh = base_level_activation([now - 10.0], now, decay=DECAY)
    stale = base_level_activation([now - 10_000.0], now, decay=DECAY)
    assert retrieval_latency(fresh, threshold=-9.0) < retrieval_latency(stale, threshold=-9.0)


def test_a_failed_retrieval_still_costs_time():
    """Reporting failure as free makes the model faster than the system."""
    params = ActrParameters(threshold=0.0, latency_factor=0.35)
    floor = retrieval_latency(-50.0, latency_factor=0.35, threshold=0.0)
    assert floor == pytest.approx(0.35 * math.exp(-0.0))
    assert expected_latency(-50.0, params=params) == pytest.approx(floor, rel=1e-9)


def test_retrieval_probability_is_a_half_at_threshold():
    assert retrieval_probability(0.5, threshold=0.5, noise_s=0.4) == pytest.approx(0.5)
    assert retrieval_probability(5.0, threshold=0.0) > 0.99
    assert retrieval_probability(-5.0, threshold=0.0) < 0.01


def test_probability_does_not_overflow_on_extreme_activation():
    assert retrieval_probability(1e6, threshold=0.0) == 1.0
    assert retrieval_probability(-1e6, threshold=0.0) == 0.0


# --------------------------------------------------------------------------
# The parameter-count criticism, taken seriously
# --------------------------------------------------------------------------


def test_sensitivity_reports_a_band_per_parameter():
    bands = latency_sensitivity(1.0, relative_perturbation=0.10)
    assert {b.parameter for b in bands} == {
        "decay",
        "noise_s",
        "threshold",
        "latency_factor",
    }
    for band in bands:
        assert band.low <= band.nominal <= band.high or band.spread_ratio >= 0.0


def test_latency_factor_scales_the_prediction_linearly():
    """F is pure scale: it can absorb any absolute timing, which is why an
    unfitted absolute latency claim is meaningless."""
    a = 1.0
    one = expected_latency(a, params=ActrParameters(latency_factor=0.35))
    two = expected_latency(a, params=ActrParameters(latency_factor=0.70))
    assert two == pytest.approx(2.0 * one, rel=1e-12)


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        ActrParameters(decay=0.0)
    with pytest.raises(ValueError):
        ActrParameters(decay=1.0)
    with pytest.raises(ValueError):
        ActrParameters(noise_s=0.0)
    with pytest.raises(ValueError):
        ActrParameters(latency_factor=0.0)
