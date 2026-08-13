"""The fitted retrieval curve, and the latency null that must stay a null.

``tools/fit_actr_retrieval.py`` fitted ACT-R's two retrieval models against
Aura's own recall and they came out differently:

* the retrieval curve fits — tau = -0.4666, s = 2.0, Brier skill 0.154;
* the latency equation does not — r^2 = 0.000037 between ln T and -A.

Both halves are pinned. The fit is pinned so the parameters stay reproducible;
the null is pinned because F is a pure scale factor that would absorb any
timing, so nothing but an explicit test stops a future reader from "fitting" it
to noise. If retrieval ever becomes genuinely activation-driven, the null test
fails and tells someone to go and fit F properly.
"""

from __future__ import annotations

import math

import pytest

from core.cognition.actr_activation import (
    DEFAULT_PARAMETERS,
    FITTED_BRIER_SKILL,
    FITTED_PARAMETERS,
    retrieval_probability,
)

pytestmark = pytest.mark.unit


def test_fitted_parameters_are_distinct_from_the_published_defaults():
    """A 'fit' identical to the defaults would mean nothing was measured."""
    assert FITTED_PARAMETERS.threshold != DEFAULT_PARAMETERS.threshold
    assert FITTED_PARAMETERS.noise_s != DEFAULT_PARAMETERS.noise_s


def test_the_latency_factor_was_not_quietly_fitted():
    """F stays at its published default: the data does not identify it.

    This is the guard against the failure the module exists to avoid — a
    confident absolute latency with no mechanism behind it.
    """
    assert FITTED_PARAMETERS.latency_factor == DEFAULT_PARAMETERS.latency_factor


def test_the_fitted_noise_is_not_at_a_grid_boundary():
    """The first fit railed at s=1.5 against a 1.5 ceiling; that is not a fit."""
    assert 0.05 < FITTED_PARAMETERS.noise_s < 6.0


def test_the_fitted_curve_discriminates_across_the_operating_range():
    """Recall probability must move meaningfully over real activations."""
    minute = -0.5 * math.log(60.0)
    year = -0.5 * math.log(365 * 86400.0)
    p_recent = retrieval_probability(
        minute,
        threshold=FITTED_PARAMETERS.threshold,
        noise_s=FITTED_PARAMETERS.noise_s,
    )
    p_old = retrieval_probability(
        year,
        threshold=FITTED_PARAMETERS.threshold,
        noise_s=FITTED_PARAMETERS.noise_s,
    )
    assert p_recent > p_old
    assert p_recent - p_old > 0.20, (
        f"the fitted curve barely separates a minute from a year: "
        f"{p_recent:.3f} vs {p_old:.3f}"
    )


def test_reported_skill_is_honest_about_being_partial():
    """Activation is 0.4 of the blend; near-perfect skill would be suspicious.

    A Brier skill close to 1.0 would mean importance had stopped contributing
    to ranking at all, which would be a defect rather than a better fit.
    """
    assert 0.02 < FITTED_BRIER_SKILL < 0.60


def test_the_latency_null_still_holds_on_the_live_ranking_path():
    """Regressing ln(T) on -A over the real ranker must find no relationship.

    Slow (it ranks real batches), but it is the test that stops the latency
    equation from being quietly resurrected. If this fails, retrieval has
    become activation-driven and F is now worth fitting.
    """
    from tools.fit_actr_retrieval import _fit_latency, _measure

    measured = _measure(trials=25, batch=30, seed=7)
    latency = _fit_latency(measured["samples"])  # type: ignore[arg-type]

    assert latency["fitted"] is False, (
        "activation now predicts retrieval latency; the ACT-R latency equation "
        f"may finally apply here — refit F. Got {latency}"
    )
    assert latency["r2"] < 0.10
