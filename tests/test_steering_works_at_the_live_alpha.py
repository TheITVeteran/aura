"""Steering at the LIVE surface alpha changes behaviour. Measured, not argued.

The quantisation analysis established that at α = 0.35 one injection is about
eighteen times smaller than the noise 4-bit weights already put into the same
residual stream (SNR ≈ 0.056). The reasonable worry followed: "steering was
injected" may not mean "steering was strong enough to matter", and the existing
A/B evidence was collected at α = 8.

So the A/B was run at 0.35, on the resident 32B, and it passed — by more than
the α = 8 run did.

    artifacts/steering/CAA_AB_ALPHA_0.35_live.json
    50 trials · 5 held-out tasks · 10 layers · 41,450 injections · 19.6 min

    steered vs terse-affect control   d = 1.879   p = 0.0002
    steered vs RICH adversarial text  d = 2.502   p = 0.0002
    distance steered↔baseline 0.2424, rich↔baseline 0.7611

The rich-adversarial control is the one that matters: a prompt STUFFED with
affect language is the cheap way to fake this result, and activation steering
at the live alpha beats it decisively.

Why the SNR did not predict the outcome, and why that is not a contradiction:
SNR measures one injection against one layer's noise at one instant. That noise
is zero-mean and uncorrelated across layers and tokens. The steering vector is
the same direction every time. Over 10 layers and hundreds of tokens the bias
accumulates and the noise cancels — which is exactly why the floor is reported
and never used to clamp α.

This file guards the artifact and the reading of it. The run itself is live and
long; these assertions are over its recorded result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "steering"
    / "CAA_AB_ALPHA_0.35_live.json"
)


@pytest.fixture(scope="module")
def report() -> dict:
    # Tracked in git: the recorded result of the live run is the subject of
    # every assertion in this file. Skipping when it is missing meant a
    # deleted artifact read as "nothing to check" instead of "the evidence
    # for the live steering alpha is gone".
    assert ARTIFACT.exists(), f"{ARTIFACT} is tracked in git and must exist"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_run_was_at_the_live_surface_alpha(report):
    """0.35 is what the worker logs on every decode; 8.0 was the old evidence."""
    assert float(report["alpha"]) == pytest.approx(0.35)


def test_it_was_a_real_run_on_the_resident_model(report):
    assert report["n_trials"] >= 50
    assert len(report["target_layers"]) >= 8
    assert len(report["held_out_tasks"]) >= 5
    assert float(report["duration_seconds"]) > 300
    assert "fused-model" in str(report["model"])


def test_steering_beats_the_rich_adversarial_prompt(report):
    """The control that matters: stuffing the prompt with affect language."""
    rich = report["analysis"]["steered_vs_rich"]
    assert float(rich["p_value"]) < 0.01
    assert float(rich["effect_size_d"]) > 1.0
    assert float(rich["ci_low"]) > 0.0  # the interval excludes "no effect"


def test_steering_beats_the_terse_affect_control(report):
    terse = report["analysis"]["steered_vs_terse"]
    assert float(terse["p_value"]) < 0.01
    assert float(terse["effect_size_d"]) > 1.0
    assert float(terse["ci_low"]) > 0.0


def test_the_verdict_is_recorded_as_passing(report):
    assert report.get("passes_adversarial_control") is True


def test_the_quantization_floor_does_not_claim_steering_is_ineffective(report):
    """The SNR is about magnitude at an instant, not about whether it works."""
    from core.consciousness.caa.quantization_floor import assess_steering_precision

    precision = assess_steering_precision(0.35, residual_norm=70.0)
    assert precision.below_floor is True
    assert precision.snr < 0.1
    # …and the behavioural result at that same alpha is significant. Both are
    # true, and the module says so rather than letting the first imply the
    # opposite of the second.
    assert float(report["analysis"]["steered_vs_rich"]["p_value"]) < 0.01


def test_the_module_carries_the_measurement_that_settles_it():
    import core.consciousness.caa.quantization_floor as module

    doc = module.__doc__ or ""
    assert "d = 2.502" in doc
    assert "does NOT predict the behavioural outcome" in doc
