"""The α = 0.35 behavioural claim is RETRACTED. This file holds the retraction.

What this file used to assert
-----------------------------
That steering at Aura's live surface alpha changes behaviour, on the strength
of ``artifacts/steering/CAA_AB_ALPHA_0.35_live.json``:

    50 trials · 5 held-out tasks · 10 layers · 41,450 injections · 19.6 min
    steered vs terse-affect control   d = 1.879   p = 0.0002
    steered vs RICH adversarial text  d = 2.502   p = 0.0002

Why it does not survive inspection
----------------------------------
The evaluation statistic scored each trial as::

    distance(steered, control) - distance(steered, baseline)

and ``tests/run_32b_steering_ab_live.py`` generated the steered and baseline
conditions from the same prompt under the same seed, toggling only the
injection. So if steering has no effect at all, ``steered == baseline``, the
subtracted term is exactly zero, and the score is ``distance(baseline,
control)`` — positive by construction, because the control deliberately uses a
different system prompt.

The null hypothesis "steering did nothing" produces a decisive pass. That is
not a weakness in the test; the null is reversed.

The artifact records it happening. All three of its saved steered samples are
word-for-word identical to their baselines, and ``affect_stats`` shows zero
positive-affect words in the steered condition against one in the rich control
— no evidence of movement in the intended affective direction, alongside
d = 2.50.

What is still supported
-----------------------
* Residual-stream injection EXECUTES (``injection_count`` = 41,450).
* The hook changes hidden activations — by mechanism.
* The mean steered↔baseline distance over all 50 trials is nonzero (0.2424), so
  some trials did differ. Nothing in that campaign says whether those
  differences exceed ordinary sampling variability, follow the intended
  affective direction, come from these vectors rather than perturbation as
  such, or generalize.

What a replacement needs, and now has a harness for
---------------------------------------------------
``core.evaluation.steering_ab`` requires a ``baseline_replicate`` (the same
unsteered condition drawn again — the model's own run-to-run variation), scores
every effect net of it, runs zero-vector / random-vector / shuffled-layer
specificity arms through the identical hook, and refuses to pass without a
significant move in a scored target behaviour.
``core.evaluation.statistics.null_effect_probe`` is the general check that
would have caught this on the day it was written.

Until such a campaign runs and passes, the behavioural claim stays retracted.
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


def test_the_artifact_is_marked_void(report):
    """A retracted result may stay on disk; it may not read as evidence."""
    assert report.get("VOID") is True
    assert "null hypothesis" in report.get("void_reason", "").lower()


def test_the_artifacts_own_samples_show_the_defect(report):
    """Steered and baseline are the same text, and the score was decisive."""
    samples = report["analysis"]["samples"]
    steered = samples["steered_black_box"]
    baseline = samples["baseline"]
    assert steered == baseline, (
        "if these ever differ the artifact was regenerated — update this file"
    )
    assert float(report["analysis"]["steered_vs_rich"]["p_value"]) < 0.01, (
        "the recorded run reported significance over identical outputs"
    )


def test_no_affect_moved_in_the_steered_condition(report):
    """Divergence is not direction, and here there was no direction at all."""
    assert report["affect_stats"]["steered"]["positive"] == 0
    assert report["affect_stats"]["steered"]["negative"] == 0


def test_what_the_injection_still_supports(report):
    """The mechanism ran. That is a different claim from the behavioural one."""
    assert int(report["injection_count"]) > 0
    assert float(report["alpha"]) == pytest.approx(0.35)
    assert "fused-model" in str(report["model"])


def test_the_old_statistic_is_gone():
    """It cannot be reached by name, so nothing can quietly keep using it."""
    import core.evaluation.statistics as statistics

    assert not hasattr(statistics, "paired_distance_comparison")


def test_the_replacement_statistic_fails_on_a_no_op_intervention():
    """The check that would have caught this, run against the shipped code."""
    from core.evaluation.statistics import (
        null_effect_probe,
        paired_effect_over_null_reference,
    )

    verdict = null_effect_probe(paired_effect_over_null_reference, n_trials=40, seed=3)

    assert not verdict.significant
    assert verdict.p_value > 0.5


def test_a_campaign_without_a_null_reference_cannot_be_analyzed():
    from core.evaluation.steering_ab import analyze_steering_ab

    with pytest.raises(ValueError, match="baseline_replicate"):
        analyze_steering_ab(
            {
                "steered_black_box": ["x"] * 6,
                "baseline": ["x"] * 6,
                "text_terse": ["y"] * 6,
                "text_rich_adversarial": ["z"] * 6,
            }
        )


def test_the_readiness_gate_refuses_the_voided_artifact(report):
    """A stale artifact must not normalize into a passing readiness metric."""
    from training.caa_32b_validation import CAA32BValidator

    normalized = CAA32BValidator._normalize_behavioral_results(dict(report))

    assert normalized["source_schema"] == "live_32b_ab_voided"
    assert normalized["black_box_prompt_hygiene_passed"] is False
    assert normalized["steered_vs_rich_prompt_effect_size"] < 0.0


def test_the_quantization_floor_no_longer_cites_the_retracted_result():
    """The SNR module leaned on d = 2.502 to argue the floor does not matter.

    It may still NAME the number — retracting a result means saying which one.
    What it may not do is present it as settling anything.
    """
    import core.consciousness.caa.quantization_floor as module

    doc = module.__doc__ or ""
    assert "RETRACTED" in doc, (
        "the module still cites its effect sizes as settled evidence"
    )
    assert "UNPROVEN" in doc
    assert "AND IT PASSED" not in doc
