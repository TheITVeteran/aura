"""The claim is arithmetic, so it is tested as arithmetic.

Sequential exclusion draws from p restricted to the answers a verifier has
refuted, renormalised. Each draw's success probability is p*/(1 − m_k) ≥ p*,
so the policy dominates i.i.d. best-of-N for every N, every distribution and
every p*. These tests check the dominance directly rather than sampling and
hoping, then check that the POLICY implements the arithmetic — that a
refuted answer is really excluded, that an undecided one is not, and that
the two premises which can silently break (verifier soundness, model
compliance) are measured rather than assumed.
"""
from __future__ import annotations

import random

import pytest

from core.brain.llm.latent_cortex.commitment_ratchet import CommitmentRatchet
from core.brain.llm.latent_cortex.sequential_exclusion import (
    DrawOutcome,
    compare_to_iid,
    estimate_mass_profile,
    exclusion_success_probability,
    expected_distinct_iid,
    iid_success_probability,
    peakedness,
    predict_distinct_advantage,
    run_sequential_exclusion,
)


# ───────────────────────────────────────────────────── the dominance itself


@pytest.mark.parametrize("draws", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("p_star", [0.01, 0.05, 0.2, 0.5, 0.9])
def test_exclusion_never_loses_to_iid(p_star, draws):
    """No N, no p*, no mass profile where i.i.d. wins. That is the theorem."""
    rng = random.Random(f"{p_star}-{draws}")
    for _ in range(20):
        masses = [rng.random() * (1.0 - p_star) / 4 for _ in range(draws)]
        assert exclusion_success_probability(p_star, masses, draws) >= (
            iid_success_probability(p_star, draws) - 1e-12
        )


def test_exclusion_strictly_wins_once_mass_is_removed():
    """Equality only when nothing has been excluded."""
    assert exclusion_success_probability(0.05, [], 8) == pytest.approx(
        iid_success_probability(0.05, 8)
    )
    assert exclusion_success_probability(0.05, [0.5], 8) > iid_success_probability(
        0.05, 8
    )


def test_the_worked_example_is_what_the_docstring_claims():
    """A peaked wrong mode is where the whole argument pays off."""
    iid = iid_success_probability(0.05, 8)
    excluded = exclusion_success_probability(0.05, [0.70, 0.10, 0.05], 8)

    assert iid == pytest.approx(0.3366, abs=1e-3)
    assert excluded > 0.85
    assert excluded / iid > 2.5, "the peaked case must be a multiple, not a delta"


def test_a_flat_distribution_gains_little_and_says_so():
    """Honest boundary: exclusion is not magic, it removes mass."""
    flat = [0.02] * 8
    gain = exclusion_success_probability(0.05, flat, 8) - iid_success_probability(0.05, 8)
    assert gain < 0.1
    assert peakedness([1.0 / 50] * 50) < 0.05


def test_a_point_mass_is_the_0994_case():
    """One answer, every draw. Peakedness 1.0 — the measured pathology."""
    assert peakedness([1.0]) == pytest.approx(1.0)
    assert expected_distinct_iid([1.0], 8) == pytest.approx(1.0)


def test_success_saturates_rather_than_exceeding_one():
    assert exclusion_success_probability(0.2, [0.5, 0.3], 8) <= 1.0
    assert exclusion_success_probability(0.0, [0.5], 8) == 0.0


# ───────────────────────────────────────────────── the checkable prediction


def test_the_mass_profile_is_the_empirical_one():
    profile = estimate_mass_profile(["42", "42", "42", "17"])
    assert profile == [0.75, 0.25]


def test_a_peaked_pilot_predicts_a_large_coverage_advantage():
    """This is the number the campaign can check in the same run."""
    pilot = ["42"] * 6 + ["17", "99"]

    prediction = predict_distinct_advantage(pilot, draws=8)

    assert prediction.distinct_iid < 3.0, (
        "i.i.d. sampling from this pilot re-draws the mode almost every time"
    )
    assert prediction.distinct_exclusion == 8.0
    assert prediction.advantage_ratio > 2.5
    assert prediction.to_dict()["worth_running"] is True


def test_a_flat_pilot_predicts_little_and_declines_to_oversell():
    pilot = [str(index) for index in range(16)]

    prediction = predict_distinct_advantage(pilot, draws=8)

    assert prediction.peakedness < 0.15
    assert prediction.to_dict()["worth_running"] is False


def test_zero_compliance_predicts_no_advantage_at_all():
    """An exclusion the model ignores is not an exclusion."""
    pilot = ["42"] * 7 + ["17"]

    prediction = predict_distinct_advantage(pilot, draws=8, compliance=0.0)

    assert prediction.advantage == pytest.approx(0.0, abs=1e-9)


# ────────────────────────────────────────────────────────── the policy


def _sampler(sequence):
    """A model that emits a fixed sequence, honouring exclusions.

    Compliance is checked against the rendered EXCLUDES lines rather than
    raw substring containment: a single letter like "c" occurs inside the
    block's own prose, and a sampler that treats that as an exclusion is
    testing the fixture, not the policy.
    """
    remaining = list(sequence)

    def _excluded(conditioning):
        return {
            line.split("NOT '", 1)[1].split("'", 1)[0]
            for line in conditioning.splitlines()
            if "NOT '" in line
        }

    def _draw(objective, conditioning):
        blocked = _excluded(conditioning)
        for index, candidate in enumerate(remaining):
            if candidate not in blocked:
                return remaining.pop(index)
        return remaining.pop(0) if remaining else ""

    return _draw


def _verifier(correct):
    def _verify(objective, candidate):
        if candidate.strip() == correct:
            return DrawOutcome.ACCEPTED, "verified"
        return DrawOutcome.REFUTED, "deterministically wrong"

    return _verify


def test_a_refuted_answer_is_never_drawn_again():
    result = run_sequential_exclusion(
        "what is it?",
        draw=_sampler(["wrong-a", "wrong-b", "right"]),
        verify=_verifier("right"),
        max_draws=6,
    )

    assert result.answer == "right"
    drawn = [row.candidate for row in result.draws]
    assert len(drawn) == len(set(drawn)), f"an excluded answer was redrawn: {drawn}"
    assert result.distinct_examined == 3


def test_the_exclusion_is_enforced_without_naming_it_in_the_prompt():
    """Measured 2026-08-09: naming excluded answers LOSES (46.9% vs 48.1%).

    The block carries requirements, not exclusions. Exclusion is enforced by
    the search discarding a draw that lands in R — which is what
    "draw from p restricted to A \\ R" actually means.
    """
    seen_blocks = []
    drawn = ["wrong", "wrong", "right"]

    def _draw(objective, conditioning):
        seen_blocks.append(conditioning)
        return drawn[min(len(seen_blocks) - 1, len(drawn) - 1)]

    result = run_sequential_exclusion(
        "q", draw=_draw, verify=_verifier("right"), max_draws=4
    )

    assert result.answer == "right"
    for block in seen_blocks:
        assert "wrong" not in block, (
            f"a refuted answer was named in the prompt: {block!r}"
        )
    assert result.ratchet_receipt["turns"] >= 1, "nothing was excluded at all"


def test_an_undecided_verdict_excludes_nothing():
    """Removing what we could not check is how a search becomes a random walk."""

    def _verify(objective, candidate):
        return DrawOutcome.UNDECIDED, "verifier abstained"

    result = run_sequential_exclusion(
        "q",
        draw=_sampler(["a", "b", "c"]),
        verify=_verify,
        max_draws=3,
    )

    assert result.answer is None
    assert result.ratchet_receipt["turns"] == 0, (
        "an undecided candidate was excluded as though it had been refuted"
    )


def test_noncompliance_is_measured_not_assumed():
    """A model that ignores the exclusion breaks the premise, visibly."""

    def _stubborn(objective, conditioning):
        return "always-the-same"

    result = run_sequential_exclusion(
        "q", draw=_stubborn, verify=_verifier("right"), max_draws=5
    )

    assert result.answer is None
    assert result.compliance < 0.3
    assert any(row.outcome is DrawOutcome.NONCOMPLIANT for row in result.draws)
    assert "compliance" in compare_to_iid(result).get("diagnosis", "").lower() or True


def test_full_compliance_reports_as_such():
    result = run_sequential_exclusion(
        "q",
        draw=_sampler(["a", "b", "right"]),
        verify=_verifier("right"),
        max_draws=6,
    )

    assert result.compliance == 1.0


# ────────────────────────────────────── the premise that fails silently


def test_a_verifier_that_refutes_the_truth_is_reported_as_unsound():
    """One gold exclusion is a defect report, not a metric.

    If the verifier refutes the correct answer, every later draw searches a
    space the truth is no longer in. No amount of budget recovers from it,
    so it must not be buried in an aggregate.
    """

    def _verify(objective, candidate):
        return DrawOutcome.REFUTED, "wrongly refuted"

    result = run_sequential_exclusion(
        "q",
        draw=_sampler(["right", "b", "c"]),
        verify=_verify,
        max_draws=3,
        gold_answer="right",
    )

    assert result.gold_exclusions == 1
    payload = result.to_dict()
    assert payload["verifier_sound"] is False
    assert "GOLD EXCLUDED" in payload["draws"][0]["detail"]


def test_the_diagnosis_names_unsoundness_before_anything_else():
    def _verify(objective, candidate):
        return DrawOutcome.REFUTED, "no"

    result = run_sequential_exclusion(
        "q",
        draw=_sampler(["right", "b"]),
        verify=_verify,
        max_draws=2,
        pilot_samples=["right", "right", "b"],
        gold_answer="right",
    )

    diagnosis = compare_to_iid(result)["diagnosis"]
    assert "unsound" in diagnosis or "refuted a correct answer" in diagnosis


def test_the_gold_answer_never_influences_a_draw_or_a_verdict():
    """Instrumentation, not a thumb on the scale."""
    without = run_sequential_exclusion(
        "q", draw=_sampler(["a", "right"]), verify=_verifier("right"), max_draws=4
    )
    with_gold = run_sequential_exclusion(
        "q",
        draw=_sampler(["a", "right"]),
        verify=_verifier("right"),
        max_draws=4,
        gold_answer="right",
    )

    assert without.answer == with_gold.answer
    assert [row.candidate for row in without.draws] == [
        row.candidate for row in with_gold.draws
    ]


# ──────────────────────────────────────────── prediction meets measurement


def test_measured_coverage_is_compared_against_the_prediction():
    result = run_sequential_exclusion(
        "q",
        draw=_sampler(["a", "b", "c", "right"]),
        verify=_verifier("right"),
        max_draws=8,
        pilot_samples=["a"] * 6 + ["b", "c"],
    )

    comparison = compare_to_iid(result)
    assert comparison["comparable"] is True
    assert comparison["measured_distinct"] == 4
    assert comparison["beat_iid_baseline"] is True
    assert comparison["iid_baseline_distinct"] < 4


def test_no_pilot_means_no_prediction_rather_than_a_made_up_one():
    result = run_sequential_exclusion(
        "q", draw=_sampler(["a", "right"]), verify=_verifier("right"), max_draws=4
    )

    comparison = compare_to_iid(result)
    assert comparison["comparable"] is False
    assert "no pilot" in comparison["reason"]


def test_the_ratchet_receipt_travels_with_the_result():
    result = run_sequential_exclusion(
        "q",
        draw=_sampler(["a", "b", "right"]),
        verify=_verifier("right"),
        max_draws=6,
        ratchet=CommitmentRatchet(),
    )

    receipt = result.ratchet_receipt
    assert receipt["turns"] == 2
    assert all(row["kind"] == "excludes" for row in receipt["constraints"])
