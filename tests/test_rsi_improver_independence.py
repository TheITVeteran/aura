"""The improver curve must be independent evidence, or there is no strong RSI.

Strong RSI is two inequalities, not one:

    C_{g+1} > C_g      the successor is more capable
    I_{g+1} > I_g      the improver got better at improving

``weight_compounding`` recorded ``improver_score = candidate_accuracy`` —
literally ``after_score`` under a second name. So a capability curve of
0.61/0.65/0.69/0.74 produced an improver curve of 0.61/0.65/0.69/0.74, both
inequalities passed, and the strong verdict rested on one measurement counted
twice.

Two fixes, both tested here: the producer now records verified gain per hour,
and the evaluator refuses a strong verdict when the curves are affinely
related — which catches the identity case and every rescaling of it.
"""

from __future__ import annotations

import pytest

from core.learning.rsi_lineage import (
    VERDICT_BOUNDED,
    VERDICT_STRONG,
    RSIGenerationRecord,
    evaluate_lineage,
    improver_curve_dependence,
    improver_efficiency,
)

pytestmark = pytest.mark.unit


def _record(after: float, improver: float, seq: int) -> RSIGenerationRecord:
    return RSIGenerationRecord(
        generation_id=f"g{seq}",
        parent_generation_id=f"g{seq - 1}" if seq else "",
        hypothesis="a successor generation improves the battery",
        intervention_type="weight_lora_sft",
        artifact_hashes={},
        baseline_score=after - 0.04,
        after_score=after,
        hidden_eval_score=after,
        regressions=[],
        promoted=True,
        rollback_performed=False,
        time_to_valid_improvement_s=3600.0,
        improver_score=improver,
    )


# --------------------------------------------------------------------------
# The efficiency measure
# --------------------------------------------------------------------------


def test_efficiency_is_gain_per_hour():
    assert improver_efficiency(
        baseline_score=0.60, after_score=0.70, cost_s=3600.0
    ) == pytest.approx(0.10)
    # Same gain, twice the cost, half the efficiency.
    assert improver_efficiency(
        baseline_score=0.60, after_score=0.70, cost_s=7200.0
    ) == pytest.approx(0.05)


def test_efficiency_falls_while_capability_rises():
    """The discrimination the capability curve cannot make.

    Diminishing returns: each generation is more capable and each improvement
    cost more to get. Any 'improver' score derived from capability alone would
    report improvement here; a real one reports the opposite.
    """
    curve = [
        improver_efficiency(baseline_score=b, after_score=a, cost_s=c)
        for b, a, c in (
            (0.60, 0.70, 3600.0),
            (0.70, 0.76, 3600.0),
            (0.76, 0.80, 7200.0),
            (0.80, 0.82, 14400.0),
        )
    ]
    assert curve == sorted(curve, reverse=True), curve
    assert curve[0] > curve[-1]


def test_unknown_cost_scores_zero_rather_than_defaulting():
    assert improver_efficiency(baseline_score=0.6, after_score=0.7, cost_s=0.0) == 0.0
    assert improver_efficiency(baseline_score=0.6, after_score=0.7, cost_s=-1.0) == 0.0


def test_a_regression_has_no_positive_efficiency():
    assert improver_efficiency(
        baseline_score=0.70, after_score=0.60, cost_s=3600.0
    ) == 0.0


# --------------------------------------------------------------------------
# The anti-circularity gate
# --------------------------------------------------------------------------


def test_identical_curves_are_refused():
    curve = [0.61, 0.65, 0.69, 0.74]
    reason = improver_curve_dependence(curve, list(curve))
    assert "identical" in reason


def test_a_rescaled_copy_is_refused_too():
    """Multiplying by two hides identity from an equality check, not from this."""
    capability = [0.61, 0.65, 0.69, 0.74]
    improver = [2.0 * c + 0.1 for c in capability]
    reason = improver_curve_dependence(capability, improver)
    assert "affine" in reason


def test_a_genuinely_independent_curve_passes():
    capability = [0.61, 0.65, 0.69, 0.74]
    improver = [0.10, 0.13, 0.11, 0.19]
    assert improver_curve_dependence(capability, improver) == ""


def test_the_strong_verdict_is_refused_on_a_circular_curve():
    """End to end: the exact shape weight_compounding used to produce."""
    records = [_record(a, a, i) for i, a in enumerate([0.61, 0.65, 0.69, 0.74])]
    verdict = evaluate_lineage(records)
    assert verdict.verdict == VERDICT_BOUNDED
    assert any("counted twice" in r for r in verdict.reasons)


def test_the_strong_verdict_still_available_with_independent_evidence():
    records = [
        _record(a, i, seq)
        for seq, (a, i) in enumerate(
            zip([0.61, 0.65, 0.69, 0.74], [0.08, 0.11, 0.14, 0.21])
        )
    ]
    verdict = evaluate_lineage(records)
    assert verdict.verdict == VERDICT_STRONG, verdict.reasons


def test_a_falling_improver_curve_blocks_strong_rsi():
    """Capability up, improver down: exactly what strong RSI must not claim."""
    records = [
        _record(a, i, seq)
        for seq, (a, i) in enumerate(
            zip([0.61, 0.65, 0.69, 0.74], [0.21, 0.14, 0.11, 0.08])
        )
    ]
    verdict = evaluate_lineage(records)
    assert verdict.verdict == VERDICT_BOUNDED
    assert any("improver curve is not strictly increasing" in r for r in verdict.reasons)


def test_the_producer_no_longer_echoes_capability():
    """weight_compounding must not pass candidate_accuracy as improver_score."""
    from pathlib import Path

    source = Path("core/learning/weight_compounding.py").read_text(encoding="utf-8")
    assert "improver_score=float(receipt.candidate_accuracy" not in source
    assert "improver_score=improver_efficiency(" in source
