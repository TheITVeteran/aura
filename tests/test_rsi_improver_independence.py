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
    PROVENANCE_AUTHORED,
    PROVENANCE_MEASURED,
    PROVENANCE_UNMEASURED,
    VERDICT_BOUNDED,
    VERDICT_STRONG,
    ImproverMeasurement,
    RSIGenerationRecord,
    evaluate_lineage,
    improver_curve_dependence,
    improver_efficiency,
    improver_rise_within_noise,
    order_invariance_violation,
)

pytestmark = pytest.mark.unit


def _record(
    after: float,
    improver: float,
    seq: int,
    provenance: str = PROVENANCE_MEASURED,
) -> RSIGenerationRecord:
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
        improver_provenance=provenance,
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


# --------------------------------------------------------------------------
# The second authorship route: a curve written by the loop counter
#
# `improver_curve_dependence` catches an improver curve that is a function of
# the *capability* curve. It cannot catch one that is a function of the
# generation *index*, and that is what
# `PrimitiveInventionEngine.improver_score` computed:
#
#     0.22 + 0.09 * generation_index + 0.17 * coverage + 0.10 * feedback
#          + 0.12 * hypothesis_quality + artifact + machinery_bonus
#
# The recorded [0.578, 0.8696, 0.9536, 1.0] reproduces exactly from it.
# --------------------------------------------------------------------------


def test_the_recorded_undeniable_curve_reproduces_from_the_old_formula():
    """The published curve was arithmetic, not observation."""

    def old_formula(index: int, coverage: float, hypothesis_quality: float, bonus: float) -> float:
        return round(
            min(
                1.0,
                0.22
                + 0.09 * index
                + 0.17 * coverage
                + 0.10 * 1.0
                + 0.12 * hypothesis_quality
                + 0.08
                + bonus,
            ),
            6,
        )

    assert old_formula(1, 0.2, 0.45, 0.0) == 0.578
    assert old_formula(2, 0.4, 0.68, 0.14) == 0.8696
    assert old_formula(3, 0.6, 0.68, 0.10) == 0.9536
    assert old_formula(4, 0.8, 0.68, 0.10) == 1.0


def test_curve_dependence_does_not_catch_index_authorship():
    """Why provenance is needed: the shape test passes on the fake curve."""
    capability = [0.51872, 0.614037, 0.811531, 1.0]
    improver = [0.578, 0.8696, 0.9536, 1.0]
    assert improver_curve_dependence(capability, improver) == ""


def test_an_authored_improver_score_cannot_reach_a_strong_verdict():
    records = [
        _record(a, i, seq, provenance=PROVENANCE_AUTHORED)
        for seq, (a, i) in enumerate(
            zip([0.61, 0.65, 0.69, 0.74], [0.08, 0.11, 0.14, 0.21])
        )
    ]
    verdict = evaluate_lineage(records)
    assert verdict.verdict == VERDICT_BOUNDED
    assert any("not measured" in reason for reason in verdict.reasons)


def test_an_unmeasured_improver_score_cannot_reach_a_strong_verdict():
    """The default. A record written by older code cannot pass by omission."""
    records = [
        _record(a, i, seq, provenance=PROVENANCE_UNMEASURED)
        for seq, (a, i) in enumerate(
            zip([0.61, 0.65, 0.69, 0.74], [0.08, 0.11, 0.14, 0.21])
        )
    ]
    assert evaluate_lineage(records).verdict == VERDICT_BOUNDED


def test_record_provenance_defaults_to_unmeasured():
    record = RSIGenerationRecord(
        generation_id="g1",
        parent_generation_id="g0",
        hypothesis="h",
        intervention_type="t",
        artifact_hashes={},
        baseline_score=0.1,
        after_score=0.2,
        hidden_eval_score=0.2,
    )
    assert record.improver_provenance == PROVENANCE_UNMEASURED


# --------------------------------------------------------------------------
# Order invariance: the structural check the old formula cannot pass
# --------------------------------------------------------------------------


def _measurement(gen: str, before: float, after: float, seconds: float) -> ImproverMeasurement:
    return ImproverMeasurement(
        generation_id=gen,
        heldout_before=before,
        heldout_after=after,
        wall_clock_samples=(seconds, seconds, seconds),
        feedback_queries=1,
    )


def test_measured_efficiency_ignores_lineage_position():
    forward = [
        _measurement("g1", 0.1, 0.3, 1.0),
        _measurement("g2", 0.3, 0.5, 2.0),
        _measurement("g3", 0.5, 0.9, 1.0),
    ]
    assert order_invariance_violation(forward) == ""
    scores = [m.efficiency() for m in forward]
    assert [m.efficiency() for m in reversed(forward)] == list(reversed(scores))


def test_a_rise_inside_the_timing_spread_is_not_a_rise():
    """Point estimates climb; the intervals overlap; the claim fails."""
    noisy = [
        ImproverMeasurement("g1", 0.1, 0.30, (1.0, 2.0), 1),
        ImproverMeasurement("g2", 0.3, 0.52, (1.0, 2.0), 1),
    ]
    assert noisy[1].efficiency() > noisy[0].efficiency()
    assert "within measurement noise" in improver_rise_within_noise(noisy)


def test_a_rise_clearing_the_spread_counts():
    clean = [
        ImproverMeasurement("g1", 0.1, 0.2, (1.0, 1.0), 1),
        ImproverMeasurement("g2", 0.2, 0.9, (1.0, 1.0), 1),
    ]
    assert improver_rise_within_noise(clean) == ""


def test_an_unmeasured_budget_scores_zero_rather_than_defaulting():
    unmeasured = ImproverMeasurement("g1", 0.1, 0.9, (), 0)
    assert not unmeasured.measured
    assert unmeasured.efficiency() == 0.0


def test_the_invention_engine_no_longer_scores_itself():
    """The improver may not hold a method that rates the improver."""
    from core.learning.autonomous_rsi import PrimitiveInventionEngine

    assert not hasattr(PrimitiveInventionEngine, "improver_score")


def test_the_custodian_measurement_takes_no_generation_index():
    """Structural: the metric cannot read the counter it used to contain."""
    import inspect

    from core.learning.autonomous_rsi import ExternalHiddenEvalCustodian

    params = inspect.signature(ExternalHiddenEvalCustodian.measure_improver).parameters
    assert not any("index" in name or "generation_index" == name for name in params)
