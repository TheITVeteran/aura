"""Aura's model of her own faculties, and of what "better" means for each.

The complaint this answers: self-editing was aimed at problems she had been
told to solve, because ``record_signal`` waits to be told and episode-level
metacognition judges one answer at a time. Nothing modelled the stack.

The properties that matter are honesty ones. An unmeasured faculty must never
read as a healthy one, a drive must not invent a target it cannot justify, and
the faculty chosen must be the one that limits the most — not merely the one
scoring worst.
"""
from __future__ import annotations

import pytest

from core.metacognition.faculty_model import (
    CognitiveSelfModel,
    Faculty,
    FacultyRegistry,
    ImprovementMetric,
    emit_improvement_signals,
    improvement_goal,
)


def _metric(metric_id="m", value=0.5, direction="higher_is_better", **kw):
    defaults = dict(floor=0.0, target=0.8, ceiling=1.0)
    if direction == "lower_is_better":
        defaults = dict(floor=1.0, target=0.2, ceiling=0.0)
    defaults.update(kw)
    return ImprovementMetric(
        metric_id=metric_id,
        unit="",
        direction=direction,
        probe=lambda: value,
        **defaults,
    )


@pytest.fixture()
def registry():
    return FacultyRegistry()


class _Loop:
    def __init__(self):
        self.signals = []

    def record_signal(self, **kwargs):
        self.signals.append(kwargs)


# --- improvement is a declared contract ---------------------------------


def test_a_metric_declares_direction_unit_and_potential():
    metric = _metric(value=0.5)

    assert metric.normalize(0.5) == 0.5
    assert metric.headroom(0.5) == 0.5
    assert metric.meets_target(0.9) is True
    assert metric.meets_target(0.5) is False


def test_lower_is_better_metrics_invert_correctly():
    metric = _metric(direction="lower_is_better", value=0.1)

    assert metric.normalize(0.0) == 1.0   # at ceiling
    assert metric.normalize(1.0) == 0.0   # at floor
    assert metric.meets_target(0.1) is True
    assert metric.meets_target(0.5) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(floor=0.0, target=1.5, ceiling=1.0),   # target above ceiling
        dict(floor=1.0, target=0.5, ceiling=2.0),   # unordered
        dict(floor=0.5, target=0.5, ceiling=0.5),   # no scale
    ],
)
def test_an_incoherent_improvement_contract_is_refused(kwargs):
    with pytest.raises(ValueError):
        ImprovementMetric("m", "", "higher_is_better", lambda: 0.5, **kwargs)


def test_a_negative_weight_is_refused():
    with pytest.raises(ValueError):
        _metric(weight=-1.0)


# --- unmeasured is never "fine" -----------------------------------------


def test_a_probe_returning_none_is_unmeasured_not_zero():
    reading = ImprovementMetric(
        "m", "", "higher_is_better", lambda: None, 0.0, 0.8, 1.0
    ).read()

    assert reading.measured is False
    assert reading.value is None
    assert reading.normalized is None
    assert "not currently measurable" in reading.reason


def test_a_raising_probe_is_unmeasured_not_a_crash():
    def _boom():
        raise RuntimeError("subsystem offline")

    reading = ImprovementMetric(
        "m", "", "higher_is_better", _boom, 0.0, 0.8, 1.0
    ).read()

    assert reading.measured is False
    assert "RuntimeError" in reading.reason


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "text", object()])
def test_a_non_finite_probe_result_is_unmeasured(bad):
    reading = ImprovementMetric(
        "m", "", "higher_is_better", lambda: bad, 0.0, 0.8, 1.0
    ).read()

    assert reading.measured is False


def test_an_unmeasurable_faculty_has_no_score_rather_than_a_good_one(registry):
    registry.declare(
        Faculty("ghost", "d", "o", metrics=(_metric(metric_id="x"),))
    )
    registry.declare(
        Faculty(
            "invisible", "d", "o",
            metrics=(ImprovementMetric("y", "", "higher_is_better", lambda: None, 0, 0.8, 1),),
        )
    )

    model = registry.assess()
    invisible = model.by_id("invisible")

    assert invisible.measurable is False
    assert invisible.headroom is None
    # None, NOT 0.0 — zero would sort last and read as "nothing to do here".
    assert invisible.priority is None


def test_blind_spots_are_reported_as_gaps_in_self_knowledge(registry):
    registry.declare(
        Faculty(
            "temporal_reasoning", "d", "o",
            metrics=(ImprovementMetric("t", "", "higher_is_better", lambda: None, 0, 0.9, 1),),
        )
    )

    model = registry.assess()

    assert model.blind_spots() == ("temporal_reasoning",)
    assert model.self_knowledge_coverage() == 0.0


def test_coverage_reports_the_measured_fraction(registry):
    registry.declare(
        Faculty(
            "mixed", "d", "o",
            metrics=(
                _metric(metric_id="ok", value=0.5),
                ImprovementMetric("dead", "", "higher_is_better", lambda: None, 0, 0.8, 1),
            ),
        )
    )

    assessment = registry.assess().by_id("mixed")

    assert assessment.coverage == 0.5
    assert assessment.measured_metrics == 1
    assert assessment.declared_metrics == 2


# --- holistic means leverage, not just deficit --------------------------


def test_leverage_counts_the_faculties_a_faculty_limits(registry):
    registry.declare(Faculty("memory", "d", "o", gates=("attention", "temporal")))
    registry.declare(Faculty("attention", "d", "o", gates=("temporal",)))
    registry.declare(Faculty("temporal", "d", "o"))

    assert registry.leverage("memory") == 3.0     # itself + 2 gated
    assert registry.leverage("attention") == 2.0
    assert registry.leverage("temporal") == 1.0


def test_leverage_is_transitive(registry):
    registry.declare(Faculty("a", "d", "o", gates=("b",)))
    registry.declare(Faculty("b", "d", "o", gates=("c",)))
    registry.declare(Faculty("c", "d", "o"))

    assert registry.leverage("a") == 3.0  # reaches b AND c


def test_a_cyclic_declaration_terminates(registry):
    registry.declare(Faculty("a", "d", "o", gates=("b",)))
    registry.declare(Faculty("b", "d", "o", gates=("a",)))

    assert registry.leverage("a") >= 1.0  # returns, does not spin


def test_the_binding_constraint_is_leverage_weighted_not_worst_score(registry):
    """A slightly-better faculty that gates the stack outranks a slightly
    worse one that gates nothing."""
    registry.declare(
        Faculty("memory", "d", "o", metrics=(_metric(value=0.42),),
                gates=("attention", "temporal"))
    )
    registry.declare(
        Faculty("attention", "d", "o", metrics=(_metric(value=0.40),))
    )
    registry.declare(Faculty("temporal", "d", "o", metrics=(_metric(value=0.9),)))

    model = registry.assess()

    # attention scores WORSE but gates nothing; memory limits the stack.
    assert model.binding_constraint == "memory"
    assert model.by_id("memory").priority > model.by_id("attention").priority


def test_a_faculty_at_potential_is_not_the_constraint(registry):
    registry.declare(Faculty("done", "d", "o", metrics=(_metric(value=1.0),)))

    model = registry.assess()

    assert model.by_id("done").headroom == 0.0
    assert model.binding_constraint is None


def test_an_empty_registry_names_no_constraint(registry):
    model = registry.assess()

    assert model.binding_constraint is None
    assert model.self_knowledge_coverage() == 0.0


# --- agency: a goal she can justify -------------------------------------


def test_the_goal_names_the_faculty_metric_and_measured_gap(registry):
    registry.declare(
        Faculty("memory", "d", "o", metrics=(_metric("recall_at_5", value=0.42),),
                gates=("attention",))
    )

    goal = improvement_goal(registry.assess())

    assert "memory" in goal["objective"]
    assert "recall_at_5" in goal["objective"]
    assert goal["faculty"] == "memory"
    assert goal["evidence"]["binding_constraint"] == "memory"
    assert goal["origin"] == "intrinsic_competence_faculty_model"


def test_with_nothing_measurable_the_goal_is_to_build_the_probe(registry):
    registry.declare(
        Faculty(
            "temporal_reasoning", "d", "o",
            metrics=(ImprovementMetric("t", "", "higher_is_better", lambda: None, 0, 0.9, 1),),
        )
    )

    goal = improvement_goal(registry.assess())

    assert goal["origin"] == "intrinsic_competence_self_knowledge"
    assert "measure" in goal["objective"]
    assert goal["faculty"] == "temporal_reasoning"


def test_with_nothing_to_want_no_goal_is_invented(registry):
    registry.declare(Faculty("fine", "d", "o", metrics=(_metric(value=1.0),)))

    assert improvement_goal(registry.assess()) is None


def test_an_empty_self_model_invents_nothing(registry):
    assert improvement_goal(registry.assess()) is None


# --- causal: it reaches the improvement loop ----------------------------


def test_the_binding_constraint_reaches_the_rsi_loop(registry):
    registry.declare(
        Faculty("memory", "d", "o", metrics=(_metric(value=0.3),), gates=("attention",))
    )
    loop = _Loop()

    emit_improvement_signals(loop, registry.assess())

    assert loop.signals[0]["metric"] == "memory"
    assert loop.signals[0]["kind"] == "faculty_headroom"
    assert loop.signals[0]["evidence"]["binding_constraint"] is True


def test_blind_spots_also_reach_the_loop(registry):
    registry.declare(
        Faculty(
            "temporal", "d", "o",
            metrics=(ImprovementMetric("t", "", "higher_is_better", lambda: None, 0, 0.9, 1),),
        )
    )
    loop = _Loop()

    emit_improvement_signals(loop, registry.assess())

    kinds = [s["kind"] for s in loop.signals]
    assert "self_knowledge_gap" in kinds


def test_a_faculty_at_potential_emits_no_signal(registry):
    registry.declare(Faculty("fine", "d", "o", metrics=(_metric(value=1.0),)))
    loop = _Loop()

    emit_improvement_signals(loop, registry.assess())

    assert loop.signals == []


def test_a_loop_without_record_signal_is_tolerated(registry):
    registry.declare(Faculty("memory", "d", "o", metrics=(_metric(value=0.3),)))

    assert emit_improvement_signals(object(), registry.assess()) == []


def test_signal_emission_is_bounded(registry):
    for index in range(10):
        registry.declare(
            Faculty(f"f{index}", "d", "o", metrics=(_metric(value=0.1),))
        )
    loop = _Loop()

    emit_improvement_signals(loop, registry.assess(), max_signals=3)

    assert len([s for s in loop.signals if s["kind"] == "faculty_headroom"]) == 3


# --- registry hygiene ----------------------------------------------------


def test_duplicate_metric_ids_are_refused(registry):
    with pytest.raises(ValueError):
        registry.declare(
            Faculty("f", "d", "o", metrics=(_metric("same"), _metric("same")))
        )


def test_the_snapshot_serializes(registry):
    registry.declare(Faculty("memory", "d", "o", metrics=(_metric(value=0.4),)))

    payload = registry.assess().as_dict()

    assert payload["schema"] == "aura.cognitive_self_model.v1"
    assert payload["faculties"][0]["faculty_id"] == "memory"
    assert "self_knowledge_coverage" in payload
