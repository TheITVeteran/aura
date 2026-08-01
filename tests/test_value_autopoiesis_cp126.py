"""Value autopoiesis: a guardrail that blocked its own repair, and evidence
that could not be interpreted becoming an identity shift."""
from __future__ import annotations

import pytest

import core.adaptation.value_autopoiesis as va
from core.adaptation.value_autopoiesis import OutcomeEvidence, ValueShift

pytestmark = pytest.mark.unit


# ── evidence must be interpretable before it moves a value ─────────────────


def test_valid_evidence_is_accepted():
    ev = OutcomeEvidence(drive_name="Curiosity", outcome_quality=0.5,
                         engagement_level=0.7, free_energy=0.2, context="ok")

    assert ev.outcome_quality == 0.5
    assert ev.drive_name == "Curiosity"


@pytest.mark.parametrize("field,value", [
    ("outcome_quality", float("nan")),
    ("outcome_quality", float("inf")),
    ("engagement_level", float("nan")),
    ("free_energy", float("-inf")),
    ("outcome_quality", "not a number"),
])
def test_uninterpretable_evidence_is_refused(field, value):
    """NaN compares False against every threshold and then poisons every mean
    it enters, so an unusable signal silently became an identity shift."""
    kwargs = dict(drive_name="Curiosity", outcome_quality=0.5,
                  engagement_level=0.7, free_energy=0.2, context="c")
    kwargs[field] = value

    with pytest.raises(ValueError):
        OutcomeEvidence(**kwargs)


def test_out_of_range_values_are_clamped_not_rejected():
    """Documented ranges are enforced; a merely out-of-range number is a
    saturated signal, not an uninterpretable one."""
    ev = OutcomeEvidence(drive_name="d", outcome_quality=9.0,
                         engagement_level=-4.0, free_energy=17.0, context="c")

    assert ev.outcome_quality == 1.0
    assert ev.engagement_level == 0.0
    assert ev.free_energy == 1.0


def test_empty_drive_name_is_refused():
    with pytest.raises(ValueError, match="drive_name"):
        OutcomeEvidence(drive_name="   ", outcome_quality=0.0,
                        engagement_level=0.0, free_energy=0.0, context="c")


def test_context_is_bounded():
    ev = OutcomeEvidence(drive_name="d", outcome_quality=0.0,
                         engagement_level=0.0, free_energy=0.0,
                         context="x" * 100_000)

    assert len(ev.context) <= 2000


# ── the guardrail must not block its own repair ────────────────────────────


def _engine():
    return va.ValueAutopoiesis.__new__(va.ValueAutopoiesis)


def _shift(name, old, new):
    return ValueShift(value_name=name, old_weight=old, new_weight=new,
                      delta=round(new - old, 4), reason="test",
                      evidence_count=5, cycle_id=1)


def test_a_shift_that_reduces_excessive_drift_is_allowed():
    """A value already outside the band has max_allowed <= 0, and the shift was
    then suppressed unconditionally — including shifts moving BACK toward
    origin. The guardrail pinned drifted values at their drifted position."""
    engine = _engine()
    engine._origin_values = {"Curiosity": 0.50}

    # Currently far outside the band; the shift moves it back toward origin.
    current = {"Curiosity": 0.90}
    shift = _shift("Curiosity", 0.90, 0.80)

    guarded = engine._apply_drift_guardrails([shift], current)

    assert guarded[0].delta != 0.0, "a repairing shift must not be suppressed"
    assert "DRIFT_REPAIR" in guarded[0].reason


def test_a_shift_that_worsens_excessive_drift_is_still_suppressed():
    """The guardrail must still do its actual job."""
    engine = _engine()
    engine._origin_values = {"Curiosity": 0.50}
    current = {"Curiosity": 0.90}
    shift = _shift("Curiosity", 0.90, 0.95)

    guarded = engine._apply_drift_guardrails([shift], current)

    assert guarded[0].delta == 0.0
    assert "DRIFT_CAPPED" in guarded[0].reason


def test_a_shift_within_the_band_passes_untouched():
    engine = _engine()
    engine._origin_values = {"Curiosity": 0.50}
    current = {"Curiosity": 0.52}
    shift = _shift("Curiosity", 0.52, 0.55)

    guarded = engine._apply_drift_guardrails([shift], current)

    assert guarded[0].delta == pytest.approx(0.03)
    assert "DRIFT" not in guarded[0].reason
