"""Habituation: the hundredth occurrence of a known fault is not the first.

Aura already de-weights repeats WITHIN one window. These tests pin the
missing half — familiarity ACROSS windows — and the three constraints that
keep it honest: the record never attenuates, the weight never reaches zero,
and a signature that goes quiet re-sensitises.
"""
from __future__ import annotations

import pytest

from core.runtime.degradation_habituation import (
    DegradationHabituation,
    get_habituation,
    signature_for,
)


@pytest.fixture
def hab() -> DegradationHabituation:
    return DegradationHabituation()


def test_an_unfamiliar_failure_lands_at_full_weight(hab):
    assert hab.multiplier("never|seen") == 1.0


def test_familiarity_reduces_the_multiplier(hab):
    sig = "cortex|TimeoutError"
    first = hab.multiplier(sig)
    for _ in range(10):
        hab.note(sig)
    assert hab.multiplier(sig) < first


def test_attenuation_never_silences_completely(hab):
    sig = "cortex|TimeoutError"
    for _ in range(500):
        hab.note(sig)
    # 0.4 residual: a chronic fault stays perceptible forever.
    assert hab.multiplier(sig) == pytest.approx(0.4, abs=1e-6)
    assert hab.attenuation(sig) == pytest.approx(0.6, abs=1e-6)


def test_the_first_occurrences_of_something_new_are_never_quieted(hab):
    """A real new cascade must land at FULL weight while it is unfolding."""
    sig = "new|CascadeError"
    for _ in range(5):
        hab.note(sig)
        assert hab.multiplier(sig) == 1.0, "a new failure must not be discounted"
    hab.note(sig)
    assert hab.multiplier(sig) < 1.0, "familiarity should begin after the allowance"


def test_growth_saturates_rather_than_accumulating(hab):
    sig = "a|B"
    for _ in range(8):
        hab.note(sig)
    after_three = hab.multiplier(sig)
    for _ in range(7):
        hab.note(sig)
    after_ten = hab.multiplier(sig)
    # Still decreasing, but the last seven moved it less than the first three.
    assert after_ten < after_three
    assert (1.0 - after_ten) - (1.0 - after_three) < (1.0 - after_three)


def test_silence_re_sensitises(hab):
    sig = "flaky|OSError"
    now = 1_000_000.0
    for _ in range(50):
        hab.note(sig, now=now)
    saturated = hab.multiplier(sig, now=now)
    # Two days of quiet.
    later = now + 48 * 3600
    assert hab.multiplier(sig, now=later) > saturated
    assert hab.multiplier(sig, now=later) > 0.97
    # Fully re-sensitised once the decay has run its course.
    assert hab.multiplier(sig, now=now + 60 * 3600) == pytest.approx(1.0, abs=1e-6)


def test_reading_does_not_increase_familiarity(hab):
    """Asking what something weighs must not make the system used to it."""
    sig = "x|Y"
    now = 1_000.0
    for _ in range(20):
        hab.note(sig, now=now)
    before = hab.multiplier(sig, now=now)
    for _ in range(20):
        hab.multiplier(sig, now=now)
    assert hab.multiplier(sig, now=now) == before


def test_distinct_signatures_do_not_share_habituation(hab):
    for _ in range(50):
        hab.note("a|TimeoutError")
    assert hab.multiplier("a|TimeoutError") < 0.5
    assert hab.multiplier("b|TimeoutError") == 1.0


def test_chronic_lists_the_persistent_faults(hab):
    now = 1_000.0
    for i in range(8):
        hab.note("cortex|TimeoutError", now=now + i * 3600)
    hab.note("rare|ValueError", now=now)
    chronic = hab.chronic(minimum_count=5)
    assert [c["signature"] for c in chronic] == ["cortex|TimeoutError"]
    assert chronic[0]["count"] == 8
    assert chronic[0]["recurring_for_h"] == pytest.approx(7.0, abs=0.01)


def test_signature_is_the_failure_class_not_the_message():
    """Message text carries ids and timings; keying on it defeats the mechanism."""
    a = signature_for("cortex", "TimeoutError")
    b = signature_for("cortex", "TimeoutError")
    assert a == b == "cortex|TimeoutError"


def test_signature_normalises_missing_parts():
    assert signature_for("", "") == "unknown|unknown"
    assert signature_for(None, None) == "unknown|unknown"
    assert signature_for("  cortex  ", "  TimeoutError ") == "cortex|TimeoutError"


def test_record_degradation_notes_through_the_canonical_key():
    """Writer and reader must key identically, or the discount never applies."""
    from core.runtime.errors import record_degradation

    hab = get_habituation()
    hab.reset_for_test()
    sig = signature_for("habituation_probe", "RuntimeError")
    assert hab.multiplier(sig) == 1.0
    for _ in range(20):
        record_degradation(
            "habituation_probe", RuntimeError("boom"), severity="warning", action="test"
        )
    assert hab.multiplier(sig) < 0.6
    hab.reset_for_test()


def test_the_degradation_record_itself_is_never_attenuated():
    """Habituation scales what is FELT, never the audit trail."""
    from core.runtime.errors import get_degradation_tracker, record_degradation

    hab = get_habituation()
    hab.reset_for_test()
    tracker = get_degradation_tracker()
    before = len(tracker._records)
    for _ in range(30):
        record_degradation(
            "habituation_record_probe",
            RuntimeError("boom"),
            severity="warning",
            action="test",
        )
    after = len(tracker._records)
    # Every occurrence is still recorded, however familiar it became.
    assert after - before == 30 or after == tracker._records.maxlen
    hab.reset_for_test()
