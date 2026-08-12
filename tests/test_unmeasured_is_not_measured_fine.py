"""An unmeasured signal must not vote "fine".

Three regulators sampled the world, and when the sample failed they carried
on with a default that means *healthy*. The degradation was recorded, so
none of this was silent — and the decision was still made, on numbers that
described nothing.

  * `somatic_throttle` defaulted to 0% CPU, 0% RAM and a governance throttle
    of 1.0. That is a perfectly idle host with unlimited quota, so a psutil
    failure or a missing compute governor made every stress test below
    unable to fire, and heavy generation was admitted during exactly the
    pressure the observer could not read.
  * `compute_orchestrator` used 40% CPU / 50% RAM and its own degradation
    called them "conservative". `CPU_NORMAL` is 50.0 and means FULL
    OPERATION, so those defaults sat comfortably inside the healthy band.
  * `emotional_regulation` left `actual_damage` at 0.0 on a read failure,
    and BOTH down-regulating branches use low damage as their grounds —
    "no real damage, hold" and "arousal exceeds actual damage, reframe". An
    unreadable nociception engine therefore suppressed a response that may
    have been entirely warranted, which is the one thing that function must
    never do.

The shape of the fix is the same each time and is the one `world_state`
already uses: record whether the signal was MEASURED, and never let an
unmeasured one carry a conclusion that depends on its value.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────── an unreadable host is treated as loaded


def test_unmeasurable_resources_do_not_read_as_an_idle_host():
    from core.agency import compute_orchestrator as co

    assert co.DEFAULT_CPU_PCT >= co.CPU_HIGH, (
        "the fallback CPU load sits in the healthy band, so an unmeasurable "
        "host disables every reduction below it"
    )
    assert co.DEFAULT_RAM_PCT >= co.RAM_HIGH


def test_the_fallbacks_are_anchored_to_the_named_thresholds():
    """Not picked. A number chosen by feel drifts away from the bands it is
    supposed to sit inside the moment either is retuned."""
    from core.agency import compute_orchestrator as co

    assert co.DEFAULT_CPU_PCT == co.CPU_HIGH
    assert co.DEFAULT_RAM_PCT == co.RAM_HIGH


def test_the_fallbacks_stay_below_critical():
    """Conservative, not catastrophic: an unobserved host should behave like
    a busy one, not like one about to fall over."""
    from core.agency import compute_orchestrator as co

    assert co.DEFAULT_CPU_PCT < co.CPU_CRITICAL
    assert co.DEFAULT_RAM_PCT < co.RAM_CRITICAL


def test_the_degradation_no_longer_calls_a_permissive_default_conservative():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "agency"
        / "compute_orchestrator.py"
    ).read_text("utf-8")

    assert "conservative static resource defaults" not in source, (
        "the degradation still describes a healthy-band default as "
        "conservative, which is how it survived review"
    )


# ─────────────────────────── an unobserved throttle input is not headroom


def test_an_unobservable_host_is_stressed_in_the_generation_throttle():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "brain"
        / "llm"
        / "somatic_throttle.py"
    ).read_text("utf-8")

    assert "hardware_measured" in source and "gov_measured" in source, (
        "the throttle no longer tracks whether its inputs were measured"
    )
    assert "unobserved" in source, (
        "an unmeasured signal can vote 'no stress' again"
    )


def test_quota_exhaustion_still_requires_a_real_measurement():
    """`gov_throttle == 0.0` is the severe cap. It must not fire on the
    default value of a governor that was never reached."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "brain"
        / "llm"
        / "somatic_throttle.py"
    ).read_text("utf-8")

    assert "if gov_throttle == 0.0 and gov_measured:" in source


# ──────────────────── unknown damage must not reappraise a feeling away


def _regulator():
    from core.affect.emotional_regulation import get_emotional_regulator

    return get_emotional_regulator()


@pytest.fixture
def blind_nociception(monkeypatch):
    """Make the nociception engine unreadable."""
    import core.affect.nociception as noc

    def _explode():
        raise RuntimeError("nociception unavailable")

    monkeypatch.setattr(noc, "get_nociception_engine", _explode)


def test_unreadable_damage_does_not_hold_or_reappraise(blind_nociception):
    """Both down-regulating branches used low damage as their reason.

    With damage unreadable, "there is no real damage" is not something the
    system knows, and acting on it suppresses a possibly-real response.
    """
    from core.affect.emotional_regulation import get_emotional_regulator

    result = get_emotional_regulator().regulate(
        arousal=0.95, valence=-0.8, deliberation=0.1
    )
    strategy = result.strategy

    assert strategy not in {"hold", "reappraise"}, (
        f"strategy {strategy!r} was chosen on damage that could not be read"
    )


def test_the_receipt_says_damage_was_not_measured(blind_nociception):
    """A 0.0 that means "unreadable" and a 0.0 that means "no damage" were
    the same number on the receipt."""
    from core.affect.emotional_regulation import get_emotional_regulator

    result = get_emotional_regulator().regulate(
        arousal=0.95, valence=-0.8, deliberation=0.1
    )
    factors = result.factors

    assert factors.get("damage_measured") is False
    assert factors.get("actual_damage") is None


def test_measured_damage_still_regulates_normally(monkeypatch):
    """The fix must not disable reappraisal when the reading works."""
    import core.affect.nociception as noc
    from core.affect.emotional_regulation import get_emotional_regulator

    class _Quiet:
        @staticmethod
        def nociceptive_pressure() -> float:
            return 0.0

    monkeypatch.setattr(noc, "get_nociception_engine", lambda: _Quiet())

    result = get_emotional_regulator().regulate(
        arousal=0.95, valence=-0.8, deliberation=0.1
    )
    assert result.factors.get("damage_measured") is True
    assert result.strategy in {"hold", "reappraise", "dampen", "escalate", "express"}
