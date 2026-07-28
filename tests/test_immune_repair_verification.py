"""A repair is verified when the recovery HELD, not when it happened.

CP126 ea9e677e: verification took 2 samples 10ms apart and accepted a 0.02
rise. Worse, the stability-window guard requires >= 3 samples, so at those
defaults it could NEVER FIRE — a guard that cannot run, which is the shape
this campaign keeps finding.
"""
from __future__ import annotations

import pytest

from core.adaptation.adaptive_immunity import get_adaptive_immune_system


@pytest.fixture()
def immune():
    return get_adaptive_immune_system()


def _verify(immune, samples, *, before=0.5, raw_success=True):
    return immune._verify_repair_success(
        raw_success=raw_success,
        health_before=before,
        health_after=samples[-1] if samples else None,
        health_samples=list(samples),
    )


def test_the_stability_guard_can_actually_fire(immune):
    """The defaults must produce enough samples for the tail check to run."""
    assert immune.cfg.verification_checks >= immune.cfg.min_verification_samples
    window = (immune.cfg.verification_checks - 1) * immune.cfg.verification_interval_s
    assert window >= immune.cfg.min_verification_window_s


def test_two_samples_no_longer_verify(immune):
    """The exact case the finding names."""
    assert _verify(immune, [0.5, 0.9]) is False


def test_a_held_recovery_verifies(immune):
    assert _verify(immune, [0.5, 0.8, 0.85, 0.9]) is True


def test_a_spike_that_decays_does_not_verify(immune):
    """It ends high, but the tail fell back — a failed repair."""
    assert _verify(immune, [0.5, 0.95, 0.51, 0.9]) is False


def test_an_insufficient_rise_does_not_verify(immune):
    assert _verify(immune, [0.5, 0.505, 0.505, 0.505]) is False


def test_a_failed_actuation_never_verifies(immune):
    assert _verify(immune, [0.5, 0.9, 0.9, 0.9], raw_success=False) is False


def test_missing_readings_never_verify(immune):
    assert immune._verify_repair_success(
        raw_success=True, health_before=None, health_after=0.9, health_samples=[0.9]
    ) is False
    assert immune._verify_repair_success(
        raw_success=True, health_before=0.5, health_after=None, health_samples=[]
    ) is False


def test_a_too_short_window_fails_closed(immune, monkeypatch):
    """Samples taken microseconds apart are one measurement repeated."""
    import dataclasses

    tight = dataclasses.replace(immune.cfg, verification_interval_s=0.001)
    monkeypatch.setattr(immune, "cfg", tight)

    assert _verify(immune, [0.5, 0.9, 0.9, 0.9]) is False


def test_recovery_is_measured_against_the_pre_repair_reading(immune):
    """A component that was already healthy gains nothing from 'repair'."""
    assert _verify(immune, [0.9, 0.91, 0.91, 0.91], before=0.9) is False


def test_unverified_outcomes_do_not_report_zero_recurrence_risk(immune):
    """CP126 9038d017: suppressed/unavailable/advisory results inherited 0.0,
    understating risk that is in fact unresolved."""
    report = immune._default_verification_report(status="suppressed", coverage_ratio=0.5)

    assert report["verified_success"] is False
    assert report["recurrence_risk"] > 0.0
