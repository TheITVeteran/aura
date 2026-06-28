"""Epistemic calibration: confidence must track warrant, not fluency.

These lock the behavior the user named — on unverifiable claims, sounding smart and being
right diverge, so warranted confidence is capped and overconfidence is flagged.
"""
from __future__ import annotations

import pytest

from core.cognition.epistemic_calibration import (
    EpistemicCalibrator,
    Verifiability,
    get_epistemic_calibrator,
)


@pytest.fixture
def cal():
    return EpistemicCalibrator()


# ── classification ──────────────────────────────────────────────────────────

def test_classifies_formal(cal):
    assert cal.classify("prove that 7 is prime") == Verifiability.FORMAL


def test_classifies_normative(cal):
    assert cal.classify("you should always tell the truth") == Verifiability.NORMATIVE


def test_classifies_speculative(cal):
    assert cal.classify("superintelligence will arrive by 2045") == Verifiability.SPECULATIVE


def test_classifies_subjective_other(cal):
    assert cal.classify("you feel angry about this") == Verifiability.SUBJECTIVE_OTHER


def test_classifies_unknowable(cal):
    assert cal.classify("what happened before the big bang") == Verifiability.UNKNOWABLE


# ── warranted-confidence ceilings ──────────────────────────────────────────

def test_unverified_formal_is_capped_low(cal):
    r = cal.calibrate("the integral equals 42", stated_confidence=0.95, tool_verified=False)
    assert r.warranted_confidence <= 0.6
    assert r.recommended_confidence <= 0.6
    assert r.overconfident


def test_tool_verified_formal_is_trusted(cal):
    r = cal.calibrate("2 + 2 = 4", stated_confidence=0.9, tool_verified=True)
    assert r.warranted_confidence >= 0.9
    assert not r.overconfident
    assert r.stance == "assert"


def test_speculative_is_capped_and_marked(cal):
    r = cal.calibrate("civilization will inevitably collapse", stated_confidence=0.9)
    assert r.warranted_confidence <= 0.4
    assert r.overconfident
    assert r.stance == "mark_speculative"


def test_normative_is_framed_as_view(cal):
    r = cal.calibrate("it is morally wrong to lie", stated_confidence=0.9)
    assert r.stance == "frame_as_view"
    assert r.warranted_confidence <= 0.6


def test_subjective_other_uses_estimate_confidence(cal):
    low = cal.calibrate("you are upset with me", stated_confidence=0.9, other_agent_confidence=0.2)
    assert low.warranted_confidence == pytest.approx(0.2)
    assert low.stance == "defer_to_person"
    assert low.overconfident


def test_evidence_raises_empirical_warrant(cal):
    none = cal.calibrate("the build is faster now", stated_confidence=0.8, evidence_count=0)
    some = cal.calibrate("the build is faster now", stated_confidence=0.8, evidence_count=4)
    assert some.warranted_confidence > none.warranted_confidence


def test_absolute_phrasing_is_itself_overreach(cal):
    r = cal.calibrate("this is definitely the best approach", stated_confidence=0.8)
    # "definitely" + a preference → warrant pulled down, flagged
    assert r.warranted_confidence <= 0.6


def test_recommended_never_exceeds_warrant(cal):
    for claim in ["x will happen", "you feel sad", "it is beautiful", "the answer is 5"]:
        r = cal.calibrate(claim, stated_confidence=1.0)
        assert r.recommended_confidence <= r.warranted_confidence + 1e-9


def test_singleton_stable():
    assert get_epistemic_calibrator() is get_epistemic_calibrator()


# ── adversarial auditor consults calibration ────────────────────────────────

def test_auditor_flags_overconfident_speculation():
    from core.cognition.adversarial_audit import AdversarialAuditor

    report = AdversarialAuditor().audit(
        "superintelligence will definitely arrive by 2045",
        stated_confidence=0.95,
    )
    names = {f.check for f in report.findings}
    assert "calibration" in names
    cal_finding = next(f for f in report.findings if f.check == "calibration")
    assert cal_finding.passed is False
    assert report.caveats  # produced a repair caveat
