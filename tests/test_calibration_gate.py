"""Tests for the epistemic-status calibration gate."""
from __future__ import annotations

from core.brain.calibration_gate import (
    CalibrationGate,
    EpistemicStatus,
    get_calibration_gate,
)


def test_confident_unsupported_claim_is_guessed_and_softened():
    gate = CalibrationGate()
    report = gate.assess("The capital of Atlantis is definitely Poseidonis.")
    assert report.overall is EpistemicStatus.GUESSED
    assert report.downgraded >= 1
    assert "not fully certain" in report.calibrated_answer.lower()


def test_hedged_claim_is_inferred_not_softened():
    gate = CalibrationGate()
    report = gate.assess("I think this might be related to caching.")
    assert report.overall is EpistemicStatus.INFERRED
    assert report.downgraded == 0


def test_source_backed_when_grounded_in_evidence():
    gate = CalibrationGate()
    report = gate.assess(
        "The retry budget is three attempts before failing closed.",
        evidence=["the retry budget allows three attempts then fails closed in the gateway"],
    )
    assert report.overall in {EpistemicStatus.SOURCE_BACKED, EpistemicStatus.KNOWN}
    assert report.confidence >= 0.8


def test_tool_verified_promotes_confidence():
    gate = CalibrationGate()

    class _V:
        checked = True
        ok = True

    report = gate.assess(
        "The result is 144.",
        verification=_V(),
        tool_verified=True,
    )
    assert any(l.status is EpistemicStatus.TOOL_VERIFIED for l in report.labels)
    assert report.confidence >= 0.85


def test_impossible_locally_flagged():
    gate = CalibrationGate()
    report = gate.assess("I just googled it and the latest version is 5.2.")
    assert report.overall is EpistemicStatus.IMPOSSIBLE_LOCALLY
    assert report.flagged_impossible >= 1
    assert "unverifiable locally" in report.calibrated_answer.lower()


def test_failed_verification_lowers_confidence():
    gate = CalibrationGate()

    class _V:
        checked = True
        ok = False

    good = gate.assess("This is the answer.")
    bad = gate.assess("This is the answer.", verification=_V())
    assert bad.confidence < good.confidence


def test_singleton():
    assert get_calibration_gate() is get_calibration_gate()
