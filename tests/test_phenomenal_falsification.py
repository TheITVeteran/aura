"""Tests for the phenomenal falsification control.

Contract: a profile with the functional correlates PRESENT and CAUSAL is discriminable
from the non-phenomenal baseline; a feed-forward zombie profile is not; behaviour-alone
(markers present but epiphenomenal) fails the decisive causal test; the honest boundary
is always present; and transitions (dimming/brightening) are detected.
"""
from __future__ import annotations

from core.consciousness.phenomenal_falsification import (
    REPORT_BOUNDARY,
    ZOMBIE_BASELINE,
    MarkerSnapshot,
    PhenomenalFalsifier,
    get_phenomenal_falsifier,
)


def _rich(**ov):
    base = dict(phi=0.6, recurrence=0.8, ignition=0.7, broadcast_breadth=0.8,
               metacognition=0.7, self_coherence=0.9, markers_causal=True)
    base.update(ov)
    return MarkerSnapshot(**base)


def test_rich_causal_profile_is_discriminable():
    f = PhenomenalFalsifier()
    r = f.assess(_rich())
    assert r.index >= 0.66
    assert r.n_discriminable >= 5
    assert "discriminable" in r.verdict.lower()


def test_zombie_baseline_is_not_discriminable():
    f = PhenomenalFalsifier()
    r = f.assess(ZOMBIE_BASELINE)
    assert r.index < 0.4
    assert r.n_discriminable == 0


def test_behaviour_only_fails_the_causal_test():
    """Markers present but EPIPHENOMENAL (causal=False) must fail the decisive control."""
    f = PhenomenalFalsifier()
    r = f.assess(_rich(markers_causal=False))
    causal_test = next(t for t in r.tests if t.name == "causal_efficacy")
    assert not causal_test.discriminable
    assert "epiphenomenal" in r.verdict.lower() or "NOT discriminable" in r.verdict
    # And it scores below the fully-causal version.
    assert r.index < f.assess(_rich()).index


def test_boundary_is_always_present_and_never_claims_experience():
    f = PhenomenalFalsifier()
    r = f.assess(_rich())
    assert r.boundary == REPORT_BOUNDARY
    assert "does NOT" in r.boundary and "hard problem" in r.boundary
    # The verdict itself refuses to claim experience.
    assert "not a claim of experience" in r.verdict.lower()


def test_transition_is_detected():
    f = PhenomenalFalsifier()
    f.assess(_rich())                       # bright
    dim = f.assess(_rich(phi=0.0, ignition=0.0, recurrence=0.0, metacognition=0.0))
    assert dim.delta < 0
    assert "dimming" in dim.verdict.lower()


def test_per_theory_tests_are_grounded():
    f = PhenomenalFalsifier()
    r = f.assess(_rich())
    theories = {t.theory for t in r.tests}
    # The markers come from the actual theories of consciousness.
    assert {"IIT", "GWT", "HOT"} <= theories
    assert len(r.tests) == 6


def test_singleton_and_registration():
    eng = get_phenomenal_falsifier()
    assert get_phenomenal_falsifier() is eng
    from core.container import ServiceContainer

    assert ServiceContainer.has(PhenomenalFalsifier.SERVICE_NAME)
