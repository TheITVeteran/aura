"""One ongoing condition is one event, not forty a minute.

Measured in Bryan's live neural feed on 2026-08-03: two "Deferring background
local endpoint" lines every two to three seconds, continuously, burying every
actual thought. The suppression existed; it compared the whole reason string,
and the reason carries live measurements, so consecutive samples at 66.6% and
66.5% read as different events.
"""

from __future__ import annotations

from core.brain.llm_health_router import _deferral_reason_kind

SAMPLES = [
    "desktop_background_headroom:Reflex:66.6%/21.3GB(need <66.0% and >=20.0GB)",
    "desktop_background_headroom:Reflex:66.5%/21.4GB(need <66.0% and >=20.0GB)",
    "desktop_background_headroom:Reflex:66.4%/21.5GB(need <66.0% and >=20.0GB)",
    "desktop_background_headroom:Reflex:67.4%/20.9GB(need <66.0% and >=20.0GB)",
    "desktop_background_headroom:Reflex:68.1%/20.4GB(need <66.0% and >=20.0GB)",
]


def test_drifting_measurements_are_one_condition():
    """These five lines are the same deferral sampled five times."""
    assert len({_deferral_reason_kind(s) for s in SAMPLES}) == 1


def test_a_different_endpoint_is_a_different_event():
    """Reflex and Brainstem defer for different budgets; both deserve a line."""
    reflex = _deferral_reason_kind(SAMPLES[0])
    brainstem = _deferral_reason_kind(
        "desktop_background_headroom:Brainstem:66.6%/21.3GB(need <62.0% and >=22.0GB)"
    )
    assert reflex != brainstem


def test_a_different_cause_is_a_different_event():
    """Collapsing distinct causes would hide a real change of state."""
    assert _deferral_reason_kind("model_lane_busy:Reflex") != _deferral_reason_kind(
        SAMPLES[0]
    )


def test_the_cause_survives_the_stripping():
    """The key must stay readable: it is what an operator sees when debugging."""
    kind = _deferral_reason_kind(SAMPLES[0])
    assert "desktop_background_headroom" in kind
    assert "Reflex" in kind


def test_it_is_total_on_junk():
    for bad in ("", None, 12345, "no digits here"):
        assert isinstance(_deferral_reason_kind(bad), str)
