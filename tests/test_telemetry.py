"""TelemetryPayload contract.

This file was three lines that constructed a payload and printed it. It
collected no tests, ran no assertions, and counted as telemetry coverage
purely because of its name.

Telemetry ids and units are a contract in this repo — `core/fsw/
telemetry_dictionary.py` says ids are never reused — so the payload carrying
them deserves an actual test.
"""

from __future__ import annotations

import pytest

from core.schemas import TelemetryPayload


def test_payload_accepts_a_nominal_reading():
    payload = TelemetryPayload(energy=80.0, curiosity=50.0, frustration=0.0, confidence=100.0)
    assert payload.energy == 80.0
    assert payload.curiosity == 50.0
    assert payload.frustration == 0.0
    assert payload.confidence == 100.0


def test_optional_channels_default_rather_than_erroring():
    """A partial reading must not fail — telemetry arrives incomplete."""
    payload = TelemetryPayload(energy=1.0, curiosity=1.0, frustration=0.0, confidence=1.0)
    assert payload.cpu_usage == 0.0
    assert payload.ram_usage == 0.0
    assert payload.coherence == 0.0
    assert payload.vitality == 0.0
    assert payload.surprise == 0.0


def test_gwt_winner_has_a_placeholder_rather_than_empty():
    payload = TelemetryPayload(energy=1.0, curiosity=1.0, frustration=0.0, confidence=1.0)
    assert payload.gwt_winner == "--"


def _payload(**overrides):
    base = {"energy": 1.0, "curiosity": 1.0, "frustration": 0.0, "confidence": 1.0}
    base.update(overrides)
    return TelemetryPayload(**base)


def test_unusable_values_clamp_rather_than_reject():
    """Clamping is deliberate, and the reason is worth pinning.

    This payload is built inside the heartbeat and published to the
    websocket. A ValidationError there would not protect the UI — it would
    kill the telemetry stream and freeze the dashboard on its last good
    frame while the runtime looked healthy. A gauge pinned at 100 is a
    visible, self-explaining wrong; a frozen dashboard is an invisible one.
    """
    assert _payload(energy="hot").energy == 100.0


def test_numeric_strings_are_coerced_not_discarded():
    assert _payload(energy="80.5").energy == 80.5


def test_out_of_range_values_are_clamped_to_the_declared_bounds():
    assert _payload(energy=1e9).energy == 100.0
    assert _payload(energy=-50.0).energy == 0.0


def test_non_finite_values_never_reach_the_browser():
    """inf and nan both validated cleanly under a bare ge=0.0 bound.

    A NaN renders as "NaN" in a gauge; an unbounded energy silently rescales
    every chart on the page.
    """
    import math

    for bad in (float("inf"), float("-inf"), float("nan")):
        value = _payload(energy=bad).energy
        assert math.isfinite(value), f"{bad!r} survived as {value!r}"
        assert 0.0 <= value <= 100.0


def test_normalised_consciousness_channels_stay_within_zero_to_one():
    assert 0.0 <= _payload(coherence=17.0).coherence <= 1.0
    assert 0.0 <= _payload(vitality=-3.0).vitality <= 1.0


def test_payload_round_trips_through_serialisation():
    payload = TelemetryPayload(
        energy=12.5, curiosity=33.0, frustration=1.5, confidence=99.0, narrative="steady"
    )
    dumped = payload.model_dump()
    restored = TelemetryPayload(**dumped)
    assert restored.model_dump() == dumped
    assert restored.narrative == "steady"


def test_declared_channels_are_present_and_named():
    """Ids are a contract; a silent rename would break every consumer."""
    fields = set(TelemetryPayload.model_fields)
    for channel in ("energy", "curiosity", "frustration", "confidence", "coherence", "vitality"):
        assert channel in fields
