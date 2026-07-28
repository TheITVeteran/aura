"""Persisted immune state is untrusted input.

CP126 e2f39609: antigens were rebuilt from disk with bare float(...), so NaN
scores, out-of-range pressures, malformed vectors, arbitrary source domains
and unbounded context entered live immune state.

NaN is the dangerous one. It propagates silently through every comparison
that decides whether to act — `nan > threshold` is False — so a poisoned
antigen reads as CALM rather than as unreadable, which is the failure this
whole campaign is about.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.adaptation.adaptive_immunity import Antigen


def _antigen(**overrides):
    payload = {"antigen_id": "a", "subsystem": "s", "vector": [0.5, 0.5]}
    payload.update(overrides)
    return Antigen.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    [
        "danger", "subsystem_need", "threat_probability", "resource_pressure",
        "error_load", "health_pressure", "temporal_pressure", "recurrence_pressure",
    ],
)
def test_a_nan_pressure_becomes_zero_not_nan(field):
    value = getattr(_antigen(**{field: float("nan")}), field)

    assert value == value          # not NaN
    assert value == 0.0


@pytest.mark.parametrize("bad", [float("inf"), -float("inf"), "text", None, object()])
def test_an_unreadable_pressure_falls_back(bad):
    assert _antigen(danger=bad).danger == 0.0


@pytest.mark.parametrize("value,expected", [(99.0, 1.0), (-5.0, 0.0), (0.4, 0.4)])
def test_pressures_are_clamped_to_the_unit_range(value, expected):
    assert _antigen(danger=value).danger == pytest.approx(expected)


def test_a_nan_in_the_vector_does_not_survive_clipping():
    """np.clip passes NaN through — that is why nan_to_num comes first."""
    antigen = _antigen(vector=[float("nan"), 2.0, -1.0])

    assert bool(np.all(np.isfinite(antigen.vector)))
    assert float(antigen.vector.max()) <= 1.0
    assert float(antigen.vector.min()) >= 0.0


# --- source_domain gates substrate repair --------------------------------


def test_an_unknown_source_domain_fails_to_the_restrictive_one():
    """source_domain decides whether substrate repair may act, so an origin
    that cannot be trusted must not unlock it."""
    assert _antigen(source_domain="substrate_but_forged").source_domain == "environment"


@pytest.mark.parametrize("domain", ["substrate", "environment"])
def test_declared_domains_are_preserved(domain):
    assert _antigen(source_domain=domain).source_domain == domain


def test_a_missing_domain_still_defaults_to_substrate():
    """Absent is the documented default and is not a forgery."""
    assert _antigen().source_domain == "substrate"


# --- bounds --------------------------------------------------------------


def test_a_nan_timestamp_does_not_corrupt_age_computations():
    stamp = _antigen(timestamp=float("nan")).timestamp

    assert stamp == stamp and stamp > 0


def test_context_is_bounded():
    antigen = _antigen(context={f"k{i}": i for i in range(500)})

    assert len(antigen.context) <= 64


def test_a_non_mapping_context_is_discarded():
    assert _antigen(context="not a mapping").context == {}


def test_oversized_text_is_bounded():
    antigen = _antigen(stack_trace="x" * 100_000, error_signature="y" * 100_000)

    assert len(antigen.stack_trace) <= 4096
    assert len(antigen.error_signature) <= 4096


def test_an_ordinary_antigen_round_trips_unchanged():
    original = _antigen(danger=0.4, source_domain="environment", subsystem="mem")
    restored = Antigen.from_dict(original.to_dict())

    assert restored.danger == pytest.approx(0.4)
    assert restored.source_domain == "environment"
    assert restored.subsystem == "mem"
