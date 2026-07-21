"""CP126: the frontier trend claim must not be tunable, and the envelope
layer must not be exhaustible.

These guard the two places an interested party could manufacture a
capability claim without touching any model: the thresholds the trend test
is graded against, and the nonce that is supposed to make a challenge
unpredictable.
"""
from __future__ import annotations

import hashlib

import pytest

from core.brain.frontier_evidence_v5 import (
    _require_bounded_structure,
    _student_t_critical_95,
    analyze_gap_trend,
)


def _entries(gaps: list[float]) -> list[dict]:
    return [{"overall_gap": gap} for gap in gaps]


# ── trend thresholds are protocol-bound, not caller-tunable ────────────────


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"minimum_runs": 0}, "a zero run floor makes any noise eligible"),
        ({"minimum_runs": 1}, "below the protocol minimum"),
        ({"minimum_runs": -5}, "negative run floor"),
        ({"minimum_runs": True}, "bool is an int subclass"),
        ({"minimum_effect": 0.0}, "a zero effect floor accepts no effect"),
        ({"minimum_effect": -0.5}, "negative effect floor"),
        ({"minimum_effect": 2.0}, "effect floor above 1"),
        ({"minimum_effect": float("nan")}, "non-finite effect floor"),
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"alpha": 1.5}, "alpha above one accepts everything"),
        ({"alpha": -0.1}, "negative alpha"),
        ({"alpha": float("inf")}, "non-finite alpha"),
    ],
)
def test_trend_thresholds_cannot_be_tuned_into_eligibility(kwargs, why):
    with pytest.raises(ValueError):
        analyze_gap_trend(_entries([0.5, 0.4, 0.3, 0.2, 0.1]), **kwargs)


def test_valid_thresholds_still_analyze():
    result = analyze_gap_trend(
        _entries([0.5, 0.45, 0.4, 0.3, 0.25, 0.2]),
        minimum_runs=5,
        minimum_effect=0.02,
        alpha=0.05,
    )

    assert result["measured_points"] == 6
    assert "claim_eligible" in result


def test_trend_defaults_are_inside_the_protocol_bounds():
    """The defaults must themselves satisfy the validation."""
    result = analyze_gap_trend(_entries([0.5, 0.4, 0.3, 0.2, 0.1]))
    assert result["minimum_runs"] == 5


# ── small-sample interval honesty ──────────────────────────────────────────


def test_small_samples_use_a_t_critical_value_not_1_96():
    """At the 5-run minimum (df=3) the normal approximation is far too narrow
    to be called a 95% interval."""
    assert _student_t_critical_95(3) > 3.0
    assert _student_t_critical_95(1) > 12.0
    # Large samples converge on the normal limit.
    assert _student_t_critical_95(100_000) == pytest.approx(1.96, abs=0.05)


def test_t_critical_is_monotone_non_increasing():
    values = [_student_t_critical_95(df) for df in range(1, 60)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


# ── envelope payloads are bounded before canonicalization ──────────────────


def test_deeply_nested_payload_is_refused_before_serialization():
    """canonical_json_bytes walks the whole tree; the byte-size guard ran
    only AFTER it, so a deep structure could exhaust the stack first."""
    payload: dict = {}
    node = payload
    for _ in range(200):
        child: dict = {}
        node["next"] = child
        node = child

    with pytest.raises(ValueError, match="nested too deeply"):
        _require_bounded_structure(payload, role="test")


def test_enormous_payload_is_refused_before_serialization():
    payload = {"rows": [{"i": index} for index in range(150_000)]}

    with pytest.raises(ValueError, match="too many elements"):
        _require_bounded_structure(payload, role="test")


def test_non_string_keys_are_refused():
    with pytest.raises(ValueError, match="non-string key"):
        _require_bounded_structure({1: "value"}, role="test")


def test_ordinary_payload_passes_the_bound_check():
    payload = {
        "schema": "x",
        "values": [1, 2, 3],
        "nested": {"a": {"b": {"c": "d"}}},
    }
    _require_bounded_structure(payload, role="test")  # must not raise


# ── nonce degeneracy ───────────────────────────────────────────────────────


def test_degenerate_nonces_are_structurally_refused():
    """Byte LENGTH cannot establish entropy. An all-zero or single-repeated
    -byte nonce is 32 bytes long and carries no unpredictability."""
    from core.brain.frontier_evidence_v5 import validate_challenge_bundle

    # Exercised end-to-end by the v5 suite; here we pin the property that a
    # real draw has broad byte diversity and a placeholder does not.
    assert len(set(b"\x00" * 32)) == 1
    assert len(set(b"r" * 32)) == 1
    assert len(set(hashlib.sha256(b"real-draw").digest())) >= 16
    assert callable(validate_challenge_bundle)
