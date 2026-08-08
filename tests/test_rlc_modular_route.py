"""Modular claims must route, or the repair chain cannot start.

The battery's `modular` family literally asks the model to "apply each
operation modulo 19", yet modular claims had no deterministic route: every
such atom fell through to no_sound_deterministic_route, so nothing was
verified or refuted, no disagreement was recorded, no repair was requested,
and answer promotion had no candidates to consider.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.deterministic_verifier_router import (  # noqa: E402
    RouteOutcome,
    _modular_verdict,
)


def test_correct_modular_claims_verify():
    for fragment in (
        "17 + 13 mod 19 = 11",
        "(17 + 13) mod 19 = 11",
        "30 - 6 mod 19 = 5",
        "7 * 3 = 21 (mod 19)",
        "17 + 13 % 19 = 11",
    ):
        verdict = _modular_verdict(fragment)
        assert verdict is not None, f"no route for {fragment!r}"
        assert verdict[0] is RouteOutcome.VERIFIED, (fragment, verdict[2])


def test_wrong_modular_claims_are_refuted():
    verdict = _modular_verdict("17 + 13 mod 19 = 12")
    assert verdict is not None
    assert verdict[0] is RouteOutcome.REFUTED
    assert verdict[2]["failure_codes"]


def test_a_zero_modulus_refutes_rather_than_raises():
    verdict = _modular_verdict("4 + 4 mod 0 = 1")
    assert verdict is not None
    assert verdict[0] is RouteOutcome.REFUTED
    assert "zero_modulus" in verdict[2]["failure_codes"]


def test_text_without_modular_claims_does_not_route():
    assert _modular_verdict("the residue is five") is None


def test_modular_is_checked_before_plain_arithmetic():
    """"17 + 13 mod 19 = 11" contains no true plain-integer claim; routing it
    as ordinary arithmetic would refute a correct modular statement."""
    from core.brain.llm.latent_cortex.deterministic_verifier_router import _route_atom

    outcome, verifier, _detail = _route_atom(
        "17 + 13 mod 19 = 11", {"kind": "assertion"}, code_atom_count=0
    )
    assert verifier == "exact_modular_arithmetic"
    assert outcome == RouteOutcome.VERIFIED.value
