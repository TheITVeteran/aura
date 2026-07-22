"""Tests for the hardened math verifier — derive the exact answer from the question."""
from __future__ import annotations

import asyncio

import pytest

from core.brain.verifiers.math_engine import MathTruthEngine, _derive_exact_answer


def _verify(candidate: str, objective: str):
    return asyncio.run(MathTruthEngine().verify(candidate, context={"objective": objective}))


# ── derivation ──────────────────────────────────────────────────────────────
def test_derive_mod():
    assert _derive_exact_answer("What is 123456 mod 7?") == ("123456 mod 7", "4")


def test_derive_power_words_and_caret():
    assert _derive_exact_answer("17 to the power of 4")[1] == "83521"
    assert _derive_exact_answer("2^10")[1] == "1024"


def test_derive_gcd():
    assert _derive_exact_answer("GCD of 1071 and 462")[1] == "21"


def test_derive_times():
    assert _derive_exact_answer("what is 137 times 248")[1] == str(137 * 248)


def test_derive_caps_huge_power():
    # exponent beyond the bound is not derived (keeps it exact + cheap)
    assert _derive_exact_answer("2 to the power of 100000") is None


def test_derive_none_for_non_arithmetic():
    assert _derive_exact_answer("explain why the sky is blue") is None


# ── verification soundness ──────────────────────────────────────────────────
def test_wrong_mod_is_hard_fail():
    r = _verify("123456 mod 7 is 5", "What is 123456 mod 7?")
    assert r.ok is False and r.checked is True


def test_right_mod_passes():
    r = _verify("The remainder is 4.", "What is 123456 mod 7?")
    assert r.ok is True and r.checked is True


def test_wrong_power_is_hard_fail():
    r = _verify("17^4 = 81000", "What is 17 to the power of 4?")
    assert r.ok is False and r.checked is True


def test_vacuous_prose_rejected_for_derivable_question():
    # The previous poison: prose with no answer used to pass. Now it fails the derived check.
    r = _verify("I am not sure; here is a guess.", "What is 123456 mod 7?")
    assert r.ok is False and r.checked is True


def test_non_derivable_prose_stays_vacuous():
    # No derivable expression and no '=' claim -> nothing to check (unchanged behavior).
    r = _verify("It depends on the context.", "Tell me about prime numbers.")
    assert r.checked is False


def test_correct_answer_among_reasoning_passes():
    r = _verify("First 17^2 = 289, then 289^2 = 83521. Answer: 83521.", "What is 17 to the power of 4?")
    assert r.ok is True and r.checked is True


def test_correct_intermediate_cannot_hide_wrong_final_answer():
    r = _verify(
        "I computed 17^4 as 83521, but my final answer is 81000.",
        "What is 17 to the power of 4?",
    )
    assert r.ok is False and r.checked is True


def test_last_answer_envelope_is_authoritative():
    r = _verify(
        "<answer>83521</answer>\nCorrection: <answer>81000</answer>",
        "What is 17 to the power of 4?",
    )
    assert r.ok is False and r.checked is True
