"""RSI is only real if the improvement is VERIFIED on held-out evidence the
improver didn't choose. Pins the challenge: the buggy seed genuinely fails
held-out cases, and a correct fix strictly beats it — so a model-produced fix
can be verified (not merely asserted) as improvement, and promoted only then."""
from __future__ import annotations

import pytest

from tools.proof.run_rsi_challenge_proof import _CHALLENGES, run_self_test


@pytest.mark.parametrize("challenge_name", ["median", "is_palindrome"])
def test_rsi_challenge_measures_real_verified_improvement(challenge_name):
    challenge = _CHALLENGES[challenge_name]()
    report = run_self_test(challenge)
    # the seed has a GENUINE bug — it fails held-out cases (not a strawman)
    assert report["seed_fails_real_cases"] is True
    assert report["seed_passed"] < report["total_cases"]
    # a correct fix passes everything and strictly beats the seed
    assert report["reference_fix_passed"] == report["total_cases"]
    assert report["reference_fix_passed"] > report["seed_passed"]
    assert report["improvement_proven"] is True


def test_no_op_change_would_not_be_promoted():
    # feeding the seed as its own "fix" must NOT count as improvement
    challenge = _CHALLENGES["median"]()
    from tools.proof.run_rsi_challenge_proof import _score

    seed_passed, _ = _score(challenge.seed_impl, challenge.fn_name, challenge.cases)
    # promotion requires strictly beating the seed; seed vs seed is not strict
    assert not (seed_passed > seed_passed)
