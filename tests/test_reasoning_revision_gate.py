"""The revision gate's one job: make a second pass safe.

The measured failure of naive self-correction is right-to-wrong regression —
a "review" flips a correct answer to an incorrect one more often than the
reverse. These tests pin that this gate CANNOT do that: a verified-correct
incumbent is never displaced except by strictly stronger verified evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.brain.reasoning_revision_gate import (
    RevisionVerdict,
    decide_revision,
    quality_bound,
)


@dataclass
class V:
    """A minimal VerificationResult-shaped verdict."""

    ok: bool
    checked: bool
    score: float = 0.5


CLEAN = V(ok=True, checked=True, score=0.9)
CLEAN_STRONGER = V(ok=True, checked=True, score=0.99)
CLEAN_WEAK = V(ok=True, checked=True, score=0.55)
FAILED = V(ok=False, checked=True, score=0.2)
UNCHECKED = V(ok=True, checked=False, score=0.9)


# ── the core guarantee: no right-to-wrong regression ───────────────────────


def test_verified_incumbent_is_never_replaced_by_a_failing_challenger():
    d = decide_revision(CLEAN, FAILED, incumbent_reliability=0.9,
                        challenger_reliability=0.9)
    assert d.verdict is RevisionVerdict.KEEP_INCUMBENT
    assert d.accept is False
    assert d.reason == "challenger_failed_verification"


def test_verified_incumbent_is_never_replaced_by_an_unchecked_challenger():
    d = decide_revision(CLEAN, UNCHECKED, incumbent_reliability=0.9)
    assert d.verdict is RevisionVerdict.KEEP_INCUMBENT
    assert d.reason == "challenger_unchecked_incumbent_verified"


def test_a_marginally_higher_score_does_not_trigger_a_revision():
    """The heart of the anti-churn rule: a slightly higher soft score is
    inside the confidence interval and must NOT flip the answer."""
    d = decide_revision(CLEAN, V(ok=True, checked=True, score=0.91),
                        incumbent_reliability=0.9, challenger_reliability=0.9)
    assert d.verdict is RevisionVerdict.KEEP_INCUMBENT
    assert d.reason == "insufficient_evidence_to_revise"


# ── it still accepts genuinely better evidence ─────────────────────────────


def test_no_incumbent_adopts_the_first_answer():
    d = decide_revision(None, CLEAN, has_incumbent=False)
    assert d.verdict is RevisionVerdict.ADOPT_FIRST
    assert d.accept is True


def test_verified_challenger_replaces_an_unverified_incumbent():
    d = decide_revision(UNCHECKED, CLEAN, challenger_reliability=0.9)
    assert d.verdict is RevisionVerdict.ACCEPT_CHALLENGER
    assert d.reason == "challenger_verified_incumbent_not"


def test_verified_challenger_replaces_a_failing_incumbent():
    d = decide_revision(FAILED, CLEAN, incumbent_reliability=0.9,
                        challenger_reliability=0.9)
    assert d.verdict is RevisionVerdict.ACCEPT_CHALLENGER


def test_a_decisively_stronger_verified_answer_is_accepted():
    """A weak-but-clean incumbent yields to a much stronger clean answer when
    a reliable verifier backs it past the margin."""
    d = decide_revision(CLEAN_WEAK, CLEAN_STRONGER,
                        incumbent_reliability=0.95, challenger_reliability=0.95)
    assert d.verdict is RevisionVerdict.ACCEPT_CHALLENGER
    assert d.reason == "challenger_lower_bound_clears_incumbent"


# ── reliability governs decisiveness ───────────────────────────────────────


def test_an_unreliable_verifier_cannot_displace_a_clean_incumbent():
    """Even a high raw score cannot flip the answer when the verifier backing
    it has no measured reliability — its interval is too wide to be decisive."""
    d = decide_revision(CLEAN_WEAK, CLEAN_STRONGER,
                        incumbent_reliability=0.9, challenger_reliability=0.1)
    assert d.verdict is RevisionVerdict.KEEP_INCUMBENT
    assert d.reason == "challenger_verifier_too_unreliable_to_displace"


def test_reliability_tightens_the_interval():
    proven = quality_bound(CLEAN, reliability=0.95)
    unmeasured = quality_bound(CLEAN, reliability=0.5)
    assert proven.half_width < unmeasured.half_width
    assert proven.upper - proven.lower < unmeasured.upper - unmeasured.lower


def test_unchecked_verdict_has_a_wide_interval_and_no_hard_ok_credit():
    bound = quality_bound(UNCHECKED, reliability=0.9)
    assert bound.checked is False
    # An unchecked verdict carries a large half-width — it is uncertainty,
    # not evidence, even when the verifier is otherwise reliable.
    assert bound.half_width >= 0.5
    checked_equivalent = quality_bound(CLEAN, reliability=0.9)
    assert bound.half_width > checked_equivalent.half_width


def test_none_verdict_is_the_weakest_evidence():
    bound = quality_bound(None)
    assert bound.checked is False
    assert bound.hard_ok is False
    assert bound.lower == 0.0


# ── both-unverified tie-breaking stays conservative ────────────────────────


def test_two_unverified_answers_keep_the_incumbent_on_a_close_call():
    d = decide_revision(V(ok=True, checked=False, score=0.6),
                        V(ok=True, checked=False, score=0.65))
    assert d.verdict is RevisionVerdict.KEEP_INCUMBENT
    assert d.reason == "both_unverified_no_clear_gain"


def test_two_unverified_answers_move_only_on_a_large_gap():
    d = decide_revision(V(ok=True, checked=False, score=0.3),
                        V(ok=True, checked=False, score=0.9))
    assert d.verdict is RevisionVerdict.ACCEPT_CHALLENGER


# ── property: a stream of revisions never leaves a verified answer ─────────


def test_monotonicity_over_a_revision_stream():
    """Once a verified-correct answer is held, no sequence of challengers —
    failing, unchecked, or weakly-scored — can dislodge it, but a strictly
    stronger verified one still can. This is the non-regression property."""
    incumbent = CLEAN
    incumbent_rel = 0.95
    challengers = [
        (FAILED, 0.9),
        (UNCHECKED, 0.9),
        (V(ok=True, checked=True, score=0.92), 0.9),  # marginal
        (V(ok=True, checked=True, score=0.5), 0.9),   # worse
        (V(ok=True, checked=True, score=0.2), 0.1),   # worse + unreliable
    ]
    for verdict, rel in challengers:
        d = decide_revision(incumbent, verdict, incumbent_reliability=incumbent_rel,
                            challenger_reliability=rel)
        assert d.accept is False, (verdict, d.reason)

    # A genuinely, decisively better verified answer is still accepted.
    d = decide_revision(V(ok=True, checked=True, score=0.6), CLEAN_STRONGER,
                        incumbent_reliability=0.95, challenger_reliability=0.95)
    assert d.accept is True


# ── the bounded multi-pass loop ────────────────────────────────────────────


import asyncio  # noqa: E402

from core.brain.reasoning_revision_gate import deliberate_best_of  # noqa: E402


def test_deliberate_keeps_the_first_verified_answer_against_worse_passes():
    """A later pass that is failing/unchecked/weaker must not replace an early
    verified-correct answer — spending more passes can only help."""
    answers = ["correct", "wrong", "unchecked", "meh"]
    verdicts = {
        "correct": CLEAN,
        "wrong": FAILED,
        "unchecked": UNCHECKED,
        "meh": V(ok=True, checked=True, score=0.6),
    }

    async def solve(i):
        return answers[i]

    async def verify(answer):
        return verdicts[answer]

    result = asyncio.run(
        deliberate_best_of(solve, verify, max_passes=4,
                           reliability_of=lambda _v: 0.95)
    )
    assert result.answer == "correct"
    assert result.verified is True
    assert result.rejected_revisions == 3


def test_deliberate_adopts_a_strictly_better_later_pass():
    answers = ["weak", "strong"]
    verdicts = {"weak": CLEAN_WEAK, "strong": CLEAN_STRONGER}

    async def solve(i):
        return answers[i]

    async def verify(answer):
        return verdicts[answer]

    result = asyncio.run(
        deliberate_best_of(solve, verify, max_passes=2,
                           reliability_of=lambda _v: 0.95)
    )
    assert result.answer == "strong"
    assert result.accepted_revisions == 1


def test_deliberate_can_stop_early_when_verified():
    calls = {"n": 0}

    async def solve(i):
        calls["n"] += 1
        return "correct"

    async def verify(answer):
        return CLEAN

    result = asyncio.run(
        deliberate_best_of(solve, verify, max_passes=5,
                           reliability_of=lambda _v: 0.95,
                           stop_when_verified=True)
    )
    assert calls["n"] == 1
    assert result.verified is True


def test_deliberate_tolerates_sync_callables_and_empty_answers():
    def solve(i):
        return "" if i == 0 else "correct"

    def verify(answer):
        return CLEAN

    result = asyncio.run(deliberate_best_of(solve, verify, max_passes=2))
    assert result.answer == "correct"
