"""Tests for cross-tier verification (weak generator, strong verifier)."""
from __future__ import annotations

import pytest

from core.brain.cross_tier_verifier import CrossTierVerifier
from core.brain.reasoning_amplifier import DeliberationEngine


def _strong(reply):
    async def _gen(prompt):
        return reply
    return _gen


@pytest.mark.asyncio
async def test_strong_tier_confirms_correct():
    v = CrossTierVerifier(_strong("VERDICT: CORRECT"))
    r = await v.verify("2+2?", "4")
    assert r.ok and not r.corrected and "verified" in r.critique


@pytest.mark.asyncio
async def test_strong_tier_corrects_wrong_answer():
    v = CrossTierVerifier(_strong("VERDICT: INCORRECT\nCORRECTED: 4"))
    r = await v.verify("2+2?", "5")
    assert r.ok and r.corrected and r.answer == "4"


@pytest.mark.asyncio
async def test_strong_tier_flags_doubt_without_correction():
    v = CrossTierVerifier(_strong("VERDICT: INCORRECT\n(no correction given)"))
    r = await v.verify("hard?", "maybe")
    assert not r.ok and not r.corrected


@pytest.mark.asyncio
async def test_unavailable_strong_tier_does_not_block():
    v = CrossTierVerifier(_strong(""))
    r = await v.verify("q", "a")
    assert r.ok and r.answer == "a" and "unavailable" in r.critique


# ── integration with the amplifier ────────────────────────────────────────

async def _all_valid(text):
    return True, []


@pytest.mark.asyncio
async def test_amplifier_uses_cross_tier_to_correct_winner():
    # the 32B converges (wrongly) on "5"; the 72B verifier corrects it to "4".
    seq = iter(["Answer: 5", "Answer: 5", "Answer: 5"])

    async def _gen(q, t):
        return next(seq)

    eng = DeliberationEngine(n_samples=3)
    cross = CrossTierVerifier(_strong("VERDICT: INCORRECT\nCORRECTED: 4"))
    r = await eng.deliberate("2+2?", _gen, cross_tier=cross, verify=_all_valid)
    assert r.answer == "4" and r.verified            # strong tier overrode the weak consensus


@pytest.mark.asyncio
async def test_amplifier_cross_tier_boosts_confidence_on_confirm():
    seq = iter(["Answer: 4", "Answer: 4", "Answer: green"])

    async def _gen(q, t):
        return next(seq)

    eng = DeliberationEngine(n_samples=3)
    cross = CrossTierVerifier(_strong("VERDICT: CORRECT"))
    r = await eng.deliberate("2+2?", _gen, cross_tier=cross, verify=_all_valid)
    assert "4" in r.answer and r.verified and r.confidence >= 0.7
