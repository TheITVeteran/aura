"""A verifier that could not run has verified nothing.

Two confirmed defects, both instances of one mistake — reporting the *absence*
of a check as a *passed* check:

  V1  ``verify_reasoning`` caught its own exceptions and returned ``(True, [])``
      — "clean, no issues found". A crashed verifier therefore marked every
      candidate verifier-clean, and ``amplify()`` set ``verified=True`` with
      full (unlifted) confidence on a completely unchecked answer.

  V2  the tier-escalation path updated ``verifier_ok`` but not
      ``verifier_checked``, so an escalated answer inherited the *previous*
      candidate's "I evaluated something" and PROOF mode presented it as
      verified.
"""
from __future__ import annotations

import pytest

from core.brain.reasoning_amplifier import (
    VerifierOutcome,
    amplify,
    verify_reasoning,
    verify_reasoning_checked,
)

# ---------------------------------------------------------------------------
# V1 — the crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crashed_verifier_reports_unknown_not_pass(monkeypatch):
    """The headline defect, at its source."""

    class _Exploding:
        def audit_reasoning(self, _text):
            raise RuntimeError("symbolic bridge is down")

    import core.reasoning.symbolic_bridge as bridge

    monkeypatch.setattr(bridge, "SymbolicBridge", _Exploding)

    outcome, issues = await verify_reasoning_checked("2 + 2 = 5, therefore the sky is green")
    assert outcome is VerifierOutcome.UNKNOWN
    assert any("unavailable" in i for i in issues)


@pytest.mark.asyncio
async def test_boolean_shim_does_not_report_a_crash_as_clean(monkeypatch):
    """The old contract returned True here. True means 'I checked; it's fine'."""

    class _Exploding:
        def audit_reasoning(self, _text):
            raise RuntimeError("symbolic bridge is down")

    import core.reasoning.symbolic_bridge as bridge

    monkeypatch.setattr(bridge, "SymbolicBridge", _Exploding)

    ok, _issues = await verify_reasoning("anything at all")
    assert ok is False, "a crashed verifier reported the reasoning as clean"


@pytest.mark.asyncio
async def test_amplify_does_not_certify_when_the_verifier_crashed():
    """An unavailable verifier must not produce verified=True."""

    async def _crashing_verifier(_text):
        raise RuntimeError("verifier exploded")

    result = await amplify(
        ["The answer is 4.", "The answer is 4.", "The answer is 7."],
        verify=_crashing_verifier,
    )

    assert result.verified is False, "crashed verifier certified the answer"
    assert result.verifier_checked is False
    assert result.valid_n == 0
    # Confidence must not be lifted as though the winner were verifier-clean.
    assert result.confidence < result.agreement or result.agreement == 0


@pytest.mark.asyncio
async def test_amplify_still_answers_when_the_verifier_is_unavailable():
    """Fail-closed on the *claim*, not on the *answer*.

    We lose the ability to certify, not the ability to respond — the candidates
    are still clustered by self-consistency.
    """

    async def _crashing_verifier(_text):
        raise RuntimeError("verifier exploded")

    result = await amplify(
        ["The answer is 4.", "The answer is 4.", "The answer is 7."],
        verify=_crashing_verifier,
    )
    assert "4" in result.answer
    assert result.n == 3


@pytest.mark.asyncio
async def test_amplify_certifies_when_the_verifier_actually_passes():
    """The fix must not break the real verified path."""

    async def _clean(_text):
        return True, []

    result = await amplify(["The answer is 4.", "The answer is 4."], verify=_clean)
    assert result.verified is True
    assert result.verifier_checked is True
    assert result.valid_n == 2


@pytest.mark.asyncio
async def test_amplify_distinguishes_checked_failure_from_unchecked():
    """'Checked and dirty' and 'not checked' must not look the same."""

    async def _dirty(_text):
        return False, ["arithmetic: 2+2=5"]

    async def _crash(_text):
        raise RuntimeError("down")

    checked = await amplify(["The answer is 4."], verify=_dirty)
    unchecked = await amplify(["The answer is 4."], verify=_crash)

    assert checked.verified is False and checked.verifier_checked is True
    assert unchecked.verified is False and unchecked.verifier_checked is False


@pytest.mark.asyncio
async def test_verifier_with_a_bad_return_shape_is_unknown_not_pass():
    """A malformed verifier response establishes nothing."""

    async def _nonsense(_text):
        return "sure, looks fine"

    result = await amplify(["The answer is 4."], verify=_nonsense)
    assert result.verified is False
    assert result.verifier_checked is False


# ---------------------------------------------------------------------------
# V2 — the stale escalation state
# ---------------------------------------------------------------------------


class _Verdict:
    """Stub verifier verdict: `ok` = did not object, `checked` = actually evaluated."""

    def __init__(self, ok: bool, checked: bool, engine: str = "stub"):
        self.ok = ok
        self.checked = checked
        self.engine = engine
        self.issues: list[str] = []


class _StubVerifier:
    """Returns a checked failure first, then a *vacuous* pass on escalation.

    This is the exact shape of the defect: the escalated verdict passes
    (``ok=True``) while having evaluated nothing (``checked=False``).
    """

    def __init__(self):
        self.calls = 0

    async def verify(self, candidate, *, task_type="", context=None):
        self.calls += 1
        if "ESCALATED" in candidate:
            return _Verdict(ok=True, checked=False)   # vacuous pass
        return _Verdict(ok=False, checked=True)       # checked, and dirty


@pytest.mark.asyncio
async def test_v2_escalation_does_not_inherit_stale_checked_state():
    """An escalated answer that was never checked must not be presented as proven.

    Before the fix, the escalation branch set ``verifier_ok = True`` but left
    ``verifier_checked`` holding the *previous* verdict's True. PROOF mode reads
    ``verifier_ok and verifier_checked`` — so a vacuous escalated pass inherited
    "I evaluated something" from a different candidate and answered as proven.
    """
    from core.brain.reasoning_amplifier_v2 import (
        AmplificationRequest,
        ReasoningAmplifierV2,
        ReasoningMode,
    )

    async def _generate(_prompt, _temp):
        return "The answer is 4."

    async def _escalate(_prompt, _temp):
        return "ESCALATED: the answer is 4."

    amp = ReasoningAmplifierV2(
        _generate,
        verifier=_StubVerifier(),
        escalate_generate=_escalate,
    )

    answer = await amp.amplify(
        AmplificationRequest(
            objective="What is 2 + 2?",
            task_type="math",
            mode=ReasoningMode.PROOF,
            time_budget_s=10.0,
        )
    )

    # PROOF mode must refuse rather than certify an unchecked escalated answer.
    assert answer.verified is False, (
        "an escalated answer the verifier never checked was reported as verified "
        "— stale verifier_checked state"
    )
