"""What she says out loud early has to survive what her governance decides late.

Speaking a reply while it is still forming buys the whole latency budget and
introduces exactly one new failure: the clauses spoken early are
pre-stabilisation, so the finished reply can disagree with words the listener
has already heard. Silence there is not neutral — it leaves them holding a
sentence Aura no longer stands behind, and every later reference to it is a
hallucination from their side of the conversation.

The interesting property is *which* text the disagreement is measured
against. "Released to the synthesiser", "sent to the client", and "actually
heard" are three different strings, and the correction context claims *the
user heard X*. If that claim is false, the fix reintroduces the bug it exists
to prevent.
"""
from __future__ import annotations

import asyncio

from core.conversation.reply_stream import ReplyStreamChannel, reconcile
from core.voice.duplex.governed_stream import govern_clause, stream_governed_reply


def test_a_late_revision_is_reported_against_what_was_heard() -> None:
    """The heard prefix, not the released one, is what she apologises for."""
    verdict = reconcile("The answer is yes", "The answer is no")
    assert verdict.diverged
    assert '"The answer is yes"' in verdict.correction_context
    assert '"The answer is no"' in verdict.correction_context


def test_an_addition_is_not_a_revision() -> None:
    """Governance adding a sentence is the ordinary case, not a correction."""
    verdict = reconcile("Yes, that works.", "Yes, that works. Though it is slower.")
    assert verdict.consistent
    assert verdict.remainder == "Though it is slower."
    assert verdict.correction_context == ""


def test_clauses_are_governed_before_they_are_spoken() -> None:
    """Nothing ungoverned reaches the synthesiser, however early it is released."""

    async def exercise() -> None:
        channel = ReplyStreamChannel()
        channel.attach_loop(asyncio.get_running_loop())
        spoken: list[str] = []

        # A real marker from the runtime's own scaffold list, not an invented
        # one — a test that strips a format nothing emits proves nothing.
        for chunk in ("The answer is yes. ", "[TOOL RESULT: web_search] rows=3 ", "Done."):
            channel.publish(chunk)
        channel.close()

        outcome = await stream_governed_reply(
            channel.drain(timeout_s=1.0),
            first_max_chars=24,
            max_chars=180,
            speak=spoken.append,
        )
        joined = " ".join(spoken)
        assert "TOOL RESULT" not in joined
        assert "The answer is yes" in joined
        assert outcome.chunks >= 1

    asyncio.run(exercise())


def test_a_refused_clause_ends_the_turn_rather_than_leaving_a_hole() -> None:
    """The sentence after a suppressed one usually depends on it.

    Speaking around the hole produces a reply that is individually governed
    clause by clause and incoherent as a whole.
    """
    safe, refusal, stop_after = govern_clause("[SKILL EXECUTION] all of it")
    assert not safe.strip()
    assert refusal
    assert stop_after


def test_an_interrupted_reply_is_not_a_governance_disagreement() -> None:
    """Two different failures that must not be confused.

    Being cut off is the listener's doing and the interruption machinery
    already records the heard prefix and hands the unheard tail forward.
    Layering a governance apology on top would apologise for a revision to
    text they never reached.
    """
    # The heard prefix is genuinely a prefix of the final reply; the reply is
    # simply longer because they stopped it. That is `consistent`, with a
    # remainder — never `diverged`.
    verdict = reconcile("So the first thing is", "So the first thing is that it depends.")
    assert verdict.consistent
    assert not verdict.diverged


def test_the_stream_stops_at_a_clause_boundary_when_asked() -> None:
    """A barge-in stops the reply between clauses, never mid-word."""

    async def exercise() -> None:
        channel = ReplyStreamChannel()
        channel.attach_loop(asyncio.get_running_loop())
        spoken: list[str] = []
        for chunk in ("First clause here. ", "Second clause here. ", "Third clause here."):
            channel.publish(chunk)
        channel.close()

        outcome = await stream_governed_reply(
            channel.drain(timeout_s=1.0),
            first_max_chars=24,
            max_chars=64,
            speak=spoken.append,
            should_continue=lambda: len(spoken) < 1,
        )
        assert outcome.refused_reason == "the listener interrupted"
        for piece in spoken:
            assert piece == piece.strip()

    asyncio.run(exercise())
