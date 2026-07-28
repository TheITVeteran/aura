"""What was observed is remembered, even when the proof does not close.

LIVE DEFECT, 2026-07-27. Bryan watched Aura hold a long exchange with
ChatGPT about her own update signals — genuinely good material — and got:

    "I routed the ChatGPT conversation through the governed web_interlocutor
     skill, but I am not claiming a successful proof:
     sent_message_not_visible_after_dom_submit. Observed 6/8 turns;
     memory=none."

Six real turns were read off the page and thrown away. The early return on
``not turn.effect_verified`` jumped straight past summarize AND persist, so
a run that failed its PROOF also lost everything it had actually seen.

The consequence was worse than a lost transcript. Asked afterwards whether
she remembered the conversation, Aura said "I remember the conversation" and
then described a completely different one — the exchange she had just had
with Bryan. With nothing retained, the only material left to answer from was
the wrong conversation.

Refusing to CLAIM a completed proof is correct and unchanged. Refusing to
REMEMBER what was observed is a separate decision, and it was never the
right one: observation and proof are different things, and discarding the
evidence because the proof failed is backwards.
"""
from __future__ import annotations

import types

import pytest

from core.capabilities.web_interlocutor import (
    WebInterlocutorResult,
    WebInterlocutorSession,
    WebInterlocutorTurn,
)


def _turn(index: int, reply: str) -> WebInterlocutorTurn:
    return WebInterlocutorTurn(
        index, "sent", reply, "before", "after", 0.0, 0.0, False, "unverified",
    )


@pytest.fixture
def session_and_writes():
    writes: list = []

    class _Gateway:
        async def write(self, request):
            writes.append(request)
            return types.SimpleNamespace(record_id="rec-1", receipt_id="rcpt-1")

    session = WebInterlocutorSession.__new__(WebInterlocutorSession)
    session.memory_gateway = _Gateway()
    return session, writes


def _failed_run() -> WebInterlocutorResult:
    """The live shape: proof failed, six turns genuinely observed."""
    result = WebInterlocutorResult(ok=False, objective="ask ChatGPT about self-awareness")
    result.status = "reply_not_observed"
    result.error = "sent_message_not_visible_after_dom_submit"
    result.turns = [
        _turn(i, f"ChatGPT turn {i}: a substantive reply about update magnitude.")
        for i in range(6)
    ]
    return result


class TestObservedTurnsAreKept:
    @pytest.mark.asyncio
    async def test_a_failed_proof_still_writes_memory(self, session_and_writes):
        session, writes = session_and_writes
        result = _failed_run()
        await session._persist_observed_transcript(result, {}, True, proven=False)
        assert writes, "six observed turns were discarded"
        assert result.memory_record_id == "rec-1"

    @pytest.mark.asyncio
    async def test_the_transcript_content_is_in_the_record(self, session_and_writes):
        session, writes = session_and_writes
        await session._persist_observed_transcript(_failed_run(), {}, True, proven=False)
        assert "update magnitude" in writes[0].content

    @pytest.mark.asyncio
    async def test_the_turn_count_is_recorded(self, session_and_writes):
        session, writes = session_and_writes
        await session._persist_observed_transcript(_failed_run(), {}, True, proven=False)
        assert writes[0].metadata["turn_count"] == 6


class TestTheRecordCarriesItsOwnProvenance:
    """A partial observation is real material and is NOT evidence of a
    completed exchange. The record says which it is, so nothing downstream
    has to remember."""

    @pytest.mark.asyncio
    async def test_an_unproven_record_is_marked(self, session_and_writes):
        session, writes = session_and_writes
        await session._persist_observed_transcript(_failed_run(), {}, True, proven=False)
        assert writes[0].metadata["proof_complete"] is False
        assert writes[0].cause == "web_interlocutor.observed_transcript_unproven"

    @pytest.mark.asyncio
    async def test_the_failing_status_is_preserved(self, session_and_writes):
        session, writes = session_and_writes
        await session._persist_observed_transcript(_failed_run(), {}, True, proven=False)
        assert writes[0].metadata["status"] == "reply_not_observed"

    @pytest.mark.asyncio
    async def test_a_proven_record_is_marked_differently(self, session_and_writes):
        session, writes = session_and_writes
        result = _failed_run()
        result.learned_summary = "The interlocutor described its update process."
        await session._persist_observed_transcript(result, {}, True, proven=True)
        assert writes[0].metadata["proof_complete"] is True
        assert writes[0].cause == "web_interlocutor.learned_summary"


class TestNothingIsInvented:
    @pytest.mark.asyncio
    async def test_a_run_with_no_observed_content_writes_nothing(self, session_and_writes):
        """Retention must not manufacture a memory of a conversation that
        never produced anything."""
        session, writes = session_and_writes
        result = WebInterlocutorResult(ok=False, objective="x")
        result.turns = [_turn(0, "   ")]
        await session._persist_observed_transcript(result, {}, True, proven=False)
        assert writes == []

    @pytest.mark.asyncio
    async def test_a_run_with_no_turns_writes_nothing(self, session_and_writes):
        session, writes = session_and_writes
        await session._persist_observed_transcript(
            WebInterlocutorResult(ok=False, objective="x"), {}, True, proven=False,
        )
        assert writes == []

    @pytest.mark.asyncio
    async def test_persist_memory_false_is_honoured(self, session_and_writes):
        session, writes = session_and_writes
        await session._persist_observed_transcript(_failed_run(), {}, False, proven=False)
        assert writes == []

    @pytest.mark.asyncio
    async def test_a_write_failure_does_not_break_the_run(self, session_and_writes):
        """Losing the memory is bad; losing the result too is worse."""
        session, _writes = session_and_writes

        class _Broken:
            async def write(self, request):
                raise RuntimeError("gateway down")

        session.memory_gateway = _Broken()
        result = _failed_run()
        await session._persist_observed_transcript(result, {}, True, proven=False)
        assert result.memory_record_id == ""


class TestTheEarlyReturnCallsIt:
    def test_the_unverified_path_persists_before_returning(self):
        """The exact line that discarded Bryan's six turns."""
        import inspect

        source = inspect.getsource(WebInterlocutorSession)
        branch = source[source.index('result.status = "reply_not_observed"'):]
        branch = branch[: branch.index("return result")]
        assert "_persist_observed_transcript" in branch
