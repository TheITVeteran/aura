"""What Aura keeps from a conversation with a stranger, and whose words they are.

Four CP126 findings about the record a run leaves behind.

033dbcf3 — a caller-set boolean enabled canned deterministic messages after
cognitive composition failed. Those messages were then summarized and persisted
with nothing marking them as canned, so a conversation Aura never composed read
back as one she did.

ba0294cc — composition hardcoded proof_evaluation_contract to False and copied
the caller's requirement into a custom key no standard gate reads.

f6059536 — a lexical summary of what a remote service said was written into
durable episodic memory with no secret filtering and no transcript binding.

7b02ba55 — a causal proof that could not run wrote causal=False, the same
value the proof writes when it ran and found no influence.
"""
from __future__ import annotations

import time

import pytest

from core.capabilities.web_interlocutor import (
    REPLY_PROVENANCE_SENT_MARKER,
    WebInterlocutorResult,
    WebInterlocutorTurn,
    _redact_remote_content,
    _transcript_digest,
)

pytestmark = pytest.mark.unit


def _turn(index=1, sent="hello", reply="hi there", authorship="cognitive"):
    return WebInterlocutorTurn(
        index=index,
        sent=sent,
        observed_reply=reply,
        before_hash="a" * 64,
        after_hash="b" * 64,
        sent_at=time.time(),
        observed_at=time.time(),
        effect_verified=True,
        verification="",
        reply_provenance=REPLY_PROVENANCE_SENT_MARKER,
        sent_authorship=authorship,
    )


# --- canned text is labelled as canned (033dbcf3) -----------------------


def test_a_turn_records_who_wrote_the_message():
    assert _turn().sent_authorship == "cognitive"
    assert _turn(authorship="deterministic_fallback").sent_authorship == (
        "deterministic_fallback"
    )


def test_a_result_defaults_to_cognitive_authorship():
    assert WebInterlocutorResult(ok=True).used_deterministic_composition is False


def test_the_deterministic_summary_says_what_it_is():
    from core.capabilities.web_interlocutor import _deterministic_learning_summary

    assert "lexical summary" in (_deterministic_learning_summary.__doc__ or "")


# --- remote content is redacted before it becomes durable (f6059536) ----


@pytest.mark.parametrize(
    "secret,kind",
    [
        ("sk-abcdefghijklmnopqrstuvwx", "api_key"),
        ("Bearer abcdefghijklmnopqrstuvwxyz123", "bearer_token"),
        ("AKIAIOSFODNN7EXAMPLE", "aws_key"),
        ("marta@example.com", "email"),
        ("4111 1111 1111 1111", "card_number"),
        ("+1 415 555 0199", "phone_number"),
    ],
)
def test_secrets_and_contacts_do_not_enter_durable_memory(secret, kind):
    redacted, found = _redact_remote_content(f"It said: {secret} — use that.")

    assert secret not in redacted
    assert kind in found
    assert f"[redacted:{kind}]" in redacted


def test_a_jwt_is_redacted():
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"
    redacted, found = _redact_remote_content(token)

    assert token not in redacted
    assert "jwt" in found


def test_ordinary_prose_is_untouched():
    text = "The ferry leaves at nine and the tide turns around eleven."

    redacted, found = _redact_remote_content(text)

    assert redacted == text
    assert found == []


def test_what_was_redacted_is_reported_not_hidden():
    _redacted, found = _redact_remote_content(
        "reach me at marta@example.com or sk-abcdefghijklmnopqrstuvwx"
    )

    assert set(found) == {"email", "api_key"}


# --- the memory is bound to the transcript it came from (f6059536) ------


def test_the_digest_changes_when_the_transcript_does():
    first = _transcript_digest([_turn(reply="the moon dominates")])
    second = _transcript_digest([_turn(reply="the sun dominates")])

    assert first != second
    assert len(first) == 64


def test_the_digest_covers_what_was_sent_too():
    assert _transcript_digest([_turn(sent="a")]) != _transcript_digest([_turn(sent="b")])


def test_turn_order_is_part_of_the_digest():
    a, b = _turn(index=1, reply="first"), _turn(index=2, reply="second")

    assert _transcript_digest([a, b]) != _transcript_digest([b, a])


def test_an_empty_transcript_still_has_a_digest():
    assert len(_transcript_digest([])) == 64


# --- an unavailable proof is not a negative result (7b02ba55) -----------


@pytest.mark.asyncio
async def test_a_causal_proof_that_cannot_run_is_not_recorded_as_no_influence(
    monkeypatch,
):
    from core.capabilities.web_interlocutor import WebInterlocutorSession

    session = WebInterlocutorSession.__new__(WebInterlocutorSession)
    session.memory_gateway = None
    result = WebInterlocutorResult(ok=True, objective="o")
    result.turns = [_turn()]

    import core.capabilities.conversation_revision as revision_module

    monkeypatch.setattr(
        revision_module,
        "revise_from_conversation",
        lambda _turns: (_ for _ in ()).throw(RuntimeError("verifier unavailable")),
    )

    await session._adjudicate_and_prove(result, {}, persist_memory=False)

    assert result.causal_influence["causal"] is None
    assert result.causal_influence["measured"] is False
    assert "unavailable" in result.causal_influence["reason"]
