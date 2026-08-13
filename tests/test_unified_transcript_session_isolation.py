from __future__ import annotations

from core.conversation.session_scope import (
    conversation_session_scope,
    conversation_session_var,
    conversation_turn_var,
)
from core.conversation.unified_transcript import UnifiedTranscript


def test_context_and_antecedents_are_partitioned_by_conversation() -> None:
    transcript = UnifiedTranscript()
    with conversation_session_scope("device-a"):
        transcript.add_text_input("My favorite animal is the orca.")
        transcript.add_text_output("I will remember that.")
    with conversation_session_scope("device-b"):
        transcript.add_text_input("My favorite animal is the octopus.")
        transcript.add_text_output("That is useful context.")

    with conversation_session_scope("device-a"):
        assert [entry.content for entry in transcript.get_context_window()] == [
            "My favorite animal is the orca.",
            "I will remember that.",
        ]
        assert transcript.preceding_turns() == (
            "My favorite animal is the orca.",
            "I will remember that.",
        )
    with conversation_session_scope("device-b"):
        assert transcript.preceding_turns() == (
            "My favorite animal is the octopus.",
            "That is useful context.",
        )


def test_authenticated_session_scope_overrides_subsystem_label() -> None:
    transcript = UnifiedTranscript()
    with conversation_session_scope("ambient"):
        transcript.add_text_input("explicit", conversation_id="target")

    assert transcript.get_entry_count(conversation_id="ambient") == 1
    assert transcript.get_entry_count(conversation_id="target") == 0


def test_explicit_session_identity_works_outside_request_scope() -> None:
    transcript = UnifiedTranscript()
    transcript.add_text_input("explicit", conversation_id="target")

    assert transcript.get_entry_count(conversation_id="target") == 1


def test_local_voice_and_text_share_one_cross_modality_context() -> None:
    transcript = UnifiedTranscript()
    transcript.add_voice_input("Can you still hear me?")
    transcript.add_text_output("Yes. The modality changed, not the conversation.")

    assert [entry.channel for entry in transcript.get_context_window()] == [
        "voice",
        "text",
    ]


def test_http_chat_uses_the_core_conversation_identity_boundary() -> None:
    from interface.routes.chat import _CHAT_DELIVERY_TURN_ID, _CHAT_REQUEST_SESSION

    assert _CHAT_REQUEST_SESSION is conversation_session_var
    assert _CHAT_DELIVERY_TURN_ID is conversation_turn_var


def test_terminal_http_exchange_enters_only_its_conversation(monkeypatch) -> None:
    from interface.routes.chat import _record_unified_transcript_exchange

    transcript = UnifiedTranscript()
    monkeypatch.setattr(UnifiedTranscript, "_instance", transcript)

    _record_unified_transcript_exchange(
        "What did I say?",
        "You said the favorite animal was the orca.",
        session_id="device-a",
        exchange_id="turn-1",
    )

    assert transcript.get_entry_count(conversation_id="device-a") == 2
    assert transcript.get_entry_count(conversation_id="device-b") == 0
    assert {
        entry.metadata["exchange_id"]
        for entry in transcript.entries_for_conversation("device-a")
    } == {"turn-1"}
