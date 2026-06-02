from core.conversation.engine import CONTEXT_HISTORY_MAX_MESSAGES, ConversationContext
from core.conversation.unified_transcript import UnifiedTranscript, _MAX_HISTORY_DEFAULT
from core.memory.conversation_persistence import MAX_HISTORY_IN_MEMORY, MAX_SESSIONS_ON_DISK


def test_conversation_retention_defaults_exceed_legacy_caps() -> None:
    assert _MAX_HISTORY_DEFAULT >= 500
    assert CONTEXT_HISTORY_MAX_MESSAGES >= 250
    assert MAX_HISTORY_IN_MEMORY >= 500
    assert MAX_SESSIONS_ON_DISK >= 200


def test_unified_transcript_retains_more_than_legacy_window() -> None:
    transcript = UnifiedTranscript()

    for idx in range(120):
        transcript.add("user", f"message {idx}")

    assert transcript.get_entry_count() == 120
    assert transcript.get_context_window(5)[-1].content == "message 119"
    assert transcript.get_summary()["max_history"] >= 500


def test_conversation_context_retains_more_than_old_rolling_window() -> None:
    context = ConversationContext("retention-test")

    for idx in range(120):
        context.add_message("user", f"turn {idx}")

    assert len(context.history) == 120
    assert context.history[0].content == "turn 0"
    assert context.history[-1].content == "turn 119"
