from core.schemas import (
    ActionResultPayload,
    AuraMessagePayload,
    ChatStreamChunkPayload,
    ChatThoughtChunkPayload,
    CognitiveThoughtPayload,
    WebsocketMessage,
)
from interface.event_bridge import (
    _map_event_to_ws_message,
    _shape_user_facing_ws_message,
)


def test_event_bridge_preserves_desktop_step_progress_metadata():
    message = _map_event_to_ws_message(
        "thoughts",
        {
            "content": "Step 2/5 open_url: retrying. transient timeout",
            "category": "ToolExecution",
            "title": "Desktop Task",
            "level": "warning",
            "step_index": 2,
            "step_total": 5,
            "action": "open_url",
            "state": "retrying",
        },
        CognitiveThoughtPayload=CognitiveThoughtPayload,
        WebsocketMessage=WebsocketMessage,
        ChatStreamChunkPayload=ChatStreamChunkPayload,
        ChatThoughtChunkPayload=ChatThoughtChunkPayload,
        AuraMessagePayload=AuraMessagePayload,
        ActionResultPayload=ActionResultPayload,
    )

    assert message["type"] == "thought"
    assert message["content"].startswith("Step 2/5 open_url")
    assert message["category"] == "ToolExecution"
    assert message["step_index"] == 2
    assert message["step_total"] == 5
    assert message["action"] == "open_url"
    assert message["state"] == "retrying"


def test_event_bridge_shapes_spoken_replies_but_not_neural_events(monkeypatch):
    calls = []

    class _Personality:
        def filter_response(self, text, *, user_facing=True):
            calls.append((text, user_facing))
            return f"shaped:{text}"

    monkeypatch.setattr(
        "core.brain.personality_engine.get_personality_engine",
        lambda: _Personality(),
    )

    thought = {"type": "thought", "content": "Goal: audit runtime"}
    reply = {"type": "aura_message", "message": "A complete reply."}
    chunk = {"type": "chat_stream_chunk", "chunk": "Goal:"}

    assert _shape_user_facing_ws_message(thought) == thought
    assert _shape_user_facing_ws_message(chunk) == chunk
    assert _shape_user_facing_ws_message(reply)["message"] == "shaped:A complete reply."
    assert calls == [("A complete reply.", True)]
