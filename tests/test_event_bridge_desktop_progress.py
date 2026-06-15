from core.schemas import (
    ActionResultPayload,
    AuraMessagePayload,
    ChatStreamChunkPayload,
    ChatThoughtChunkPayload,
    CognitiveThoughtPayload,
    WebsocketMessage,
)
from interface.event_bridge import _map_event_to_ws_message


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
