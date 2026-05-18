"""core/schemas.py
Strict Pydantic payloads for all internal state passing in the new Zenith architecture.
"""

import time
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class WebsocketMessage(BaseModel):
    """Base schema for any message sent down the websocket."""
    model_config = ConfigDict(extra='allow') # allow extra fields to prevent stripping
    
    type: str = Field(..., description="The type of the message (e.g. 'thought', 'telemetry')")

class TelemetryPayload(WebsocketMessage):
    type: str = "telemetry"
    energy: float = Field(default=100.0, ge=0.0)
    curiosity: float = Field(default=50.0, ge=0.0)
    frustration: float = Field(default=0.0, ge=0.0)
    confidence: float = Field(default=100.0, ge=0.0)
    cpu_usage: float = Field(default=0.0, ge=0.0)
    ram_usage: float = Field(default=0.0, ge=0.0)
    
    # Consciousness Fields (v6)
    gwt_winner: str = "--"
    coherence: float = Field(default=0.0, ge=0.0)
    vitality: float = Field(default=0.0, ge=0.0)
    surprise: float = Field(default=0.0, ge=0.0)
    narrative: str = ""
    
    consciousness: dict[str, Any] = Field(default_factory=dict)
    mycelial: dict[str, Any] = Field(default_factory=dict)
    
class CognitiveThoughtPayload(WebsocketMessage):
    type: str = "thought"
    content: str
    urgency: str = "NORMAL"
    cognitive_phase: str | None = None

class ChatStreamChunkPayload(WebsocketMessage):
    type: str = "chat_stream_chunk"
    chunk: str

class ChatThoughtChunkPayload(WebsocketMessage):
    type: str = "chat_thought_chunk"
    content: str

class AuraMessagePayload(WebsocketMessage):
    """Used for non-streaming responses, autonomic messages, and reflexes."""
    type: str = "aura_message"
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class ActionResultPayload(WebsocketMessage):
    type: str = "action_result"
    tool: str
    result: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class UserMessagePayload(WebsocketMessage):
    type: str = "user_message"
    content: str

class ErrorPayload(WebsocketMessage):
    type: str = "error"
    message: str
class ChatStreamEvent(BaseModel):
    """Internal event for structured chat streaming."""
    type: str  # "token", "thought", "meta", "error", "end"
    content: str | None = None
    metadata: dict[str, Any] | None = None

class ToolInvocation(BaseModel):
    name: str = Field(..., description="The tool to invoke (python_sandbox, web_search)")
    payload: str = Field(..., description="The script or query for the tool")

class ShardResponse(BaseModel):
    """Strict schema for autonomous cognitive shards."""
    model_config = ConfigDict(extra='allow') # allow extra fields like 'thought' from LLMs
    
    analysis: str = Field(..., description="Internal cognitive monologue/analysis.", validation_alias=AliasChoices('analysis', 'thought'))
    action_type: str = Field(..., description="One of: 'observation', 'tool_use', 'conclusion', 'thought'")
    tools: list[ToolInvocation] = Field(default_factory=list, description="Array of tools to execute simultaneously.")
    tool_name: str | None = Field(None, description="[Legacy] The tool to invoke")
    tool_payload: str | None = Field(None, description="[Legacy] The script or query for the tool")
    conclusion: str = Field(..., description="Final takeaway or message.")

class IPCMessage(BaseModel):
    """Strictly validated payload for inter-process communication and task queues."""
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
    
    priority: int = Field(default=20)
    timestamp: float = Field(default_factory=time.monotonic)
    sequence: int = Field(default=0)
    payload: Any = Field(...)
    origin: str = Field(default="background")

    def __lt__(self, other: object) -> bool:
        other_key = _ipc_sort_key(other)
        if other_key is None:
            return False
        return _ipc_sort_key(self) < other_key


def _ipc_sort_key(message: object) -> tuple[int, float, int] | None:
    try:
        priority = int(message.priority)
        timestamp = float(message.timestamp)
        sequence = int(getattr(message, "sequence", 0))
    except (AttributeError, TypeError, ValueError):
        return None
    return priority, timestamp, sequence
