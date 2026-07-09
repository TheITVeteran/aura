from typing import Any, Dict, List, Optional, TypedDict


class Task(TypedDict):
    """Standard executable task unit."""

    id: str
    skill: str
    params: Dict[str, Any]
    priority: int
    context: Optional[Dict[str, Any]]

class PlanStep(TypedDict):
    """Single step in a plan."""

    name: str
    arguments: Dict[str, Any]

class Plan(TypedDict):
    """Execution plan."""

    goal: str
    tool_calls: List[PlanStep]

class Message(TypedDict):
    """Standard internal message."""

    sender: str
    recipient: str
    content: str
    type: str # 'text', 'command', 'error'
    timestamp: float
