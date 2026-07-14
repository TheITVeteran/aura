"""Goal-domain public API without eager runtime construction imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .goal_engine import GoalEngine as GoalEngine

__all__ = ["GoalEngine"]


def __getattr__(name: str) -> Any:
    if name != "GoalEngine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .goal_engine import GoalEngine

    globals()[name] = GoalEngine
    return GoalEngine
