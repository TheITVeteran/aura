"""Compatibility shim for the package-backed goal engine surface.

The canonical GoalEngine now lives in `core.goals.goal_engine`. This module
stays only to preserve direct file-path imports and older tooling without
carrying a second competing GoalEngine implementation.
"""
from uuid import uuid4

from core.goals import GoalEngine


class Goal:
    """Legacy lightweight goal record kept for compatibility helpers/tests."""

    def __init__(self, objective: str, metric: str = "sub", cost: float = 0.1, id: str | None = None):
        self.objective = objective
        self.metric = metric
        self.cost = cost
        self.id = id if id is not None else str(uuid4())
        self.score = 0.0


__all__ = ["Goal", "GoalEngine"]
