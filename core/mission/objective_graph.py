"""core/mission/objective_graph.py — Mission Objective Graph.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set

logger = logging.getLogger("Aura.ObjectiveGraph")


@dataclass
class Milestone:
    milestone_id: str
    description: str
    status: str = "pending"  # "pending", "in_progress", "completed", "blocked"
    dependencies: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)


class ObjectiveGraph:
    """Manages active milestones, task chains, and resolving blockers."""

    def __init__(self) -> None:
        self.milestones: Dict[str, Milestone] = {}

    def add_milestone(self, ms: Milestone) -> None:
        self.milestones[ms.milestone_id] = ms
        logger.info("🎯 Milestone added: %s (Status: %s)", ms.description, ms.status)

    def set_status(self, milestone_id: str, status: str) -> None:
        if milestone_id in self.milestones:
            self.milestones[milestone_id].status = status
            logger.info("🎯 Milestone [%s] updated to: %s", milestone_id, status)

    def is_blocked(self, milestone_id: str) -> bool:
        if milestone_id not in self.milestones:
            return False
        ms = self.milestones[milestone_id]
        if ms.blockers:
            return True
        # Check if any dependencies are not completed
        for dep in ms.dependencies:
            if dep in self.milestones and self.milestones[dep].status != "completed":
                return True
        return False
