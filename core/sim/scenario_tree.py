"""core/sim/scenario_tree.py — Speculative Decision Scenario Tree.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("Aura.ScenarioTree")


@dataclass
class ScenarioNode:
    decision_id: str
    choice_label: str
    utility_score: float
    probability: float
    branches: List[ScenarioNode] = field(default_factory=list)


class ScenarioTreeBuilder:
    """Constructs branching scenario nodes for decision analysis."""

    @staticmethod
    def build_tree(action_choices: List[str]) -> ScenarioNode:
        logger.info("🌲 Constructing scenario tree for action choices: %s", ", ".join(action_choices))
        root = ScenarioNode("root", "initial_state", 1.0, 1.0)
        
        # Build one node per choice branch
        for i, choice in enumerate(action_choices):
            child = ScenarioNode(
                decision_id=f"branch_{i}",
                choice_label=choice,
                utility_score=0.85 - (i * 0.10),
                probability=1.0 / len(action_choices),
            )
            root.branches.append(child)
            # Add secondary outcome branches (success/failure)
            child.branches.append(ScenarioNode(f"branch_{i}_ok", f"{choice}_success", 0.90, 0.70))
            child.branches.append(ScenarioNode(f"branch_{i}_err", f"{choice}_failure", 0.10, 0.30))

        return root
