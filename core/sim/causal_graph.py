"""core/sim/causal_graph.py — Causal Intervention Graph.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Set

logger = logging.getLogger("Aura.CausalGraphSim")


@dataclass
class InterventionNode:
    node_id: str
    impact_level: float
    probability: float


class CausalInterventionGraph:
    """Graph mapping interventions to estimated downstream side-effects."""

    def __init__(self) -> None:
        self.nodes: Dict[str, InterventionNode] = {}
        self.links: Dict[str, List[str]] = {}

    def register_node(self, node: InterventionNode) -> None:
        self.nodes[node.node_id] = node

    def add_link(self, parent_id: str, child_id: str) -> None:
        self.links.setdefault(parent_id, []).append(child_id)

    def simulate_intervention(self, node_id: str, visited: Set[str] | None = None) -> float:
        """Propagates probability through causal pathways."""
        if visited is None:
            visited = set()
        if node_id not in self.nodes or node_id in visited:
            return 0.0

        visited.add(node_id)
        current = self.nodes[node_id]
        total_impact = current.impact_level * current.probability

        children = self.links.get(node_id, [])
        for ch in children:
            total_impact += 0.50 * self.simulate_intervention(ch, visited)

        return total_impact
