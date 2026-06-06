"""core/social/relationship_graph.py
Social relationship graph representing user nodes and associations.
"""
from typing import Dict, List, Any


class RelationshipGraph:
    """Tracks interpersonal nodes and connection edges."""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[str, List[str]] = {}

    def add_person(self, name: str, details: Dict[str, Any]) -> None:
        self._nodes[name] = details
        if name not in self._edges:
            self._edges[name] = []

    def associate(self, name_a: str, name_b: str) -> None:
        if name_a in self._edges and name_b in self._nodes:
            self._edges[name_a].append(name_b)

    def get_connections(self, name: str) -> List[str]:
        return self._edges.get(name, [])
