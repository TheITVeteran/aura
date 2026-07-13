"""Exact-agent person view composed from canonical relational authority."""
from __future__ import annotations

from typing import Any

from core.social.relational_memory import RelationalMemoryAuthority
from core.social.relationship_graph import RelationshipGraph


class PersonModel:
    """Read-only compatibility view for one explicitly identified person.

    This class no longer constructs parallel trust, preference, memory,
    reciprocity, boundary, or Theory-of-Mind stores.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        relationship_graph: RelationshipGraph | None = None,
        authority: RelationalMemoryAuthority | None = None,
    ) -> None:
        normalized = " ".join(str(agent_id or "").strip().split())[:160]
        if not normalized:
            raise ValueError("PersonModel requires an exact non-empty agent_id")
        self.agent_id = normalized
        self.name = normalized
        self.relationships = relationship_graph or RelationshipGraph(authority=authority)

    def get_social_status(self) -> dict[str, Any]:
        """Return content-bounded topology status without invented trust scores."""
        node = self.relationships.get_node(self.agent_id)
        if node is None:
            return {
                "agent_id": self.agent_id,
                "authorized": False,
                "interaction_evidence_count": 0,
                "relationship_confidence": 0.0,
                "boundary_flags": {},
            }
        return {
            "agent_id": self.agent_id,
            "authorized": True,
            "interaction_evidence_count": node.interaction_count,
            "relationship_confidence": node.confidence,
            "boundary_flags": dict(node.boundary_flags),
            "relation_types": list(node.relation_types),
        }

    def validate_action(self, channel: str, params: dict[str, Any]) -> bool:
        """Apply explicit channel boundaries; never derive permission from rapport."""
        del params
        node = self.relationships.get_node(self.agent_id)
        if node is None:
            return False
        normalized_channel = "_".join(str(channel or "").strip().casefold().split())[:80]
        if not normalized_channel:
            return False
        boundaries = node.boundary_flags
        return not (
            boundaries.get(f"block_{normalized_channel}", False)
            or boundaries.get(f"do_not_{normalized_channel}", False)
        )
