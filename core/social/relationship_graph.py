"""core/social/relationship_graph.py
===================================
Durable RelationshipGraph for user and peer nodes.
Saves preferences, boundary flags, digests, sentiment scores, and shared projects.
"""

from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.runtime.atomic_writer import atomic_write_text
from core.social.relationship_model import get_store
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Social.RelationshipGraph")


@dataclass
class RelationshipNode:
    node_id: str
    name: str
    node_type: str = "user"  # "user" or "peer"
    sentiment_score: float = 0.5  # 0..1
    preferences: Dict[str, Any] = field(default_factory=dict)
    boundary_flags: Dict[str, bool] = field(default_factory=dict)
    shared_projects: List[str] = field(default_factory=list)
    digests: List[str] = field(default_factory=list)
    last_interaction: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RelationshipGraph:
    """Graph structure maintaining nodes representing people and agents."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or (Path.home() / ".aura" / "data" / "social_graph")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.nodes: Dict[str, RelationshipNode] = {}
        self._load_all_nodes()

    def _path(self, node_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(node_id or "unknown")).strip("._")
        if not safe_id:
            safe_id = "unknown"
        return self.storage_dir / f"{safe_id[:120]}.json"

    def _load_all_nodes(self) -> None:
        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                node = RelationshipNode(**data)
                self.nodes[node.node_id] = node
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                record_degradation("relationship_graph", e, severity="warning", action="skipped unreadable social graph node")
                logger.error("Failed to load social node %s: %s", path.name, e)

    def get_or_create_node(self, node_id: str, name: str, node_type: str = "user") -> RelationshipNode:
        """Retrieve or create a new social relationship node in the graph."""
        if node_id in self.nodes:
            return self.nodes[node_id]
        if node_type not in {"user", "peer"}:
            node_type = "peer"
        
        # Mirror trust/commitment store if present
        store = get_store()
        dossier = store.get(node_id)
        preferences = {}
        if dossier:
            preferences = dossier.style_preferences
            
        node = RelationshipNode(
            node_id=node_id,
            name=name,
            node_type=node_type,
            preferences=preferences
        )
        self.save_node(node)
        return node

    def save_node(self, node: RelationshipNode) -> None:
        """Atomically persist a node to the social graph database."""
        self.nodes[node.node_id] = node
        path = self._path(node.node_id)
        try:
            atomic_write_text(path, json.dumps(node.to_dict(), indent=2, sort_keys=True, default=str))
        except (OSError, TypeError, ValueError) as e:
            record_degradation("relationship_graph", e, severity="warning", action="kept social graph node in memory after persistence failure")
            logger.error("Failed to save social node %s: %s", node.node_id, e)

    def record_interaction(self, node_id: str, sentiment_delta: float, digest: Optional[str] = None) -> None:
        """Update node sentiment, touch timestamp, and append a log digest."""
        node = self.nodes.get(node_id)
        if not node:
            return
        
        node.sentiment_score = max(0.0, min(1.0, node.sentiment_score + sentiment_delta))
        node.last_interaction = time.time()
        if digest:
            node.digests.append(digest)
            if len(node.digests) > 20:
                node.digests.pop(0)
        self.save_node(node)

    def set_boundary_flag(self, node_id: str, flag: str, active: bool) -> None:
        node = self.nodes.get(node_id)
        if not node:
            return
        node.boundary_flags[flag] = active
        self.save_node(node)

    def link_project(self, node_id: str, project_id: str) -> None:
        node = self.nodes.get(node_id)
        if not node:
            return
        if project_id not in node.shared_projects:
            node.shared_projects.append(project_id)
            self.save_node(node)


# Singleton
_graph_instance: Optional[RelationshipGraph] = None


def get_relationship_graph() -> RelationshipGraph:
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = RelationshipGraph()
    return _graph_instance
