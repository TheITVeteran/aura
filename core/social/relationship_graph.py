"""core/social/relationship_graph.py

Persistent social relationship graph: the companion's memory of people.

Each person is a node with mirrored preferences, a clamped sentiment
score that interactions modulate, explicit boundary flags (consent
machinery: "do not disturb after midnight" is causal, not decorative),
shared projects, and a bounded log of interaction digests. Nodes
persist as one JSON file per node so relationship memory survives
restarts and is inspectable/exportable per person — which is also what
makes per-person deletion (privacy) trivial.

The legacy in-memory association API (add_person/associate/
get_connections) is kept for existing callers.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.runtime.atomic_writer import atomic_write_text

logger = logging.getLogger("Aura.Social.RelationshipGraph")

_MAX_DIGESTS = 200
_NODE_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class RelationshipNode:
    node_id: str
    name: str = ""
    node_type: str = "person"
    sentiment_score: float = 0.5
    preferences: Dict[str, Any] = field(default_factory=dict)
    boundary_flags: Dict[str, bool] = field(default_factory=dict)
    shared_projects: List[str] = field(default_factory=list)
    digests: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type,
            "sentiment_score": self.sentiment_score,
            "preferences": dict(self.preferences),
            "boundary_flags": dict(self.boundary_flags),
            "shared_projects": list(self.shared_projects),
            "digests": list(self.digests),
            "connections": list(self.connections),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipNode":
        return cls(
            node_id=str(data.get("node_id") or ""),
            name=str(data.get("name") or ""),
            node_type=str(data.get("node_type") or "person"),
            sentiment_score=float(data.get("sentiment_score", 0.5)),
            preferences=dict(data.get("preferences") or {}),
            boundary_flags=dict(data.get("boundary_flags") or {}),
            shared_projects=list(data.get("shared_projects") or []),
            digests=list(data.get("digests") or [])[-_MAX_DIGESTS:],
            connections=list(data.get("connections") or []),
        )


class RelationshipGraph:
    """Tracks interpersonal nodes with persistence and sentiment dynamics."""

    def __init__(self, storage_dir: str | Path | None = None):
        self.storage_dir: Optional[Path] = Path(storage_dir) if storage_dir else None
        self.nodes: Dict[str, RelationshipNode] = {}
        self._lock = threading.Lock()
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    # ── persistence ───────────────────────────────────────────────────

    def _path(self, node_id: str) -> Path:
        if self.storage_dir is None:
            raise RuntimeError("RelationshipGraph has no storage_dir configured")
        safe = _NODE_ID_SAFE.sub("_", str(node_id)) or "node"
        return self.storage_dir / f"{safe}.json"

    def _load_all(self) -> None:
        assert self.storage_dir is not None
        for path in sorted(self.storage_dir.glob("*.json")):
            try:
                node = RelationshipNode.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if node.node_id:
                    self.nodes[node.node_id] = node
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Skipping unreadable relationship node %s: %s", path, exc)

    def _persist(self, node: RelationshipNode) -> None:
        if self.storage_dir is None:
            return
        try:
            atomic_write_text(
                self._path(node.node_id),
                json.dumps(node.to_dict(), indent=2, sort_keys=True),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Failed to persist relationship node %s: %s", node.node_id, exc)

    # ── node lifecycle ────────────────────────────────────────────────

    def get_or_create_node(
        self,
        node_id: str,
        *,
        name: str = "",
        node_type: str = "person",
        preferences: Dict[str, Any] | None = None,
    ) -> RelationshipNode:
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                node = RelationshipNode(
                    node_id=node_id,
                    name=name or node_id,
                    node_type=node_type,
                    preferences=dict(preferences or {}),
                )
                self.nodes[node_id] = node
                self._persist(node)
            return node

    def record_interaction(
        self, node_id: str, *, sentiment_delta: float = 0.0, digest: str = ""
    ) -> RelationshipNode:
        """Modulate sentiment (clamped to [0,1]) and append a digest."""
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                node = RelationshipNode(node_id=node_id, name=node_id)
                self.nodes[node_id] = node
            node.sentiment_score = max(
                0.0, min(1.0, node.sentiment_score + float(sentiment_delta))
            )
            text = str(digest or "").strip()
            if text:
                node.digests.append(text)
                if len(node.digests) > _MAX_DIGESTS:
                    node.digests = node.digests[-_MAX_DIGESTS:]
            self._persist(node)
            return node

    def set_boundary_flag(self, node_id: str, flag: str, value: bool) -> None:
        """Boundaries are explicit consent state, persisted immediately."""
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                node = RelationshipNode(node_id=node_id, name=node_id)
                self.nodes[node_id] = node
            node.boundary_flags[str(flag)] = bool(value)
            self._persist(node)

    def link_project(self, node_id: str, project_id: str) -> None:
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                node = RelationshipNode(node_id=node_id, name=node_id)
                self.nodes[node_id] = node
            project = str(project_id)
            if project not in node.shared_projects:
                node.shared_projects.append(project)
            self._persist(node)

    def forget_node(self, node_id: str) -> bool:
        """Privacy: remove a person's node from memory and disk."""
        with self._lock:
            removed = self.nodes.pop(node_id, None) is not None
            if self.storage_dir is not None:
                try:
                    self._path(node_id).unlink(missing_ok=True)
                except OSError as exc:
                    logger.error("Failed to delete relationship node %s: %s", node_id, exc)
            return removed

    # ── legacy association API (existing callers) ─────────────────────

    def add_person(self, name: str, details: Dict[str, Any]) -> None:
        node = self.get_or_create_node(name, name=name)
        with self._lock:
            node.preferences.update(details or {})
            self._persist(node)

    def associate(self, name_a: str, name_b: str) -> None:
        with self._lock:
            node_a = self.nodes.get(name_a)
            if node_a is not None and name_b in self.nodes:
                if name_b not in node_a.connections:
                    node_a.connections.append(name_b)
                    self._persist(node_a)

    def get_connections(self, name: str) -> List[str]:
        node = self.nodes.get(name)
        return list(node.connections) if node else []
