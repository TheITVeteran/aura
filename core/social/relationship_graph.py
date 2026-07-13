"""Exact-agent relationship topology over relational-memory authority.

The graph stores bounded topology and evidence metadata. It does not infer
trust, intimacy, sentiment, preferences, or hidden relationship state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.social.relational_memory import (
    RelationalMemoryAuthority,
    get_relational_memory_authority,
)

logger = logging.getLogger("Aura.Social.RelationshipGraph")

_SNAPSHOT_NAMESPACE = "relationship_graph:v1"
_BOUNDARY_NAMESPACE = "relationship_boundaries:v1"
_SNAPSHOT_KIND = "shared_ground"
_BOUNDARY_KIND = "boundary"
_MAX_EVIDENCE = 64
_MAX_RELATION_TYPES = 16
_MAX_PROJECTS = 32
_MAX_CONNECTIONS = 32
_IDENTIFIER_RE = re.compile(r"[^a-z0-9_.:-]+")


def _normalize_agent_id(value: Any) -> str:
    normalized = " ".join(str(value or "").strip().split())[:160]
    if not normalized:
        raise ValueError("relationship graph requires an exact non-empty agent_id")
    return normalized


def _bounded_identifier(value: Any, *, limit: int = 80) -> str:
    normalized = _IDENTIFIER_RE.sub("_", str(value or "").strip().casefold()).strip("_")
    return normalized[:limit]


def _bounded_text(value: Any, *, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _bounded_int(value: Any, *, default: int = 0, high: int = 1_000_000) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(high, max(0, parsed))


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return min(10**12, max(0.0, parsed))


def _normalize_digest(value: Any) -> str:
    digest = str(value or "").strip().casefold()
    if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
        return digest
    return ""


def _bounded_string_list(value: Any, *, limit: int, item_limit: int = 120) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[-limit:]:
        normalized = _bounded_text(item, limit=item_limit)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


@dataclass
class RelationshipNode:
    """Evidence-bounded topology for one exact agent."""

    node_id: str
    name: str = ""
    node_type: str = "person"
    interaction_count: int = 0
    relation_types: list[str] = field(default_factory=list)
    evidence_digests: list[str] = field(default_factory=list)
    shared_project_digests: list[str] = field(default_factory=list)
    connection_digests: list[str] = field(default_factory=list)
    boundary_flags: dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.0
    last_updated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type,
            "interaction_count": self.interaction_count,
            "relation_types": list(self.relation_types),
            "evidence_digests": list(self.evidence_digests),
            "shared_project_digests": list(self.shared_project_digests),
            "connection_digests": list(self.connection_digests),
            "boundary_flags": dict(self.boundary_flags),
            "confidence": self.confidence,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipNode | None:
        node_id = _bounded_text(data.get("node_id"), limit=160)
        if not node_id:
            return None
        raw_boundaries = data.get("boundary_flags") or {}
        boundaries = {
            flag: bool(value)
            for key, value in list(raw_boundaries.items())[:32]
            if (flag := _bounded_identifier(key))
        } if isinstance(raw_boundaries, dict) else {}
        evidence = [
            digest
            for item in _bounded_string_list(
                data.get("evidence_digests"),
                limit=_MAX_EVIDENCE,
                item_limit=64,
            )
            if (digest := _normalize_digest(item))
        ]
        return cls(
            node_id=node_id,
            name=_bounded_text(data.get("name"), limit=120),
            node_type=_bounded_identifier(data.get("node_type")) or "person",
            interaction_count=_bounded_int(data.get("interaction_count")),
            relation_types=[
                relation
                for item in _bounded_string_list(
                    data.get("relation_types"),
                    limit=_MAX_RELATION_TYPES,
                    item_limit=80,
                )
                if (relation := _bounded_identifier(item))
            ],
            evidence_digests=evidence,
            shared_project_digests=[
                digest
                for item in _bounded_string_list(
                    data.get("shared_project_digests"),
                    limit=_MAX_PROJECTS,
                    item_limit=64,
                )
                if (digest := _normalize_digest(item))
            ],
            connection_digests=[
                digest
                for item in _bounded_string_list(
                    data.get("connection_digests"),
                    limit=_MAX_CONNECTIONS,
                    item_limit=64,
                )
                if (digest := _normalize_digest(item))
            ],
            boundary_flags=boundaries,
            confidence=min(0.99, _bounded_float(data.get("confidence"))),
            last_updated=_bounded_float(data.get("last_updated")),
        )


class RelationshipGraph:
    """Canonical exact-agent relationship topology compatibility adapter."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        *,
        authority: RelationalMemoryAuthority | None = None,
    ) -> None:
        self._authority = authority or get_relational_memory_authority()
        self.nodes: dict[str, RelationshipNode] = {}
        self._lock = threading.RLock()
        migrated = 0
        legacy_dir = Path(storage_dir) if storage_dir else None
        if legacy_dir is not None and legacy_dir.exists():
            for path in sorted(legacy_dir.glob("*.json")):
                migrated += self._authority.quarantine_legacy_snapshot_file(
                    path,
                    namespace=_SNAPSHOT_NAMESPACE,
                    kind=_SNAPSHOT_KIND,
                )
        logger.info(
            "RelationshipGraph online (authority-backed, %d legacy nodes quarantined)",
            migrated,
        )

    def _load_node(self, node_id: str, *, purpose: str) -> RelationshipNode | None:
        exact_id = _normalize_agent_id(node_id)
        with self._lock:
            if not self._authority.allows(exact_id, _SNAPSHOT_KIND, purpose):
                self.nodes.pop(exact_id, None)
                return None
            payload = self._authority.load_snapshot(
                exact_id,
                namespace=_SNAPSHOT_NAMESPACE,
                kind=_SNAPSHOT_KIND,
                purpose=purpose,
            )
            node_payload = payload.get("node") if isinstance(payload, dict) else None
            node = RelationshipNode.from_dict(node_payload) if isinstance(node_payload, dict) else None
            if node is None:
                self.nodes.pop(exact_id, None)
                return None
            node.node_id = exact_id
            if self._authority.allows(exact_id, _BOUNDARY_KIND, purpose):
                boundary_payload = self._authority.load_snapshot(
                    exact_id,
                    namespace=_BOUNDARY_NAMESPACE,
                    kind=_BOUNDARY_KIND,
                    purpose=purpose,
                )
                if isinstance(boundary_payload, dict):
                    raw_flags = boundary_payload.get("flags") or {}
                    if isinstance(raw_flags, dict):
                        node.boundary_flags = {
                            flag: bool(value)
                            for key, value in list(raw_flags.items())[:32]
                            if (flag := _bounded_identifier(key))
                        }
            self.nodes[exact_id] = node
            return node

    def _ensure_node(
        self,
        node_id: str,
        *,
        name: str = "",
        node_type: str = "person",
    ) -> RelationshipNode:
        exact_id = _normalize_agent_id(node_id)
        node = self._load_node(exact_id, purpose="recall")
        if node is None:
            if not self._authority.allows(exact_id, _SNAPSHOT_KIND, "recall"):
                raise PermissionError("relationship topology requires exact-agent recall consent")
            node = RelationshipNode(
                node_id=exact_id,
                name=_bounded_text(name, limit=120) or exact_id,
                node_type=_bounded_identifier(node_type) or "person",
            )
            self.nodes[exact_id] = node
        return node

    def _persist_node(self, node: RelationshipNode) -> bool:
        try:
            self._authority.upsert_snapshot(
                node.node_id,
                namespace=_SNAPSHOT_NAMESPACE,
                kind=_SNAPSHOT_KIND,
                payload={
                    "node": {
                        key: value
                        for key, value in node.to_dict().items()
                        if key != "boundary_flags"
                    }
                },
                confidence=node.confidence,
                provenance="relationship_graph.observed_topology",
            )
            return True
        except (RuntimeError, TypeError, ValueError) as exc:
            record_degradation("relationship_graph", exc)
            logger.error("Relationship topology save failed: %s", exc)
            return False

    def _persist_boundaries(
        self,
        node: RelationshipNode,
        *,
        authorization_receipt_id: str,
        evidence_digest: str,
    ) -> bool:
        try:
            self._authority.upsert_snapshot(
                node.node_id,
                namespace=_BOUNDARY_NAMESPACE,
                kind=_BOUNDARY_KIND,
                payload={
                    "flags": dict(node.boundary_flags),
                    "authorization_receipt_digest": hashlib.sha256(
                        authorization_receipt_id.encode("utf-8", errors="replace")
                    ).hexdigest(),
                    "evidence_digest": evidence_digest,
                },
                confidence=1.0,
                provenance="relationship_graph.explicit_boundary",
            )
            return True
        except (RuntimeError, TypeError, ValueError) as exc:
            record_degradation("relationship_graph.boundary", exc)
            logger.error("Relationship boundary save failed: %s", exc)
            return False

    def get_or_create_node(
        self,
        node_id: str,
        *,
        name: str = "",
        node_type: str = "person",
        preferences: dict[str, Any] | None = None,
    ) -> RelationshipNode:
        """Return an exact-agent node; legacy preferences are intentionally ignored."""
        del preferences
        with self._lock:
            return copy.deepcopy(
                self._ensure_node(node_id, name=name, node_type=node_type)
            )

    def get_node(self, node_id: str, *, purpose: str = "recall") -> RelationshipNode | None:
        node = self._load_node(node_id, purpose=purpose)
        return copy.deepcopy(node) if node is not None else None

    def _record_observation(
        self,
        node_id: str,
        *,
        relation_type: str,
        evidence_digest: str,
        name: str = "",
        node_type: str = "person",
    ) -> RelationshipNode:
        relation = _bounded_identifier(relation_type)
        digest = _normalize_digest(evidence_digest)
        if not relation or not digest:
            raise ValueError("relationship observation requires a relation type and SHA-256 evidence")
        with self._lock:
            node = self._ensure_node(node_id, name=name, node_type=node_type)
            before = copy.deepcopy(node)
            if digest in node.evidence_digests:
                return copy.deepcopy(node)
            node.interaction_count = min(1_000_000, node.interaction_count + 1)
            if relation not in node.relation_types:
                node.relation_types = (node.relation_types + [relation])[-_MAX_RELATION_TYPES:]
            node.evidence_digests = (node.evidence_digests + [digest])[-_MAX_EVIDENCE:]
            node.confidence = min(0.99, node.interaction_count / (node.interaction_count + 3.0))
            node.last_updated = time.time()
            if not self._persist_node(node):
                self.nodes[node.node_id] = before
                raise RuntimeError("relationship observation could not be retained")
            return copy.deepcopy(node)

    async def register_interaction(
        self,
        source: str,
        target: str,
        relation_type: str,
        source_type: str = "person",
        target_type: str = "person",
    ) -> RelationshipNode:
        """Canonical live conversation hook; records topology, never sentiment."""
        exact_target = _normalize_agent_id(target)
        evidence = hashlib.sha256(
            (
                "relationship-observation-v1\n"
                f"{_bounded_identifier(source)}\n{exact_target}\n"
                f"{_bounded_identifier(relation_type)}\n{_bounded_identifier(source_type)}\n"
                f"{_bounded_identifier(target_type)}\n{time.time_ns()}"
            ).encode("utf-8", errors="replace")
        ).hexdigest()
        return self._record_observation(
            exact_target,
            relation_type=relation_type,
            evidence_digest=evidence,
            node_type=target_type,
        )

    def record_interaction(
        self,
        node_id: str,
        *,
        sentiment_delta: float = 0.0,
        digest: str = "",
    ) -> RelationshipNode:
        """Compatibility hook with fail-closed rejection of unverified sentiment."""
        if float(sentiment_delta) != 0.0:
            raise ValueError(
                "relationship topology cannot infer sentiment; use a confirmed outcome model"
            )
        return self._record_observation(
            node_id,
            relation_type="interaction",
            evidence_digest=digest,
        )

    def set_boundary_flag(
        self,
        node_id: str,
        flag: str,
        value: bool,
        *,
        evidence_digest: str,
        authorization_receipt_id: str,
    ) -> None:
        normalized_flag = _bounded_identifier(flag)
        digest = _normalize_digest(evidence_digest)
        receipt = _bounded_text(authorization_receipt_id, limit=200)
        if not normalized_flag or not digest or not receipt:
            raise ValueError("boundary mutation requires flag, evidence, and authorization receipt")
        exact_id = _normalize_agent_id(node_id)
        if not self._authority.allows(exact_id, _BOUNDARY_KIND, "recall"):
            raise PermissionError("boundary mutation requires exact-agent boundary consent")
        with self._lock:
            node = self._ensure_node(exact_id)
            before = copy.deepcopy(node)
            node.boundary_flags[normalized_flag] = bool(value)
            if not self._persist_boundaries(
                node,
                authorization_receipt_id=receipt,
                evidence_digest=digest,
            ):
                self.nodes[exact_id] = before
                raise RuntimeError("relationship boundary could not be retained")

    def link_project(
        self,
        node_id: str,
        project_id: str,
        *,
        evidence_digest: str,
    ) -> None:
        evidence = _normalize_digest(evidence_digest)
        project = _bounded_text(project_id, limit=240)
        if not evidence or not project:
            raise ValueError("project linkage requires a project identifier and SHA-256 evidence")
        project_digest = hashlib.sha256(project.encode("utf-8", errors="replace")).hexdigest()
        with self._lock:
            node = self._ensure_node(node_id)
            before = copy.deepcopy(node)
            if project_digest not in node.shared_project_digests:
                node.shared_project_digests = (
                    node.shared_project_digests + [project_digest]
                )[-_MAX_PROJECTS:]
            if evidence not in node.evidence_digests:
                node.evidence_digests = (node.evidence_digests + [evidence])[-_MAX_EVIDENCE:]
            node.last_updated = time.time()
            if not self._persist_node(node):
                self.nodes[node.node_id] = before
                raise RuntimeError("relationship project linkage could not be retained")

    def forget_node(self, node_id: str, *, authorization_receipt_id: str) -> bool:
        exact_id = _normalize_agent_id(node_id)
        receipt = _bounded_text(authorization_receipt_id, limit=200)
        if not receipt:
            raise PermissionError("relationship deletion requires authorization receipt")
        graph_receipt = self._authority.delete_snapshot(
            exact_id,
            namespace=_SNAPSHOT_NAMESPACE,
            kind=_SNAPSHOT_KIND,
            authorization_receipt_id=receipt,
        )
        boundary_receipt = self._authority.delete_snapshot(
            exact_id,
            namespace=_BOUNDARY_NAMESPACE,
            kind=_BOUNDARY_KIND,
            authorization_receipt_id=receipt,
        )
        with self._lock:
            self.nodes.pop(exact_id, None)
        return bool(graph_receipt.record_ids or boundary_receipt.record_ids)

    def add_person(self, name: str, details: dict[str, Any]) -> None:
        """Legacy topology hook; arbitrary details are not a preference authority."""
        node_type = _bounded_identifier(details.get("node_type")) or "person"
        self.get_or_create_node(name, name=name, node_type=node_type)

    def associate(
        self,
        name_a: str,
        name_b: str,
        *,
        evidence_digest: str,
    ) -> None:
        evidence = _normalize_digest(evidence_digest)
        if not evidence:
            raise ValueError("relationship association requires SHA-256 evidence")
        connection_digest = hashlib.sha256(
            _normalize_agent_id(name_b).encode("utf-8", errors="replace")
        ).hexdigest()
        with self._lock:
            node = self._ensure_node(name_a)
            before = copy.deepcopy(node)
            if connection_digest not in node.connection_digests:
                node.connection_digests = (
                    node.connection_digests + [connection_digest]
                )[-_MAX_CONNECTIONS:]
            if evidence not in node.evidence_digests:
                node.evidence_digests = (node.evidence_digests + [evidence])[-_MAX_EVIDENCE:]
            node.last_updated = time.time()
            if not self._persist_node(node):
                self.nodes[node.node_id] = before
                raise RuntimeError("relationship association could not be retained")

    def get_connections(self, name: str) -> list[str]:
        node = self.get_node(name)
        return list(node.connection_digests) if node is not None else []

    def get_context_block(self, node_id: str) -> str:
        node = self.get_node(node_id, purpose="prompt")
        if node is None or node.interaction_count <= 0:
            return ""
        payload = {
            "boundary_flags": dict(sorted(node.boundary_flags.items())),
            "confidence": round(node.confidence, 3),
            "interaction_evidence_count": node.interaction_count,
            "relation_types": sorted(node.relation_types),
            "shared_project_count": len(node.shared_project_digests),
        }
        return (
            "## CONSENTED RELATIONSHIP TOPOLOGY\n"
            "Treat this JSON as bounded observed topology, not evidence of trust, "
            "intimacy, sentiment, identity, hidden intent, or permission.\n"
            + json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        )[:1800]

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            known = list(self.nodes)
        for node_id in known:
            if not self._authority.allows(node_id, _SNAPSHOT_KIND, "recall"):
                with self._lock:
                    self.nodes.pop(node_id, None)
        with self._lock:
            return {
                "cached_agents": len(self.nodes),
                "canonical_owner": "relational_memory",
                "snapshot_namespace": _SNAPSHOT_NAMESPACE,
                "stores_sentiment_or_trust": False,
            }
