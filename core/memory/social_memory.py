"""Compatibility adapter over the identity-scoped relational memory authority."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.base_module import AuraBaseModule
from core.social.relational_memory import (
    RelationalMemoryAuthority,
    get_relational_memory_authority,
)


@dataclass(frozen=True)
class RelationshipMilestone:
    description: str
    timestamp: float = 0.0
    importance: float = 0.5
    record_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "record_id": self.record_id,
        }


class SocialMemory(AuraBaseModule):  # type: ignore[misc]
    """Legacy API with no independent storage or relationship-depth owner."""

    def __init__(
        self,
        data_path: Path | None = None,
        *,
        authority: RelationalMemoryAuthority | None = None,
    ) -> None:
        super().__init__("SocialMemory")
        self.data_path = Path(data_path) if data_path is not None else None
        self.authority = authority or get_relational_memory_authority()

    @property
    def active_user_id(self) -> str:
        return self.authority.active_agent_id

    @property
    def milestones(self) -> list[RelationshipMilestone]:
        records = self.authority.query(
            self.active_user_id,
            kinds=["milestone"],
            purpose="recall",
            limit=100,
        )
        return [
            RelationshipMilestone(
                description=record.content,
                timestamp=record.created_at,
                importance=record.confidence,
                record_id=record.record_id,
            )
            for record in records
        ]

    @property
    def shared_context_keys(self) -> list[str]:
        return [
            record.content
            for record in self.authority.query(
                self.active_user_id,
                kinds=["shared_ground"],
                purpose="recall",
                limit=100,
            )
        ]

    @property
    def relationship_depth(self) -> float:
        records = self.authority.query(
            self.active_user_id,
            kinds=["milestone", "outcome", "repair", "shared_ground"],
            purpose="recall",
            limit=100,
        )
        if not records:
            return 0.0
        evidence = sum(record.confidence for record in records)
        return min(1.0, evidence / 20.0)

    @relationship_depth.setter
    def relationship_depth(self, _value: Any) -> None:
        # Depth is derived from consented evidence; passive increments are ignored.
        return

    def save(self) -> bool:
        return self.authority.save()

    def record_milestone(
        self,
        description: str,
        importance: float = 0.5,
        *,
        user_id: str | None = None,
        provenance: str = "social_memory_adapter",
    ) -> RelationshipMilestone:
        agent_id = str(user_id or self.active_user_id)
        record, _receipt = self.authority.record(
            agent_id,
            kind="milestone",
            content=description,
            confidence=importance,
            provenance=provenance,
        )
        return RelationshipMilestone(
            description=record.content,
            timestamp=record.created_at or time.time(),
            importance=record.confidence,
            record_id=record.record_id,
        )

    def add_shared_context(
        self,
        key: str,
        *,
        user_id: str | None = None,
    ) -> str:
        record, _receipt = self.authority.record(
            str(user_id or self.active_user_id),
            kind="shared_ground",
            content=key,
            provenance="social_memory_adapter",
        )
        return record.record_id

    def get_social_context(self, user_id: str | None = None) -> str:
        return self.authority.prompt_block(str(user_id or self.active_user_id))

    def get_status(self) -> dict[str, Any]:
        status = dict(self.authority.status())
        status["module"] = "SocialMemoryAdapter"
        status["canonical_owner"] = "relational_memory"
        return status
