from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .aura_now import OwnershipState


@dataclass
class OwnershipTracker:
    """Tracks mineness and action attribution from prediction match."""

    def assess(
        self,
        *,
        intended_action: str = "",
        predicted_outcome: str = "",
        actual_outcome: str = "",
        tool_failed: bool = False,
        external_override: bool = False,
        memory_influence: bool = False,
    ) -> OwnershipState:
        intended = bool(str(intended_action or "").strip())
        predicted = str(predicted_outcome or "").strip().lower()
        actual = str(actual_outcome or "").strip().lower()
        if not intended:
            match = 0.35
        elif predicted and actual:
            shared = set(predicted.split()) & set(actual.split())
            match = len(shared) / max(1, len(set(predicted.split()) | set(actual.split())))
        else:
            match = 0.65
        if tool_failed:
            match = min(match, 0.25)
        if external_override:
            match = min(match, 0.15)
        continuity = 0.15 if memory_influence else 0.05
        agency = max(0.0, min(1.0, (0.25 if intended else 0.0) + match * 0.65 + continuity))
        mineness = max(0.0, min(1.0, agency - (0.25 if tool_failed else 0.0) - (0.25 if external_override else 0.0)))
        if external_override:
            attribution = "external_override"
            reason = "external override reduced ownership"
        elif tool_failed:
            attribution = "tool_mismatch"
            reason = "intended action diverged from tool result"
        elif agency > 0.7:
            attribution = "self_authored"
            reason = ""
        else:
            attribution = "mixed"
            reason = "partial prediction match"
        return OwnershipState(
            mineness=round(mineness, 4),
            agency_confidence=round(agency, 4),
            predicted_action_match=round(max(0.0, min(1.0, match)), 4),
            attribution=attribution,
            mismatch_reason=reason,
        )


def assess_ownership(**kwargs: Any) -> OwnershipState:
    return OwnershipTracker().assess(**kwargs)
