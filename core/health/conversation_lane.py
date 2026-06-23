"""Shared conversation-lane health semantics.

The desktop UI and transport heartbeats need to distinguish a broken chat lane
from a healthy lane that is currently occupied by the foreground generation.
"""
from __future__ import annotations

from typing import Any


def conversation_lane_is_busy(lane: dict[str, Any] | None) -> bool:
    """Return true when the foreground conversation lane is actively working."""

    if not isinstance(lane, dict):
        return False
    state = str(lane.get("state", "") or "").strip().lower()
    if state != "ready":
        return False
    if bool(lane.get("warmup_in_flight", False)):
        return False
    blockers = {
        str(item or "").strip()
        for item in (lane.get("readiness_blockers") or [])
        if str(item or "").strip()
    }
    reason = str(lane.get("last_failure_reason", "") or lane.get("last_error", "") or "").strip()
    try:
        active_generations = int(lane.get("active_generations", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        active_generations = 0
    return (
        active_generations > 0
        or "active_generation_in_flight" in blockers
        or reason == "active_generation_in_flight"
    )


def conversation_lane_is_available(lane: dict[str, Any] | None) -> bool:
    """Return true when the lane is ready or legitimately busy answering."""

    if not isinstance(lane, dict):
        return False
    return bool(lane.get("conversation_ready", False)) or conversation_lane_is_busy(lane)
