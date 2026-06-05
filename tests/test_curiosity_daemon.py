from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.agi.curiosity_daemon import AutonomousCuriosityDaemon
from core.epistemic_tracker import EpistemicGap, EpistemicProfile


class _ProfileTracker:
    def __init__(self, profile: EpistemicProfile) -> None:
        self.profile = profile
        self.calls: list[dict[str, Any]] = []

    def get_profile(self, *, force_refresh: bool = False) -> EpistemicProfile:
        self.calls.append({"force_refresh": force_refresh})
        return self.profile


class _CapabilityEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        *,
        context: dict[str, Any],
    ) -> None:
        self.calls.append((tool_name, params, context))


class _WillGate:
    def __init__(self, token: str) -> None:
        self.token = token
        self.requests: list[str] = []

    async def request_background_token(self, scope: str) -> str:
        self.requests.append(scope)
        return self.token


@pytest.mark.asyncio
async def test_curiosity_daemon_exploration_loop_routes_urgent_gap() -> None:
    gap = EpistemicGap(
        domain="testing",
        description="Limited knowledge about local test databases",
        urgency=0.9,
        detected_at=0.0,
        gap_type="unknown",
        seed_question="What is the local database syntax?",
    )
    profile = EpistemicProfile(
        timestamp=0.0,
        strong_nodes=[],
        weak_nodes=[],
        contradictions=[],
        gaps=[gap],
        overall_confidence=0.5,
        most_urgent_gap=gap,
    )
    tracker = _ProfileTracker(profile)
    capability_engine = _CapabilityEngine()
    will_gate = _WillGate("capability-token-123")

    daemon = AutonomousCuriosityDaemon(tracker=tracker, interval_seconds=10)

    await daemon.start(capability_engine=capability_engine, will_gate=will_gate)
    await asyncio.sleep(0.05)
    await daemon.stop()

    assert tracker.calls == [{"force_refresh": True}]
    assert will_gate.requests == ["research:testing"]
    assert capability_engine.calls == [
        (
            "web_search",
            {"query": "What is the local database syntax?"},
            {"capability_token_id": "capability-token-123"},
        )
    ]
