"""The visible-conversation-probe guard must not downgrade an idle-but-served lane.

Regression test for the heartbeat false-positive: a warm Cortex lane that already
served a visible turn but has since been idle >300s was being flipped to
not-ready with `visible_conversation_probe_missing`. The guard's intent (and the
authoritative mlx_client guard) is to fire ONLY when the lane has never served a
visible turn (anchor <= 0), not merely when idle.
"""
from __future__ import annotations

import time

from core.brain.inference_gate import InferenceGate


class _ReadyClient:
    """Fake MLX lane: ready, with a configurable visible-turn anchor."""

    def __init__(self, anchor: float):
        self._anchor = anchor

    def get_lane_status(self):
        now = time.time()
        return {
            "state": "ready",
            "last_error": "",
            "conversation_ready": True,
            "warmup_attempted": True,
            "warmup_in_flight": False,
            "last_transition_at": now,
            "last_ready_at": now,
            "last_progress_at": now,
            "last_heartbeat": now,
            "last_token_progress_at": now,
            "last_generation_completed_at": self._anchor,
            "last_user_facing_completed_at": self._anchor,
            "last_visible_readiness_at": self._anchor,
            "readiness_blockers": [],
            "ready": True,
        }

    def is_alive(self):
        return True


def _blockers(gate: InferenceGate) -> list[str]:
    return list(gate.get_conversation_status().get("readiness_blockers") or [])


def test_idle_but_served_lane_not_flagged():
    # Served a visible turn 10 minutes ago, idle since → anchor > 0 but stale.
    gate = InferenceGate()
    gate._mlx_client = _ReadyClient(time.time() - 600.0)
    assert "visible_conversation_probe_missing" not in _blockers(gate)


def test_never_served_lane_is_flagged():
    # Claims ready but never served a visible turn → anchor <= 0 (true zombie/cold).
    gate = InferenceGate()
    gate._mlx_client = _ReadyClient(0.0)
    assert "visible_conversation_probe_missing" in _blockers(gate)
