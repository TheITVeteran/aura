import time

from core.brain.llm.local_server_client import LocalServerClient


def _ready_client() -> LocalServerClient:
    client = LocalServerClient("/models/Aura-32B-Zenith")
    client._lane_state = "ready"
    client._last_ready_at = time.time()
    client._last_progress_at = time.time()
    client._last_generation_completed_at = time.time()
    client.is_alive = lambda: True  # type: ignore[method-assign]
    return client


def test_local_server_status_blocks_runtime_identity_mismatch():
    client = _ready_client()
    client._runtime_identity_ok = False
    client._detected_runtime_models = ["unrelated/raw-assistant-runtime"]

    status = client.get_lane_status()

    assert status["conversation_ready"] is False
    assert "runtime_identity_mismatch" in status["readiness_blockers"]


def test_local_server_status_blocks_missing_visible_conversation_probe():
    client = _ready_client()
    client._runtime_identity_ok = True
    client._last_generation_completed_at = 0.0

    status = client.get_lane_status()

    assert status["conversation_ready"] is False
    assert "visible_conversation_probe_missing" in status["readiness_blockers"]


def test_local_server_status_ready_requires_identity_and_visible_probe():
    client = _ready_client()
    client._runtime_identity_ok = True

    status = client.get_lane_status()

    assert status["conversation_ready"] is True
    assert status["readiness_blockers"] == []
