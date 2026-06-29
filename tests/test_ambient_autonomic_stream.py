from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.autonomic.reflection_loop import AutonomicReflectionLoop
from core.container import ServiceContainer
from core.perception.ambient_developer_stream import AmbientDeveloperStream
from core.runtime.timescale_bridge import TimescaleBridge


class _CompletedProcess:
    returncode = 0

    def __init__(self, stdout: str):
        self.stdout = stdout


class _WorldState:
    def __init__(self):
        self.events = []

    def record_event(self, description, *, source, salience, ttl):
        self.events.append(
            {
                "description": description,
                "source": source,
                "salience": salience,
                "ttl": ttl,
            }
        )


@pytest.mark.asyncio
async def test_ambient_developer_stream_collects_repo_logs_and_feeds_timescale(
    monkeypatch,
    tmp_path,
):
    ServiceContainer.clear()
    repo = tmp_path
    core_dir = repo / "core"
    log_dir = repo / "logs"
    core_dir.mkdir()
    log_dir.mkdir()
    (core_dir / "mind.py").write_text("print('changed')\n", encoding="utf-8")
    (log_dir / "aura.log").write_text(
        "INFO ok\nERROR live desktop conversation unhealthy\n",
        encoding="utf-8",
    )

    class Gateway:
        def run(self, argv, **kwargs):
            assert kwargs["read_only"] is True
            assert kwargs["source"] == "ambient_developer_stream.git_status"
            return _CompletedProcess(" M core/mind.py\n")

    monkeypatch.setattr(
        "core.runtime.subprocess_gateway.get_subprocess_gateway",
        lambda: Gateway(),
    )
    world = _WorldState()
    bridge = TimescaleBridge(sample_interval_s=0)
    ServiceContainer.register_instance("world_state", world, required=False)
    ServiceContainer.register_instance("timescale_bridge", bridge, required=False)

    stream = AmbientDeveloperStream(
        project_root=repo,
        watch_roots=(core_dir,),
        log_roots=(log_dir,),
        sample_interval_s=5.0,
        max_scan_files=50,
        recent_window_s=3600.0,
    )

    try:
        frame = await stream.sample_once()

        assert frame.git_dirty_count == 1
        assert frame.recent_files
        assert frame.log_events
        assert "review_recent_log_errors" in frame.repair_candidates
        assert world.events
        status = bridge.get_status()
        assert status["observations"] == 1
        assert status["latest_observation"]["ambient_event_count"] >= 3
        assert "repair candidates" in status["latest_observation"]["ambient_summary"]
    finally:
        ServiceContainer.clear()


def test_timescale_bridge_samples_perceptual_and_ambient_sources_independently():
    bridge = TimescaleBridge(sample_interval_s=10.0)
    perceptual = SimpleNamespace(
        timestamp=100.0,
        screen=SimpleNamespace(active_app="Aura Zenith", window_title="Chat", screen_changed=False),
        system=SimpleNamespace(cpu_percent=10.0, memory_percent=50.0, thermal_pressure=0.0),
        user=SimpleNamespace(idle_seconds=0.0),
        audio=SimpleNamespace(voice_activity=False),
        novelty_score=lambda: 0.0,
        threat_score=lambda: 0.0,
        social_signal=lambda: 0.4,
    )
    ambient = SimpleNamespace(
        timestamp=101.0,
        to_dict=lambda: {
            "git_dirty_count": 1,
            "recent_files": [],
            "log_events": [],
            "summary": "1 tracked repo change(s)",
            "repair_candidates": ["run_targeted_tests_for_recent_changes"],
        },
    )

    bridge.ingest_perceptual_frame(perceptual)
    bridge.ingest_ambient_developer_frame(ambient)

    status = bridge.get_status()
    assert status["observations"] == 2
    reconciliation = bridge.reconcile_foreground_turn("status", now=120.0).to_dict()
    assert reconciliation["ambient_event_count"] == 1
    assert reconciliation["ambient_repair_candidates"] == ["run_targeted_tests_for_recent_changes"]


@pytest.mark.asyncio
async def test_autonomic_reflection_loop_writes_to_dream_journal(monkeypatch, tmp_path):
    ServiceContainer.clear()
    reflections = []

    class DreamJournal:
        def append_autonomic_reflection(self, reflection):
            reflections.append(reflection)

    frame = SimpleNamespace(
        frame_id=9,
        to_dict=lambda: {
            "summary": "1 recent warning/error log line(s)",
            "git_dirty_count": 0,
            "log_events": [{"path": "logs/aura.log", "line": "ERROR something"}],
            "repair_candidates": ["review_recent_log_errors"],
        },
    )
    stream = SimpleNamespace(latest_frame=frame)
    ServiceContainer.register_instance("ambient_developer_stream", stream, required=False)
    ServiceContainer.register_instance("dream_journal", DreamJournal(), required=False)
    loop = AutonomicReflectionLoop(interval_s=30.0, journal_path=tmp_path / "autonomic.jsonl")

    try:
        reflection = await loop.reflect_once()

        assert reflection is not None
        assert reflection.log_event_count == 1
        assert "queue diagnosis" in reflection.self_correction_note
        assert reflections
        assert reflections[0]["repair_candidates"] == ["review_recent_log_errors"]
    finally:
        ServiceContainer.clear()
