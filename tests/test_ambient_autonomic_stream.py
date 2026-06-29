from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from core.autonomic.reflection_loop import AutonomicReflectionLoop
from core.container import ServiceContainer
from core.kernel.upgrades_10x import NativeMultimodalBridge
from core.perception.ambient_developer_stream import (
    AmbientDeveloperStream,
    render_ambient_developer_prompt_block,
)
from core.runtime.timescale_bridge import TimescaleBridge
from core.state.aura_state import AuraState


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
    terminal_dir = repo / "terminal"
    core_dir.mkdir()
    log_dir.mkdir()
    terminal_dir.mkdir()
    (core_dir / "mind.py").write_text("print('changed')\n", encoding="utf-8")
    (log_dir / "aura.log").write_text(
        "INFO ok\nERROR live desktop conversation unhealthy\n",
        encoding="utf-8",
    )
    (terminal_dir / "aura-terminal.log").write_text(
        "boot ok\nTraceback: foreground worker crashed\n",
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
    monkeypatch.setattr(
        "core.perception.ambient_developer_stream.psutil.net_connections",
        lambda kind: [
            SimpleNamespace(status="LISTEN", laddr=("127.0.0.1", 8000), raddr=()),
            SimpleNamespace(status="ESTABLISHED", laddr=("127.0.0.1", 50000), raddr=("93.184.216.34", 443)),
        ],
    )
    monkeypatch.setattr(
        "core.perception.ambient_developer_stream.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=91.0, available=2 * 1024**3),
    )
    monkeypatch.setattr("core.perception.ambient_developer_stream.psutil.cpu_percent", lambda interval=None: 91.0)
    monkeypatch.setattr("core.perception.ambient_developer_stream.psutil.sensors_battery", lambda: None)
    world = _WorldState()
    bridge = TimescaleBridge(sample_interval_s=0)
    ServiceContainer.register_instance("world_state", world, required=False)
    ServiceContainer.register_instance("timescale_bridge", bridge, required=False)

    stream = AmbientDeveloperStream(
        project_root=repo,
        watch_roots=(core_dir,),
        log_roots=(log_dir,),
        terminal_roots=(terminal_dir,),
        sample_interval_s=5.0,
        max_scan_files=50,
        recent_window_s=3600.0,
    )

    try:
        frame = await stream.sample_once()

        assert frame.git_dirty_count == 1
        assert frame.recent_files
        assert frame.log_events
        assert frame.terminal_events
        assert frame.network_events
        assert frame.resource_interrupts
        assert "review_recent_log_errors" in frame.repair_candidates
        assert "review_recent_terminal_errors" in frame.repair_candidates
        assert "reduce_background_compute_until_memory_recovers" in frame.repair_candidates
        assert world.events
        status = bridge.get_status()
        assert status["observations"] == 1
        assert frame.event_count >= 7
        assert status["latest_observation"]["ambient_event_count"] >= 3
        assert "repair candidates" in status["latest_observation"]["ambient_summary"]
        prompt_block = render_ambient_developer_prompt_block(frame)
        assert "Resource interrupts" in prompt_block
    finally:
        ServiceContainer.clear()


def test_ambient_network_permission_boundary_is_observed_without_degradation(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "core.perception.ambient_developer_stream.psutil.net_connections",
        Mock(side_effect=psutil.AccessDenied()),
    )
    stream = AmbientDeveloperStream(
        project_root=tmp_path,
        watch_roots=(),
        log_roots=(),
        terminal_roots=(),
    )

    events = stream._collect_network_events()

    assert len(events) == 1
    assert events[0].kind == "socket_visibility_unavailable"
    assert events[0].count == 0


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
async def test_native_multimodal_bridge_binds_ambient_frame_into_world_state(monkeypatch):
    ServiceContainer.clear()
    state = AuraState()
    frame = SimpleNamespace(
        frame_id=4,
        timestamp=123.0,
        summary="terminal warning/error line(s); resource interrupt(s)",
        event_count=2,
        repair_candidates=("review_recent_terminal_errors",),
        resource_interrupts=(
            SimpleNamespace(to_dict=lambda: {"kind": "memory_pressure", "severity": "warning"}),
        ),
        network_events=(
            SimpleNamespace(to_dict=lambda: {"kind": "established_connections", "count": 1}),
        ),
    )
    stream = SimpleNamespace(latest_frame=frame)
    monkeypatch.setattr(
        "core.perception.ambient_developer_stream.get_ambient_developer_stream",
        lambda: stream,
    )
    bridge = NativeMultimodalBridge(SimpleNamespace(organs={}))

    try:
        await bridge.execute(state, "Perception")

        percept = state.world.recent_percepts[-1]
        assert percept["role"] == "ambient_developer_stream"
        assert percept["frame_id"] == 4
        assert percept["repair_candidates"] == ["review_recent_terminal_errors"]
        assert percept["resource_interrupts"][0]["kind"] == "memory_pressure"
        assert percept["network_events"][0]["kind"] == "established_connections"
    finally:
        ServiceContainer.clear()


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


def test_dream_journal_compiles_autobiographical_mythos(monkeypatch, tmp_path):
    from core.adaptation import dream_journal
    from core.identity.identity_ledger import IdentityLedger

    class Gateway:
        def write_text(self, path, text, *, source):
            assert source == "adaptation.dream_journal.autobiographical_mythos"
            path.write_text(text, encoding="utf-8")

    journal = dream_journal.DreamJournal.__new__(dream_journal.DreamJournal)
    journal.journal_dir = tmp_path
    journal.journal_file = tmp_path / "dream_journal.txt"
    journal.autonomic_reflection_file = tmp_path / "autonomic_reflections.jsonl"
    journal.mythos_file = tmp_path / "autobiographical_mythos.json"
    journal.journal_file.write_text(
        "=== Dream: now ===\nrepair and continuity through the desktop body\n",
        encoding="utf-8",
    )
    journal.autonomic_reflection_file.write_text(
        '{"self_correction_note":"queue diagnosis and evidence receipts"}\n',
        encoding="utf-8",
    )
    ledger = IdentityLedger(root=tmp_path / "identity")
    ledger.commitments.add("verify live desktop conversation path")
    ledger.preferences.set("runtime_reliability", "daily-use", reason="closeout")
    ledger.versioning.snapshot({"phase": "after-crsm-caa-closure"})
    monkeypatch.setattr(dream_journal, "get_file_write_gateway", lambda: Gateway())

    payload = journal.compile_autobiographical_mythos(identity_ledger=ledger)
    block = journal.get_autobiographical_mythos_block()

    assert payload["schema"] == "aura.autobiographical_mythos.v1"
    assert "repair" in payload["motifs"]
    assert "continuity" in payload["motifs"]
    assert "identity ledger" in payload["narrative"]
    assert "AUTOBIOGRAPHICAL MYTHOS" in block
