import asyncio
import subprocess
from pathlib import Path

import pytest

from core.container import ServiceContainer
from core.event_bus import EventPriority, get_event_bus
from core.perception.multimodal_sync import Modality, MultimodalSynchronizer
from core.perception.perception_daemon import PerceptionDaemon
from core.perception.perception_runtime import PerceptionRuntime


@pytest.mark.asyncio
async def test_perception_daemon_lifecycle():
    daemon = PerceptionDaemon(check_interval_s=0.01)
    
    assert daemon.running is False
    await daemon.start()
    assert daemon.running is True
    
    # Wait for one poll cycle
    await asyncio.sleep(0.02)
    
    await daemon.stop()
    assert daemon.running is False


@pytest.mark.asyncio
async def test_perception_daemon_moment_buffering_and_privacy(monkeypatch):
    daemon = PerceptionDaemon(check_interval_s=0.1)
    
    bus = get_event_bus()
    published_events = []
    
    def _capture_publish_threadsafe(topic, data, priority=EventPriority.COGNITIVE):
        published_events.append((topic, data, priority))
        
    monkeypatch.setattr(bus, "publish_threadsafe", _capture_publish_threadsafe)
    
    # Register regular moment
    moment = daemon.register_moment("test_source", "Hello Bryan", {"custom": 123})
    assert moment["source"] == "test_source"
    assert moment["content"] == "Hello Bryan"
    assert moment["metadata"]["custom"] == 123
    
    # Register sensitive moment (should trigger privacy redaction)
    sensitive_moment = daemon.register_moment("test_source", "Here is my secret password")
    assert sensitive_moment["content"] == "<redacted: privacy policy>"
    assert sensitive_moment["metadata"]["redacted"] is True
    
    # Verify buffers
    recent = daemon.get_recent_moments(duration_seconds=5.0)
    assert len(recent) == 2
    assert recent[0]["content"] == "Hello Bryan"
    assert recent[1]["content"] == "<redacted: privacy policy>"
    
    # Verify event bus publishing
    assert len(published_events) == 2
    assert published_events[0][0] == "aura/perception/moment"
    assert published_events[0][1]["content"] == "Hello Bryan"
    assert published_events[0][2] == EventPriority.AUTONOMIC


def test_perception_daemon_bridges_redacted_semantics_into_canonical_fusion() -> None:
    ServiceContainer.clear()
    synchronizer = MultimodalSynchronizer()
    ServiceContainer.register_instance(
        "multimodal_synchronizer",
        synchronizer,
        required=False,
    )
    daemon = PerceptionDaemon()

    try:
        daemon.register_moment(
            "browser",
            "Private project tab (https://secret.example)",
            {"tabs": [{"title": "Private project tab", "url": "https://secret.example"}]},
        )
        frame = synchronizer.fuse("daemon-fusion")

        assert frame.has_usable(Modality.TEXT) is True
        assert frame.belief("browser.open_tab_count").value == 1
        assert "browser_titles_and_urls_not_retained" in frame.observations[
            Modality.TEXT
        ].quality_flags
        status_text = repr(synchronizer.get_status())
        assert "Private project tab" not in status_text
        assert "secret.example" not in status_text
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_perception_daemon_entity_tracking():
    daemon = PerceptionDaemon()
    
    # Track file entity
    ent1_id = daemon.track_entity("file", "main.py", {"size": 100})
    assert ent1_id.startswith("ent-")
    
    # Repeated track resolves to the same ID
    ent2_id = daemon.track_entity("file", "main.py", {"size": 200})
    assert ent1_id == ent2_id
    
    # Verify metadata update
    entity = daemon._entities[ent1_id]
    assert entity["metadata"]["size"] == 200
    
    # Track different entity
    ent3_id = daemon.track_entity("browser_tab", "Stanford expense portal")
    assert ent3_id != ent1_id


@pytest.mark.asyncio
async def test_perception_runtime_daemon_integration(monkeypatch):
    # Setup PerceptionRuntime
    async def _mock_decide(**kwargs):
        return {"approved": True, "receipt_id": "r1"}
        
    runtime = PerceptionRuntime(governance_decide=_mock_decide)
    assert runtime.daemon is not None
    
    moments_registered = []
    monkeypatch.setattr(runtime.daemon, "register_moment", lambda src, content, meta=None: moments_registered.append((src, content, meta)))
    
    # Open movie session should log to daemon
    runtime.open_movie_session(title="The Matrix", privacy_mode=True)
    assert len(moments_registered) == 1
    assert moments_registered[0][0] == "movie_session"
    assert "Started watching: The Matrix" in moments_registered[0][1]
    
    # Update focus should update daemon focus/attention
    runtime.shared_attention.update_focus(user="Safari", aura="thinking", confidence=0.8)
    assert runtime.daemon.user_focus == "Safari"
    assert runtime.daemon.aura_focus == "thinking"
    assert runtime.daemon.joint_attention_score == 0.8


@pytest.mark.asyncio
async def test_perception_daemon_active_perceive():
    daemon = PerceptionDaemon()
    
    # Active perceive file check
    res = await daemon.active_perceive("file_status", "pyproject.toml")
    assert res["ok"] is True
    assert "pyproject.toml" in res["result"]
    
    # Invalid path
    res_bad = await daemon.active_perceive("file_status", "non_existent_file.xyz")
    assert res_bad["ok"] is False
    assert res_bad["error"] == "file_not_found"


def test_perception_daemon_file_scan_is_bounded_and_prunes_heavy_dirs(monkeypatch, tmp_path):
    daemon = PerceptionDaemon(check_interval_s=10.0)
    recent = tmp_path / "recent.txt"
    recent.write_text("changed", encoding="utf-8")
    nested = tmp_path / ".venv" / "ignored.txt"
    nested.parent.mkdir()
    nested.write_text("ignored", encoding="utf-8")

    monkeypatch.setenv("AURA_PERCEPTION_FILE_SCAN_MAX_FILES", "2")
    monkeypatch.setenv("AURA_PERCEPTION_FILE_SCAN_MAX_SECONDS", "1.0")

    found = daemon._scan_recent_file_mutations(tmp_path, interval_s=10.0)

    assert str(recent) in found
    assert str(nested) not in found


@pytest.mark.asyncio
async def test_perception_daemon_file_scan_runs_off_event_loop(monkeypatch, tmp_path):
    daemon = PerceptionDaemon(check_interval_s=0.1)
    calls: list[tuple[Path, float]] = []

    async def _fake_sleep(_delay):
        daemon.running = False

    async def _fake_to_thread(func, *args):
        calls.append((args[0], args[1]))
        return []

    async def _none():
        return None

    monkeypatch.setattr("core.perception.perception_daemon.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr("core.perception.perception_daemon.asyncio.to_thread", _fake_to_thread)
    monkeypatch.setattr(daemon, "_check_active_window", _none)
    monkeypatch.setattr(daemon, "_check_clipboard", _none)
    daemon._file_scan_root = tmp_path
    daemon.running = True

    await daemon._main_perceptual_loop()

    assert calls == [(tmp_path, 0.1)]


@pytest.mark.asyncio
async def test_perception_daemon_active_window_timeout_does_not_kill_main_loop(monkeypatch):
    daemon = PerceptionDaemon(check_interval_s=0.1)
    checks = {"active_window": 0, "clipboard": 0}

    async def _fake_sleep(_delay):
        if checks["active_window"] >= 1:
            daemon.running = False

    async def _timeout_window():
        checks["active_window"] += 1
        raise subprocess.TimeoutExpired(cmd=["osascript"], timeout=1.5)

    async def _clipboard_probe():
        checks["clipboard"] += 1
        return None

    monkeypatch.setattr("core.perception.perception_daemon.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr(daemon, "_check_active_window", _timeout_window)
    monkeypatch.setattr(daemon, "_check_clipboard", _clipboard_probe)
    daemon.running = True

    await daemon._main_perceptual_loop()

    assert checks["active_window"] == 1
    assert checks["clipboard"] == 0
    assert daemon.running is False
