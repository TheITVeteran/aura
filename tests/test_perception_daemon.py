import asyncio
import pytest
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.perception.perception_daemon import PerceptionDaemon, get_perception_daemon
from core.perception.perception_runtime import PerceptionRuntime
from core.event_bus import get_event_bus, EventPriority


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
    
    # Clear event bus publish mock if needed
    bus = get_event_bus()
    published_events = []
    
    def _mock_publish_threadsafe(topic, data, priority=EventPriority.COGNITIVE):
        published_events.append((topic, data, priority))
        
    monkeypatch.setattr(bus, "publish_threadsafe", _mock_publish_threadsafe)
    
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
    
    # Mock daemon register_moment
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
