"""Tests for the layering-clean runtime-settings accessor and the first dead
setting wired through it: voice.output_enabled (docs/SETTINGS_WIRING_AUDIT.md)."""
import asyncio
import json

import pytest

from core.runtime import runtime_settings as rs


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_SETTINGS_PATH", str(tmp_path / "runtime.json"))
    rs.clear_runtime_settings_cache()
    yield
    rs.clear_runtime_settings_cache()


def _write_settings(tmp_path, data):
    (tmp_path / "runtime.json").write_text(json.dumps(data), encoding="utf-8")
    rs.clear_runtime_settings_cache()


def test_missing_file_returns_default(tmp_path):
    assert rs.get_runtime_setting("voice.output_enabled", True) is True
    assert rs.get_runtime_setting("voice.output_enabled", False) is False
    assert rs.get_runtime_setting("nope.key") is None


def test_present_key_returned(tmp_path):
    _write_settings(tmp_path, {"voice.output_enabled": False})
    assert rs.get_runtime_setting("voice.output_enabled", True) is False


def test_missing_key_uses_default(tmp_path):
    _write_settings(tmp_path, {"other.key": 1})
    assert rs.get_runtime_setting("voice.output_enabled", True) is True


def test_legacy_false_value_cannot_override_protected_agency_default(tmp_path):
    _write_settings(tmp_path, {"autonomy.actions_enabled": False})

    assert rs.get_runtime_setting("autonomy.actions_enabled", False) is True
    assert rs.autonomous_actions_admitted("autonomous_initiative_loop") == (
        True,
        "autonomous_agency_invariant",
    )


def test_corrupt_file_falls_back_to_default(tmp_path):
    (tmp_path / "runtime.json").write_text("{ not valid json", encoding="utf-8")
    rs.clear_runtime_settings_cache()
    assert rs.get_runtime_setting("voice.output_enabled", True) is True


def test_corrupt_file_activates_conservative_governance_without_erasing_agency(tmp_path):
    (tmp_path / "runtime.json").write_text("{ not valid json", encoding="utf-8")
    rs.clear_runtime_settings_cache()

    assert rs.get_runtime_setting("autonomy.actions_enabled", True) is True
    assert rs.get_runtime_setting("autonomy.level", "full") == "paused"
    assert rs.get_runtime_setting("governance.approval_mode", "none") == "all"
    assert rs.get_runtime_setting("safety.safe_mode", False) is True
    assert rs.get_runtime_setting("privacy.mode", "standard") == "isolated"
    assert rs.autonomous_actions_admitted("curiosity") == (
        True,
        "autonomous_agency_invariant",
    )


def test_deleted_settings_after_valid_read_fail_closed(tmp_path):
    _write_settings(
        tmp_path,
        {
            "autonomy.actions_enabled": True,
            "governance.approval_mode": "none",
        },
    )
    assert rs.get_runtime_setting("autonomy.actions_enabled", False) is True
    (tmp_path / "runtime.json").unlink()

    assert rs.get_runtime_setting("autonomy.actions_enabled", True) is True
    assert rs.get_runtime_setting("governance.approval_mode", "none") == "all"


def test_versioned_state_is_verified_against_its_audit_chain(tmp_path):
    from core.runtime.settings_control_plane import RuntimeSettingsStore

    path = tmp_path / "runtime.json"
    store = RuntimeSettingsStore(path)
    store.patch(
        {"notify.enabled": False},
        expected_revision=0,
        request_id="runtime-reader-versioned-state",
    )
    rs.clear_runtime_settings_cache()

    assert rs.get_runtime_setting("notify.enabled", True) is False

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["values"]["autonomy.actions_enabled"] = True
    envelope["values"]["governance.approval_mode"] = "none"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    rs.clear_runtime_settings_cache()

    assert rs.get_runtime_setting("autonomy.actions_enabled", True) is True
    assert rs.get_runtime_setting("governance.approval_mode", "none") == "all"


def test_live_update_reflected(tmp_path):
    _write_settings(tmp_path, {"voice.output_enabled": True})
    assert rs.get_runtime_setting("voice.output_enabled", True) is True
    _write_settings(tmp_path, {"voice.output_enabled": False})
    assert rs.get_runtime_setting("voice.output_enabled", True) is False


def test_voice_output_predicate_default_on(tmp_path):
    from core.senses.voice_engine import _user_voice_output_enabled

    assert _user_voice_output_enabled() is True


def test_voice_output_disabled_suppresses_synthesis(tmp_path):
    _write_settings(tmp_path, {"voice.output_enabled": False})

    from core.senses.voice_engine import SovereignVoiceEngine, _user_voice_output_enabled

    assert _user_voice_output_enabled() is False

    # Uninitialized instance: synthesize_speech must short-circuit at the user
    # gate (before acquiring locks or touching the TTS engine), returning "".
    engine = object.__new__(SovereignVoiceEngine)
    engine.speaking_enabled = True
    result = asyncio.run(engine.synthesize_speech("hello there"))
    assert result == ""


def test_voice_input_predicate_default_on(tmp_path):
    from core.voice.local_voice_cortex import _user_voice_input_enabled

    assert _user_voice_input_enabled() is True


def test_voice_input_disabled_never_opens_microphone(tmp_path):
    import threading
    from unittest.mock import MagicMock

    _write_settings(tmp_path, {"voice.input_enabled": False})

    from core.voice.local_voice_cortex import LocalVoiceCortex, _user_voice_input_enabled

    assert _user_voice_input_enabled() is False

    # Uninitialized cortex with just enough wired to pass the init guard and
    # enter the listen loop; the user gate must skip opening the mic stream.
    cortex = object.__new__(LocalVoiceCortex)
    cortex.audio_interface = MagicMock()
    cortex.audio_queue = MagicMock()
    cortex.vad = MagicMock()
    cortex._shutdown_event = threading.Event()
    cortex.is_listening = True

    async def run():
        try:
            await asyncio.wait_for(cortex.listen_loop(), timeout=0.3)
        except TimeoutError:
            pass
        finally:
            cortex.is_listening = False

    asyncio.run(run())
    cortex.audio_interface.open.assert_not_called()


# ── notify.enabled + quiet hours ────────────────────────────────────────────

def test_quiet_hours_window_logic():
    from datetime import datetime

    from core.senses.notifications import _within_quiet_hours

    # Wrap-around window 22:00 -> 08:00.
    assert _within_quiet_hours(datetime(2026, 6, 21, 23, 0), "22:00", "08:00") is True
    assert _within_quiet_hours(datetime(2026, 6, 21, 3, 0), "22:00", "08:00") is True
    assert _within_quiet_hours(datetime(2026, 6, 21, 12, 0), "22:00", "08:00") is False
    # Same-day window 09:00 -> 17:00.
    assert _within_quiet_hours(datetime(2026, 6, 21, 12, 0), "09:00", "17:00") is True
    assert _within_quiet_hours(datetime(2026, 6, 21, 20, 0), "09:00", "17:00") is False
    # Zero-length / malformed windows = quiet hours effectively off.
    assert _within_quiet_hours(datetime(2026, 6, 21, 12, 0), "10:00", "10:00") is False
    assert _within_quiet_hours(datetime(2026, 6, 21, 12, 0), "bad", "08:00") is False


def test_notify_disabled_suppresses_send(tmp_path, monkeypatch):
    _write_settings(tmp_path, {"notify.enabled": False})

    from core.senses import notifications as notif

    called = {"run": False}

    class _SpyGateway:
        def run(self, *a, **k):
            called["run"] = True
            raise AssertionError("subprocess must not run when notifications disabled")

    monkeypatch.setattr(notif, "get_subprocess_gateway", lambda: _SpyGateway())
    assert notif._notifications_allowed() is False
    notif.DesktopNotifier.send("Aura", "hello")  # must short-circuit, no raise
    assert called["run"] is False


# ── model.local_path / model.deep_path ──────────────────────────────────────

def test_model_local_path_override_used_when_present(tmp_path):
    from core.brain.llm import model_registry as mr

    real_model = tmp_path / "mlx-model"
    real_model.mkdir()
    _write_settings(tmp_path, {"model.local_path": str(real_model)})

    assert mr._user_model_path_override(mr.ACTIVE_MODEL) == str(real_model)
    assert mr.get_runtime_model_path(mr.ACTIVE_MODEL) == str(real_model)


def test_model_path_override_ignored_when_file_missing(tmp_path):
    from core.brain.llm import model_registry as mr

    _write_settings(tmp_path, {"model.local_path": str(tmp_path / "nope")})
    assert mr._user_model_path_override(mr.ACTIVE_MODEL) is None


def test_model_path_override_ignores_retired_external_artifacts(tmp_path):
    from core.brain.llm import model_registry as mr

    retired_model = tmp_path / "old-model.gguf"
    retired_model.write_text("x", encoding="utf-8")
    _write_settings(tmp_path, {"model.local_path": str(retired_model)})

    assert mr._user_model_path_override(mr.ACTIVE_MODEL) is None


def test_model_path_override_none_for_unrelated_lane(tmp_path):
    from core.brain.llm import model_registry as mr

    real_model = tmp_path / "mlx-model"
    real_model.mkdir()
    _write_settings(tmp_path, {"model.local_path": str(real_model)})
    # A lane that is neither the primary nor deep model gets no override.
    assert mr._user_model_path_override("some-other-lane-model") is None


# ── dev.developer_mode gates /trace ─────────────────────────────────────────

def test_trace_route_gated_on_developer_mode(tmp_path):
    from fastapi import HTTPException

    from interface.routes.dashboard import trace

    # Default off → 403.
    with pytest.raises(HTTPException) as exc_off:
        asyncio.run(trace(receipt_id="x", _=None))
    assert exc_off.value.status_code == 403

    # Enabled → no longer gated (proceeds; resolves to a non-403 outcome).
    _write_settings(tmp_path, {"dev.developer_mode": True})
    with pytest.raises(HTTPException) as exc_on:
        asyncio.run(trace(receipt_id="x", _=None))
    assert exc_on.value.status_code != 403


# ── permissions.camera / permissions.screen ─────────────────────────────────

def test_permission_gates_default_allowed(tmp_path):
    from core.runtime import permission_gates as pg

    assert pg.camera_allowed() is True
    assert pg.screen_allowed() is True
    assert pg.workspace_files_allowed() is True


def test_permission_gates_reflect_disabled(tmp_path):
    from core.runtime import permission_gates as pg

    _write_settings(tmp_path, {"permissions.camera": False, "permissions.screen": False})
    assert pg.camera_allowed() is False
    assert pg.screen_allowed() is False


def test_camera_capture_blocked_when_permission_off(tmp_path):
    _write_settings(tmp_path, {"permissions.camera": False})

    from core.perception.sensory_integration import VisionSystem

    vision = object.__new__(VisionSystem)
    result = asyncio.run(vision.capture())
    assert result.get("error") == "camera_permission_denied"


def test_screen_sensor_blocked_when_permission_off(tmp_path):
    _write_settings(tmp_path, {"permissions.screen": False})

    from core.body.screen_sensor import ScreenSensor

    sensor = object.__new__(ScreenSensor)
    result = asyncio.run(sensor.read())
    assert result.get("available") is False
    assert "permissions.screen" in str(result.get("error", ""))


def test_screenshot_tool_denied_when_screen_permission_off(tmp_path):
    from types import SimpleNamespace

    _write_settings(tmp_path, {"permissions.screen": False})

    from core.tools.computer_use import ComputerUseSkill

    skill = object.__new__(ComputerUseSkill)
    action = SimpleNamespace(target="screen", payload={})
    with pytest.raises(PermissionError):
        asyncio.run(skill._default_screenshot(action))


# ── autonomy.proactive_messaging ────────────────────────────────────────────

def test_proactive_messaging_toggle(tmp_path):
    from core.autonomy.proactive_communication import _proactive_messaging_enabled

    assert _proactive_messaging_enabled() is True
    _write_settings(tmp_path, {"autonomy.proactive_messaging": False})
    assert _proactive_messaging_enabled() is False


# ── autonomy.self_modification (blocked / staged / open) ─────────────────────

def test_self_modification_blocked_refuses_proposals(tmp_path):
    from core.self_modification.growth_ladder import GrowthLadder

    _write_settings(tmp_path, {"autonomy.self_modification": "blocked"})
    ladder = GrowthLadder(state_path=tmp_path / "gl.json")
    result = asyncio.run(
        ladder.propose_modification(
            proposal_id="p1",
            modification_type="behavioral_adjustment",
            level=0,
            description="d",
            predicted_stability_risk=0.0,
            predicted_welfare_cost=0.0,
        )
    )
    assert result is False
    assert ladder._proposals == []  # refused before anything was recorded


def test_self_modification_staged_passes_the_policy_gate(tmp_path):
    from core.self_modification.growth_ladder import GrowthLadder

    _write_settings(tmp_path, {"autonomy.self_modification": "staged"})
    ladder = GrowthLadder(state_path=tmp_path / "gl.json")
    asyncio.run(
        ladder.propose_modification(
            proposal_id="p2",
            modification_type="behavioral_adjustment",
            level=0,
            description="d",
            predicted_stability_risk=0.0,
            predicted_welfare_cost=0.0,
        )
    )
    # Default policy "staged" does not refuse at the gate: the proposal is recorded.
    assert any(p.id == "p2" for p in ladder._proposals)


# ── model.cloud_fallback_enabled (authoritative over caller request) ─────────

def test_cloud_fallback_setting_default_off(tmp_path):
    from core.runtime.runtime_settings import get_runtime_setting

    # Default off → a caller requesting cloud fallback is still gated to False
    # (allow_cloud_fallback = requested AND this setting).
    assert bool(get_runtime_setting("model.cloud_fallback_enabled", False)) is False
    _write_settings(tmp_path, {"model.cloud_fallback_enabled": True})
    assert bool(get_runtime_setting("model.cloud_fallback_enabled", False)) is True


# ── memory.retention_days (recency horizon in the sovereign pruner) ──────────

def test_memory_retention_days_extends_recency_horizon(tmp_path):
    import time
    from types import SimpleNamespace

    from core.memory.sovereign_pruner import SovereignPruner

    pruner = object.__new__(SovereignPruner)
    old_mem = SimpleNamespace(
        timestamp=time.time() - 180 * 86400,  # 180 days old
        emotional_weight=0.0,
        identity_relevance=0.0,
        referenced_count=0,
        content="x",
    )
    _write_settings(tmp_path, {"memory.retention_days": 365})
    score_long = pruner._score_memory(old_mem, {})
    _write_settings(tmp_path, {"memory.retention_days": 90})
    score_short = pruner._score_memory(old_mem, {})
    # A longer retention horizon keeps an old memory's recency competitive.
    assert score_long > score_short


# ── privacy.mode (world bridge gate) ────────────────────────────────────────

def test_privacy_mode_isolated_blocks_world_bridge(tmp_path):
    from core.embodiment.world_bridge import Channel, WorldBridge

    _write_settings(tmp_path, {"privacy.mode": "isolated"})
    bridge = WorldBridge()
    result = asyncio.run(bridge.call(Channel.SOCIAL_POST, action="post", intent="share"))
    assert result.ok is False
    assert result.error == "privacy_mode_isolated"


def test_privacy_mode_private_blocks_social_post(tmp_path):
    from core.embodiment.world_bridge import Channel, WorldBridge

    _write_settings(tmp_path, {"privacy.mode": "private"})
    bridge = WorldBridge()
    result = asyncio.run(bridge.call(Channel.SOCIAL_POST, action="post", intent="share"))
    assert result.ok is False
    assert result.error == "privacy_mode_private"


def test_privacy_mode_standard_does_not_block_at_privacy_gate(tmp_path):
    from core.embodiment.world_bridge import Channel, WorldBridge

    # Default "standard": the privacy gate must not fire (call proceeds to the
    # normal permission/will path, which denies for other reasons here).
    bridge = WorldBridge()
    result = asyncio.run(bridge.call(Channel.SOCIAL_POST, action="post", intent="share"))
    assert result.ok is False
    assert result.error not in ("privacy_mode_isolated", "privacy_mode_private")


# ── voice.output_rate (TTS speech-rate multiplier) ──────────────────────────

def test_voice_output_rate_multiplier_applied(tmp_path):
    import threading
    from unittest.mock import MagicMock

    from core.perception.sensory_integration import SpeechSystem

    _write_settings(tmp_path, {"voice.output_rate": 2.0})
    speech = object.__new__(SpeechSystem)
    speech.tts_available = True
    speech._lock = threading.Lock()
    engine = MagicMock()
    speech._engine = engine

    result = asyncio.run(speech.speak("hello", rate=150, save_path=str(tmp_path / "out.wav")))
    assert result.get("success") is True
    # 150 base wpm * 2.0 multiplier = 300
    engine.setProperty.assert_any_call("rate", 300)
