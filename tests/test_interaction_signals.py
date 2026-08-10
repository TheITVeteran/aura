import asyncio
import base64
import sys
import time
from types import SimpleNamespace

import pytest

from core.senses.interaction_signals import InteractionSignalsEngine, decode_data_url_image
from interface.routes import privacy as privacy_routes
from interface.routes.interaction_signals import _camera_signal_allowed


def test_decode_data_url_image_round_trip():
    raw = b"fake-jpeg-payload"
    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    assert decode_data_url_image(data_url) == raw


def test_interaction_signals_typing_hesitation_biases_concise_guidance():
    engine = InteractionSignalsEngine()
    engine._typing = engine._update_typing_state(
        {
            "timestamp": time.time(),
            "active": False,
            "session_ms": 5400,
            "key_count": 18,
            "correction_count": 5,
            "max_pause_ms": 2100,
            "pause_before_submit_ms": 1900,
            "message_chars": 24,
            "submitted": True,
        }
    )

    fused = engine._compute_fused_state()
    guidance = engine.get_prompt_guidance()

    assert fused.hesitation > 0.5
    assert fused.verbosity_bias == "concise"
    assert "LIVE HUMAN SIGNALS" in guidance
    assert "question pressure" in guidance.lower()


def test_interaction_signals_voice_and_vision_raise_attention_and_engagement():
    engine = InteractionSignalsEngine()
    now = time.time()
    engine._voice = engine._update_voice_state(
        {
            "timestamp": now,
            "speech_ratio": 0.92,
            "rms_avg": 0.065,
            "rms_std": 0.009,
            "peak_avg": 0.34,
            "zcr_avg": 0.13,
            "clipping_ratio": 0.0,
        }
    )
    engine._vision = engine._update_vision_state(
        {
            "updated_at": now,
            "face_present": True,
            "face_count": 1,
            "face_area_ratio": 0.18,
            "gaze_direction": "center",
            "head_pose": "center",
            "attention_available": 0.86,
            "eyes_detected": 2,
        }
    )

    fused = engine._compute_fused_state()

    assert engine._voice.label in {"activated", "steady"}
    assert fused.engagement > 0.45
    assert fused.attention_available > 0.75
    assert "voice" in fused.active_modalities
    assert "vision" in fused.active_modalities


@pytest.mark.asyncio
async def test_interaction_voice_signal_updates_world_state_without_transcript():
    from core.world_state import get_world_state

    ws = get_world_state()
    previous = {
        "voice_activity_detected": ws.voice_activity_detected,
        "last_voice_activity_at": getattr(ws, "last_voice_activity_at", 0.0),
        "ambient_audio_level": ws.ambient_audio_level,
        "last_audio_source_assessment": dict(ws.last_audio_source_assessment),
    }
    try:
        ws.voice_activity_detected = False
        ws.last_voice_activity_at = 0.0
        ws.ambient_audio_level = 0.0
        ws.last_audio_source_assessment = {}

        engine = InteractionSignalsEngine()
        await engine.publish_voice(
            {
                "timestamp": time.time(),
                "speech_ratio": 0.84,
                "rms_avg": 0.22,
                "rms_std": 0.02,
                "peak_avg": 0.38,
                "zcr_avg": 0.12,
            }
        )

        assert ws.voice_activity_detected is True
        assert ws.last_voice_activity_at > 0.0
        assert ws.ambient_audio_level >= 0.22
        assert ws.last_audio_source_assessment["source"] == "browser_voice_signal"
        assert ws.last_audio_source_assessment["transcript_available"] is False
        await engine.stop()
    finally:
        ws.voice_activity_detected = previous["voice_activity_detected"]
        ws.last_voice_activity_at = previous["last_voice_activity_at"]
        ws.ambient_audio_level = previous["ambient_audio_level"]
        ws.last_audio_source_assessment = previous["last_audio_source_assessment"]


def test_interaction_signals_defers_inprocess_cv2_after_pyav_load(monkeypatch):
    engine = InteractionSignalsEngine()

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace())
    monkeypatch.delenv("AURA_ALLOW_INPROCESS_CV2_WITH_STT", raising=False)

    result = engine._analyze_vision_frame_sync(b"not-a-real-jpeg", {})

    assert result["method"] == "vision_backend_deferred"
    assert result["reason"] == "cv2_blocked_in_main_process_after_pyav_load"
    assert engine._vision_backend_ready is False


@pytest.mark.asyncio
async def test_interaction_signals_async_publish_updates_queue_consumers():
    engine = InteractionSignalsEngine()
    await engine.publish_typing(
        {
            "timestamp": time.time(),
            "active": True,
            "session_ms": 1200,
            "key_count": 14,
            "correction_count": 1,
            "max_pause_ms": 240,
            "pause_before_submit_ms": 0,
            "message_chars": 19,
            "submitted": False,
        }
    )

    for _ in range(20):
        await asyncio.sleep(0.01)
        status = engine.get_status()
        if status["typing"]["message_chars"] == 19:
            break
    else:
        status = engine.get_status()

    await engine.stop()

    assert status["typing"]["message_chars"] == 19
    assert status["typing"]["label"] in {"flowing", "rapid", "considered"}
    assert status["queues"]["typing_depth"] == 0


@pytest.mark.asyncio
async def test_isolated_camera_privacy_keeps_native_and_browser_vision_available(monkeypatch):
    original_state = privacy_routes.get_browser_camera_privacy()
    smc = SimpleNamespace(camera_enabled=False)
    vision_buffer = SimpleNamespace(
        camera_enabled=False,
        camera_capture_enabled=False,
        _camera_lease=None,
    )

    def fake_get(name, default=None):
        if name == "sensory_motor_cortex":
            return smc
        if name == "continuous_vision":
            return vision_buffer
        return default

    monkeypatch.setattr(privacy_routes, "get_runtime_service", fake_get)
    monkeypatch.setattr(
        privacy_routes,
        "_commit_camera_permission",
        lambda enabled: asyncio.sleep(0, result=bool(enabled)),
    )

    import core.runtime.boot_safety as boot_safety

    monkeypatch.setattr(
        boot_safety,
        "main_process_camera_policy",
        lambda enabled: (False, "denied for tests"),
    )

    import core.perception.camera_authority as camera_authority

    authority = SimpleNamespace(
        state=lambda: {
            "backend_available": True,
            "transport": "sidecar",
        },
        revoke_owner_permission=lambda: {"released": False, "holder": None},
    )
    monkeypatch.setattr(camera_authority, "get_camera_authority", lambda: authority)
    monkeypatch.setattr(
        privacy_routes,
        "_vision_worker_readiness",
        lambda: {
            "schema": "aura.mlx_vision.readiness.v1",
            "ready": False,
            "reason": "not_started",
        },
    )

    try:
        response = await privacy_routes.api_privacy_camera(
            privacy_routes.PrivacyPayload(enabled=True), None
        )

        assert response["ok"] is True
        assert response["enabled"] is True
        assert response["mode"] == "isolated_sidecar"
        assert response["native_capture_enabled"] is True
        assert response["vision_worker"]["ready"] is False
        assert response["vision_worker"]["reason"] == "not_started"
        assert smc.camera_enabled is False
        assert vision_buffer.camera_enabled is False
        assert vision_buffer.camera_capture_enabled is True
        assert _camera_signal_allowed() is True
    finally:
        privacy_routes.set_browser_camera_privacy(
            enabled=bool(original_state.get("enabled", False)),
            mode=str(original_state.get("mode", "off")),
            reason=original_state.get("reason"),
        )


@pytest.mark.asyncio
async def test_camera_privacy_off_revokes_hardware_and_all_visible_states(monkeypatch):
    original_state = privacy_routes.get_browser_camera_privacy()
    lease = object()
    smc = SimpleNamespace(camera_enabled=True)
    vision_buffer = SimpleNamespace(
        camera_enabled=True,
        camera_capture_enabled=True,
        _camera_lease=lease,
    )
    committed: list[bool] = []
    revoked: list[bool] = []

    def fake_get(name, default=None):
        if name == "sensory_motor_cortex":
            return smc
        if name == "continuous_vision":
            return vision_buffer
        return default

    async def commit(enabled):
        committed.append(bool(enabled))
        return bool(enabled)

    authority = SimpleNamespace(
        revoke_owner_permission=lambda: (
            revoked.append(True)
            or {"released": True, "holder": "continuous_vision"}
        ),
        state=lambda: {
            "backend_available": True,
            "transport": "sidecar",
        },
    )
    monkeypatch.setattr(privacy_routes, "get_runtime_service", fake_get)
    monkeypatch.setattr(privacy_routes, "_commit_camera_permission", commit)
    import core.perception.camera_authority as camera_authority
    import core.runtime.boot_safety as boot_safety

    monkeypatch.setattr(camera_authority, "get_camera_authority", lambda: authority)
    monkeypatch.setattr(
        boot_safety,
        "main_process_camera_policy",
        lambda enabled: (False, "disabled"),
    )

    try:
        response = await privacy_routes.api_privacy_camera(
            privacy_routes.PrivacyPayload(enabled=False), None
        )

        assert committed == [False]
        assert revoked == [True]
        assert response["mode"] == "off"
        assert response["revocation"]["holder"] == "continuous_vision"
        assert smc.camera_enabled is False
        assert vision_buffer.camera_enabled is False
        assert vision_buffer.camera_capture_enabled is False
        assert vision_buffer._camera_lease is None
        assert _camera_signal_allowed() is False
    finally:
        privacy_routes.set_browser_camera_privacy(
            enabled=bool(original_state.get("enabled", False)),
            mode=str(original_state.get("mode", "off")),
            reason=original_state.get("reason"),
        )


@pytest.mark.asyncio
async def test_camera_privacy_commits_the_canonical_owner_setting(
    monkeypatch, tmp_path
):
    import interface.routes.settings as settings_routes

    store = settings_routes.SettingsStore(tmp_path / "runtime.json")
    monkeypatch.setattr(settings_routes, "_STORE", store)

    assert await privacy_routes._commit_camera_permission(False) is False
    assert store.snapshot().values["permissions.camera"] is False

    assert await privacy_routes._commit_camera_permission(True) is True
    assert store.snapshot().values["permissions.camera"] is True


def test_settings_control_plane_applies_camera_hardware_state(
    monkeypatch, tmp_path
):
    import interface.routes.settings as settings_routes

    store = settings_routes.SettingsStore(tmp_path / "runtime.json")
    applied: list[tuple[bool, str | None]] = []

    def apply(enabled, *, reason=None):
        applied.append((bool(enabled), reason))
        return {
            "mode": "off" if not enabled else "isolated_sidecar",
            "transport": "sidecar",
            "native_capture_enabled": bool(enabled),
        }

    monkeypatch.setattr(privacy_routes, "apply_camera_runtime_state", apply)
    monkeypatch.setattr(settings_routes, "SettingsStore", lambda: store)
    monkeypatch.setattr(settings_routes, "_STORE", None)
    wired_store = settings_routes.get_settings()

    result = wired_store.patch(
        {"permissions.camera": False},
        expected_revision=wired_store.snapshot().revision,
        actor="test",
    )

    assert applied == [
        (False, "applied from the transactional runtime settings control plane")
    ]
    assert result.application["permissions.camera"]["status"] == "applied"
    assert result.application["permissions.camera"]["owner"] == "camera_runtime"
