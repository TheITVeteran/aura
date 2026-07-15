from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.executive.action_confirmation import (
    ActionConfirmationRegistry,
    action_confirmation_fingerprint,
)
from interface.routes import interaction_signals, performance, privacy, settings, system


@pytest.mark.asyncio
async def test_performance_frame_reports_bad_payload_without_throwing(monkeypatch):
    class _Guard:
        def record_frame(self, *_args, **_kwargs):
            self.recorded = True

        def report(self):
            return {"motion_throttled": False}

    import core.runtime.performance_guard as performance_guard

    guard = _Guard()
    monkeypatch.setattr(performance_guard, "get_guard", lambda: guard)

    response = await performance.frame({"duration_ms": "bad-number", "source": "ui"}, _=None)
    payload = json.loads(response.body)

    assert payload["ok"] is False
    assert "bad-number" in payload["error"]


@pytest.mark.asyncio
async def test_vision_signal_rejects_invalid_image_payload(monkeypatch):
    published = {"vision": False}

    async def _publish_vision_frame(*_args, **_kwargs):
        published["vision"] = True

    monkeypatch.setattr(interaction_signals, "_camera_signal_allowed", lambda: True)
    monkeypatch.setattr(
        interaction_signals,
        "_get_engine",
        lambda: SimpleNamespace(publish_vision_frame=_publish_vision_frame),
    )

    with pytest.raises(HTTPException) as exc_info:
        await interaction_signals.api_signal_vision(
            interaction_signals.VisionSignalPayload(
                timestamp=1.0,
                frame_data_url="data:image/jpeg;base64,not-valid-base64",
            ),
            _=None,
        )

    assert exc_info.value.status_code == 400
    assert published["vision"] is False


def test_settings_store_subscriber_failure_does_not_block_valid_write(monkeypatch, tmp_path):
    settings_path = tmp_path / "runtime.json"
    monkeypatch.setattr(settings, "_SETTINGS_PATH", settings_path)
    store = settings.SettingsStore()
    calls = []

    def _failing_subscriber(key, previous, value):
        calls.append((key, previous, value))
        raise RuntimeError("subscriber unavailable")

    store.subscribe(_failing_subscriber)

    assert store.set("theme.reduced_motion", True) is True
    assert calls and calls[0][0] == "theme.reduced_motion"
    state = json.loads(settings_path.read_text(encoding="utf-8"))
    assert state["schema"] == "aura.runtime_settings"
    assert state["revision"] == 1
    assert state["values"]["theme.reduced_motion"] is True


@pytest.mark.asyncio
async def test_settings_api_uses_atomic_cas_and_frontend_acknowledgements(
    monkeypatch,
    tmp_path,
):
    store = settings.SettingsStore(tmp_path / "runtime.json")
    monkeypatch.setattr(settings, "_STORE", store)

    initial = json.loads((await settings.get_all(_=None)).body)
    assert initial["revision"] == 0
    assert initial["control_plane"]["cas_required"] is True

    patched_response = await settings.patch_settings(
        {
            "expected_revision": 0,
            "request_id": "route-patch",
            "changes": {"theme.reduced_motion": True},
        },
        _=None,
    )
    patched = json.loads(patched_response.body)
    assert patched_response.status_code == 200
    assert patched["revision"] == 1
    assert patched["application"]["theme.reduced_motion"]["status"] == "awaiting_frontend"

    acknowledged_response = await settings.acknowledge_settings_application(
        {
            "settings_receipt_hash": patched["receipt"]["receipt_hash"],
            "acknowledgements": {
                "theme.reduced_motion": {
                    "status": "applied",
                    "detail": "desktop animation policy applied",
                }
            },
        },
        _=None,
    )
    acknowledged = json.loads(acknowledged_response.body)
    assert acknowledged_response.status_code == 200
    assert acknowledged["application"]["theme.reduced_motion"]["status"] == "applied"

    stale_response = await settings.patch_settings(
        {
            "expected_revision": 0,
            "request_id": "route-stale",
            "changes": {"notify.enabled": False},
        },
        _=None,
    )
    stale = json.loads(stale_response.body)
    assert stale_response.status_code == 409
    assert stale == {
        "error": "settings_revision_conflict",
        "expected_revision": 0,
        "current_revision": 1,
        "retryable": True,
    }

    integrity = json.loads((await settings.settings_integrity(_=None)).body)
    assert integrity["ok"] is True
    assert integrity["application_entries"] == 2


@pytest.mark.asyncio
async def test_settings_confirmation_endpoint_authorizes_only_supplied_challenge(
    monkeypatch,
):
    registry = ActionConfirmationRegistry()
    fingerprint = action_confirmation_fingerprint(
        tool_name="desktop_task",
        arguments={"objective": "open Notes"},
        source="desktop_ui",
        risk_level="high",
        effect_scope="foreground_desktop_control",
    )
    challenge = registry.issue(
        action_fingerprint=fingerprint,
        tool_name="desktop_task",
    )
    conscience = SimpleNamespace(
        acknowledge_user_authorization=lambda: None,
        fresh_user_authorization_window_s=lambda: 60.0,
    )
    monkeypatch.setattr(
        "core.executive.action_confirmation.get_action_confirmation_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        "core.ethics.conscience.get_conscience",
        lambda: conscience,
    )

    response = await settings.acknowledge_fresh_auth(
        {"challenge_id": challenge["challenge_id"]},
        _=None,
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["one_time"] is True
    assert payload["action_bound"] is True
    assert registry.consume_authorized(fingerprint)[0] is True
    assert registry.consume_authorized(fingerprint)[0] is False


@pytest.mark.asyncio
async def test_settings_confirmation_revoke_cancels_unconsumed_authorization(
    monkeypatch,
):
    registry = ActionConfirmationRegistry()
    fingerprint = action_confirmation_fingerprint(
        tool_name="desktop_task",
        arguments={"objective": "open Notes"},
        source="desktop_ui",
        risk_level="high",
        effect_scope="foreground_desktop_control",
    )
    challenge = registry.issue(
        action_fingerprint=fingerprint,
        tool_name="desktop_task",
    )
    registry.authorize(challenge["challenge_id"])
    monkeypatch.setattr(
        "core.executive.action_confirmation.get_action_confirmation_registry",
        lambda: registry,
    )

    response = await settings.revoke_action_confirmation(
        {"challenge_id": challenge["challenge_id"]},
        _=None,
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload == {"ok": True, "cancelled": True}
    assert registry.consume_authorized(fingerprint)[0] is False


@pytest.mark.asyncio
async def test_privacy_source_download_reports_bundle_failure(monkeypatch):
    import utils.bundler as bundler

    called = {"write_bundle": False}

    def _fail_write_bundle(*_args, **_kwargs):
        called["write_bundle"] = True
        raise RuntimeError("bundle writer unavailable")

    monkeypatch.setattr(bundler, "write_bundle", _fail_write_bundle)

    with pytest.raises(HTTPException) as exc_info:
        await privacy.api_source_download(_=None)

    assert called["write_bundle"] is True
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_control_plane_diagnostics_exposes_reconciler_freshness(monkeypatch):
    plane = SimpleNamespace(
        get_status=lambda: {
            "alive": True,
            "ready": False,
            "services": {"event_loop_monitor": {"observed_state": "ready"}},
        }
    )
    monkeypatch.setattr(system, "_require_internal", lambda _request: None)
    monkeypatch.setattr(
        system.ServiceContainer,
        "peek",
        staticmethod(
            lambda name, default=None: plane
            if name == "runtime_control_plane"
            else default
        ),
    )
    monkeypatch.setattr(
        system.scheduler,
        "get_health",
        lambda: {
            "task_details": {
                "runtime_control_plane_reconcile": {
                    "status": "ok",
                    "freshness": "fresh",
                }
            }
        },
    )

    response = await system.api_system_control_plane(SimpleNamespace())
    payload = json.loads(response.body)

    assert payload["available"] is True
    assert payload["control_plane"]["ready"] is False
    assert payload["reconcile_task"]["freshness"] == "fresh"
