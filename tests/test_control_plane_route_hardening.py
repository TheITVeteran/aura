from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

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
    assert json.loads(settings_path.read_text(encoding="utf-8"))["theme.reduced_motion"] is True


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
