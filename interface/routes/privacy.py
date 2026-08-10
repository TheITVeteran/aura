"""interface/routes/privacy.py
──────────────────────────────
Extracted from server.py — Privacy toggles, voice endpoints,
and source download.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service
from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Privacy")

router = APIRouter()
_SOURCE_DOWNLOAD_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError)

# ── Voice Engine Accessor ─────────────────────────────────────
# The voice engine factory is set by the main server lifespan.
# This module provides a getter/setter so system.py can also access it.

_voice_engine_fn: Callable | None = None
_browser_camera_privacy: dict[str, Any] = {
    "enabled": False,
    "mode": "off",
    "reason": None,
    "vision_worker": {
        "schema": "aura.mlx_vision.readiness.v1",
        "ready": False,
        "reason": "not_observed",
    },
}


def set_voice_engine_fn(fn: Callable | None) -> None:
    global _voice_engine_fn
    _voice_engine_fn = fn


def get_voice_engine_fn() -> Callable | None:
    return _voice_engine_fn


def set_browser_camera_privacy(
    *,
    enabled: bool,
    mode: str = "off",
    reason: str | None = None,
    vision_worker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global _browser_camera_privacy
    prior_worker = dict(_browser_camera_privacy.get("vision_worker") or {})
    _browser_camera_privacy = {
        "enabled": bool(enabled),
        "mode": str(mode or ("browser_only" if enabled else "off")),
        "reason": reason,
        "vision_worker": dict(
            prior_worker if vision_worker is None else vision_worker
        ),
    }
    return get_browser_camera_privacy()


def get_browser_camera_privacy() -> dict[str, Any]:
    state = dict(_browser_camera_privacy)
    state["vision_worker"] = dict(state.get("vision_worker") or {})
    return state


def _vision_worker_readiness() -> dict[str, Any]:
    """Inspect sight readiness without loading the model or opening a camera."""
    try:
        from core.brain.llm.mlx_vision_client import get_vision_client

        return dict(get_vision_client().readiness_status())
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "schema": "aura.mlx_vision.readiness.v1",
            "ready": False,
            "reason": f"readiness_unavailable:{type(exc).__name__}",
        }


async def _commit_camera_permission(enabled: bool) -> bool:
    """Commit the owner switch through the canonical transactional store."""
    from interface.routes.settings import get_settings

    committed = await asyncio.to_thread(
        get_settings().set,
        "permissions.camera",
        bool(enabled),
    )
    return bool(committed)


def apply_camera_runtime_state(
    enabled: bool,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Apply a committed camera decision to hardware and visible state."""
    from core.perception.camera_authority import get_camera_authority
    from core.runtime.boot_safety import main_process_camera_policy

    enabled = bool(enabled)
    authority = get_camera_authority()
    revoked: dict[str, Any] = {"released": False, "holder": None}
    if not enabled:
        revoked = authority.revoke_owner_permission()

    authority_state = authority.state()
    main_allowed, main_reason = main_process_camera_policy(enabled)
    backend_available = bool(authority_state.get("backend_available"))
    transport = str(authority_state.get("transport") or "none")

    smc = get_runtime_service("sensory_motor_cortex", default=None)
    vision_buffer = get_runtime_service("continuous_vision", default=None)
    if smc is not None:
        smc.camera_enabled = bool(enabled and main_allowed)
    if vision_buffer is not None:
        # Keep this compatibility field honest about direct in-process use.
        vision_buffer.camera_enabled = bool(enabled and main_allowed)
        # Actual capture may be supplied by the isolated sidecar.
        vision_buffer.camera_capture_enabled = bool(enabled and backend_available)
        if not enabled:
            vision_buffer._camera_lease = None

    if not enabled:
        mode = "off"
        state_reason = reason
    elif transport == "sidecar" and backend_available:
        mode = "isolated_sidecar"
        state_reason = reason or main_reason
    elif transport == "in_process" and backend_available:
        mode = "full"
        state_reason = reason
    else:
        # Browser-supplied vision signals remain a usable, separately visible
        # transport even when this host has no native camera backend.
        mode = "browser_only"
        state_reason = reason or main_reason or "native_camera_backend_unavailable"

    vision_worker = _vision_worker_readiness()
    browser_state = set_browser_camera_privacy(
        enabled=enabled,
        mode=mode,
        reason=state_reason,
        vision_worker=vision_worker,
    )
    result = {
        "ok": True,
        "enabled": enabled,
        "mode": browser_state["mode"],
        "reason": browser_state["reason"],
        "transport": transport,
        "native_capture_enabled": bool(enabled and backend_available),
        "main_process_capture_enabled": bool(enabled and main_allowed),
        "vision_worker": vision_worker,
        "revocation": revoked,
    }
    logger.info(
        "\U0001f512 Privacy: Camera %s mode=%s transport=%s",
        "enabled" if enabled else "disabled",
        mode,
        transport,
    )
    return result


async def apply_camera_privacy(
    enabled: bool,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Commit and apply one owner camera decision end to end."""
    enabled = bool(enabled)
    committed = await _commit_camera_permission(enabled)
    if committed is not enabled:
        raise RuntimeError("camera_permission_commit_mismatch")
    # A no-op settings transaction does not notify subscribers. Reapply here
    # so an already-selected value repairs stale hardware/UI state as well.
    return await asyncio.to_thread(
        apply_camera_runtime_state,
        enabled,
        reason=reason,
    )


# ── Models ────────────────────────────────────────────────────

class PrivacyPayload(BaseModel):
    enabled: bool
    # True when the owner pressed the UI's voice-conversation control, as opposed
    # to the microphone coming up for ambient wake-word listening. Only the
    # former is an invitation to be answered without a wake word, so this
    # defaults False and every existing caller keeps the wake-word boundary.
    conversation: bool = False


# ── Routes ────────────────────────────────────────────────────

@router.post("/privacy/camera")
async def api_privacy_camera(payload: PrivacyPayload, _: None = Depends(_require_internal)):
    """Toggle the visual cortex camera processing."""
    return await apply_camera_privacy(payload.enabled)


@router.post("/privacy/microphone")
async def api_privacy_microphone(payload: PrivacyPayload, _: None = Depends(_require_internal)):
    """Toggle voice I/O and make enablement operational, not just declarative."""
    enabled = payload.enabled
    voice = _voice_engine_fn() if _voice_engine_fn else None
    if voice:
        if enabled:
            from core.runtime.runtime_settings import get_runtime_setting

            if not bool(get_runtime_setting("voice.input_enabled", True)):
                return JSONResponse(
                    {
                        "ok": False,
                        "enabled": False,
                        "microphone_enabled": False,
                        "speaking_enabled": bool(
                            getattr(voice, "speaking_enabled", True)
                        ),
                        "listening": bool(getattr(voice, "_mic_listening", False)),
                        "listening_started": False,
                        "error": "microphone_disabled_by_runtime_setting",
                    },
                    status_code=409,
                )
        voice.microphone_enabled = enabled
        listening_started = False
        start_error: str | None = None
        if enabled and hasattr(voice, "start_listening"):
            try:
                start_result = voice.start_listening()
                if inspect.isawaitable(start_result):
                    start_result = await start_result
                listening_started = bool(start_result)
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation(
                    "privacy",
                    exc,
                    action="rejected microphone enablement after listener startup failed",
                )
                start_error = f"{type(exc).__name__}: {exc}"
                listening_started = False
            if not listening_started:
                voice.microphone_enabled = False
        elif enabled:
            start_error = "microphone_listener_unavailable"
            voice.microphone_enabled = False
        elif not enabled and hasattr(voice, "stop_listening"):
            voice.stop_listening()
        logger.info("\U0001f512 Privacy: Microphone %s", 'enabled' if enabled else 'disabled')
        # An owner who pressed "Start voice conversation" has already declared
        # that the speech is for her; requiring a wake word per utterance made
        # her hear him and not answer. Ambient enablement keeps the boundary.
        if enabled and payload.conversation and bool(
            getattr(voice, "microphone_enabled", False)
        ):
            begin = getattr(voice, "begin_owner_voice_conversation", None)
            if callable(begin):
                begin()
        elif not enabled:
            end = getattr(voice, "end_owner_voice_conversation", None)
            if callable(end):
                end()
        listening = bool(getattr(voice, "_mic_listening", False))
        ok = bool((not enabled) or listening_started or listening)
        return {
            "ok": ok,
            "enabled": bool(getattr(voice, "microphone_enabled", enabled)),
            "microphone_enabled": getattr(voice, "microphone_enabled", enabled),
            "speaking_enabled": getattr(voice, "speaking_enabled", True),
            "listening": listening,
            "listening_started": listening_started,
            "error": start_error or (None if ok else "microphone_start_failed"),
        }
    return JSONResponse({"error": "VoiceEngine unavailable"}, status_code=503)


@router.post("/voice/chunk")
async def api_voice_chunk(
    _request: Request,
    _: None = Depends(_require_internal),
):
    """Reject the retired unleased browser PCM ingress."""
    raise HTTPException(
        status_code=410,
        detail="Legacy voice transport retired; use authenticated /ws/voice.",
    )


@router.get("/source")
async def api_source_download(
    _: None = Depends(_require_internal),
):
    """Bundle and return the current source code as a download."""
    project_root = config.paths.project_root
    try:
        import tempfile as _tf

        from utils.bundler import write_bundle
        with _tf.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            out = Path(tmp.name)
        write_bundle(project_root, out, lite=True)
        return FileResponse(
            str(out),
            media_type="text/plain",
            filename=f"aura_source_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        )
    except _SOURCE_DOWNLOAD_ERRORS as exc:
        record_degradation('privacy', exc)
        logger.error("Source download failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Source bundle generation failed",
        ) from exc


@router.get("/stream/voice")
async def voice_sse_stream(
    request: Request,
    _: None = Depends(_require_internal),
):
    """Server-Sent Events stream for voice pipeline output."""
    async def gen():
        sse_q: asyncio.Queue = asyncio.Queue(maxsize=50)
        voice = _voice_engine_fn() if _voice_engine_fn else None
        if voice and hasattr(voice, "subscribe"):
            await voice.subscribe(sse_q)
        try:
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(sse_q.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except TimeoutError:
                    yield "data: {\"type\":\"ping\"}\n\n"
        finally:
            if voice and hasattr(voice, "unsubscribe"):
                await voice.unsubscribe(sse_q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
