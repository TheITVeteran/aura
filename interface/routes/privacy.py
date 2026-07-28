"""interface/routes/privacy.py
──────────────────────────────
Extracted from server.py — Privacy toggles, voice endpoints,
and source download.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.config import config
from core.runtime.service_registry import get_runtime_service

from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Privacy")

router = APIRouter()
_SOURCE_DOWNLOAD_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError)

# ── Voice Engine Accessor ─────────────────────────────────────
# The voice engine factory is set by the main server lifespan.
# This module provides a getter/setter so system.py can also access it.

_voice_engine_fn: Optional[Callable] = None
_browser_camera_privacy: Dict[str, Any] = {
    "enabled": False,
    "mode": "off",
    "reason": None,
}


def set_voice_engine_fn(fn: Optional[Callable]) -> None:
    global _voice_engine_fn
    _voice_engine_fn = fn


def get_voice_engine_fn() -> Optional[Callable]:
    return _voice_engine_fn


def set_browser_camera_privacy(*, enabled: bool, mode: str = "off", reason: Optional[str] = None) -> Dict[str, Any]:
    global _browser_camera_privacy
    _browser_camera_privacy = {
        "enabled": bool(enabled),
        "mode": str(mode or ("browser_only" if enabled else "off")),
        "reason": reason,
    }
    return dict(_browser_camera_privacy)


def get_browser_camera_privacy() -> Dict[str, Any]:
    return dict(_browser_camera_privacy)


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
    enabled = payload.enabled
    smc = get_runtime_service("sensory_motor_cortex", default=None)
    vision_buffer = get_runtime_service("continuous_vision", default=None)

    if not smc and not vision_buffer:
        return JSONResponse({"error": "Camera systems unavailable"}, status_code=503)

    if enabled:
        from core.runtime.boot_safety import main_process_camera_policy

        camera_allowed, reason = main_process_camera_policy(True)
        if not camera_allowed:
            if smc is not None:
                smc.camera_enabled = False
            if vision_buffer is not None:
                vision_buffer.camera_enabled = False
            browser_state = set_browser_camera_privacy(
                enabled=True,
                mode="browser_only",
                reason=reason,
            )
            logger.warning(
                "\U0001f512 Privacy: Main-process camera denied (%s); browser-only camera remains available",
                reason,
            )
            return {
                "ok": True,
                "enabled": True,
                "mode": browser_state["mode"],
                "reason": browser_state["reason"],
            }

    if smc is not None:
        smc.camera_enabled = enabled
    if vision_buffer is not None:
        vision_buffer.camera_enabled = enabled
    browser_state = set_browser_camera_privacy(
        enabled=enabled,
        mode="full" if enabled else "off",
        reason=None,
    )
    logger.info("\U0001f512 Privacy: Camera %s", 'enabled' if enabled else 'disabled')
    return {
        "ok": True,
        "enabled": enabled,
        "mode": browser_state["mode"],
        "reason": browser_state["reason"],
    }


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
async def api_voice_chunk(request: Request):
    """Receive raw PCM audio chunk from browser AudioWorklet.
    M-01 FIX: Size limit enforced before reading body."""
    content_length = int(request.headers.get("content-length", 0))
    MAX_VOICE_CHUNK = 512 * 1024  # 512KB max
    if content_length > MAX_VOICE_CHUNK:
        raise HTTPException(status_code=413, detail="Voice chunk too large")
    chunk = await request.body()
    if len(chunk) > MAX_VOICE_CHUNK:
        raise HTTPException(status_code=413, detail="Voice chunk too large")
    voice = _voice_engine_fn() if _voice_engine_fn else None
    if voice and hasattr(voice, "feed_chunk"):
        # Audio on this path exists only because the owner pressed the UI's
        # voice control and is deliberately speaking to her — categorically
        # different from a microphone that happens to be listening. That
        # distinction is what the wake-word boundary was missing: measured live,
        # Whisper transcribed the owner correctly, every utterance was filed as a
        # `transcript_candidate` requiring a wake-word session, and nothing ever
        # answered. She could hear him and would not respond.
        note = getattr(voice, "note_owner_voice_chunk", None)
        if callable(note):
            note()
        await voice.feed_chunk(chunk)
    return JSONResponse({"ok": True})


@router.get("/source")
async def api_source_download(
    _: None = Depends(_require_internal),
):
    """Bundle and return the current source code as a download."""
    PROJECT_ROOT = config.paths.project_root
    try:
        from utils.bundler import write_bundle
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            out = Path(tmp.name)
        write_bundle(PROJECT_ROOT, out, lite=True)
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
async def voice_sse_stream(request: Request):
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
                except asyncio.TimeoutError:
                    yield "data: {\"type\":\"ping\"}\n\n"
        finally:
            if voice and hasattr(voice, "unsubscribe"):
                await voice.unsubscribe(sse_q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
