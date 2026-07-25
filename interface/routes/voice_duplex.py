"""interface/routes/voice_duplex.py — the /ws/voice transport.

A dedicated socket rather than a mode on ``/ws``. Three reasons, all
practical: voice pushes continuous binary in both directions and would
starve the chat socket's event queue; a voice session owns models and tasks
whose lifetime must be exactly the socket's; and a failure in the voice lane
must never be able to drop the text conversation.

Authentication mirrors ``/ws`` exactly — same local-origin trust, same paired
-device token flow — with one addition: a paired device also needs the
explicit ``voice`` scope, because a microphone is a materially larger grant
than a chat box.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.runtime.errors import record_degradation
from core.voice.duplex.config import DuplexConfig
from core.voice.duplex.session import DuplexVoiceSession

logger = logging.getLogger("Aura.Routes.VoiceDuplex")

router = APIRouter()

_VOICE_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

# One live session per socket. Tracked so the runtime can report and close
# them, and so a reload cannot strand model handles.
_SESSIONS: dict[str, DuplexVoiceSession] = {}


def active_sessions() -> dict[str, dict[str, Any]]:
    """Status of every live voice session, for health surfaces."""
    return {sid: session.status() for sid, session in _SESSIONS.items()}


def _voice_output_permitted() -> bool:
    """Respect the user's runtime toggles.

    Voice is the most intrusive output channel Aura has — it makes noise in
    a room. If the user turned it off, the lane refuses to open rather than
    opening silently and surprising them later.
    """
    try:
        from core.runtime.runtime_settings import get_runtime_setting

        return bool(get_runtime_setting("voice.output_enabled", True))
    except _VOICE_ERRORS as exc:
        record_degradation(
            "voice_duplex.route",
            exc,
            action="assumed voice output permitted; runtime settings unreadable",
            severity="warning",
        )
        return True


def _voice_input_permitted() -> bool:
    try:
        from core.runtime.runtime_settings import get_runtime_setting

        return bool(get_runtime_setting("voice.input_enabled", True))
    except _VOICE_ERRORS as exc:
        record_degradation(
            "voice_duplex.route",
            exc,
            action="assumed voice input permitted; runtime settings unreadable",
            severity="warning",
        )
        return True


@router.websocket("/ws/voice")
async def voice_duplex_endpoint(ws: WebSocket) -> None:
    # Import here rather than at module scope: interface.server imports this
    # module, so a top-level import would be circular.
    from interface.auth import device_for_request, request_has_allowed_local_browser_origin
    from interface.server import _live_device_scopes, _verify_ws_device_token, config

    await ws.accept()

    host = ws.client.host if ws.client else "unknown"
    is_local = host in ("127.0.0.1", "::1", "localhost")
    local_origin_ok = request_has_allowed_local_browser_origin(ws)
    expected = str(getattr(config, "api_token", "") or "")

    authenticated = is_local and local_origin_ok
    device_session = None
    explicit_token: str | None = None

    if not authenticated:
        device_session = device_for_request(ws)
        authenticated = device_session is not None

    try:
        if not authenticated:
            # Same 5-second credential exchange the chat socket uses.
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
                data = json.loads(raw)
            except TimeoutError:
                await ws.close(code=4001, reason="Auth Timeout")
                return
            except json.JSONDecodeError:
                await ws.close(code=4001, reason="Invalid Auth Payload")
                return

            token = str(data.get("token", "") or "")
            if data.get("type") == "auth" and token.startswith("adt1."):
                device_session = _verify_ws_device_token(token)
                if device_session is not None:
                    explicit_token = token
                    authenticated = True
            elif data.get("type") == "auth" and expected:
                import hmac

                authenticated = hmac.compare_digest(token, expected)

            if not authenticated:
                await ws.send_text(json.dumps({"type": "voice.error", "message": "Unauthorized"}))
                await ws.close(code=4001, reason="Unauthorized")
                return

        # A microphone is a bigger grant than a chat box, so a paired device
        # needs the voice scope explicitly. Deny by default.
        if device_session is not None:
            if "voice" not in _live_device_scopes(device_session.device_id):
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "voice.error",
                            "status": "paired_device_voice_scope_denied",
                            "message": (
                                "Voice is not enabled for this paired device. Ask the "
                                "owner to grant the voice scope."
                            ),
                        }
                    )
                )
                await ws.close(code=4003, reason="Voice scope required")
                return

        if not _voice_input_permitted() or not _voice_output_permitted():
            await ws.send_text(
                json.dumps(
                    {
                        "type": "voice.error",
                        "status": "voice_disabled_in_settings",
                        "message": "Voice is turned off in Runtime Settings.",
                    }
                )
            )
            await ws.close(code=4004, reason="Voice disabled")
            return

        session_id = uuid.uuid4().hex[:12]
        send_lock = asyncio.Lock()

        async def send_json(payload: dict[str, Any]) -> None:
            # Serialise sends: the session emits from several tasks at once
            # (reflex loop, turn loop, filler loop) and Starlette's socket
            # is not safe against interleaved concurrent writes.
            async with send_lock:
                await ws.send_text(json.dumps(payload, default=str))

        async def send_binary(payload: bytes) -> None:
            async with send_lock:
                await ws.send_bytes(payload)

        session = DuplexVoiceSession(
            session_id=session_id,
            send_json=send_json,
            send_binary=send_binary,
            config=DuplexConfig.load(),
        )
        _SESSIONS[session_id] = session

        try:
            await session.start()
            logger.info("Voice session %s opened from %s", session_id, host)

            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                # Re-check a paired device's authorisation every frame, so a
                # revoked grant closes the microphone immediately rather than
                # at the end of an open-ended streaming session.
                if device_session is not None:
                    refreshed = (
                        _verify_ws_device_token(explicit_token)
                        if explicit_token
                        else device_for_request(ws)
                    )
                    if (
                        refreshed is None
                        or refreshed.device_id != device_session.device_id
                        or "voice" not in _live_device_scopes(refreshed.device_id)
                    ):
                        await send_json(
                            {
                                "type": "voice.error",
                                "status": "paired_device_session_revoked",
                                "message": "This device's voice authorization was revoked.",
                            }
                        )
                        await ws.close(code=4003, reason="Voice authorization revoked")
                        break
                    device_session = refreshed

                if "bytes" in message and message["bytes"]:
                    await session.feed_audio(message["bytes"])
                elif "text" in message and message["text"]:
                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        await session.handle_command(payload)

        finally:
            _SESSIONS.pop(session_id, None)
            with contextlib.suppress(*_VOICE_ERRORS, asyncio.CancelledError):
                await session.close()
            logger.info("Voice session %s closed", session_id)

    except WebSocketDisconnect as exc:
        logger.debug("Voice socket disconnected: %s", exc)
    except _VOICE_ERRORS as exc:
        record_degradation(
            "voice_duplex.route",
            exc,
            action="closed the voice socket after a transport error",
        )
        with contextlib.suppress(RuntimeError):
            await ws.close(code=1011, reason="Voice lane error")
