"""interface/routes/ambient.py
────────────────────────────
The bubble's only server surface: read her state, dismiss a message, tell
her where she was parked, and ask her what she already saw.

Everything that DECIDES is behind these — whether she observes, whether she
speaks, and whether a window is private. This module reads and forwards; it
never grants. A route that could make her speak would be a second authority
for unprompted speech, and there is exactly one.

The recall endpoint is the latency payoff and the reason the loop exists: a
question about the screen is answered from what she saw a moment ago instead
of from a capture that starts when the question arrives.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.runtime.errors import record_degradation
from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Ambient")

router = APIRouter()

_AMBIENT_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError)


class BubblePosition(BaseModel):
    x: float = Field(default=0.0)
    y: float = Field(default=0.0)


class RecallRequest(BaseModel):
    question: str = Field(default="", max_length=2000)


def _presence() -> Any:
    from core.perception.ambient_presence import get_ambient_presence

    return get_ambient_presence()


@router.get("/api/ambient/state")
async def ambient_state(surface: str = "") -> JSONResponse:
    """What the surface should show. Safe to poll.

    ``surface`` names the caller — the bubble passes ``surface=bubble``. It
    does two things, and both are about the overlay rather than the message:
    it marks the bubble as a live host that can draw, and it collects a queued
    highlight exactly once. The restrained chat window reads the same endpoint
    without a surface and so never takes a rectangle it cannot draw.
    """
    try:
        return JSONResponse(_presence().state(surface=surface))
    except _AMBIENT_ERRORS as exc:
        record_degradation(
            "ambient_routes", exc, severity="warning",
            action="bubble state unavailable; the surface keeps its last render",
        )
        # 200 with an explicit unavailable flag, not 500: the bubble treats a
        # failed poll as "keep showing what she last said", and a 500 here
        # would be indistinguishable from her having withdrawn it.
        return JSONResponse(
            {
                "mode": "unknown",
                "has_utterance": False,
                "utterance": "",
                "available": False,
                "reason": type(exc).__name__,
            }
        )


@router.post("/api/ambient/clear", dependencies=[Depends(_require_internal)])
async def ambient_clear() -> JSONResponse:
    """The person dismissed the message. It does not come back."""
    try:
        _presence().clear_utterance()
        return JSONResponse({"ok": True})
    except _AMBIENT_ERRORS as exc:
        record_degradation("ambient_routes", exc, severity="warning",
                           action="dismissal not applied")
        return JSONResponse({"ok": False, "error": type(exc).__name__}, status_code=503)


@router.post("/api/ambient/position", dependencies=[Depends(_require_internal)])
async def ambient_position(payload: BubblePosition) -> JSONResponse:
    """Where she was parked, so it survives a restart."""
    try:
        x, y = _presence().move_bubble(payload.x, payload.y)
        return JSONResponse({"ok": True, "position": [x, y]})
    except _AMBIENT_ERRORS as exc:
        record_degradation("ambient_routes", exc, severity="debug",
                           action="bubble position not retained")
        return JSONResponse({"ok": False}, status_code=503)


@router.post("/api/ambient/visibility", dependencies=[Depends(_require_internal)])
async def ambient_visibility(payload: dict) -> JSONResponse:
    """Hide or show her. Hiding stops observation, not just display.

    This is the one control that must not be cosmetic: a "hidden" companion
    that keeps reading the screen is worse than a visible one.
    """
    from core.perception.ambient_presence import PresenceMode

    wanted = str((payload or {}).get("mode") or "").strip().lower()
    if wanted not in {mode.value for mode in PresenceMode}:
        return JSONResponse(
            {"ok": False, "error": f"unknown mode {wanted!r}"}, status_code=400
        )
    try:
        applied = _presence().set_mode(wanted)
        return JSONResponse({"ok": True, "mode": applied.value})
    except _AMBIENT_ERRORS as exc:
        record_degradation("ambient_routes", exc, severity="warning",
                           action="visibility change not applied")
        return JSONResponse({"ok": False}, status_code=503)


@router.post("/api/ambient/recall", dependencies=[Depends(_require_internal)])
async def ambient_recall(payload: RecallRequest) -> JSONResponse:
    """What she already saw that bears on this question.

    Returns ``fresh: false`` with an empty body when she has not looked —
    which the caller must read as "capture now", never as "there is nothing
    on the screen". Those are different answers and only one of them is a
    fact about the world.
    """
    question = str(payload.question or "").strip()
    if not question:
        return JSONResponse({"ok": False, "error": "question required"}, status_code=400)
    try:
        recalled = _presence().recall_for(question)
    except _AMBIENT_ERRORS as exc:
        record_degradation("ambient_routes", exc, severity="warning",
                           action="ambient recall failed; caller must capture fresh")
        recalled = ""
    return JSONResponse(
        {
            "ok": True,
            "fresh": bool(recalled),
            "observation": recalled,
            "note": (
                ""
                if recalled
                else "she has not looked recently; capture before answering"
            ),
        }
    )
