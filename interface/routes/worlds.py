"""interface/routes/worlds.py
──────────────────────────
HTTP surface for Aura's persistent worlds.

Read endpoints (list / inspect / render) are available to the owner AND
to paired conversation devices — watching Aura's worlds from the phone
is part of the conversation surface. Mutating endpoints (step, impulse)
are owner-only: devices observe, they do not rewrite her worlds.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from core.worlds import PhysicsError, get_world_host
from interface.routes.devices import _owner_authenticated

logger = logging.getLogger("Aura.Server.Worlds")

router = APIRouter()


def _require_owner(request: Request) -> None:
    if not _owner_authenticated(request):
        raise HTTPException(status_code=403, detail="World mutation is owner-only")


@router.get("/worlds")
async def list_worlds() -> dict[str, Any]:
    return {"ok": True, "worlds": get_world_host().list_worlds()}


@router.get("/worlds/{world_id}/render")
async def render_world(world_id: str) -> dict[str, Any]:
    try:
        return {"ok": True, **get_world_host().render_state(world_id)}
    except PhysicsError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/worlds/{world_id}")
async def inspect_world(world_id: str) -> dict[str, Any]:
    try:
        return {"ok": True, **get_world_host().inspect(world_id)}
    except PhysicsError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/worlds/{world_id}/step")
async def step_world(world_id: str, request: Request) -> dict[str, Any]:
    _require_owner(request)
    payload = {}
    try:
        payload = await request.json()
    except ValueError:
        payload = {}
    ticks = int(payload.get("ticks", 120) or 120)
    try:
        summary = await get_world_host().step_world(world_id, ticks)
    except PhysicsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "world": summary}
