"""interface/routes/devices.py
────────────────────────────
LAN device pairing — the HTTP surface for core/security/device_pairing.py.

Flow:
1. Owner (desktop, localhost) POSTs /api/devices/pair/begin → 8-digit code.
2. Phone opens http://<lan-ip>:8000/pair, enters the code, POSTs
   /api/devices/pair/complete → receives its device token once and a
   session cookie, then lands on the normal chat UI.
3. Owner can list and revoke devices at any time.

Begin/list/revoke are owner-only. Complete is reachable unauthenticated
by design (it IS the authentication ceremony) but is rate-limited and
bounded by the pairing code's TTL and attempt budget.
"""
from __future__ import annotations

import hmac
import logging
import socket
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.config import config
from core.runtime.errors import record_degradation
from core.security.device_pairing import (
    PairingDisabledError,
    PairingError,
    get_device_registry,
)
from interface.auth import (
    DEVICE_SESSION_COOKIE_NAME,
    DEVICE_SESSION_COOKIE_TTL_SECS,
    _allow_local_without_token,
    _check_rate_limit,
    _extract_request_token,
    local_owner_principal_id,
)

logger = logging.getLogger("Aura.Server.Devices")

router = APIRouter()

_LAN_PROBE_ERRORS = (OSError, socket.gaierror, ValueError)


def _owner_authenticated(request: Request) -> bool:
    """Pairing administration requires an owner-present surface: the
    desktop UI on localhost, or the master API token."""
    expected = config.api_token
    supplied = _extract_request_token(request)
    if expected and supplied and hmac.compare_digest(supplied, expected):
        return True
    return _allow_local_without_token(request, protected_route=True)


def _require_owner(request: Request) -> None:
    if not _owner_authenticated(request):
        raise HTTPException(status_code=403, detail="Device administration is owner-only")


def _lan_addresses() -> list[str]:
    """Best-effort local addresses for the pairing hint. No packets leave
    the host — a UDP connect() only selects a route."""
    addresses: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1: never actually sent
            primary = probe.getsockname()[0]
            if primary and not primary.startswith("127."):
                addresses.append(primary)
    except _LAN_PROBE_ERRORS as exc:
        record_degradation("devices.lan_probe", exc)
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidate = info[4][0]
            if candidate not in addresses and not candidate.startswith("127."):
                addresses.append(candidate)
    except _LAN_PROBE_ERRORS as exc:
        record_degradation("devices.lan_probe", exc)
    return addresses


class PairCompleteRequest(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    device_name: str = Field(default="device", max_length=64)


@router.post("/devices/pair/begin")
async def pair_begin(request: Request) -> dict[str, Any]:
    _require_owner(request)
    try:
        principal_id = local_owner_principal_id()
        if not principal_id:
            raise HTTPException(
                status_code=409,
                detail="Primary operator identity is unavailable; pairing cannot bind a principal",
            )
        challenge = get_device_registry().begin_pairing(principal_id)
    except PairingDisabledError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    host_header = str(request.headers.get("host", "") or "")
    port = host_header.rsplit(":", 1)[1] if ":" in host_header else "8000"
    if not port.isdigit():
        port = "8000"
    scheme = str(getattr(request.url, "scheme", "http") or "http")
    return {
        "ok": True,
        **challenge,
        "pair_urls": [
            f"{scheme}://{addr}:{port}/pair" for addr in _lan_addresses()
        ],
    }


@router.post("/devices/pair/cancel")
async def pair_cancel(request: Request) -> dict[str, Any]:
    _require_owner(request)
    get_device_registry().cancel_pairing()
    return {"ok": True}


@router.post("/devices/pair/complete")
async def pair_complete(body: PairCompleteRequest, request: Request) -> JSONResponse:
    _check_rate_limit(request)
    try:
        issued = await get_device_registry().complete_pairing(body.code, body.device_name)
    except PairingDisabledError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except PairingError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    logger.info("Paired new device %s (%s)", issued["device_id"], issued["name"])
    response = JSONResponse({"ok": True, **issued})
    response.set_cookie(
        DEVICE_SESSION_COOKIE_NAME,
        issued["token"],
        max_age=DEVICE_SESSION_COOKIE_TTL_SECS,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


@router.get("/devices")
async def list_devices(request: Request) -> dict[str, Any]:
    _require_owner(request)
    registry = get_device_registry()
    await registry.flush_last_seen()
    return {"ok": True, "devices": registry.list_devices()}


@router.post("/devices/grant-scope")
async def grant_device_scope(request: Request) -> dict[str, Any]:
    """Owner-only scope widening (e.g. voice for a trusted phone).
    Deny-by-default stands: nothing is granted at pairing time."""
    _require_owner(request)
    payload = await request.json()
    device_id = str(payload.get("device_id", "") or "")
    scope = str(payload.get("scope", "") or "")
    try:
        granted = await get_device_registry().grant_scope(device_id, scope)
    except PairingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not granted:
        raise HTTPException(status_code=404, detail="Unknown or revoked device")
    logger.info("Granted scope '%s' to device %s", scope, device_id)
    return {"ok": True, "device_id": device_id, "scope": scope}


@router.post("/devices/revoke-scope")
async def revoke_device_scope(request: Request) -> dict[str, Any]:
    _require_owner(request)
    payload = await request.json()
    device_id = str(payload.get("device_id", "") or "")
    scope = str(payload.get("scope", "") or "")
    revoked = await get_device_registry().revoke_scope(device_id, scope)
    if not revoked:
        raise HTTPException(status_code=404, detail="Unknown device")
    return {"ok": True, "device_id": device_id, "scope": scope}


@router.post("/devices/revoke")
async def revoke_device(request: Request) -> dict[str, Any]:
    _require_owner(request)
    payload = await request.json()
    device_id = str(payload.get("device_id", "") or "")
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id required")
    revoked = await get_device_registry().revoke_device(device_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="Unknown device")
    return {"ok": True, "device_id": device_id}
