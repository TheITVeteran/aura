"""interface/routes/allostasis.py — the predictive-interoception receipt surface.

Read-only windows onto the allostasis engine (core/autonomic/allostasis.py):

  GET /api/allostasis            → tier, narrative, vitals, felt contribution,
                                   open forecasts, allostatic load, calibration
  GET /api/allostasis/forecasts  → the falsifiable ledger view: open predictions
                                   with deadlines, recently resolved with outcomes

This surface exists so the forecasts are publicly checkable receipts: every
prediction Aura makes about her own body carries a deadline and is scored when
that deadline passes. No mutation endpoints — regulation decisions belong to
the engine's governed policy, not to HTTP callers.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Routes.Allostasis")

router = APIRouter(prefix="/allostasis", tags=["allostasis"])

_ROUTE_ERRORS = (
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


def _engine() -> Any:
    return ServiceContainer.get("allostasis_engine", default=None)


@router.get("")
async def allostasis_status() -> JSONResponse:
    """Full engine status: tier, narrative, vitals, forecasts, calibration."""
    engine = _engine()
    if engine is None:
        return JSONResponse(
            {"available": False, "reason": "allostasis engine not registered"},
            status_code=503,
        )
    try:
        payload = engine.status()
        return JSONResponse({"available": True, **payload})
    except _ROUTE_ERRORS as exc:
        logger.warning("Allostasis status unavailable: %s", exc)
        return JSONResponse(
            {"available": False, "reason": f"{type(exc).__name__}: {exc}"},
            status_code=500,
        )


@router.get("/forecasts")
async def allostasis_forecasts() -> JSONResponse:
    """The falsifiable ledger: open predictions and recent outcomes."""
    engine = _engine()
    if engine is None:
        return JSONResponse(
            {"available": False, "reason": "allostasis engine not registered"},
            status_code=503,
        )
    try:
        status = engine.status()
        return JSONResponse(
            {
                "available": True,
                "narrative": status.get("narrative", ""),
                "tier": status.get("tier", ""),
                "open": status.get("open_forecasts", []),
                "recently_resolved": status.get("recently_resolved", []),
                "calibration": status.get("calibration", {}),
            }
        )
    except _ROUTE_ERRORS as exc:
        logger.warning("Allostasis forecasts unavailable: %s", exc)
        return JSONResponse(
            {"available": False, "reason": f"{type(exc).__name__}: {exc}"},
            status_code=500,
        )
