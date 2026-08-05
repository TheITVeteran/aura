"""Authenticated operator surface for Reality Reach acceptance evidence."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.container import ServiceContainer
from core.reality_reach.acceptance import AcceptanceError, AcceptanceEvidenceClass
from core.reality_reach.acceptance_service import ScalarAcceptanceRequest
from interface.auth import _require_internal, _verify_token

router = APIRouter(prefix="/reality-reach/acceptance")


class ScalarAcceptancePayload(BaseModel):
    campaign_id: Annotated[str, Field(min_length=1, max_length=128)]
    connector_id: Annotated[str, Field(min_length=1, max_length=128)]
    adapter_id: Annotated[str, Field(min_length=1, max_length=256)]
    target: float
    expected_source_commit_sha256: Annotated[
        str,
        Field(pattern=r"^sha256:[0-9a-f]{64}$"),
    ]
    evidence_class: AcceptanceEvidenceClass
    scenario_id: Annotated[str, Field(max_length=128)] = ""
    simulated_channel_ids: tuple[
        Annotated[str, Field(min_length=1, max_length=256)], ...
    ] = ()
    deadline_s: Annotated[float, Field(ge=0.5, le=60.0)] = 5.0
    sample_interval_s: Annotated[float, Field(ge=0.01, le=0.5)] = 0.1
    effect_hold_s: Annotated[float, Field(ge=0.05, le=5.0)] = 0.25


def _service() -> Any:
    service = ServiceContainer.get("reality_acceptance", default=None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="reality_acceptance_service_unavailable",
        )
    return service


@router.get("/status")
async def acceptance_status(
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token),
) -> JSONResponse:
    return JSONResponse(_service().status())


@router.post("/run")
async def run_acceptance(
    payload: ScalarAcceptancePayload,
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token),
) -> JSONResponse:
    try:
        result = await _service().run(ScalarAcceptanceRequest(**payload.model_dump()))
    except AcceptanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(result, status_code=201)


__all__ = ["ScalarAcceptancePayload", "acceptance_status", "router", "run_acceptance"]
