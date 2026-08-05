"""Authenticated operator surface for Reality Reach acceptance evidence."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.container import ServiceContainer
from core.reality_reach.acceptance import AcceptanceError, AcceptanceEvidenceClass
from core.reality_reach.acceptance_service import (
    ScalarAcceptanceMandateRequest,
    ScalarAcceptanceRequest,
)
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
    mandate_sha256: Annotated[
        str,
        Field(pattern=r"^(?:|sha256:[0-9a-f]{64})$"),
    ] = ""
    scenario_id: Annotated[str, Field(max_length=128)] = ""
    simulated_channel_ids: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=256)], ...],
        Field(max_length=63),
    ] = ()
    deadline_s: Annotated[float, Field(ge=0.5, le=60.0)] = 5.0
    sample_interval_s: Annotated[float, Field(ge=0.01, le=0.5)] = 0.1
    effect_hold_s: Annotated[float, Field(ge=0.05, le=5.0)] = 0.25


class ScalarAcceptancePreflightPayload(BaseModel):
    adapter_id: Annotated[str, Field(min_length=1, max_length=256)]


class ScalarAcceptanceMandatePayload(BaseModel):
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
    simulated_channel_ids: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=256)], ...],
        Field(max_length=63),
    ] = ()


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


@router.post("/preflight")
async def acceptance_preflight(
    payload: ScalarAcceptancePreflightPayload,
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token),
) -> JSONResponse:
    try:
        result = await _service().preflight(payload.adapter_id)
    except AcceptanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(result)


@router.post("/acoustic/provision")
async def provision_acoustic_acceptance_adapter(
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token),
) -> JSONResponse:
    try:
        result = await _service().provision_macos_acoustic_adapter()
    except AcceptanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(result, status_code=201)


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


@router.post("/mandate")
async def precommit_acceptance_mandate(
    payload: ScalarAcceptanceMandatePayload,
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token),
) -> JSONResponse:
    try:
        result = await _service().precommit(
            ScalarAcceptanceMandateRequest(**payload.model_dump())
        )
    except AcceptanceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(result, status_code=201)


__all__ = [
    "ScalarAcceptanceMandatePayload",
    "ScalarAcceptancePayload",
    "ScalarAcceptancePreflightPayload",
    "acceptance_preflight",
    "acceptance_status",
    "precommit_acceptance_mandate",
    "provision_acoustic_acceptance_adapter",
    "router",
    "run_acceptance",
]
