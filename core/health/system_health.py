from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.runtime.errors import record_degradation
from core.runtime.health_contract import evaluate_health, runtime_health_report
from core.runtime.service_registry import get_runtime_container_health_report, get_runtime_service

router = APIRouter()

_SYSTEM_HEALTH_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _current_state_for_scan() -> object:
    """Return the best available state object for health scans."""
    repo = get_runtime_service("state_repository", default=None)
    state = getattr(repo, "_current", None) if repo is not None else None
    if state is not None:
        return state
    from core.state.aura_state import AuraState

    return AuraState()


@router.get("/")
@router.get("/report")
async def get_full_health_report() -> JSONResponse:
    """Comprehensive health report gated by the canonical runtime contract."""
    try:
        contract = runtime_health_report()
        container_report = get_runtime_container_health_report()
    except _SYSTEM_HEALTH_ERRORS as exc:
        record_degradation("system_health", exc)
        return JSONResponse(
            {
                "status": "dead",
                "healthy": False,
                "operational": False,
                "error": str(exc),
                "required_probes": {"all_passed": False},
                "container": {},
            },
            status_code=503,
        )

    return JSONResponse(
        {
            "status": contract.get("status", "unknown"),
            "healthy": bool(contract.get("healthy", False)),
            "operational": bool(contract.get("operational", False)),
            "required_probes": contract.get("required_probes", {}),
            "contract": contract,
            "container": container_report,
        },
        status_code=int(contract.get("status_code", 503)),
    )


@router.get("/runtime")
@router.get("/contract")
async def get_runtime_health_contract() -> JSONResponse:
    """Canonical runtime contract: what must be alive for Aura to be healthy."""
    verdict = evaluate_health()
    return JSONResponse(verdict.to_report(), status_code=verdict.status_code)


@router.get("/threads")
async def get_thread_summary() -> JSONResponse:
    """Live thread histogram grouped by pool — observability for leak hunts."""
    from core.runtime.thread_inspector import thread_summary

    return JSONResponse(thread_summary())


@router.get("/v2")
async def get_health_v2() -> JSONResponse:
    """Extended system health endpoints via the [ZENITH] Tricorder."""
    contract = runtime_health_report()
    tricorder = get_runtime_service("tricorder", default=None)
    if not tricorder:
        return JSONResponse(
            {
                "status": "error",
                "healthy": False,
                "message": "Tricorder organ not found.",
                "runtime_contract": contract,
            },
            status_code=503,
        )

    # Trigger a real-time scan against the canonical state repository when
    # available, with a fresh AuraState fallback during cold boot/tests.
    state = _current_state_for_scan()
    report = await tricorder.scan(state)

    # Add legacy metadata for compatibility, but do not allow the Tricorder
    # alone to imply system health when required runtime probes are down.
    contract_ok = bool(contract.get("operational", False)) and bool(
        (contract.get("required_probes") or {}).get("all_passed", False)
    )
    report["runtime_contract"] = contract
    report["healthy"] = bool(getattr(tricorder, "healthy", False) and contract_ok)
    report["legacy_status"] = "ok" if report["healthy"] else "degraded"
    status_code = 200 if report["healthy"] else 503
    return JSONResponse(report, status_code=status_code)
