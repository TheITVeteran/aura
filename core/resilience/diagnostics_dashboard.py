"""core/resilience/diagnostics_dashboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runtime diagnostics endpoint for reliability-grade observability.

Mounts at /api/diagnostics/reliability and exposes a unified view of all
reliability hardening subsystems:
- Fault taxonomy with live RPN scores
- SLO burn rates and error budgets
- State machine current states
- Circuit breaker states
- TMR divergence log
- Contract violation counts
- Chaos experiment results
- Watchdog heartbeat status
- Tracing statistics

Usage:
    # In FastAPI app setup:
    from core.resilience.diagnostics_dashboard import create_diagnostics_router
    app.include_router(create_diagnostics_router(), prefix="/api/diagnostics")
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("Aura.Diagnostics.Reliability")


def collect_reliability_diagnostics() -> dict[str, Any]:
    """Collect diagnostics from all reliability hardening subsystems.

    Returns a single dict suitable for JSON serialization and API response.
    """
    diagnostics: dict[str, Any] = {
        "timestamp": time.time(),
        "version": "1.0.0",
        "subsystems": {},
    }

    # 1. Fault Taxonomy
    try:
        from core.resilience.fault_taxonomy import get_fault_registry
        registry = get_fault_registry()
        diagnostics["subsystems"]["fault_taxonomy"] = {
            "status": registry.status(),
            "rpn_report": registry.rpn_report()[:10],  # Top 10 by risk
        }
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
        diagnostics["subsystems"]["fault_taxonomy"] = {"error": str(exc)}

    # 2. FMEA Registry
    try:
        from core.resilience.fmea_registry import get_fmea_registry
        fmea = get_fmea_registry()
        diagnostics["subsystems"]["fmea"] = {
            "coverage": fmea.coverage_summary(),
            "unmitigated": fmea.unmitigated_faults(),
            "high_risk": fmea.faults_above_rpn(30),
        }
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
        diagnostics["subsystems"]["fmea"] = {"error": str(exc)}

    # 3. SLO Monitor
    try:
        from slo.slo_monitor import get_slo_monitor
        monitor = get_slo_monitor()
        diagnostics["subsystems"]["slo_monitor"] = monitor.status()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
        diagnostics["subsystems"]["slo_monitor"] = {"error": str(exc)}

    # 4. Contract Tracker
    try:
        from core.resilience.contracts import get_contract_tracker
        tracker = get_contract_tracker()
        diagnostics["subsystems"]["contracts"] = tracker.status()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
        diagnostics["subsystems"]["contracts"] = {"error": str(exc)}

    # 5. Tracing
    try:
        from core.observability.tracing import get_tracer
        tracer = get_tracer()
        diagnostics["subsystems"]["tracing"] = tracer.status()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
        diagnostics["subsystems"]["tracing"] = {"error": str(exc)}

    # 6. Degradation Tracker (existing)
    try:
        from core.runtime.errors import get_degradation_tracker
        tracker = get_degradation_tracker()
        diagnostics["subsystems"]["degradation"] = tracker.status()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
        diagnostics["subsystems"]["degradation"] = {"error": str(exc)}

    # 7. Empirical fault evidence + probability drift (FMEA that learns)
    try:
        from core.resilience.fault_evidence import get_fault_evidence_store
        from core.resilience.fault_taxonomy import get_fault_registry
        store = get_fault_evidence_store()
        drift = store.drift_report(get_fault_registry().all_definitions())
        diagnostics["subsystems"]["fault_evidence"] = {
            "status": store.status(),
            "probability_drift": [f.to_dict() for f in drift[:10]],
            "drift_count": len(drift),
        }
        # Diagnostics access is the designated off-hot-path flush window.
        store.flush()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
        diagnostics["subsystems"]["fault_evidence"] = {"error": str(exc)}

    # 8. Verified shutdown lifecycle
    try:
        from core.runtime.shutdown_coordinator import get_shutdown_coordinator
        diagnostics["subsystems"]["lifecycle"] = get_shutdown_coordinator().get_status()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
        diagnostics["subsystems"]["lifecycle"] = {"error": str(exc)}

    # 9. Canonical lifecycle/resource control plane
    try:
        from core.runtime.operator_control_plane import (
            collect_runtime_control_plane_status,
        )

        diagnostics["subsystems"]["runtime_control_plane"] = (
            collect_runtime_control_plane_status()
        )
    except (
        ImportError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:
        diagnostics["subsystems"]["runtime_control_plane"] = {"error": str(exc)}

    # Summary
    subsystems = diagnostics["subsystems"]
    errors = sum(1 for v in subsystems.values() if isinstance(v, dict) and "error" in v)
    diagnostics["summary"] = {
        "total_subsystems": len(subsystems),
        "healthy_subsystems": len(subsystems) - errors,
        "errored_subsystems": errors,
        "overall_status": "DEGRADED" if errors > 0 else "HEALTHY",
    }

    return diagnostics


def create_diagnostics_router() -> Any:
    """Create a FastAPI router for the reliability diagnostics endpoint.

    Returns a FastAPI APIRouter. Import is deferred to avoid hard
    dependency on FastAPI at module level.
    """
    import asyncio

    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router = APIRouter(tags=["reliability-diagnostics"])

    # Every collector runs in a worker thread: they take subsystem locks and
    # walk bounded histories, and the live event loop must never inherit a
    # subsystem's lock contention (or worse) as loop stall time.

    @router.get("/reliability", response_class=JSONResponse, response_model=None)
    async def reliability_diagnostics() -> Any:
        """Full reliability diagnostics report."""
        payload = await asyncio.to_thread(collect_reliability_diagnostics)
        return JSONResponse(content=payload)

    @router.get("/reliability/faults", response_class=JSONResponse, response_model=None)
    async def fault_report() -> Any:
        """Fault taxonomy report."""
        try:
            from core.resilience.fault_taxonomy import get_fault_registry
            payload = await asyncio.to_thread(get_fault_registry().status)
            return JSONResponse(content=payload)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @router.get("/reliability/slos", response_class=JSONResponse, response_model=None)
    async def slo_report() -> Any:
        """SLO burn rates and budget status."""
        try:
            from slo.slo_monitor import get_slo_monitor
            payload = await asyncio.to_thread(get_slo_monitor().status)
            return JSONResponse(content=payload)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @router.get("/reliability/fmea", response_class=JSONResponse, response_model=None)
    async def fmea_report() -> Any:
        """FMEA coverage and risk report."""
        try:
            from core.resilience.fmea_registry import get_fmea_registry

            def _collect() -> dict[str, Any]:
                fmea = get_fmea_registry()
                return {
                    "coverage": fmea.coverage_summary(),
                    "report": fmea.full_report(),
                }

            payload = await asyncio.to_thread(_collect)
            return JSONResponse(content=payload)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @router.get("/reliability/contracts", response_class=JSONResponse, response_model=None)
    async def contract_report() -> Any:
        """Design-by-contract violation report."""
        try:
            from core.resilience.contracts import get_contract_tracker
            payload = await asyncio.to_thread(get_contract_tracker().status)
            return JSONResponse(content=payload)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @router.get("/reliability/drift", response_class=JSONResponse, response_model=None)
    async def drift_report() -> Any:
        """Empirical probability drift: static FMEA bands vs observed rates."""
        try:
            from core.resilience.fault_evidence import get_fault_evidence_store
            from core.resilience.fault_taxonomy import get_fault_registry

            def _collect() -> dict[str, Any]:
                store = get_fault_evidence_store()
                drift = store.drift_report(get_fault_registry().all_definitions())
                store.flush()
                return {
                    "evidence": store.status(),
                    "probability_drift": [f.to_dict() for f in drift],
                }

            payload = await asyncio.to_thread(_collect)
            return JSONResponse(content=payload)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @router.get("/reliability/traces", response_class=JSONResponse, response_model=None)
    async def trace_report() -> Any:
        """Tracing statistics and recent spans."""
        try:
            from core.observability.tracing import get_tracer

            def _collect() -> dict[str, Any]:
                tracer = get_tracer()
                status = tracer.status()
                status["recent_spans"] = [
                    s.to_otlp_dict() for s in tracer.recent_spans(10)
                ]
                return status

            payload = await asyncio.to_thread(_collect)
            return JSONResponse(content=payload)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError, OSError) as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @router.get(
        "/reliability/control-plane",
        response_class=JSONResponse,
        response_model=None,
    )
    async def control_plane_report() -> Any:
        """Desired state, resource leases, conditions, circuits, and blockers."""
        try:
            from core.runtime.operator_control_plane import (
                collect_runtime_control_plane_status,
            )

            payload = await asyncio.to_thread(collect_runtime_control_plane_status)
            return JSONResponse(content=payload)
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            KeyError,
            OSError,
        ) as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    return router
