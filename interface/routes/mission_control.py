"""interface/routes/mission_control.py — Mission Control API.

Provides real-time state streams for the Sovereign cockpit GUI.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from core.kernel.leviathan_kernel import get_leviathan_kernel
from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.MissionControl")
router = APIRouter(prefix="/mission_control", tags=["mission_control"])
_MISSION_CONTROL_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@router.get("/status")
async def status(_: None = Depends(_require_internal)) -> JSONResponse:
    """Returns active campaigns, swarm workers, tool logs, and active risks."""
    kernel = get_leviathan_kernel()
    
    # Read statistics from registered organs
    perception = kernel.get_subsystem("perception")
    world_model = kernel.get_subsystem("world_model")
    council = kernel.get_subsystem("council")
    swarm = kernel.get_subsystem("swarm")
    lab = kernel.get_subsystem("lab")
    factory = kernel.get_subsystem("factory")
    simulator = kernel.get_subsystem("simulator")
    memory = kernel.get_subsystem("memory")
    forge = kernel.get_subsystem("forge")
    auditor = kernel.get_subsystem("auditor")
    cloud_body = kernel.get_subsystem("cloud_body")
    kernel_health = kernel.health_status()

    status_payload = {
        "timestamp": time.time(),
        "kernel_online": kernel_health["healthy"],
        "kernel_health": kernel_health,
        "active_missions": kernel.active_missions,
        "perception_scans": getattr(perception, "last_scan_count", 0) if perception else 0,
        "claims_tracked": len(world_model.graph.nodes) if (world_model and hasattr(world_model, "graph")) else 0,
        "council_debates": len(getattr(council, "debate_history", [])) if council else 0,
        "swarm_workers": len(getattr(swarm, "active_workers", {})) if swarm else 0,
        "research_memos": len(getattr(lab, "memos", {})) if lab else 0,
        "factory_patches": len(getattr(factory, "history", [])) if factory else 0,
        "simulations_run": getattr(simulator, "runs", 0) if simulator else 0,
        "memories_committed": len(getattr(memory, "memories", {})) if memory else 0,
        "weaknesses_identified": len(getattr(forge, "weaknesses", [])) if forge else 0,
        "cloud_compute_cost": getattr(cloud_body, "current_cost", 0.0) if cloud_body else 0.0,
        "audit_failures": getattr(auditor, "failures_detected", 0) if auditor else 0,
    }
    
    return JSONResponse(status_payload)


@router.post("/run_mission")
async def run_mission(payload: Dict[str, Any], _: None = Depends(_require_internal)) -> JSONResponse:
    """Invokes a new long-horizon campaign through the Leviathan Kernel."""
    objective = payload.get("objective")
    if not objective:
        raise HTTPException(status_code=400, detail="Missing objective parameter")

    kernel = get_leviathan_kernel()
    kernel.active_missions.append(objective)

    try:
        # Run background task or synchronous wait depending on request
        # For control panel requests, we execute and return the consensus result
        result = await kernel.execute_mission(objective, constraints=payload.get("constraints"))
        return JSONResponse({"ok": True, "result": result})
    except _MISSION_CONTROL_RECOVERABLE_ERRORS as e:
        logger.error("Error running mission via cockpit API: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
    finally:
        if objective in kernel.active_missions:
            kernel.active_missions.remove(objective)
