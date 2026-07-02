#!/usr/bin/env python3
"""Generate a live background-autonomy proof artifact.

This proof is intentionally narrower than ``tools/live_boot_proof.py``. It
boots the same Aura runtime, verifies the full desktop/background organs through
the runtime health contract, samples the autonomy conductor long enough for
immediate jobs to run, and shuts down without forcing a foreground Cortex chat.

The output directory is the live artifact required by the remaining closeout
contract:

    artifacts/current/background_autonomy/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

from tools.live_boot_proof import LiveProof  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "artifacts" / "current" / "background_autonomy"

REQUIRED_COMPONENTS = (
    "pneuma",
    "mhaf",
    "curiosity",
    "proactive_communication",
    "autonomous_initiative",
    "subjective_choice",
    "ambient_life_director",
    "research",
    "self_healing",
    "self_modification",
    "consciousness_stream",
    "autonomy_conductor",
    "overt_action",
    "deliberation",
    "wake_word",
    "screen_perception",
    "perceptual_pump",
    "cognitive_situation",
    "imagination_engine",
    "timescale_bridge",
    "ambient_developer_stream",
    "autonomic_reflection_loop",
)

DISALLOWED_DEFER_REASONS = {
    "background_cognition_disabled",
    "foreground_only_runtime",
    "proof_run_active",
    "shutdown_requested",
}


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _get_json(base: str, path: str, *, timeout: float = 8.0) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        response = client.get(f"{base}{path}")
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"value": payload}


def _component_running(status: Any) -> bool:
    if not isinstance(status, dict):
        return False
    if "running" in status:
        return bool(status.get("running"))
    if "online" in status:
        return bool(status.get("online"))
    if "active" in status:
        return bool(status.get("active"))
    return False


def evaluate_background_autonomy(health: dict[str, Any]) -> dict[str, Any]:
    full_runtime = health.get("full_runtime")
    full_runtime = full_runtime if isinstance(full_runtime, dict) else {}
    background = full_runtime.get("background_cognition")
    background = background if isinstance(background, dict) else {}
    components = full_runtime.get("components")
    components = components if isinstance(components, dict) else {}

    component_status: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name in REQUIRED_COMPONENTS:
        status = components.get(name)
        running = _component_running(status)
        component_status[name] = {
            "running": running,
            "status": status if isinstance(status, dict) else {},
        }
        if not running:
            missing.append(name)

    conductor = component_status["autonomy_conductor"]["status"]
    jobs = conductor.get("jobs") if isinstance(conductor, dict) else {}
    jobs = jobs if isinstance(jobs, dict) else {}
    delegated_jobs = {
        name: job
        for name, job in jobs.items()
        if isinstance(job, dict) and str(job.get("policy") or "") == "delegated"
    }
    constitutive_jobs = {
        name: job
        for name, job in jobs.items()
        if isinstance(job, dict) and str(job.get("policy") or "") == "constitutive"
    }
    job_statuses = {
        name: str(job.get("last_status") or "unknown")
        for name, job in jobs.items()
        if isinstance(job, dict)
    }
    deferred_reasons = {
        name: str((job.get("last_result") or {}).get("reason") or "")
        for name, job in jobs.items()
        if isinstance(job, dict) and str(job.get("last_status") or "") == "deferred"
    }
    bad_deferred = {
        name: reason
        for name, reason in deferred_reasons.items()
        if reason in DISALLOWED_DEFER_REASONS
    }

    initiative = component_status["autonomous_initiative"]["status"]
    core_tasks = initiative.get("core_tasks") if isinstance(initiative, dict) else {}
    core_tasks = core_tasks if isinstance(core_tasks, dict) else {}
    inactive_core_tasks = [name for name, alive in sorted(core_tasks.items()) if not bool(alive)]

    report = {
        "schema": "aura.background_autonomy_proof.v1",
        "full_runtime_expected": bool(full_runtime.get("full_runtime_expected")),
        "full_runtime_ready": bool(health.get("full_runtime_ready") or full_runtime.get("ready")),
        "full_runtime_blockers": list(full_runtime.get("blockers") or []),
        "background_enabled": bool(background.get("enabled")),
        "background_active": bool(background.get("active")),
        "background_loops_allowed": bool(background.get("loops_allowed")),
        "background_work_admission": str(background.get("work_admission") or ""),
        "background_work_defer_reason": str(background.get("work_defer_reason") or ""),
        "missing_components": missing,
        "running_component_count": len(REQUIRED_COMPONENTS) - len(missing),
        "required_component_count": len(REQUIRED_COMPONENTS),
        "autonomy_conductor_active": bool(conductor.get("active")) if isinstance(conductor, dict) else False,
        "autonomy_jobs_count": len(jobs),
        "delegated_jobs": sorted(delegated_jobs),
        "constitutive_jobs": sorted(constitutive_jobs),
        "job_statuses": job_statuses,
        "deferred_reasons": deferred_reasons,
        "bad_deferred_reasons": bad_deferred,
        "autonomous_initiative_core_tasks": core_tasks,
        "inactive_core_tasks": inactive_core_tasks,
    }

    pass_conditions = {
        "full_runtime_expected": report["full_runtime_expected"],
        "full_runtime_ready": report["full_runtime_ready"],
        "background_enabled": report["background_enabled"],
        "background_active": report["background_active"],
        "background_loops_allowed": report["background_loops_allowed"],
        "all_required_components_running": not missing,
        "autonomy_conductor_active": report["autonomy_conductor_active"],
        "has_delegated_overt_action": "overt_action_cycle" in delegated_jobs,
        "has_deliberation_job": "internal_deliberation_cycle" in jobs,
        "has_constitutive_jobs": bool(constitutive_jobs),
        "no_forbidden_defer_reasons": not bad_deferred,
        "initiative_core_tasks_alive": not inactive_core_tasks,
    }
    report["pass_conditions"] = pass_conditions
    report["passed"] = all(pass_conditions.values())
    return report


def _prepare_env() -> dict[str, str]:
    env = dict(os.environ)
    env["AURA_BACKGROUND_BOOT_GRACE_S"] = "0"
    env["AURA_AMBIENT_STREAM_INTERVAL_S"] = "2"
    env["AURA_AUTONOMIC_REFLECTION_INTERVAL_S"] = "5"
    env["AURA_OVERT_ACTION_INTERVAL_S"] = "5"
    env["AURA_ENABLE_BACKGROUND_COGNITION"] = "1"
    env["AURA_DEFERRED_CORTEX_PREWARM"] = "1"
    env["AURA_EAGER_CORTEX_WARMUP"] = "0"
    env.pop("AURA_FOREGROUND_ONLY", None)
    env.pop("AURA_PROOF_RUN", None)
    return env


def run_live_proof(
    *,
    port: int,
    mode: str,
    boot_timeout: float,
    observe_seconds: float,
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    previous_env = dict(os.environ)
    proof_env = _prepare_env()
    os.environ.clear()
    os.environ.update(proof_env)
    proof = LiveProof(
        port=port,
        mode=mode,
        boot_timeout_s=boot_timeout,
        skip_desktop=True,
        restart_continuity=False,
        conversation_soak_turns=0,
        proof_dir=out_dir,
    )
    snapshots_path = out_dir / "HEALTH_SNAPSHOTS.jsonl"
    snapshots_path.write_text("", encoding="utf-8")
    final_health: dict[str, Any] = {}
    desktop_access: dict[str, Any] = {}
    passed = False
    shutdown_ok = False
    try:
        boot_ok = proof.boot()
        if boot_ok:
            deadline = time.monotonic() + max(0.0, observe_seconds)
            while time.monotonic() <= deadline:
                final_health = _get_json(proof.base, "/api/health")
                with snapshots_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"at": time.time(), "health": final_health}, default=str) + "\n")
                time.sleep(2.0)
            try:
                desktop_access = _get_json(proof.base, "/api/system/desktop-access", timeout=24.0)
            except (httpx.HTTPError, ValueError) as exc:
                desktop_access = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            report = evaluate_background_autonomy(final_health)
            report["desktop_access"] = desktop_access
            report["live_proof_transcript"] = str(proof.transcript_path)
            report["live_proof_stdout"] = str(proof.stdout_path)
            report["peak_rss_mb"] = round(proof.peak_rss_mb, 1)
            _json_dump(out_dir / "BACKGROUND_AUTONOMY_REPORT.json", report)
            _json_dump(
                out_dir / "AUTONOMY_CONDUCTOR.json",
                {
                    "schema": "aura.background_autonomy_conductor.v1",
                    "autonomy_conductor_active": report.get("autonomy_conductor_active"),
                    "autonomy_jobs_count": report.get("autonomy_jobs_count"),
                    "delegated_jobs": report.get("delegated_jobs"),
                    "constitutive_jobs": report.get("constitutive_jobs"),
                    "job_statuses": report.get("job_statuses"),
                    "deferred_reasons": report.get("deferred_reasons"),
                    "bad_deferred_reasons": report.get("bad_deferred_reasons"),
                    "pass_conditions": report.get("pass_conditions"),
                },
            )
            passed = bool(report.get("passed"))
        else:
            _json_dump(
                out_dir / "BACKGROUND_AUTONOMY_REPORT.json",
                {
                    "schema": "aura.background_autonomy_proof.v1",
                    "passed": False,
                    "boot_failed": True,
                    "steps": proof.steps,
                    "live_proof_transcript": str(proof.transcript_path),
                    "live_proof_stdout": str(proof.stdout_path),
                },
            )
    finally:
        if proof.proc is not None and proof.proc.poll() is None:
            shutdown_ok = proof.shutdown(step="background_autonomy_shutdown")
        stream_ok = proof.scan_runtime_stream()
        manifest = {
            "schema": "aura.background_autonomy_manifest.v1",
            "generated_at": time.time(),
            "mode": mode,
            "port": port,
            "observe_seconds": observe_seconds,
            "passed": bool(passed and shutdown_ok and stream_ok),
            "background_report": "BACKGROUND_AUTONOMY_REPORT.json",
            "health_snapshots": "HEALTH_SNAPSHOTS.jsonl",
            "live_proof_transcript": str(proof.transcript_path),
            "live_proof_stdout": str(proof.stdout_path),
            "shutdown_ok": shutdown_ok,
            "stream_ok": stream_ok,
        }
        _json_dump(out_dir / "MANIFEST.json", manifest)
        os.environ.clear()
        os.environ.update(previous_env)

    return 0 if (passed and shutdown_ok and stream_ok) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=("desktop", "headless"), default="desktop")
    parser.add_argument("--boot-timeout", type=float, default=420.0)
    parser.add_argument("--observe-seconds", type=float, default=45.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    try:
        get_subprocess_gateway().run(
            [os.environ.get("PYTHON", "python"), "-u", "aura_main.py", "--stop"],
            cwd=ROOT,
            timeout=90,
            offline_tooling=True,
            source="certification_tooling:background_autonomy_proof.stop_runtime",
        )
    except (OSError, subprocess.SubprocessError):
        pass

    return run_live_proof(
        port=args.port,
        mode=args.mode,
        boot_timeout=args.boot_timeout,
        observe_seconds=args.observe_seconds,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
