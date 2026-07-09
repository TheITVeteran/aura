#!/usr/bin/env python3
"""Proof: runtime faults become autonomous repair work, not inert telemetry.

This is a bounded non-mutating proof. It deliberately injects a representative
runtime degradation into the real repair routing surfaces, then verifies that:

* the degradation hits resilience pressure
* the self-modification error-intake lane receives the fault
* the autonomous repair executor runs a repair cycle
* adaptive-immune patch proposals schedule through the same executor
* duplicate repair requests are cooled down instead of storming
* RSI repair-lab artifacts still prove strict held-out improvement

It does not claim unrestricted live source mutation. It proves the runtime
plumbing that turns observed defects into governed repair attempts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.resilience.autonomous_repair_executor import (  # noqa: E402
    AutonomousRepairExecutor,
    AutonomousRepairRequest,
    set_autonomous_repair_executor_for_tests,
)
from core.resilience.degradation_repair import DegradationRepairRouter  # noqa: E402


class ProofResilienceEngine:
    def __init__(self) -> None:
        self.failures: list[dict[str, Any]] = []

    def record_failure(self, domain: str, severity: float, stakes: float) -> SimpleNamespace:
        self.failures.append(
            {
                "domain": domain,
                "severity": round(float(severity), 4),
                "stakes": round(float(stakes), 4),
            }
        )
        return SimpleNamespace(value="strain")


class ProofSelfModificationEngine:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.cycles = 0

    def on_error(self, error: BaseException, context: dict[str, Any], skill_name=None, goal=None) -> None:
        self.errors.append(
            {
                "error_type": type(error).__qualname__,
                "error": str(error),
                "context": dict(context),
                "skill_name": skill_name,
                "goal": goal,
            }
        )

    async def run_autonomous_cycle(self) -> dict[str, Any]:
        self.cycles += 1
        await asyncio.sleep(0)
        return {
            "success": True,
            "bugs_found": 1,
            "fixes_applied": 1,
            "auto_repair_mode": "safe_autonomous",
            "validation": "proof-cycle",
        }


class ProofImmuneSystem:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def observe_event(self, event: dict[str, Any], **kwargs: Any) -> SimpleNamespace:
        self.events.append({"event": dict(event), "kwargs": dict(kwargs)})
        await asyncio.sleep(0)
        return SimpleNamespace(selected_artifact=None)


async def _wait_for_executor(executor: AutonomousRepairExecutor, *, previous_started: int = 0) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if executor.stats["started"] > previous_started and executor.last_result is not None:
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("autonomous repair executor did not produce a result")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _service_getter(
    *,
    resilience: ProofResilienceEngine,
    self_modification: ProofSelfModificationEngine,
    immune: ProofImmuneSystem,
):
    def _get(name: str) -> Any | None:
        return {
            "resilience_engine": resilience,
            "self_modification_engine": self_modification,
            "adaptive_immune_system": immune,
        }.get(name)

    return _get


async def run_proof(*, out: Path | None = None, rsi_dir: Path | None = None) -> dict[str, Any]:
    resilience = ProofResilienceEngine()
    self_modification = ProofSelfModificationEngine()
    immune = ProofImmuneSystem()
    services = _service_getter(
        resilience=resilience,
        self_modification=self_modification,
        immune=immune,
    )
    executor = AutonomousRepairExecutor(
        service_getter=services,
        cooldown_seconds=0.0,
        cycle_timeout_seconds=5.0,
    )
    router = DegradationRepairRouter(service_getter=services, cooldown_seconds=0.0)

    record = SimpleNamespace(
        subsystem="proof.cognitive_engine",
        severity="warning",
        error_type="TimeoutError",
        error_message="full-mind desktop reply timed out",
        action="prove autonomous repair routing",
    )
    incident = SimpleNamespace(incident_id="INC-PROOF-AUTOREPAIR", occurrence_count=6)

    set_autonomous_repair_executor_for_tests(executor)
    try:
        route_action = router.route(
            record=record,
            error=TimeoutError(record.error_message),
            incident=incident,
            extra={"repair_requested": True, "proof": "autonomous_repair"},
        )
        await _wait_for_executor(executor)

        cooldown_executor = AutonomousRepairExecutor(
            service_getter=services,
            cooldown_seconds=60.0,
            cycle_timeout_seconds=5.0,
        )
        cooldown_request = AutonomousRepairRequest(
            subsystem="proof.tool_lane",
            error_type="RuntimeError",
            error_message="repeat web tool routing failure",
            goal="Prove cooldown prevents repair storm",
        )
        cooldown_first = await cooldown_executor.execute_now(cooldown_request)
        cooldown_second = await cooldown_executor.execute_now(cooldown_request)

        before_patch_started = executor.stats["started"]
        patch_result = await executor.attempt_patch_for_antigen(
            SimpleNamespace(
                artifact_id="proof-effector-1",
                kind=SimpleNamespace(value="patch_proposal"),
                component="proof.runtime_engine",
                notes="runtime patch proposal from immune effector",
            ),
            SimpleNamespace(
                subsystem="proof.runtime_engine",
                error_signature="RuntimeError",
                source="proof",
                antigen_id="proof-antigen-1",
                danger=0.7,
            ),
        )
        await _wait_for_executor(executor, previous_started=before_patch_started)
    finally:
        set_autonomous_repair_executor_for_tests(None)

    rsi_dir = rsi_dir or (REPO_ROOT / "artifacts" / "live_proof" / "rsi")
    rsi_dir.mkdir(parents=True, exist_ok=True)
    from tools.proof.run_rsi_challenge_proof import _CHALLENGES, run_repair_lab

    rsi_reports: dict[str, dict[str, Any]] = {}
    for challenge_name in ("median", "is_palindrome"):
        challenge = _CHALLENGES[challenge_name]()
        rsi_out = rsi_dir / f"{challenge_name}_repair_lab_autonomous_proof.json"
        rsi_report = run_repair_lab(challenge, out=rsi_out)
        rsi_reports[challenge_name] = {
            "artifact": str(rsi_out),
            "sha256": _sha256_file(rsi_out),
            "improvement_proven": bool(rsi_report.get("improvement_proven")),
            "promoted": bool(rsi_report.get("promoted")),
            "seed_passed": rsi_report.get("seed_passed"),
            "improved_passed": rsi_report.get("improved_passed"),
            "promoted_candidate": rsi_report.get("promoted_candidate"),
        }

    checks = {
        "resilience_pressure_recorded": bool(resilience.failures),
        "self_modification_error_intake": len(self_modification.errors) >= 2,
        "autonomous_cycle_completed": executor.stats["completed"] >= 2,
        "immune_event_scheduled": route_action.immune_status == "scheduled" and bool(immune.events),
        "immune_patch_scheduled": patch_result.get("status") == "scheduled",
        "cooldown_prevents_storm": cooldown_first.get("status") == "completed"
        and cooldown_second.get("status") == "cooldown",
        "rsi_median_improved": bool(rsi_reports["median"]["improvement_proven"]),
        "rsi_palindrome_improved": bool(rsi_reports["is_palindrome"]["improvement_proven"]),
    }
    report = {
        "proof": "autonomous_repair_and_rsi_live_plumbing",
        "passed": all(checks.values()),
        "checks": checks,
        "route_action": route_action.to_dict(),
        "resilience_failures": resilience.failures,
        "self_modification_errors": self_modification.errors,
        "executor_stats": dict(executor.stats),
        "executor_last_result": executor.last_result,
        "immune_events": immune.events,
        "immune_patch_result": patch_result,
        "cooldown": {
            "first": cooldown_first,
            "second": cooldown_second,
            "stats": dict(cooldown_executor.stats),
        },
        "rsi_reports": rsi_reports,
        "policy": (
            "bounded non-mutating proof: runtime faults route to autonomous repair; "
            "RSI candidates are promoted only in artifacts after held-out sandbox improvement"
        ),
        "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "live_proof" / "autonomous_repair_proof.json",
    )
    parser.add_argument(
        "--rsi-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "live_proof" / "rsi",
    )
    args = parser.parse_args()
    report = asyncio.run(run_proof(out=args.out, rsi_dir=args.rsi_dir))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
