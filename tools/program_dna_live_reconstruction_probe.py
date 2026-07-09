#!/usr/bin/env python3
"""Live proof: Aura reconstructs runnable programs from spec, verified honestly.

Unlike the offline behavioral-equivalence battery — which proves the differential
harness is real and can fail — this probe exercises the GENUINE capability:
``ProgramDNAReconstructionEngine.reconstruct_executable_via_cognition`` asks the
live model to implement each archetype from docs + a few input/output examples
(NO source), then a sandbox that fails wrong code checks the implementation
against HELD-OUT observations the model never saw.

The reported number is honest reconstruction coverage: how many archetypes the
model actually reconstructed to full held-out equivalence. A partial score is a
real result, not a failure to hide. Requires the live model, so it is marked
``live`` and skipped in the offline suite.

    python tools/program_dna_live_reconstruction_probe.py --out artifacts/program_dna/live_reconstruction.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.self_improvement.program_dna import ProgramDNAReconstructionEngine
from tools.program_dna.behavioral_equivalence_battery import scenarios


def _llm_generation_available() -> tuple[bool, str]:
    """Return whether the live reconstruction path has a real generator.

    The reconstruction engine first prefers the local un-steered code model
    (same on-device weights, no persona steering), then falls back to a runtime
    service. A cold proof process may have no ServiceContainer entries yet, so
    checking only the container incorrectly downgrades the proof to conjecture.
    This probe should verify the actual generation surface without loading the
    large model eagerly.
    """
    try:
        from core.brain.llm.local_code_model import LocalCodeModel

        candidate = LocalCodeModel()
        if candidate.is_available():
            return True, "local_code_model_available"
    except (ImportError, RuntimeError, OSError, TypeError, ValueError):
        pass

    try:
        from core.container import ServiceContainer
    except (ImportError, RuntimeError):
        return False, "no_service_container_or_local_code_model"
    for service_name in ("inference_gate", "llm_router", "cognitive_engine"):
        try:
            if ServiceContainer.get(service_name, default=None) is not None:
                return True, f"service_container:{service_name}"
        except (AttributeError, RuntimeError):
            continue
    return False, "no_llm_router_registered"


def _spec_docs(scenario: Any) -> list[str]:
    return [
        *scenario.docs,
        *scenario.ui_notes,
        *scenario.api_observations,
        *scenario.file_formats,
        *scenario.workflows,
        *scenario.permissions,
    ]


def _scenario_max_tokens(scenario: Any) -> int:
    # Keep live proof calls bounded. The complex app prompt can otherwise run a
    # long MLX generation that coroutine cancellation cannot stop immediately.
    if getattr(scenario, "category", "") == "complex_local_app":
        return 640
    return 900


def _build_report(
    *,
    results: list[dict[str, Any]],
    router_available: bool,
    router_reason: str,
) -> dict[str, Any]:
    supported = [r for r in results if r["status"] == "supported"]
    total_cases = sum(r["held_out_total"] for r in results)
    passed_cases = sum(r["held_out_passed"] for r in results)
    return {
        "battery": "program_dna_live_cognition_reconstruction",
        "scenario_count": len(results),
        "fully_reconstructed": len(supported),
        "held_out_cases": total_cases,
        "passed_cases": passed_cases,
        "case_equivalence": passed_cases / total_cases if total_cases else 0.0,
        "policy": "spec-only (docs + examples); no source, no decompilation; held-out differential verification in sandbox",
        "honesty": "partial coverage is reported as-is; epistemic status is supported/refuted/conjecture per held-out outcome",
        "router_available": router_available,
        "router_reason": router_reason,
        "results": results,
    }


def _write_report(out: Path | None, report: dict[str, Any]) -> None:
    if not out:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _result_from_outcome(scenario: Any, outcome: dict[str, Any], held_out: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "category": scenario.category,
        "status": outcome.get("status"),
        "reason": outcome.get("reason", ""),
        "held_out_passed": outcome.get("held_out_passed", 0),
        "held_out_total": outcome.get("held_out_total", len(held_out)),
        "equivalence": outcome.get("equivalence", 0.0),
        "failures": outcome.get("failures", []),
        "repair_attempts_used": outcome.get("repair_attempts_used", 0),
        "synthesis_provenance": outcome.get("synthesis_provenance", ""),
        "candidate_code_sha256": hashlib.sha256(
            str(outcome.get("code") or "").encode("utf-8")
        ).hexdigest() if outcome.get("code") else "",
        "candidate_code_excerpt": str(outcome.get("code") or "")[:1200],
    }


def _run_scenario_subprocess(scenario: Any, *, scenario_timeout_s: float) -> dict[str, Any]:
    """Run one scenario in a child process so native MLX generation is killable."""

    timeout_s = max(20.0, float(scenario_timeout_s) + 35.0)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--only-scenario",
        str(scenario.name),
        "--inline",
        "--scenario-timeout-s",
        str(float(scenario_timeout_s)),
    ]
    held_out = [
        {"input": case, "expected": scenario.original(case)}
        for case in scenario.held_out_cases
    ]
    # Managed gateway instead of raw subprocess: same bounded timeout, plus
    # the repo's standard receipts/kill discipline (subprocess-usage ratchet).
    from core.tasks.managed_command import run_project_command

    proc = run_project_command(tuple(cmd), timeout_s=timeout_s)
    if proc.timed_out:
        return _result_from_outcome(
            scenario,
            {
                "status": "conjecture",
                "reason": f"process_timeout:{timeout_s:.1f}s",
                "held_out_passed": 0,
                "held_out_total": len(held_out),
                "equivalence": 0.0,
                "failures": [],
            },
            held_out,
        )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _result_from_outcome(
            scenario,
            {
                "status": "conjecture",
                "reason": f"subprocess_invalid_json:exit={proc.returncode}",
                "held_out_passed": 0,
                "held_out_total": len(held_out),
                "equivalence": 0.0,
                "failures": [{"stderr": proc.stderr[-2000:]}],
            },
            held_out,
        )
    result = next((r for r in report.get("results", []) if r.get("name") == scenario.name), None)
    if isinstance(result, dict):
        return result
    return _result_from_outcome(
        scenario,
        {
            "status": "conjecture",
            "reason": "subprocess_missing_scenario_result",
            "held_out_passed": 0,
            "held_out_total": len(held_out),
            "equivalence": 0.0,
            "failures": [],
        },
        held_out,
    )


async def run_live_reconstruction(
    *,
    out: Path | None = None,
    scenario_timeout_s: float = 75.0,
    only_scenario: str | None = None,
    inline: bool = False,
) -> dict[str, Any]:
    engine = ProgramDNAReconstructionEngine(project_root=REPO_ROOT)
    results: list[dict[str, Any]] = []
    router_available, router_reason = _llm_generation_available()

    selected_scenarios = [
        scenario for scenario in scenarios()
        if not only_scenario or scenario.name == only_scenario
    ]
    for scenario in selected_scenarios:
        held_out = [
            {"input": case, "expected": scenario.original(case)}
            for case in scenario.held_out_cases
        ]
        if (
            not inline
            and not only_scenario
            and scenario.category == "complex_local_app"
            and router_available
        ):
            results.append(
                _run_scenario_subprocess(
                    scenario,
                    scenario_timeout_s=scenario_timeout_s,
                )
            )
            _write_report(
                out,
                _build_report(
                    results=results,
                    router_available=router_available,
                    router_reason=router_reason,
                ),
            )
            continue
        if router_available:
            try:
                outcome = await asyncio.wait_for(
                    engine.reconstruct_executable_via_cognition(
                        target=scenario.name,
                        spec_docs=_spec_docs(scenario),
                        train_examples=scenario.behavior_examples,
                        held_out=held_out,
                        fn_name="reconstructed",
                        authorization="educational",
                        objective=f"clean-room reconstruction study for {scenario.category}",
                        max_tokens=_scenario_max_tokens(scenario),
                        max_repair_attempts=2,
                    ),
                    timeout=max(5.0, float(scenario_timeout_s)),
                )
            except (asyncio.TimeoutError, TimeoutError):
                outcome = {
                    "status": "conjecture",
                    "held_out_passed": 0,
                    "held_out_total": len(held_out),
                    "equivalence": 0.0,
                    "failures": [],
                    "reason": f"scenario_timeout:{max(5.0, float(scenario_timeout_s)):.1f}s",
                }
        else:
            outcome = {
                "status": "conjecture",
                "held_out_passed": 0,
                "held_out_total": len(held_out),
                "equivalence": 0.0,
                "failures": [],
                "reason": router_reason,
            }
        results.append(_result_from_outcome(scenario, outcome, held_out))
        _write_report(
            out,
            _build_report(
                results=results,
                router_available=router_available,
                router_reason=router_reason,
            ),
        )

    return _build_report(
        results=results,
        router_available=router_available,
        router_reason=router_reason,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scenario-timeout-s", type=float, default=75.0)
    parser.add_argument("--only-scenario", default=None)
    parser.add_argument("--inline", action="store_true")
    args = parser.parse_args()

    report = asyncio.run(
        run_live_reconstruction(
            out=args.out,
            scenario_timeout_s=args.scenario_timeout_s,
            only_scenario=args.only_scenario,
            inline=args.inline,
        )
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        _write_report(args.out, report)
    print(encoded)
    # The probe SUCCEEDS if it ran and produced honest numbers; coverage is a
    # measurement, not a pass/fail gate (the model may honestly miss archetypes).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
