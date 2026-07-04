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
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.self_improvement.program_dna import ProgramDNAReconstructionEngine
from tools.program_dna.behavioral_equivalence_battery import scenarios


def _spec_docs(scenario: Any) -> list[str]:
    return [
        *scenario.docs,
        *scenario.ui_notes,
        *scenario.api_observations,
        *scenario.file_formats,
        *scenario.workflows,
        *scenario.permissions,
    ]


async def run_live_reconstruction(*, out: Path | None = None) -> dict[str, Any]:
    engine = ProgramDNAReconstructionEngine(project_root=REPO_ROOT)
    results: list[dict[str, Any]] = []

    for scenario in scenarios():
        held_out = [
            {"input": case, "expected": scenario.original(case)}
            for case in scenario.held_out_cases
        ]
        outcome = await engine.reconstruct_executable_via_cognition(
            target=scenario.name,
            spec_docs=_spec_docs(scenario),
            train_examples=scenario.behavior_examples,
            held_out=held_out,
            fn_name="reconstructed",
            authorization="educational",
            objective=f"clean-room reconstruction study for {scenario.category}",
        )
        results.append(
            {
                "name": scenario.name,
                "category": scenario.category,
                "status": outcome.get("status"),
                "held_out_passed": outcome.get("held_out_passed", 0),
                "held_out_total": outcome.get("held_out_total", len(held_out)),
                "equivalence": outcome.get("equivalence", 0.0),
                "failures": outcome.get("failures", []),
            }
        )

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
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = asyncio.run(run_live_reconstruction(out=args.out))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    # The probe SUCCEEDS if it ran and produced honest numbers; coverage is a
    # measurement, not a pass/fail gate (the model may honestly miss archetypes).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
