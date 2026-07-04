"""Program DNA hidden-source behavioral equivalence battery skill."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill


class ProgramDNAEquivalenceBatteryInput(BaseModel):
    out_path: str = Field(
        "artifacts/current/program_dna_behavioral_equivalence_latest.json",
        description="Where to write the battery JSON artifact.",
    )
    include_results: bool = Field(
        True,
        description="Return per-scenario results in the skill response.",
    )


class ProgramDNAEquivalenceBatterySkill(BaseSkill):
    name = "program_dna_equivalence_battery"
    description = (
        "Run the hidden-source Program DNA behavioral equivalence battery across "
        "CLI, GUI, file converter, web, DB, mocked-auth, and missing-doc archetypes."
    )
    input_model = ProgramDNAEquivalenceBatteryInput
    timeout_seconds = 20.0
    metabolic_cost = 1
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ProgramDNAEquivalenceBatteryInput(**params)
        elif not isinstance(params, ProgramDNAEquivalenceBatteryInput):
            params = ProgramDNAEquivalenceBatteryInput.model_validate(params)

        battery = importlib.import_module("tools.program_dna.behavioral_equivalence_battery")
        report = await battery.run_battery(project_root=Path.cwd())
        out_path = Path(params.out_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        slim_report = dict(report)
        if not params.include_results:
            slim_report.pop("results", None)
        passed = bool(report.get("ok"))
        return {
            "ok": passed,
            "skill": self.name,
            "artifact": str(out_path),
            "summary": (
                "Program DNA hidden-source equivalence battery "
                f"{'passed' if passed else 'failed'}: "
                f"{report.get('passed_scenarios')}/{report.get('scenario_count')} scenarios, "
                f"{report.get('passed_cases')}/{report.get('held_out_cases')} held-out cases."
            ),
            "result": slim_report,
        }


__all__ = ["ProgramDNAEquivalenceBatteryInput", "ProgramDNAEquivalenceBatterySkill"]
