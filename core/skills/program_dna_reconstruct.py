"""Program DNA reconstruction skill.

Live capability surface for authorized clean-room reconstruction and mechanism
study of programs from observable behavior, open/user-owned source, metadata,
UI notes, host/Aura interaction traces, and research evidence.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.service_registry import get_runtime_service
from core.service_names import ServiceNames
from core.skills.base_skill import BaseSkill


class ProgramDNAInput(BaseModel):
    target: str = Field(..., description="Program/app/library name or target path label.")
    authorization: str = Field(
        "unspecified",
        description=(
            "open_source | owner_authorized | explicit_permission | internal | educational | "
            "user_owned | public_observation | external_observation | host_observation | "
            "defensive_analysis | security_research"
        ),
    )
    analysis_mode: str = Field(
        "reconstruct",
        description="reconstruct | reverse_engineer | study | observe | monitor | defensive_analysis",
    )
    source_paths: list[str] = Field(default_factory=list)
    observed_behaviors: list[str] = Field(default_factory=list)
    ui_notes: list[str] = Field(default_factory=list)
    research_notes: list[str] = Field(default_factory=list)
    similar_programs: list[str] = Field(default_factory=list)
    api_observations: list[str] = Field(default_factory=list)
    file_formats: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    workflows: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    study_questions: list[str] = Field(default_factory=list)
    interaction_observations: list[str] = Field(default_factory=list)
    aura_interactions: list[str] = Field(default_factory=list)
    host_interactions: list[str] = Field(default_factory=list)
    network_observations: list[str] = Field(default_factory=list)
    hardware_observations: list[str] = Field(default_factory=list)
    process_observations: list[str] = Field(default_factory=list)
    security_observations: list[str] = Field(default_factory=list)
    compatibility_targets: list[str] = Field(default_factory=list)
    target_stack: str = "python"
    enable_binary_static_analysis: bool = False
    capture_live_host_snapshot: bool = False
    emit_scaffold: bool = False
    output_dir: str | None = None


class ProgramDNAReconstructSkill(BaseSkill):
    name = "program_dna_reconstruct"
    description = (
        "Authorized clean-room reconstruction and mechanism study of a program's behavior DNA "
        "from source, metadata, UI/UX observations, Aura/host/network/hardware interactions, "
        "research notes, and similar-program hints."
    )
    input_model = ProgramDNAInput
    timeout_seconds = 45.0
    metabolic_cost = 2
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ProgramDNAInput(**params)
        elif not isinstance(params, ProgramDNAInput):
            params = ProgramDNAInput.model_validate(params)

        engine = get_runtime_service(ServiceNames.PROGRAM_DNA_RECONSTRUCTION, default=None)
        if engine is None:
            program_dna = importlib.import_module("core.self_improvement.program_dna")

            engine = program_dna.register_program_dna_reconstruction_engine(project_root=Path.cwd())

        # Runnable reverse-engineering: observe a REAL host binary, reconstruct
        # its behavior via cognition, and VERIFY against held-out real outputs.
        # Preferred whenever the target is a known safe host binary — that is
        # the strongest, verifiable answer — for both the explicit
        # reverse_engineer mode and the default reconstruct mode.
        if params.analysis_mode in {"reverse_engineer", "reconstruct"}:
            reverse = await self._reverse_engineer_host(engine, params.target)
            if reverse is not None:
                return reverse

        result = await engine.reconstruct(params.model_dump())
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        feature_names = [feature.get("name") for feature in payload.get("features", [])]
        return {
            "ok": bool(payload.get("ok")),
            "skill": self.name,
            "target": payload.get("target_name"),
            "features": feature_names,
            "result": payload,
            "summary": self._summary(payload, feature_names),
        }

    async def _reverse_engineer_host(self, engine: Any, target_label: str) -> dict[str, Any] | None:
        """Runnable reverse-engineering of a real host binary, verified against
        held-out real outputs. Returns None if the target is not a known safe
        host binary (caller falls back to structural reconstruction)."""
        try:
            from core.self_improvement.host_reconstruction import (
                resolve_target,
                reverse_engineer_host_binary,
            )
        except ImportError:
            return None
        target = resolve_target(target_label)
        if target is None:
            return None
        report = await reverse_engineer_host_binary(engine, target)
        status = report.get("status")
        return {
            "ok": status == "supported",
            "skill": self.name,
            "target": report.get("target"),
            "result": report,
            "summary": (
                f"Reverse-engineered {report.get('target')} from behavior only "
                f"(no source): {report.get('held_out_passed')}/{report.get('held_out_total')} "
                f"held-out cases reproduced — epistemic status: {status}."
            ),
        }

    def _summary(self, payload: dict[str, Any], feature_names: list[str]) -> str:
        if not payload.get("ok"):
            reasons = ", ".join(payload.get("blocked_reasons") or ["blocked"])
            return f"Program DNA reconstruction blocked: {reasons}"
        scaffold = payload.get("scaffold_path")
        suffix = f"; scaffold emitted at {scaffold}" if scaffold else ""
        return (
            f"Program DNA captured for {payload.get('target_name')}: "
            f"{len(payload.get('evidence', []))} evidence item(s), "
            f"{len(feature_names)} inferred feature(s){suffix}."
        )


__all__ = ["ProgramDNAInput", "ProgramDNAReconstructSkill"]
