import json
from pathlib import Path

import pytest

from core.capability_engine import CapabilityEngine
from core.container import ServiceContainer
from core.self_improvement.program_dna import (
    ProgramDNAReconstructionEngine,
    register_program_dna_reconstruction_engine,
)
from core.service_names import ServiceNames
from core.skills.program_dna_reconstruct import ProgramDNAReconstructSkill


def _write_toy_program(root: Path) -> Path:
    app = root / "toy_notes"
    src = app / "src" / "toy_notes"
    src.mkdir(parents=True)
    (app / "pyproject.toml").write_text(
        """
[project]
name = "toy-notes"
dependencies = ["reportlab"]
""".strip(),
        encoding="utf-8",
    )
    (src / "app.py").write_text(
        '''
"""Toy notes app used for Program DNA reconstruction tests."""

class NoteStore:
    def create_note(self, title: str, body: str) -> dict:
        return {"title": title, "body": body}

    def search_notes(self, query: str) -> list[dict]:
        return []

def export_pdf(note: dict, output_path: str) -> str:
    return output_path
'''.strip(),
        encoding="utf-8",
    )
    return app


@pytest.mark.asyncio
async def test_program_dna_engine_builds_clean_room_blueprint_and_scaffold(tmp_path):
    toy = _write_toy_program(tmp_path)
    out = tmp_path / "dna_out"
    engine = ProgramDNAReconstructionEngine(project_root=tmp_path)

    result = await engine.reconstruct(
        {
            "target": "Toy Notes",
            "authorization": "open_source",
            "source_paths": [str(toy), str(toy / "pyproject.toml"), str(toy / "src" / "toy_notes" / "app.py")],
            "observed_behaviors": [
                "User can create a note, search notes, and export a note as PDF.",
            ],
            "ui_notes": [
                "Main window has note editor, search field, and export button.",
            ],
            "api_observations": ["POST /notes creates a note and GET /notes?q= filters notes."],
            "file_formats": ["Exports PDF and imports JSON backup files."],
            "logs": ["Background worker logs export failures and retries."],
            "tests": ["Golden-file test compares exported PDF receipt metadata."],
            "workflows": ["Create note -> edit body -> save -> export PDF -> verify file exists."],
            "permissions": ["Filesystem write permission is required for PDF export."],
            "compatibility_targets": ["modern local-first desktop app", "headless batch export"],
            "similar_programs": ["Apple Notes", "Notion"],
            "emit_scaffold": True,
            "output_dir": str(out),
            "target_stack": "python",
        }
    )

    assert result.ok is True
    feature_names = {feature.name for feature in result.features}
    assert {
        "document_creation",
        "search_and_retrieval",
        "export_pipeline",
        "persistence",
        "api_surface",
        "file_format_inference",
        "background_service",
        "permissions_model",
    } <= feature_names
    assert result.genome is not None
    assert result.genome.workflow_graph
    assert result.genome.file_formats
    assert result.genome.api_surface
    assert result.verification_plan is not None
    assert result.verification_plan.black_box_tests
    assert result.verification_plan.ui_tests
    assert result.verification_plan.golden_file_tests
    assert result.verification_plan.edge_case_tests
    assert result.verification_plan.security_checks
    assert result.verification_plan.scaffold_syntax_ok is True
    assert result.blueprint is not None
    assert "Do not bypass DRM" in result.blueprint.safety_boundary[0]
    assert result.scaffold_path
    scaffold = Path(result.scaffold_path)
    assert (scaffold / "PROGRAM_DNA_BLUEPRINT.json").exists()
    assert (scaffold / "PROGRAM_GENOME.json").exists()
    assert (scaffold / "VERIFICATION_PLAN.json").exists()
    assert (scaffold / "src" / "__init__.py").exists()
    assert (scaffold / "src" / "program.py").exists()
    assert (scaffold / "tests" / "conftest.py").exists()
    assert (scaffold / "tests" / "test_program_contract.py").exists()
    blueprint = json.loads((scaffold / "PROGRAM_DNA_BLUEPRINT.json").read_text(encoding="utf-8"))
    assert blueprint["target_name"] == "Toy Notes"
    genome = json.loads((scaffold / "PROGRAM_GENOME.json").read_text(encoding="utf-8"))
    assert genome["compatibility_targets"] == ["modern local-first desktop app", "headless batch export"]
    plan = json.loads((scaffold / "VERIFICATION_PLAN.json").read_text(encoding="utf-8"))
    assert plan["scaffold_syntax_ok"] is True


@pytest.mark.asyncio
async def test_program_dna_engine_blocks_unauthorized_or_abusive_reconstruction(tmp_path):
    engine = ProgramDNAReconstructionEngine(project_root=tmp_path)

    result = await engine.reconstruct(
        {
            "target": "Closed App",
            "authorization": "unspecified",
            "objective": "steal source and bypass DRM",
        }
    )

    assert result.ok is False
    assert "authorization_required_for_program_reconstruction" in result.blocked_reasons
    assert "prohibited_reverse_engineering_or_abuse_intent" in result.blocked_reasons


@pytest.mark.asyncio
async def test_program_dna_public_observation_study_maps_interaction_surfaces(tmp_path):
    engine = ProgramDNAReconstructionEngine(project_root=tmp_path)

    result = await engine.reconstruct(
        {
            "target": "Observed Calendar Widget",
            "authorization": "public_observation",
            "analysis_mode": "study",
            "objective": "study how this public widget works from visible behavior and public docs",
            "observed_behaviors": [
                "The widget accepts a date input, renders a month grid, and highlights events from imported JSON.",
                "When offline, the widget still shows cached local events and reports that sync is unavailable.",
            ],
            "ui_notes": [
                "Visible controls include next month, previous month, today, import, export, and settings.",
            ],
            "research_notes": [
                "Similar open widgets use a state machine around selected_date, visible_month, and event_store.",
            ],
            "study_questions": [
                "How does the visible state machine handle imported events and offline cache misses?",
                "Which parts can be implemented from public behavior without copying hidden source?",
            ],
            "interaction_observations": [
                "The UI receives clicks and key input, reads JSON files, writes exported calendars, and emits status events.",
            ],
            "aura_interactions": [
                "Aura may call the replacement through /api/skill/execute with a JSON payload and must receive a receipt.",
            ],
            "host_interactions": [
                "The app reads only user-selected files and writes only to an authorized output directory.",
            ],
            "network_observations": [
                "The app may call a calendar sync endpoint, but local cached rendering must work without network.",
            ],
            "hardware_observations": [
                "The app needs screen and keyboard interaction when driven visually; it does not require camera or microphone.",
            ],
            "process_observations": [
                "No persistent daemon is visible; import/export jobs should terminate after producing a receipt.",
            ],
            "security_observations": [
                "Defensive security study: token handling must be mocked or user-provided; public observation must not inspect private credentials.",
            ],
            "compatibility_targets": ["clean-room local widget", "public-behavior compatible CLI export"],
        }
    )

    assert result.ok is True
    feature_names = {feature.name for feature in result.features}
    assert {
        "study_model",
        "interaction_surface",
        "aura_interaction_surface",
        "host_hardware_interaction",
        "network_interaction",
        "process_observation",
        "defensive_security_analysis",
        "file_format_inference",
    } <= feature_names
    assert result.genome is not None
    assert result.genome.analysis_mode == "study"
    assert result.genome.study_questions
    assert result.genome.interaction_surfaces
    assert result.genome.aura_interaction_surface
    assert result.genome.host_touchpoints
    assert result.genome.network_surface
    assert result.genome.hardware_surface
    assert result.genome.defensive_observations
    assert any("public observation" in risk.lower() for risk in result.genome.hidden_state_risks)
    assert result.blueprint is not None
    assert "mechanism study" in result.blueprint.reconstruction_strategy
    assert any("Public observation" in unknown for unknown in result.blueprint.unknowns)
    assert result.verification_plan is not None
    interaction_names = {item["name"] for item in result.verification_plan.interaction_tests}
    assert "aura_touchpoints_governed" in interaction_names
    assert "network_surface_bounded" in interaction_names
    assert "hardware_permission_boundary" in interaction_names


@pytest.mark.asyncio
async def test_program_dna_dual_use_study_requires_defensive_authorization(tmp_path):
    engine = ProgramDNAReconstructionEngine(project_root=tmp_path)

    blocked = await engine.reconstruct(
        {
            "target": "Unknown Payload",
            "authorization": "public_observation",
            "objective": "build malware payload behavior from public observation",
        }
    )
    allowed = await engine.reconstruct(
        {
            "target": "Unknown Payload",
            "authorization": "security_research",
            "analysis_mode": "defensive_analysis",
            "objective": "defensive study to understand malware behavior on my host and protect Aura",
            "security_observations": [
                "Suspicious process attempts network beaconing and filesystem writes; preserve forensic receipts only.",
            ],
            "process_observations": [
                "Process spawned a child worker and exited after the network attempt was blocked.",
            ],
            "network_observations": [
                "Outbound socket attempt was blocked; record endpoint metadata without packet payloads.",
            ],
        }
    )

    assert blocked.ok is False
    assert "dual_use_security_intent_requires_defensive_authorization" in blocked.blocked_reasons
    assert allowed.ok is True
    assert allowed.genome is not None
    assert allowed.genome.defensive_observations
    assert allowed.verification_plan is not None
    assert allowed.verification_plan.security_checks


def test_program_dna_engine_registers_canonical_service(tmp_path, monkeypatch):
    from core.self_improvement import program_dna as program_dna_module

    monkeypatch.setattr(program_dna_module, "_PROGRAM_DNA_INSTANCE", None)
    ServiceContainer.clear()

    engine = register_program_dna_reconstruction_engine(project_root=tmp_path)

    assert ServiceContainer.get(ServiceNames.PROGRAM_DNA_RECONSTRUCTION) is engine
    ServiceContainer.clear()


def test_program_dna_skill_is_live_capability_engine_surface():
    engine = CapabilityEngine(orchestrator=None)

    assert "program_dna_reconstruct" in engine.get_available_skills()
    assert "program_dna_equivalence_battery" in engine.get_available_skills()
    assert "program_dna_reconstruct" in engine.active_skills
    assert "program_dna_equivalence_battery" in engine.active_skills
    selected = engine.detect_intent("Can you capture the program DNA and rebuild this app clean-room?")
    assert "program_dna_reconstruct" in selected
    selected_battery = engine.detect_intent("Run the Program DNA hidden-source behavioral equivalence battery.")
    assert "program_dna_equivalence_battery" in selected_battery


@pytest.mark.asyncio
async def test_program_dna_skill_exposes_live_capability(tmp_path, monkeypatch):
    toy = _write_toy_program(tmp_path)
    out = tmp_path / "skill_out"
    engine = ProgramDNAReconstructionEngine(project_root=tmp_path)
    ServiceContainer.clear()
    ServiceContainer.register_instance(ServiceNames.PROGRAM_DNA_RECONSTRUCTION, engine, required=False)

    result = await ProgramDNAReconstructSkill().execute(
        {
            "target": "Toy Notes Skill",
            "authorization": "open_source",
            "source_paths": [str(toy)],
            "observed_behaviors": ["Create notes and export PDF files."],
            "ui_notes": ["Editor with export button."],
            "emit_scaffold": True,
            "output_dir": str(out),
        },
        context={"surface": "test"},
    )

    assert result["ok"] is True
    assert "document_creation" in result["features"]
    assert "export_pipeline" in result["features"]
    assert Path(result["result"]["scaffold_path"]).exists()
    ServiceContainer.clear()
