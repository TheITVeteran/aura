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
    assert (scaffold / "src" / "program.py").exists()
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
    assert "program_dna_reconstruct" in engine.active_skills
    selected = engine.detect_intent("Can you capture the program DNA and rebuild this app clean-room?")
    assert "program_dna_reconstruct" in selected


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
