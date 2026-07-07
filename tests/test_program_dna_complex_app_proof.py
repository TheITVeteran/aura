from __future__ import annotations

import pytest

from tools.proof.run_program_dna_complex_app_proof import run_proof


@pytest.mark.asyncio
async def test_program_dna_complex_app_proof_builds_auditable_workspace(tmp_path):
    report = await run_proof(out_dir=tmp_path / "complex_app")

    assert report["passed"] is True
    assert report["genome_ok"] is True
    assert report["replacement_tests_passed"] is True
    assert report["mutant_rejected"] is True
    assert report["held_out_cases"] >= 3
    workspace = tmp_path / "complex_app" / "replacement_workspace"
    assert (workspace / "src" / "knowledge_vault.py").exists()
    assert (workspace / "tests" / "test_behavioral_equivalence.py").exists()
    assert (tmp_path / "complex_app" / "EVIDENCE.json").exists()
    assert (tmp_path / "complex_app" / "RECEIPT.json").exists()
    assert (tmp_path / "complex_app" / "STANDARDS_REVIEW.json").exists()
