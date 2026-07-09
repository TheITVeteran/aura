from __future__ import annotations

import pytest

from tools.proof.run_autonomous_repair_proof import run_proof


@pytest.mark.asyncio
async def test_autonomous_repair_proof_routes_faults_and_rsi_artifacts(tmp_path):
    report = await run_proof(
        out=tmp_path / "autonomous_repair_proof.json",
        rsi_dir=tmp_path / "rsi",
    )

    assert report["passed"] is True
    assert report["checks"]["resilience_pressure_recorded"] is True
    assert report["checks"]["self_modification_error_intake"] is True
    assert report["checks"]["autonomous_cycle_completed"] is True
    assert report["checks"]["immune_event_scheduled"] is True
    assert report["checks"]["immune_patch_scheduled"] is True
    assert report["checks"]["cooldown_prevents_storm"] is True
    assert report["checks"]["rsi_median_improved"] is True
    assert report["checks"]["rsi_palindrome_improved"] is True
    assert (tmp_path / "autonomous_repair_proof.json").exists()
    assert (tmp_path / "rsi" / "median_repair_lab_autonomous_proof.json").exists()
    assert (tmp_path / "rsi" / "is_palindrome_repair_lab_autonomous_proof.json").exists()
