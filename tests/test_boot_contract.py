from __future__ import annotations

from pathlib import Path

from core.runtime.boot_contract import (
    BOOT_SERVICE_REQUIREMENTS,
    CANONICAL_PROOF_ARTIFACT_DIRS,
    BootServiceRequirement,
    boot_contract_report,
    validate_boot_contract,
)
from tools.arch_map import build_architecture_report


ROOT = Path(__file__).resolve().parents[1]


def test_boot_contract_passes_for_current_repo() -> None:
    issues = validate_boot_contract(ROOT)

    assert issues == []
    assert {item.name for item in BOOT_SERVICE_REQUIREMENTS} >= {
        "unified_will",
        "being_runtime",
        "aura_now",
        "memory_write_gateway",
        "state_gateway",
        "inference_gate",
        "llm_router",
        "capability_engine",
    }
    assert "person_box_proof" in CANONICAL_PROOF_ARTIFACT_DIRS


def test_boot_contract_detects_missing_evidence_token(tmp_path) -> None:
    owner = tmp_path / "core" / "demo.py"
    owner.parent.mkdir(parents=True)
    owner.write_text("present = True\n", encoding="utf-8")

    issues = validate_boot_contract(
        tmp_path,
        requirements=(
            BootServiceRequirement(
                name="demo",
                owner_file="core/demo.py",
                required_for="test",
                failure_policy="fail-closed",
                evidence_tokens=("missing_token",),
            ),
        ),
    )

    assert len(issues) == 1
    assert issues[0].code == "BOOT_CONTRACT_TOKEN_MISSING"


def test_architecture_report_includes_boot_contract() -> None:
    report = build_architecture_report()

    assert report["schema"] == "aura.architecture.dependency_map.v2"
    assert report["boot_contract"] == boot_contract_report(ROOT)
    assert report["boot_contract"]["ok"] is True
