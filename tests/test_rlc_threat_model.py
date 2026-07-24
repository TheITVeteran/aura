"""SPARK-003: the threat model must stay executable, complete, and honest."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256
from core.brain.llm.latent_cortex.threat_model import (
    REQUIRED_THREAT_IDS,
    THREATS,
    MitigationCheck,
    ThreatEntry,
    ThreatModelError,
    validate_threat_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_validates_against_this_checkout() -> None:
    receipt = validate_threat_model(REPO_ROOT)
    assert receipt["threat_count"] == len(REQUIRED_THREAT_IDS) == 12
    assert receipt["check_count"] >= 2 * len(REQUIRED_THREAT_IDS)
    body = {key: value for key, value in receipt.items() if key != "registry_sha256"}
    assert receipt["registry_sha256"] == canonical_sha256(body)


def test_every_required_threat_is_covered_with_proofs_and_residue() -> None:
    by_id = {entry.threat_id: entry for entry in THREATS}
    assert set(by_id) == set(REQUIRED_THREAT_IDS)
    for entry in THREATS:
        assert len(entry.checks) >= 2, entry.threat_id
        assert entry.residual_risk, entry.threat_id


def test_bound_checks_resolve_to_real_test_functions() -> None:
    for entry in THREATS:
        for check in entry.checks:
            text = (REPO_ROOT / check.test_file).read_text(encoding="utf-8")
            assert f"def {check.test_name}(" in text, (
                entry.threat_id,
                check.test_file,
                check.test_name,
            )


def test_missing_mitigation_module_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ThreatModelError) as error:
        validate_threat_model(tmp_path)
    assert error.value.code == "threat_model_mitigation_missing"


def test_phantom_check_fails_closed(tmp_path: Path) -> None:
    entry = THREATS[0]
    for module in entry.mitigations:
        target = tmp_path / module
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# stub\n")
    check = entry.checks[0]
    test_target = tmp_path / check.test_file
    test_target.parent.mkdir(parents=True, exist_ok=True)
    test_target.write_text("def test_something_else():\n    pass\n")
    with pytest.raises(ThreatModelError):
        validate_threat_model(tmp_path)


def test_entry_contracts_fail_closed() -> None:
    with pytest.raises(ThreatModelError):
        MitigationCheck("not_tests/file.py", "test_x")
    with pytest.raises(ThreatModelError):
        MitigationCheck("tests/test_x.py", "not_a_test")
    with pytest.raises(ThreatModelError):
        ThreatEntry(
            threat_id="x",
            name="x",
            failure_mode="too short",
            mitigations=("core/x.py",),
            checks=(MitigationCheck("tests/test_x.py", "test_x"),),
            residual_risk="also too short....",
        )
