from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.adaptation.verified_modification_plasticity import (
    VerifiedModificationEvidence,
    VerifiedModificationPlasticityBridge,
)


def _evidence(**overrides):
    values = {
        "change_id": "change-001",
        "objective": "Correct an off-by-one error without changing prior behavior.",
        "verified_solution": "Use the bounded index and preserve the existing empty-input contract.",
        "risk_tier": 1,
        "harness_passed": True,
        "behavioral_validation_passed": True,
        "retention_validation_passed": True,
        "rollback_passed": True,
        "governance_receipt_id": "will-123",
    }
    values.update(overrides)
    return VerifiedModificationEvidence(**values)


def test_bridge_rejects_training_without_retention_evidence(tmp_path: Path) -> None:
    bridge = VerifiedModificationPlasticityBridge(tmp_path)
    receipt = asyncio.run(
        bridge.handoff(_evidence(retention_validation_passed=False), enqueue_lora=False)
    )
    assert receipt.status == "rejected"
    assert "retention_validation_passed" in receipt.reason
    assert not list((tmp_path / "traces").glob("*.jsonl"))


def test_bridge_records_verified_trace_without_claiming_weight_change(tmp_path: Path) -> None:
    bridge = VerifiedModificationPlasticityBridge(tmp_path)
    receipt = asyncio.run(bridge.handoff(_evidence(), enqueue_lora=False))
    assert receipt.status == "queued_for_parametric_validation"
    assert receipt.lora_status == "not_requested"
    trace = json.loads(Path(receipt.trace_path).read_text(encoding="utf-8").strip())
    assert trace["metadata"]["retention_validation_passed"] is True
    assert trace["risk_tier"] == "tier1"


def test_bridge_refuses_proposal_only_and_sealed_changes(tmp_path: Path) -> None:
    bridge = VerifiedModificationPlasticityBridge(tmp_path)
    for tier in (2, 3):
        receipt = asyncio.run(
            bridge.handoff(
                _evidence(change_id=f"change-{tier}", risk_tier=tier), enqueue_lora=False
            )
        )
        assert receipt.status == "rejected"
        assert "risk_tier_not_trainable" in receipt.reason
