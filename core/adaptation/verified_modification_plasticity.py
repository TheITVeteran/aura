"""Proof-gated handoff from successful code repair to parametric learning.

This module never edits base weights.  It records a durable, verifier-backed
training candidate and may enqueue it with the canonical LoRA owner.  Adapter
training and promotion remain separate governed stages with their own
behavioral validation and rollback contracts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from core.learning.synthetic_data_flywheel import SyntheticDataFlywheel, VerifiedTrace
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root


@dataclass(frozen=True)
class VerifiedModificationEvidence:
    change_id: str
    objective: str
    verified_solution: str
    risk_tier: int
    harness_passed: bool
    behavioral_validation_passed: bool
    retention_validation_passed: bool
    rollback_passed: bool
    governance_receipt_id: str
    verifier: str = "safe_modification_harness"
    score: float = 1.0

    @property
    def eligible(self) -> bool:
        return (
            bool(self.change_id and self.objective and self.verified_solution)
            and self.risk_tier <= 1
            and self.harness_passed
            and self.behavioral_validation_passed
            and self.retention_validation_passed
            and self.rollback_passed
            and bool(self.governance_receipt_id)
            and self.score >= 0.8
        )


@dataclass(frozen=True)
class PlasticityHandoffReceipt:
    change_id: str
    status: str
    reason: str
    created_at: float
    trace_path: str = ""
    lora_status: str = "not_requested"


class VerifiedModificationPlasticityBridge:
    """Creates training candidates only from fully verified low-risk changes."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(
            root or state_root() / "data" / "learning" / "verified_modifications"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipt_path = self.root / "handoffs.jsonl"
        self.flywheel = SyntheticDataFlywheel(self.root / "traces")

    async def handoff(
        self,
        evidence: VerifiedModificationEvidence,
        *,
        enqueue_lora: bool = True,
    ) -> PlasticityHandoffReceipt:
        if not evidence.eligible:
            return self._record(
                PlasticityHandoffReceipt(
                    change_id=evidence.change_id,
                    status="rejected",
                    reason=self._rejection_reason(evidence),
                    created_at=time.time(),
                )
            )

        trace = VerifiedTrace(
            trace_id=evidence.change_id,
            task_type="verified_self_modification",
            prompt=evidence.objective,
            response=evidence.verified_solution,
            verifier=evidence.verifier,
            score=evidence.score,
            risk_tier=f"tier{evidence.risk_tier}",
            metadata={
                "governance_receipt_id": evidence.governance_receipt_id,
                "behavioral_validation_passed": True,
                "retention_validation_passed": True,
                "rollback_passed": True,
            },
        )
        trace_path = self.flywheel.write_jsonl(
            [trace], self.root / "traces" / f"{evidence.change_id}.jsonl"
        )
        lora_status = "not_requested"
        if enqueue_lora:
            from core.adaptation.online_lora_governor import get_online_lora_governor

            lora_receipt = await get_online_lora_governor().maybe_update_from_reflection(
                evidence.verified_solution,
                conversation_context=evidence.objective,
                will_receipt_id=evidence.governance_receipt_id,
            )
            lora_status = lora_receipt.status

        return self._record(
            PlasticityHandoffReceipt(
                change_id=evidence.change_id,
                status="queued_for_parametric_validation",
                reason="verified trace recorded; adapter owner retains training and promotion authority",
                created_at=time.time(),
                trace_path=str(trace_path),
                lora_status=lora_status,
            )
        )

    def _record(self, receipt: PlasticityHandoffReceipt) -> PlasticityHandoffReceipt:
        get_file_write_gateway().append_text(
            self.receipt_path,
            json.dumps(asdict(receipt), sort_keys=True) + "\n",
            encoding="utf-8",
            source="adaptation.verified_modification_plasticity.receipt",
        )
        return receipt

    @staticmethod
    def _rejection_reason(evidence: VerifiedModificationEvidence) -> str:
        missing: list[str] = []
        for name in (
            "harness_passed",
            "behavioral_validation_passed",
            "retention_validation_passed",
            "rollback_passed",
        ):
            if not getattr(evidence, name):
                missing.append(name)
        if evidence.risk_tier > 1:
            missing.append("risk_tier_not_trainable")
        if not evidence.governance_receipt_id:
            missing.append("governance_receipt_id")
        if evidence.score < 0.8:
            missing.append("minimum_verifier_score")
        return "missing_or_failed:" + ",".join(missing or ["required_content"])


__all__ = [
    "PlasticityHandoffReceipt",
    "VerifiedModificationEvidence",
    "VerifiedModificationPlasticityBridge",
]
