"""research/meta_learning_loop.py — Recursive Self-Improvement (RSI) Lab
=======================================================================
This implements the overarching "Outer Loop" of Aura's Phase 22 architecture.
Rather than modifying herself directly and risking regression, this engine:
1. Receives candidate artifacts (skills, heuristics, parameter changes).
2. Evaluates them against simulation environments or objective heuristics.
3. Gates them through a Promotion Protocol, generating PR-ready metadata.

This satisfies Phase 22.10.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.RSILab")

RSI_LAB_IO_ERRORS = (OSError, TypeError, ValueError)

@dataclass
class CandidateArtifact:
    """An artifact proposed for core integration."""
    id: str
    artifact_type: str  # 'heuristic', 'skill', 'prompt_tweak'
    content: Any
    rationale: str
    status: str = 'pending_eval'  # pending_eval, passed, failed, promoted
    score: float = 0.0
    evaluation_report: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

class RSILab:
    """The laboratory for safe Recursive Self-Improvement."""
    name = "rsi_lab"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        from core.config import config
        # Store RSI experiments separately from operational data
        self.lab_dir = config.paths.data_dir / "rsi_lab"
        self.lab_dir.mkdir(parents=True, exist_ok=True)
        
        self.candidates: Dict[str, CandidateArtifact] = {}
        self._load()

    def submit_candidate(self, artifact_type: str, content: Any, rationale: str) -> str:
        """Submit a new artifact for evaluation."""
        candidate_id = f"cand_{int(time.time())}_{len(self.candidates)}"
        candidate = CandidateArtifact(
            id=candidate_id,
            artifact_type=artifact_type,
            content=content,
            rationale=rationale
        )
        self.candidates[candidate_id] = candidate
        self._save()
        logger.info(f"🧪 RSI Lab received new {artifact_type} candidate: {candidate_id}")
        return candidate_id

    async def evaluate_pending_candidates(self) -> int:
        """
        Run the evaluation loop. This gates changes from entering the core
        without validation.
        """
        pending = [c for c in self.candidates.values() if c.status == 'pending_eval']
        if not pending:
            return 0
            
        logger.info(f"🧪 Evaluating {len(pending)} pending RSI candidates...")
        evaluated_count = 0
        
        for candidate in pending:
            report = self._evaluate_candidate(candidate)
            candidate.score = float(report["score"])
            candidate.evaluation_report = report
            candidate.status = 'passed' if candidate.score >= 0.72 and report["blocking_failures"] == [] else 'failed'
            logger.info(f"🧪 Candidate {candidate.id} evaluated. Score: {candidate.score:.2f} -> {candidate.status}")
            evaluated_count += 1
            
        self._save()
        return evaluated_count

    def _evaluate_candidate(self, candidate: CandidateArtifact) -> Dict[str, Any]:
        content = candidate.content
        evidence = content.get("evidence", {}) if isinstance(content, dict) else {}
        checks = {
            "typed_artifact": candidate.artifact_type in {"heuristic", "skill", "prompt_tweak", "policy_patch"},
            "rationale_present": len(str(candidate.rationale).strip()) >= 20,
            "provenance_present": bool(evidence.get("provenance") or evidence.get("source_trace")),
            "validation_command_present": bool(evidence.get("validation_command") or evidence.get("commands")),
            "validation_passed": self._validation_passed(evidence),
            "rollback_plan_present": bool(evidence.get("rollback_plan") or evidence.get("revert_plan")),
            "receipt_present": bool(evidence.get("receipt_id") or evidence.get("receipts")),
            "risk_bounded": self._risk_is_bounded(evidence),
            "artifact_contract_valid": self._artifact_contract_valid(candidate),
            "absolute_language_avoided": "always" not in f"{candidate.rationale} {candidate.content}".lower(),
        }
        weights = {
            "typed_artifact": 0.06,
            "rationale_present": 0.08,
            "provenance_present": 0.12,
            "validation_command_present": 0.12,
            "validation_passed": 0.18,
            "rollback_plan_present": 0.10,
            "receipt_present": 0.12,
            "risk_bounded": 0.10,
            "artifact_contract_valid": 0.08,
            "absolute_language_avoided": 0.04,
        }
        score = sum(weight for name, weight in weights.items() if checks[name])
        blocking = [
            name for name in (
                "typed_artifact",
                "validation_command_present",
                "validation_passed",
                "risk_bounded",
                "artifact_contract_valid",
            )
            if not checks[name]
        ]
        return {
            "score": round(score, 4),
            "checks": checks,
            "blocking_failures": blocking,
            "evaluated_at": time.time(),
        }

    @staticmethod
    def _validation_passed(evidence: Dict[str, Any]) -> bool:
        if evidence.get("validation_passed") is True:
            return True
        results = evidence.get("validation_results")
        if isinstance(results, list) and results:
            return all(bool(item.get("passed")) for item in results if isinstance(item, dict))
        return False

    @staticmethod
    def _risk_is_bounded(evidence: Dict[str, Any]) -> bool:
        risk = evidence.get("risk")
        if isinstance(risk, str):
            return risk.lower() in {"low", "bounded", "contained"}
        if isinstance(risk, dict):
            return risk.get("level") in {"low", "bounded", "contained"} and bool(risk.get("blast_radius"))
        return False

    @staticmethod
    def _artifact_contract_valid(candidate: CandidateArtifact) -> bool:
        content = candidate.content
        if candidate.artifact_type == "heuristic":
            text = content.get("rule", "") if isinstance(content, dict) else str(content)
            return 20 <= len(text) <= 2000 and "because" in f"{candidate.rationale} {text}".lower()
        if candidate.artifact_type == "skill":
            return (
                isinstance(content, dict)
                and isinstance(content.get("steps"), list)
                and len(content["steps"]) >= 2
                and bool(content.get("inputs") or content.get("tool_contract"))
            )
        if candidate.artifact_type in {"prompt_tweak", "policy_patch"}:
            return isinstance(content, dict) and bool(content.get("diff") or content.get("patch"))
        return False

    def get_promotable_artifacts(self) -> List[CandidateArtifact]:
        """Fetch candidates that passed evaluation and are ready for promotion."""
        return [c for c in self.candidates.values() if c.status == 'passed']

    def promote(self, candidate_id: str):
        """Mark as promoted. The actual integration is handled by the caller."""
        if candidate_id in self.candidates:
            self.candidates[candidate_id].status = 'promoted'
            self._save()
            logger.info(f"🚀 Candidate {candidate_id} promoted to core!")

    def _save(self):
        try:
            data = {k: asdict(v) for k, v in self.candidates.items()}
            get_file_write_gateway().write_text(
                self.lab_dir / "candidates.json",
                json.dumps(data, indent=4),
                source="research.meta_learning_loop.candidates",
            )
        except RSI_LAB_IO_ERRORS as e:
            logger.error(f"Failed to save RSI Lab candidates: {e}")

    def _load(self):
        file_path = self.lab_dir / "candidates.json"
        if not file_path.exists():
            return
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self.candidates = {k: CandidateArtifact(**v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to load RSI Lab candidates: {e}")

def register_rsi_lab(orchestrator=None):
    from core.container import ServiceContainer
    lab = RSILab(orchestrator)
    ServiceContainer.register_instance("rsi_lab", lab)
    return lab
