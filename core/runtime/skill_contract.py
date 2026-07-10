"""SkillContract + SkillVerifier framework.

Every Aura skill, per the audit, must declare:

    name, version, inputs, outputs, preconditions, postconditions,
    required_tools, required_permissions, timeout_seconds, retry_policy,
    rollback_supported, verifier, benchmark, memory_policy,
    autonomy_level_required

Skill execution must yield a typed ``SkillExecutionResult`` whose status
distinguishes success_verified / success_unverified / partial_success /
failed_recoverable / failed_fatal / blocked_by_policy / needs_human_approval.

A skill without a registered verifier is recorded as ``unverified`` and is
flagged by the conformance suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence


class SkillStatus(str, Enum):
    SUCCESS_VERIFIED = "success_verified"
    SUCCESS_UNVERIFIED = "success_unverified"
    PARTIAL_SUCCESS = "partial_success"
    FAILED_RECOVERABLE = "failed_recoverable"
    FAILED_FATAL = "failed_fatal"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    NEEDS_HUMAN_APPROVAL = "needs_human_approval"


@dataclass(frozen=True)
class ActionExpectation:
    """Operational acceptance criteria for a consequential action.

    This is the contract that prevents shallow "the action fired" completions
    from being reported as verified success.
    """

    objective: str = ""
    acceptance_criteria: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    required_evidence_present: List[str] = field(default_factory=list)
    user_visible_effect: Optional[str] = None
    repair_hint: str = ""
    rollback_hint: str = ""
    allow_partial: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "acceptance_criteria": list(self.acceptance_criteria),
            "required_evidence": list(self.required_evidence),
            "required_evidence_present": list(self.required_evidence_present),
            "user_visible_effect": self.user_visible_effect,
            "repair_hint": self.repair_hint,
            "rollback_hint": self.rollback_hint,
            "allow_partial": self.allow_partial,
        }


@dataclass(frozen=True)
class ExpectationVerdict:
    passed: bool
    status: SkillStatus
    satisfied_criteria: List[str] = field(default_factory=list)
    missing_criteria: List[str] = field(default_factory=list)
    present_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    next_step: str = ""

    def to_evidence(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status.value,
            "satisfied_criteria": list(self.satisfied_criteria),
            "missing_criteria": list(self.missing_criteria),
            "present_evidence": list(self.present_evidence),
            "missing_evidence": list(self.missing_evidence),
            "next_step": self.next_step,
        }


@dataclass
class SkillContract:
    name: str
    version: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    retry_policy: str = "none"
    rollback_supported: bool = False
    verifier: Optional[str] = None
    benchmark: Optional[str] = None
    memory_policy: str = "session"
    autonomy_level_required: int = 0


@dataclass
class SkillExecutionResult:
    skill: str
    status: SkillStatus
    output: Any = None
    receipt_id: Optional[str] = None
    verification_evidence: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    expectation: Optional[ActionExpectation] = None

    @property
    def ok(self) -> bool:
        return self.status == SkillStatus.SUCCESS_VERIFIED


class VerifierMissing(RuntimeError):
    pass  # no-op: intentional


def _normalize_label(value: Any) -> str:
    return str(value).strip().casefold()


def _truthy_path(mapping: Dict[str, Any], key: str) -> bool:
    found, current = _path_value(mapping, key)
    return bool(found and current)


def _path_value(mapping: Dict[str, Any], key: str) -> tuple[bool, Any]:
    current: Any = mapping
    for part in str(key).split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _criteria_claims(evidence: Dict[str, Any]) -> set[str]:
    claims: set[str] = set()
    for key in ("satisfied_criteria", "criteria_satisfied"):
        raw = evidence.get(key)
        if isinstance(raw, (list, tuple, set)):
            claims.update(_normalize_label(item) for item in raw)
    for key in ("criteria", "criteria_results"):
        raw = evidence.get(key)
        if isinstance(raw, dict):
            claims.update(_normalize_label(item) for item, passed in raw.items() if passed)
    return {claim for claim in claims if claim}


def evaluate_action_expectation(result: SkillExecutionResult) -> Optional[ExpectationVerdict]:
    expectation = result.expectation
    if expectation is None:
        return None

    evidence = dict(result.verification_evidence or {})
    if isinstance(result.output, dict):
        combined = {**result.output, **evidence}
    else:
        combined = evidence

    claims = _criteria_claims(combined)
    satisfied: List[str] = []
    missing_criteria: List[str] = []
    for criterion in expectation.acceptance_criteria:
        label = _normalize_label(criterion)
        if label in claims or _truthy_path(combined, str(criterion)):
            satisfied.append(str(criterion))
        else:
            missing_criteria.append(str(criterion))

    present_evidence: List[str] = []
    missing_evidence: List[str] = []
    for key in expectation.required_evidence:
        if _truthy_path(combined, key):
            present_evidence.append(key)
        else:
            missing_evidence.append(key)

    for key in expectation.required_evidence_present:
        found, _value = _path_value(combined, key)
        if found:
            present_evidence.append(key)
        else:
            missing_evidence.append(key)

    if expectation.user_visible_effect:
        visible_proven = bool(
            combined.get("user_visible_effect")
            or combined.get("effect_verified")
            or combined.get("visible_effect_verified")
        )
        if visible_proven:
            present_evidence.append("user_visible_effect")
        else:
            missing_criteria.append(f"user-visible effect: {expectation.user_visible_effect}")

    passed = not missing_criteria and not missing_evidence
    if passed:
        status = result.status
        next_step = ""
    elif missing_criteria:
        status = SkillStatus.PARTIAL_SUCCESS if expectation.allow_partial else SkillStatus.FAILED_RECOVERABLE
        next_step = expectation.repair_hint or "repair_missing_acceptance_criteria"
    else:
        status = SkillStatus.SUCCESS_UNVERIFIED
        next_step = expectation.repair_hint or "collect_missing_verification_evidence"

    return ExpectationVerdict(
        passed=passed,
        status=status,
        satisfied_criteria=satisfied,
        missing_criteria=missing_criteria,
        present_evidence=present_evidence,
        missing_evidence=missing_evidence,
        next_step=next_step,
    )


def apply_action_expectation(result: SkillExecutionResult) -> SkillExecutionResult:
    verdict = evaluate_action_expectation(result)
    if verdict is None:
        return result

    evidence = dict(result.verification_evidence or {})
    if result.expectation is not None:
        evidence["action_expectation"] = result.expectation.to_dict()
    evidence["expectation_verdict"] = verdict.to_evidence()

    if result.status not in {SkillStatus.SUCCESS_VERIFIED, SkillStatus.SUCCESS_UNVERIFIED}:
        return replace(result, verification_evidence=evidence)
    if verdict.passed and verdict.status == result.status:
        return replace(result, verification_evidence=evidence)

    missing_parts = verdict.missing_criteria or verdict.missing_evidence
    failure_reason = result.failure_reason
    if missing_parts:
        failure_reason = (
            "expectation incomplete: " + "; ".join(missing_parts)
            if not failure_reason
            else f"{failure_reason}; expectation incomplete: " + "; ".join(missing_parts)
        )

    return replace(
        result,
        status=verdict.status,
        verification_evidence=evidence,
        failure_reason=failure_reason,
    )


def apply_action_expectation_payload(
    skill: str,
    payload: Dict[str, Any],
    expectation: Optional[ActionExpectation],
) -> Dict[str, Any]:
    """Apply one expectation to a dictionary result without emitting receipts.

    Runtime boundaries can share the exact same status/evidence semantics and
    remain responsible for emitting their own domain-specific receipt chain.
    """
    if expectation is None or not isinstance(payload, dict) or not payload.get("ok", True):
        return payload

    raw_status = str(payload.get("status") or "").strip()
    try:
        status = SkillStatus(raw_status) if raw_status else SkillStatus.SUCCESS_VERIFIED
    except ValueError:
        status = SkillStatus.SUCCESS_VERIFIED
    evidence = payload.get("verification_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    checked = apply_action_expectation(
        SkillExecutionResult(
            skill=skill,
            status=status,
            output=payload,
            receipt_id=str(payload.get("receipt_id") or "") or None,
            verification_evidence=evidence,
            expectation=expectation,
        )
    )
    result = dict(payload)
    result["verification_evidence"] = checked.verification_evidence
    result["expectation_verdict"] = dict(
        checked.verification_evidence.get("expectation_verdict") or {}
    )
    result["action_expectation"] = expectation.to_dict()
    result["status"] = checked.status.value
    result["ok"] = checked.ok
    if not checked.ok and checked.failure_reason and not result.get("error"):
        result["error"] = checked.failure_reason
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    def __init__(self):
        self._contracts: Dict[str, SkillContract] = {}
        self._verifiers: Dict[str, Callable[[SkillExecutionResult], SkillExecutionResult]] = {}

    def register(self, contract: SkillContract) -> None:
        self._contracts[contract.name] = contract

    def register_verifier(
        self,
        name: str,
        verifier: Callable[[SkillExecutionResult], SkillExecutionResult],
    ) -> None:
        self._verifiers[name] = verifier

    def get(self, name: str) -> Optional[SkillContract]:
        return self._contracts.get(name)

    def all(self) -> Sequence[SkillContract]:
        return list(self._contracts.values())

    def verify(self, result: SkillExecutionResult) -> SkillExecutionResult:
        verifier = self._verifiers.get(result.skill)
        if verifier is None:
            if result.status == SkillStatus.SUCCESS_VERIFIED:
                # Cannot self-verify without a verifier present.
                return apply_action_expectation(SkillExecutionResult(
                    skill=result.skill,
                    status=SkillStatus.SUCCESS_UNVERIFIED,
                    output=result.output,
                    receipt_id=result.receipt_id,
                    verification_evidence=result.verification_evidence,
                    failure_reason="no verifier registered",
                    expectation=result.expectation,
                ))
            return result
        verified = verifier(result)
        if verified.expectation is None and result.expectation is not None:
            verified = replace(verified, expectation=result.expectation)
        return apply_action_expectation(verified)

    def unverified_skills(self) -> List[str]:
        return [name for name in self._contracts if name not in self._verifiers]


_global_skills: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    global _global_skills
    if _global_skills is None:
        _global_skills = SkillRegistry()
    return _global_skills


def reset_skill_registry() -> None:
    global _global_skills
    _global_skills = None
