"""Machine-readable inventory of cognitive candidate and selection gates.

The inventory separates causal safety boundaries from deterministic selectors,
advisory modifiers, action authority, and certification gates. That distinction
keeps fail-soft salience inputs from being mistaken for permission while making
every fail-closed attention/workspace boundary receipt- and health-visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CognitiveGateSurface:
    surface_id: str
    owner_file: str
    owner_class: str
    callable_name: str
    lane: str
    role: str
    failure_policy: str
    receipt_kind: str | None
    health_service: str | None
    boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


COGNITIVE_GATE_SURFACES: tuple[CognitiveGateSurface, ...] = (
    CognitiveGateSurface(
        "workspace.global_inhibition.submit",
        "core/consciousness/global_workspace.py",
        "GlobalWorkspace",
        "submit",
        "workspace_candidate_admission",
        "safety_boundary",
        "fail_closed",
        "workspace_gate",
        "global_workspace",
        "CTX2-GATE-001/002",
    ),
    CognitiveGateSurface(
        "workspace.global_inhibition.revalidate",
        "core/consciousness/global_workspace.py",
        "GlobalWorkspace",
        "run_competition",
        "workspace_candidate_admission",
        "safety_boundary",
        "fail_closed",
        "workspace_gate",
        "global_workspace",
        "CTX2-GATE-002",
    ),
    CognitiveGateSurface(
        "attention.focus_rigidity",
        "core/consciousness/attention_schema.py",
        "AttentionSchema",
        "set_focus",
        "attention_focus",
        "safety_boundary",
        "fail_closed",
        "workspace_gate",
        "attention_schema",
        "CTX2-GATE-002",
    ),
    CognitiveGateSurface(
        "attention.message_context",
        "core/consciousness/attention_gate.py",
        "AttentionGate",
        "gate_context",
        "conversation_context",
        "selection_boundary",
        "reject_callback_failure",
        None,
        "attention_gate",
        "bounded prompt selection; identity and current user context are protected",
    ),
    CognitiveGateSurface(
        "attention.prompt_block",
        "core/brain/llm/context_gate.py",
        "AttentionalContextGate",
        "should_include_block",
        "conversation_context",
        "selection_boundary",
        "reject_callback_failure",
        None,
        None,
        "bounded prompt selection; essential blocks are explicit policy",
    ),
    CognitiveGateSurface(
        "phenomenal.coalition_ignition",
        "core/phenomenal_substrate/global_workspace.py",
        "GlobalWorkspace",
        "compete",
        "phenomenal_simulation",
        "pure_selection",
        "empty_input_rejects",
        None,
        None,
        "deterministic inner-model calculation; not live workspace authority",
    ),
    CognitiveGateSurface(
        "drafts.generate",
        "core/consciousness/multiple_drafts.py",
        "MultipleDraftsEngine",
        "submit_input",
        "interpretation_drafts",
        "advisory_generation",
        "modifier_failure_lowers_evidence",
        None,
        None,
        "neural-mesh data is advisory and cannot authorize an effect",
    ),
    CognitiveGateSurface(
        "drafts.probe",
        "core/consciousness/multiple_drafts.py",
        "MultipleDraftsEngine",
        "probe",
        "interpretation_drafts",
        "pure_selection",
        "empty_input_rejects",
        None,
        None,
        "deterministic draft selection; not action authority",
    ),
    CognitiveGateSurface(
        "animal.quorum",
        "core/consciousness/animal_cognition.py",
        "QuorumDecisionGate",
        "check_quorum",
        "advisory_decision",
        "pure_selection",
        "empty_or_zero_vote_rejects",
        None,
        None,
        "weighted quorum only; callers still require canonical action authority",
    ),
    CognitiveGateSurface(
        "somatic.marker",
        "core/consciousness/somatic_marker_gate.py",
        "SomaticMarkerGate",
        "evaluate",
        "action_signal",
        "advisory_modifier",
        "modifier_failure_is_neutral",
        None,
        None,
        "somatic evidence cannot authorize an effect by itself",
    ),
    CognitiveGateSurface(
        "legacy.executive_inhibitor",
        "core/consciousness/executive_inhibitor.py",
        "ExecutiveInhibitor",
        "authorize",
        "legacy_action_authority",
        "action_authority",
        "tracked_authority_debt",
        None,
        None,
        "no production caller; migration belongs to canonical authority/effect spine",
    ),
    CognitiveGateSurface(
        "substrate.action_authority",
        "core/consciousness/substrate_authority.py",
        "SubstrateAuthority",
        "authorize",
        "action_authority",
        "action_authority",
        "tracked_authority_debt",
        None,
        "substrate_authority",
        "action-effect authority is governed by the canonical authority/effect workstream",
    ),
    CognitiveGateSurface(
        "caa.readiness",
        "core/consciousness/caa/readiness_gate.py",
        "ReadinessGate",
        "evaluate",
        "certification",
        "certification_gate",
        "validator_failure_cannot_reach_production",
        None,
        None,
        "CAA proof readiness; not candidate admission",
    ),
)


def inventory_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "requirement": "CTX2-GATE-002",
        "surfaces": [surface.to_dict() for surface in COGNITIVE_GATE_SURFACES],
    }


def validate_inventory_contract() -> list[str]:
    issues: list[str] = []
    identifiers = [surface.surface_id for surface in COGNITIVE_GATE_SURFACES]
    if len(identifiers) != len(set(identifiers)):
        issues.append("duplicate cognitive gate surface_id")
    for surface in COGNITIVE_GATE_SURFACES:
        if surface.role == "safety_boundary":
            if surface.failure_policy != "fail_closed":
                issues.append(f"{surface.surface_id}: safety boundary is not fail_closed")
            if not surface.receipt_kind:
                issues.append(f"{surface.surface_id}: safety boundary lacks receipts")
            if not surface.health_service:
                issues.append(f"{surface.surface_id}: safety boundary lacks lane health")
    return issues
