"""Canonical structural fingerprints for latent branch independence.

Surface text is deliberately absent from this measurement. Two candidates
that differ only in wording receive the same fingerprint when their admitted
premises, dependency topology, executable algorithms, state-transition shape,
predicted consequences, and failure conditions are the same.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.cognitive_operators import (
    CognitiveOperator,
    validate_operator_receipt,
)

STRUCTURAL_DIVERSITY_SCHEMA = "aura.rlc.structural_diversity.v1"
_FACETS = (
    "premises",
    "dependencies",
    "algorithms",
    "intermediate_states",
    "predictions",
    "failure_conditions",
)


@dataclass(frozen=True, slots=True)
class OperatorStructure:
    premise_modes: tuple[str, ...]
    dependencies: tuple[str, ...]
    intermediate_pattern: str
    predictions: tuple[str, ...]
    failure_conditions: tuple[str, ...]


_STRUCTURE: dict[CognitiveOperator, OperatorStructure] = {
    CognitiveOperator.DIRECT_DERIVATION: OperatorStructure(
        ("current_state", "action_control"),
        ("action_control->communication_slot",),
        "single_target_transition",
        ("communication_slot_moves_toward_control",),
        ("communication_slot_unavailable", "noncausal_control_write"),
    ),
    CognitiveOperator.CONSTRUCTIVE_SOLUTION: OperatorStructure(
        ("current_state", "fixed_anchor", "action_control"),
        ("anchor+action_control->each_mutable_slot", "earlier_stage->later_stage"),
        "progressive_multislot_scaffold",
        ("all_mutable_slots_form_ordered_scaffold",),
        ("mutable_workspace_empty", "scaffold_stage_unchanged"),
    ),
    CognitiveOperator.COUNTEREXAMPLE: OperatorStructure(
        ("current_hypothesis", "fixed_anchor", "action_control"),
        ("hypothesis_delta+anchor+action_control->terminal_mutable_slot",),
        "signed_hypothesis_reversal",
        ("terminal_hypothesis_delta_reverses",),
        ("counterexample_slot_unavailable", "hypothesis_reversal_noncausal"),
    ),
    CognitiveOperator.INVERSE_REASONING: OperatorStructure(
        ("current_state", "action_control", "terminal_state"),
        ("later_slot->earlier_slot", "action_control->each_mutable_slot"),
        "reverse_slot_transport",
        ("terminal_structure_propagates_toward_initial_slots",),
        ("reverse_path_empty", "slot_transport_noncausal"),
    ),
    CognitiveOperator.CAUSAL_SIMULATION: OperatorStructure(
        ("current_state", "fixed_anchor", "prior_state", "action_control"),
        ("prior_slot+current_slot->next_delta", "next_delta->current_slot"),
        "finite_difference_rollout",
        ("local_state_velocity_advances_one_step",),
        ("causal_predecessor_missing", "finite_difference_noncausal"),
    ),
    CognitiveOperator.FORMALIZATION: OperatorStructure(
        ("current_state", "fixed_anchor", "action_control_axis"),
        ("current_slot+control_axis->projection", "projection+anchor->formal_slot"),
        "control_axis_projection",
        ("mutable_state_is_expressed_on_control_axis",),
        ("control_axis_degenerate", "projection_noncausal"),
    ),
    CognitiveOperator.ANALOGY_MAPPING: OperatorStructure(
        ("source_relation", "target_anchor", "action_control"),
        ("paired_slot-anchor->relation", "relation+target_anchor->target_slot"),
        "paired_relation_transport",
        ("source_relation_is_transferred_to_paired_target",),
        ("analogy_pair_unavailable", "relation_transport_noncausal"),
    ),
    CognitiveOperator.ASSUMPTION_REMOVAL: OperatorStructure(
        ("current_state", "fixed_anchor", "action_control_axis"),
        ("all_slots->max_alignment", "max_alignment->selected_slot_subtraction"),
        "max_alignment_subtraction",
        ("strongest_control_aligned_component_is_removed",),
        ("aligned_component_absent", "assumption_subtraction_noncausal"),
    ),
    CognitiveOperator.BOUNDARY_CASE: OperatorStructure(
        ("initial_boundary", "terminal_boundary", "fixed_anchor", "action_control"),
        ("boundary_delta+signed_control->boundary_endpoint",),
        "signed_boundary_extrapolation",
        ("opposite_boundary_directions_are_extrapolated",),
        ("boundary_endpoint_unavailable", "boundary_extrapolation_noncausal"),
    ),
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _trend(before: Any, after: Any) -> str:
    if (
        isinstance(before, bool)
        or isinstance(after, bool)
        or not isinstance(before, (int, float))
        or not isinstance(after, (int, float))
        or not math.isfinite(float(before))
        or not math.isfinite(float(after))
    ):
        return "unmeasured"
    delta = float(after) - float(before)
    if abs(delta) <= 1e-9:
        return "stable"
    return "increased" if delta > 0.0 else "decreased"


def _premise_commitments(cognitive_slots: Any) -> list[str]:
    if not isinstance(cognitive_slots, list):
        raise ValueError("cognitive slot premises must be a list")
    premises: list[str] = []
    for row in cognitive_slots:
        if not isinstance(row, dict):
            raise ValueError("cognitive slot premise is invalid")
        slot = row.get("slot")
        source = row.get("source")
        digest = row.get("text_sha256")
        if (
            type(slot) is not int
            or slot < 0
            or not isinstance(source, str)
            or not source
            or not _is_sha256(digest)
        ):
            raise ValueError("cognitive slot premise is incomplete")
        premises.append(f"slot:{slot}:source:{source}:content:{digest}")
    return sorted(set(premises))


def _action_observations(action_trace: Any) -> dict[int, dict[str, str]]:
    if not isinstance(action_trace, list):
        raise ValueError("cognitive action trace must be a list")
    observations: dict[int, dict[str, str]] = {}
    previous_step = -1
    for row in action_trace:
        if not isinstance(row, dict):
            raise ValueError("cognitive action row is invalid")
        transition = row.get("transition")
        before = row.get("state_before")
        after = row.get("state_after")
        step_index = transition.get("step_index") if isinstance(transition, dict) else None
        if (
            not isinstance(transition, dict)
            or type(step_index) is not int
            or step_index <= previous_step
            or not isinstance(transition.get("action"), str)
            or not isinstance(transition.get("outcome"), str)
            or not isinstance(before, dict)
            or not isinstance(after, dict)
        ):
            raise ValueError("cognitive action evidence is incomplete")
        previous_step = step_index
        observations[step_index] = {
            "action": transition["action"],
            "outcome": transition["outcome"],
            "residual_trend": _trend(before.get("residual"), after.get("residual")),
            "disagreement_trend": _trend(
                before.get("disagreement"), after.get("disagreement")
            ),
        }
    return observations


def _candidate_by_branch(branch_isolation: Any, n_branches: int) -> dict[int, str]:
    if not isinstance(branch_isolation, dict):
        raise ValueError("branch isolation evidence is missing")
    candidates = branch_isolation.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != n_branches:
        raise ValueError("branch candidate evidence has wrong cardinality")
    result: dict[int, str] = {}
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError("branch candidate evidence is invalid")
        index = row.get("index")
        digest = row.get("candidate_sha256")
        if (
            type(index) is not int
            or not 0 <= index < n_branches
            or index in result
            or not _is_sha256(digest)
        ):
            raise ValueError("branch candidate commitment is invalid")
        result[index] = digest
    return result


def _observed_candidates_by_branch(
    branch_isolation: dict[str, Any], n_branches: int
) -> dict[int, str]:
    """Return only well-formed candidate commitments from uncertified evidence.

    An incomplete isolation interval is an expected bounded-run outcome, not a
    malformed episode.  Its partial commitments remain diagnostic, but they
    cannot satisfy the positive structural-diversity certificate.
    """

    candidates = branch_isolation.get("candidates")
    if not isinstance(candidates, list):
        return {}
    result: dict[int, str] = {}
    for row in candidates:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        digest = row.get("candidate_sha256")
        if (
            type(index) is int
            and 0 <= index < n_branches
            and index not in result
            and _is_sha256(digest)
        ):
            result[index] = digest
    return result


def build_structural_diversity_receipt(
    *,
    n_branches: int,
    cognitive_slots: Any,
    operator_trace: Any,
    action_trace: Any,
    branch_isolation: Any,
) -> dict[str, Any]:
    """Build a wording-independent, service-reconstructible diversity proof."""

    if type(n_branches) is not int or not 1 <= n_branches <= 64:
        raise ValueError("structural branch cardinality is invalid")
    if not isinstance(operator_trace, list) or not operator_trace:
        raise ValueError("operator trace is required for structural measurement")
    premises = _premise_commitments(cognitive_slots)
    observations = _action_observations(action_trace)
    if not isinstance(branch_isolation, dict):
        raise ValueError("branch isolation evidence is missing")
    isolation_certified = branch_isolation.get("certified") is True
    candidates = (
        _candidate_by_branch(branch_isolation, n_branches)
        if isolation_certified
        else _observed_candidates_by_branch(branch_isolation, n_branches)
    )
    candidate_commitments_complete = len(candidates) == n_branches
    by_branch: dict[int, list[dict[str, Any]]] = {index: [] for index in range(n_branches)}
    for raw in operator_trace:
        row = validate_operator_receipt(raw)
        index = row["branch_index"]
        if index not in by_branch:
            raise ValueError("operator receipt names an unknown branch")
        observation = observations.get(row["action_step"])
        if observation is None or observation["action"] != row["action"]:
            raise ValueError("operator receipt is not bound to its action transition")
        by_branch[index].append(row)

    branches: list[dict[str, Any]] = []
    for index in range(n_branches):
        rows = sorted(
            by_branch[index], key=lambda row: (row["action_step"], row["receipt_sha256"])
        )
        if not rows:
            raise ValueError("every branch needs an executable structural trace")
        role_path = [row["role"] for row in rows]
        operator_path = [row["operator"] for row in rows]
        facets: dict[str, list[str]] = {
            "premises": sorted(
                set(premises)
                | {
                    f"step:{row['action_step']}:usage:{mode}"
                    for row in rows
                    for mode in _STRUCTURE[CognitiveOperator(row["operator"])].premise_modes
                }
                | {
                    f"protected_slot:{slot}"
                    for row in rows
                    for slot in row["protected_slots"]
                }
            ),
            "dependencies": sorted(
                {
                    f"step:{row['action_step']}:template:{item}"
                    for row in rows
                    for item in _STRUCTURE[CognitiveOperator(row["operator"])].dependencies
                }
                | {
                    f"step:{row['action_step']}:changed_slots:"
                    + ",".join(str(slot) for slot in row["changed_slots"])
                    for row in rows
                }
            ),
            "algorithms": sorted(
                {
                    f"step:{row['action_step']}:action:{row['action']}:"
                    f"operator:{row['operator']}:transform:{row['transform']}"
                    for row in rows
                }
            ),
            "intermediate_states": sorted(
                {
                    f"step:{row['action_step']}:pattern:"
                    f"{_STRUCTURE[CognitiveOperator(row['operator'])].intermediate_pattern}:"
                    f"changed_count:{len(row['changed_slots'])}:"
                    f"residual:{observations[row['action_step']]['residual_trend']}:"
                    f"disagreement:{observations[row['action_step']]['disagreement_trend']}"
                    for row in rows
                }
            ),
            "predictions": sorted(
                {
                    f"step:{row['action_step']}:{prediction}"
                    for row in rows
                    for prediction in _STRUCTURE[
                        CognitiveOperator(row["operator"])
                    ].predictions
                }
            ),
            "failure_conditions": sorted(
                {
                    f"step:{row['action_step']}:program:{condition}"
                    for row in rows
                    for condition in _STRUCTURE[
                        CognitiveOperator(row["operator"])
                    ].failure_conditions
                }
                | {
                    f"step:{row['action_step']}:outcome:"
                    f"{observations[row['action_step']]['outcome']}"
                    for row in rows
                }
            ),
        }
        facet_sha256 = {name: _canonical_sha256(facets[name]) for name in _FACETS}
        structural_sha256 = _canonical_sha256(facet_sha256)
        branches.append(
            {
                "index": index,
                "role_path": role_path,
                "operator_path": operator_path,
                "facets": facets,
                "facet_sha256": facet_sha256,
                "structural_sha256": structural_sha256,
                "state_commitment_sha256": _canonical_sha256(
                    {
                        "candidate": candidates.get(index),
                        "candidate_available": index in candidates,
                        "outputs": [row["output_sha256"] for row in rows],
                    }
                ),
            }
        )

    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(branches):
        for right in branches[left_index + 1 :]:
            differing = [
                facet
                for facet in _FACETS
                if left["facet_sha256"][facet] != right["facet_sha256"][facet]
            ]
            equal = [facet for facet in _FACETS if facet not in differing]
            independent = (
                len(differing) >= 3
                and "dependencies" in differing
                and "algorithms" in differing
                and (
                    "intermediate_states" in differing
                    or "predictions" in differing
                    or "failure_conditions" in differing
                )
            )
            pairwise.append(
                {
                    "left": left["index"],
                    "right": right["index"],
                    "differing_facets": differing,
                    "equal_facets": equal,
                    "distance": round(len(differing) / len(_FACETS), 6),
                    "independent": independent,
                    "reason": (
                        "causal_structure_differs"
                        if independent
                        else "same_or_insufficiently_distinct_causal_structure"
                    ),
                }
            )

    groups_by_fingerprint: dict[str, list[int]] = {}
    for branch in branches:
        groups_by_fingerprint.setdefault(branch["structural_sha256"], []).append(
            branch["index"]
        )
    support_classes = sorted(groups_by_fingerprint.values(), key=lambda group: group[0])
    duplicate_groups = [group for group in support_classes if len(group) > 1]
    structural_independence_observed = not duplicate_groups and all(
        row["independent"] for row in pairwise
    )
    certified = (
        isolation_certified
        and candidate_commitments_complete
        and structural_independence_observed
    )
    payload = {
        "schema": STRUCTURAL_DIVERSITY_SCHEMA,
        "facet_names": list(_FACETS),
        "wording_counted": False,
        "n_branches": n_branches,
        "branch_isolation_sha256": _canonical_sha256(branch_isolation),
        "branch_isolation_certified": isolation_certified,
        "candidate_commitments_complete": candidate_commitments_complete,
        "structural_independence_observed": structural_independence_observed,
        "branches": branches,
        "pairwise": pairwise,
        "support_classes": support_classes,
        "independent_support_count": len(support_classes),
        "duplicate_groups": duplicate_groups,
        "certified": certified,
        "reason": (
            "certified_structurally_independent"
            if certified
            else "branch_isolation_unproven"
            if not isolation_certified
            else "candidate_commitments_incomplete"
            if not candidate_commitments_complete
            else "duplicate_or_insufficiently_distinct_structure"
        ),
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def validate_structural_diversity_receipt(
    value: Any,
    *,
    n_branches: int,
    cognitive_slots: Any,
    operator_trace: Any,
    action_trace: Any,
    branch_isolation: Any,
) -> dict[str, Any]:
    """Reconstruct the receipt from primary traces and require exact equality."""

    if not isinstance(value, dict):
        raise ValueError("structural diversity receipt is missing")
    expected = build_structural_diversity_receipt(
        n_branches=n_branches,
        cognitive_slots=cognitive_slots,
        operator_trace=operator_trace,
        action_trace=action_trace,
        branch_isolation=branch_isolation,
    )
    if value != expected:
        raise ValueError("structural diversity receipt differs from reconstruction")
    if value.get("wording_counted") is not False:
        raise ValueError("surface wording must not count as structural evidence")
    if value.get("certified") is not True:
        raise ValueError("branch structure is not independently certified")
    return dict(value)


__all__ = [
    "STRUCTURAL_DIVERSITY_SCHEMA",
    "build_structural_diversity_receipt",
    "validate_structural_diversity_receipt",
]
