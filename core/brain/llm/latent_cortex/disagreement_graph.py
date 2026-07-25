"""Machine-checkable localization of branch disagreements.

SPARK-047 joins two evidence planes that previously stopped short of diagnosis:

* cognitive-operator receipts identify the exact causal program each branch ran;
* atomic decompositions identify exact, hash-bound decoded claims and declared
  dependencies without publishing private hidden-state reasoning.

The graph finds the longest exact shared prefix in each plane and records the
first differing transition or claim. Exact equality is intentionally strict:
different wording is not promoted to semantic equivalence. This milestone is
diagnostic only; operation selection and repair authority belong to later gates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
    validate_atomic_decomposition_envelope,
)
from core.brain.llm.latent_cortex.cognitive_operators import (
    validate_operator_receipt,
)

DISAGREEMENT_GRAPH_SCHEMA = "aura.rlc.disagreement_graph.v1"
MAX_BRANCHES = 64


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def decompose_branch_candidates(
    candidates: Mapping[int, str],
    *,
    objective: str,
) -> dict[str, dict[str, Any]]:
    """Create source-reconstructed atomic envelopes for contiguous branches."""

    if not isinstance(candidates, Mapping):
        raise TypeError("branch candidates must be a mapping")
    if not isinstance(objective, str):
        raise TypeError("objective must be a string")
    indices = sorted(candidates)
    if indices != list(range(len(indices))) or len(indices) > MAX_BRANCHES:
        raise ValueError("branch candidate indices must be contiguous and bounded")
    result: dict[str, dict[str, Any]] = {}
    for index in indices:
        candidate = candidates[index]
        if not isinstance(candidate, str):
            raise TypeError("branch candidate must be a string")
        result[str(index)] = build_atomic_decomposition(candidate, objective=objective)
    return result


def _action_inventory(action_trace: Any) -> dict[int, dict[str, str]]:
    if not isinstance(action_trace, list):
        raise ValueError("cognitive action trace must be a list")
    inventory: dict[int, dict[str, str]] = {}
    for expected_step, row in enumerate(action_trace):
        if not isinstance(row, Mapping):
            raise ValueError("cognitive action row is invalid")
        transition = row.get("transition")
        if (
            not isinstance(transition, Mapping)
            or transition.get("step_index") != expected_step
            or not isinstance(transition.get("action"), str)
            or not transition["action"]
            or not isinstance(transition.get("outcome"), str)
            or not transition["outcome"]
        ):
            raise ValueError("cognitive action transition is incomplete")
        inventory[expected_step] = {
            "action": str(transition["action"]),
            "outcome": str(transition["outcome"]),
        }
    return inventory


def _operator_timelines(
    *,
    n_branches: int,
    operator_trace: Any,
    actions: Mapping[int, Mapping[str, str]],
) -> dict[int, list[dict[str, Any]]]:
    if not isinstance(operator_trace, list):
        raise ValueError("cognitive operator trace must be a list")
    timelines: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(n_branches)
    }
    seen: set[tuple[int, int]] = set()
    for raw in operator_trace:
        row = validate_operator_receipt(raw)
        branch = row["branch_index"]
        step = row["action_step"]
        if branch not in timelines:
            raise ValueError("operator receipt names an unknown branch")
        if (branch, step) in seen:
            raise ValueError("branch has duplicate operator transition")
        seen.add((branch, step))
        action = actions.get(step)
        if action is None or action["action"] != row["action"]:
            raise ValueError("operator transition is not bound to its action")
        structure = {
            "action_step": step,
            "action": row["action"],
            "outcome": action["outcome"],
            "role": row["role"],
            "operator": row["operator"],
            "transform": row["transform"],
            "changed_slots": list(row["changed_slots"]),
            "protected_slots": list(row["protected_slots"]),
        }
        timelines[branch].append(
            {
                **structure,
                "program_sha256": _canonical_sha256(structure),
                "input_state_sha256": row["input_sha256"],
                "output_state_sha256": row["output_sha256"],
                "operator_receipt_sha256": row["receipt_sha256"],
            }
        )
    for rows in timelines.values():
        rows.sort(key=lambda item: int(item["action_step"]))
    return timelines


def _candidate_inventory(
    *,
    n_branches: int,
    candidate_decompositions: Any,
    blind_review: Any,
) -> tuple[dict[int, dict[str, Any]], str, str]:
    if candidate_decompositions in (None, {}):
        return {}, "", "decoded_candidates_unavailable"
    if not isinstance(candidate_decompositions, Mapping):
        raise ValueError("candidate decompositions must be a mapping")
    expected_keys = {str(index) for index in range(n_branches)}
    if set(candidate_decompositions) != expected_keys:
        raise ValueError("candidate decomposition coverage is incomplete")
    if not isinstance(blind_review, Mapping):
        raise ValueError("candidate decompositions require blind-review binding")
    rows = blind_review.get("rows")
    if not isinstance(rows, list) or len(rows) != n_branches:
        raise ValueError("blind-review candidate binding is incomplete")
    commitments: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("blind-review candidate row is invalid")
        branch = row.get("branch")
        digest = row.get("candidate_sha256")
        if (
            type(branch) is not int
            or branch not in range(n_branches)
            or branch in commitments
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("blind-review candidate commitment is invalid")
        commitments[branch] = digest
    result: dict[int, dict[str, Any]] = {}
    binding_rows: list[dict[str, Any]] = []
    for index in range(n_branches):
        decomposition = validate_atomic_decomposition_envelope(
            candidate_decompositions[str(index)]
        )
        if decomposition["source_sha256"] != commitments[index]:
            raise ValueError("atomic candidate differs from blind-review commitment")
        result[index] = decomposition
        binding_rows.append(
            {
                "branch": index,
                "source_sha256": decomposition["source_sha256"],
                "decomposition_sha256": decomposition["receipt_sha256"],
            }
        )
    return (
        result,
        _canonical_sha256(binding_rows),
        "worker_source_reconstructed_hash_bound_for_service_validation",
    )


def _first_program_divergence(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    shared = 0
    while (
        shared < len(left)
        and shared < len(right)
        and left[shared]["action_step"] == right[shared]["action_step"]
        and left[shared]["program_sha256"] == right[shared]["program_sha256"]
    ):
        shared += 1
    if shared == len(left) == len(right):
        return shared, {
            "available": False,
            "reason": "causal_programs_exactly_equal",
        }
    left_row = left[shared] if shared < len(left) else None
    right_row = right[shared] if shared < len(right) else None
    fields = (
        "action_step",
        "action",
        "outcome",
        "role",
        "operator",
        "transform",
        "changed_slots",
        "protected_slots",
    )
    differing_fields = [
        field
        for field in fields
        if (None if left_row is None else left_row[field])
        != (None if right_row is None else right_row[field])
    ]
    steps = [
        int(row["action_step"])
        for row in (left_row, right_row)
        if row is not None
    ]
    return shared, {
        "available": True,
        "kind": "causal_transition",
        "operator_ordinal": shared,
        "action_step": min(steps),
        "differing_fields": differing_fields,
        "left": left_row,
        "right": right_row,
        "exact_program_comparison": True,
    }


def _atom_signature(atom: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "kind": atom["kind"],
            "text_sha256": atom["text_sha256"],
            "dependency_cues": atom["dependency_cues"],
        }
    )


def _touching_transitions(
    decomposition: Mapping[str, Any],
    atom_id: str | None,
) -> list[dict[str, Any]]:
    if atom_id is None:
        return []
    return [
        dict(row)
        for row in decomposition["transitions"]
        if row["output_id"] == atom_id or atom_id in row["premise_ids"]
    ]


def _atom_summary(atom: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if atom is None:
        return None
    return {
        "atom_id": atom["atom_id"],
        "kind": atom["kind"],
        "start": atom["start"],
        "end": atom["end"],
        "text_sha256": atom["text_sha256"],
        "dependency_cues": list(atom["dependency_cues"]),
        "atom_sha256": atom["atom_sha256"],
    }


def _first_candidate_divergence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    left_atoms = left["atoms"]
    right_atoms = right["atoms"]
    shared = 0
    while (
        shared < len(left_atoms)
        and shared < len(right_atoms)
        and _atom_signature(left_atoms[shared]) == _atom_signature(right_atoms[shared])
    ):
        shared += 1
    if shared == len(left_atoms) == len(right_atoms):
        return shared, {
            "available": False,
            "reason": "decoded_claim_graphs_exactly_equal",
        }
    left_atom = left_atoms[shared] if shared < len(left_atoms) else None
    right_atom = right_atoms[shared] if shared < len(right_atoms) else None
    left_summary = _atom_summary(left_atom)
    right_summary = _atom_summary(right_atom)
    left_transitions = _touching_transitions(
        left, None if left_atom is None else left_atom["atom_id"]
    )
    right_transitions = _touching_transitions(
        right, None if right_atom is None else right_atom["atom_id"]
    )
    cues = {
        cue
        for summary in (left_summary, right_summary)
        if summary is not None
        for cue in summary["dependency_cues"]
    }
    kinds = {
        summary["kind"]
        for summary in (left_summary, right_summary)
        if summary is not None
    }
    dispute_kind = (
        "assumption"
        if "condition" in cues or "condition" in kinds
        else "dependency_transition"
        if left_transitions != right_transitions
        else "claim"
    )
    return shared, {
        "available": True,
        "kind": dispute_kind,
        "atom_ordinal": shared,
        "left": left_summary,
        "right": right_summary,
        "left_transitions": left_transitions,
        "right_transitions": right_transitions,
        "exact_content_committed": True,
        "semantic_equivalence_claimed": False,
    }


def build_disagreement_graph_receipt(
    *,
    n_branches: int,
    operator_trace: Any,
    action_trace: Any,
    structural_diversity: Any,
    candidate_decompositions: Any = None,
    blind_review: Any = None,
) -> dict[str, Any]:
    """Build pairwise shared-prefix and first-divergence evidence."""

    if type(n_branches) is not int or not 1 <= n_branches <= MAX_BRANCHES:
        raise ValueError("disagreement graph branch count is invalid")
    if not isinstance(structural_diversity, Mapping):
        raise ValueError("structural diversity receipt is required")
    structural_sha = structural_diversity.get("receipt_sha256")
    if (
        structural_diversity.get("n_branches") != n_branches
        or not isinstance(structural_sha, str)
        or len(structural_sha) != 64
        or any(character not in "0123456789abcdef" for character in structural_sha)
    ):
        raise ValueError("structural diversity commitment is invalid")
    actions = _action_inventory(action_trace)
    timelines = _operator_timelines(
        n_branches=n_branches,
        operator_trace=operator_trace,
        actions=actions,
    )
    candidates, candidate_binding_sha, candidate_status = _candidate_inventory(
        n_branches=n_branches,
        candidate_decompositions=candidate_decompositions,
        blind_review=blind_review,
    )
    branches = [
        {
            "index": index,
            "operator_transition_count": len(timelines[index]),
            "operator_program_sha256": _canonical_sha256(
                [row["program_sha256"] for row in timelines[index]]
            ),
            "candidate_decomposition_sha256": (
                candidates[index]["receipt_sha256"] if candidates else ""
            ),
        }
        for index in range(n_branches)
    ]
    pairwise: list[dict[str, Any]] = []
    for left in range(n_branches):
        for right in range(left + 1, n_branches):
            causal_prefix, causal = _first_program_divergence(
                timelines[left], timelines[right]
            )
            if candidates:
                claim_prefix, claim = _first_candidate_divergence(
                    candidates[left], candidates[right]
                )
            else:
                claim_prefix, claim = 0, {
                    "available": False,
                    "reason": candidate_status,
                }
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "shared_causal_program_prefix": causal_prefix,
                    "shared_exact_claim_prefix": claim_prefix,
                    "causal_divergence": causal,
                    "candidate_divergence": claim,
                    "localized": bool(causal["available"] or claim["available"]),
                }
            )
    localized = [row for row in pairwise if row["localized"]]
    earliest_causal = [
        (
            int(row["causal_divergence"]["action_step"]),
            int(row["left"]),
            int(row["right"]),
        )
        for row in localized
        if row["causal_divergence"]["available"]
    ]
    earliest_claim = [
        (
            int(row["candidate_divergence"]["atom_ordinal"]),
            int(row["left"]),
            int(row["right"]),
        )
        for row in localized
        if row["candidate_divergence"]["available"]
    ]
    payload = {
        "schema": DISAGREEMENT_GRAPH_SCHEMA,
        "n_branches": n_branches,
        "structural_diversity_sha256": structural_sha,
        "candidate_binding_sha256": candidate_binding_sha,
        "candidate_evidence_status": candidate_status,
        "candidate_decompositions": (
            {str(index): candidates[index] for index in range(n_branches)}
            if candidates
            else {}
        ),
        "branches": branches,
        "pairwise": pairwise,
        "localized_pair_count": len(localized),
        "global_earliest": {
            "causal": (
                {
                    "action_step": min(earliest_causal)[0],
                    "left": min(earliest_causal)[1],
                    "right": min(earliest_causal)[2],
                }
                if earliest_causal
                else None
            ),
            "candidate": (
                {
                    "atom_ordinal": min(earliest_claim)[0],
                    "left": min(earliest_claim)[1],
                    "right": min(earliest_claim)[2],
                }
                if earliest_claim
                else None
            ),
        },
        "localized": bool(localized),
        "reason": (
            "earliest_exact_disagreement_localized"
            if localized
            else "no_exact_disagreement_observed"
        ),
        "surface_wording_authority": "none",
        "hidden_reasoning_exposed": False,
        "semantic_equivalence_claimed": False,
        "selection_effect": "none",
        "repair_effect": "none",
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def validate_disagreement_graph_receipt(
    value: Any,
    *,
    n_branches: int,
    operator_trace: Any,
    action_trace: Any,
    structural_diversity: Any,
    blind_review: Any = None,
) -> dict[str, Any]:
    """Reconstruct the graph from primary receipts and require exact equality."""

    if not isinstance(value, Mapping):
        raise ValueError("disagreement graph receipt is missing")
    expected = build_disagreement_graph_receipt(
        n_branches=n_branches,
        operator_trace=operator_trace,
        action_trace=action_trace,
        structural_diversity=structural_diversity,
        candidate_decompositions=value.get("candidate_decompositions"),
        blind_review=blind_review,
    )
    if dict(value) != expected:
        raise ValueError("disagreement graph differs from reconstruction")
    if (
        value.get("surface_wording_authority") != "none"
        or value.get("hidden_reasoning_exposed") is not False
        or value.get("semantic_equivalence_claimed") is not False
        or value.get("selection_effect") != "none"
        or value.get("repair_effect") != "none"
    ):
        raise ValueError("disagreement graph exceeded diagnostic authority")
    return dict(value)


__all__ = [
    "DISAGREEMENT_GRAPH_SCHEMA",
    "build_disagreement_graph_receipt",
    "decompose_branch_candidates",
    "validate_disagreement_graph_receipt",
]
