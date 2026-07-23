"""Machine-checkable evidence and hypothesis continuity for latent recurrence."""

from __future__ import annotations

import hashlib
import json
from typing import Any

RECURRENT_GROUNDING_SCHEMA = "aura.rlc.recurrent_grounding.v1"
_ATTENTION_POLICY = "prompt_kv_then_context_prefix_then_hypothesis_v1"
_EVIDENCE_POLICY = "post_prelude_sealed_immutable_every_step_v1"
_HYPOTHESIS_POLICY = "persistent_hidden_state_no_prose_reencoding_v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_recurrent_grounding_receipt(
    *,
    input_tokens_sha256: str,
    input_token_count: int,
    cognitive_slots: list[dict[str, Any]],
    branches: list[Any],
    n_slots: int,
    comm_slot: int,
    selected_branch: int,
) -> dict[str, Any]:
    """Commit the topology and hidden-state transitions without leaking tensors."""

    evidence_slots = [
        {
            "slot": int(row["slot"]),
            "context_index": int(row["context_index"]),
            "source": str(row["source"]),
            "text_sha256": str(row["text_sha256"]),
        }
        for row in cognitive_slots
    ]
    evidence_indices = tuple(row["slot"] for row in evidence_slots)
    hypothesis_slots = [
        index
        for index in range(n_slots)
        if index != comm_slot and index not in evidence_indices
    ]
    branch_rows = []
    for branch in branches:
        transitions = [dict(row) for row in branch.recurrent_grounding_trace]
        branch_rows.append(
            {
                "branch_index": int(branch.index),
                "role": str(branch.role),
                "evidence_anchor_sha256": str(branch.evidence_anchor_sha256),
                "initial_hypothesis_sha256": str(
                    branch.initial_hypothesis_sha256
                ),
                "transitions": transitions,
            }
        )
    selected = next(
        (row for row in branch_rows if row["branch_index"] == selected_branch),
        None,
    )
    evidence_precedes_hypothesis = not evidence_slots or (
        bool(hypothesis_slots)
        and max(evidence_indices) < min(hypothesis_slots)
        and min(evidence_indices) > comm_slot
    )
    payload = {
        "schema": RECURRENT_GROUNDING_SCHEMA,
        "attention_policy": _ATTENTION_POLICY,
        "evidence_policy": _EVIDENCE_POLICY,
        "hypothesis_policy": _HYPOTHESIS_POLICY,
        "prompt_evidence": {
            "input_tokens_sha256": input_tokens_sha256,
            "input_token_count": input_token_count,
            "kv_policy": "shared_read_only_prompt_cache_v1",
        },
        "n_slots": n_slots,
        "n_branches": len(branch_rows),
        "comm_slot": comm_slot,
        "evidence_slots": evidence_slots,
        "hypothesis_slots": hypothesis_slots,
        "evidence_precedes_hypothesis": evidence_precedes_hypothesis,
        "evidence_embedded_once": True,
        "prose_reencoded_between_steps": False,
        "branches": branch_rows,
        "selected_branch": selected_branch,
        "selected_transition_count": (
            len(selected["transitions"]) if selected is not None else 0
        ),
        "all_evidence_invariant": all(
            transition.get("evidence_unchanged") is True
            for row in branch_rows
            for transition in row["transitions"]
        ),
        "selected_hypothesis_causal": bool(
            selected is not None
            and any(
                transition.get("hypothesis_changed") is True
                for transition in selected["transitions"]
            )
        ),
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    validate_recurrent_grounding_receipt(
        receipt,
        input_tokens_sha256=input_tokens_sha256,
        input_token_count=input_token_count,
        cognitive_slots=cognitive_slots,
        n_slots=n_slots,
        n_branches=len(branch_rows),
        selected_branch=selected_branch,
    )
    return receipt


def validate_recurrent_grounding_receipt(
    value: Any,
    *,
    input_tokens_sha256: str,
    input_token_count: int,
    cognitive_slots: list[dict[str, Any]],
    n_slots: int,
    n_branches: int,
    selected_branch: int,
) -> dict[str, Any]:
    """Independently replay the public recurrent-grounding contract."""

    if not isinstance(value, dict):
        raise ValueError("recurrent grounding receipt must be a mapping")
    required = {
        "schema",
        "attention_policy",
        "evidence_policy",
        "hypothesis_policy",
        "prompt_evidence",
        "n_slots",
        "n_branches",
        "comm_slot",
        "evidence_slots",
        "hypothesis_slots",
        "evidence_precedes_hypothesis",
        "evidence_embedded_once",
        "prose_reencoded_between_steps",
        "branches",
        "selected_branch",
        "selected_transition_count",
        "all_evidence_invariant",
        "selected_hypothesis_causal",
        "receipt_sha256",
    }
    if set(value) != required:
        raise ValueError("recurrent grounding receipt fields do not match contract")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("recurrent grounding receipt commitment mismatch")
    prompt = value["prompt_evidence"]
    if (
        value["schema"] != RECURRENT_GROUNDING_SCHEMA
        or value["attention_policy"] != _ATTENTION_POLICY
        or value["evidence_policy"] != _EVIDENCE_POLICY
        or value["hypothesis_policy"] != _HYPOTHESIS_POLICY
        or not isinstance(prompt, dict)
        or set(prompt) != {"input_tokens_sha256", "input_token_count", "kv_policy"}
        or prompt["input_tokens_sha256"] != input_tokens_sha256
        or prompt["input_token_count"] != input_token_count
        or prompt["kv_policy"] != "shared_read_only_prompt_cache_v1"
        or not _is_sha256(input_tokens_sha256)
        or type(input_token_count) is not int
        or input_token_count <= 0
        or value["n_slots"] != n_slots
        or value["n_branches"] != n_branches
        or value["comm_slot"] != 0
        or value["selected_branch"] != selected_branch
        or value["evidence_embedded_once"] is not True
        or value["prose_reencoded_between_steps"] is not False
    ):
        raise ValueError("recurrent grounding topology or prompt binding is invalid")

    raw_expected = sorted(cognitive_slots, key=lambda row: int(row["slot"]))
    expected_evidence = [
        {
            "slot": int(row["slot"]),
            "context_index": int(row["context_index"]),
            "source": str(row["source"]),
            "text_sha256": str(row["text_sha256"]),
        }
        for row in raw_expected
    ]
    evidence = value["evidence_slots"]
    if evidence != expected_evidence:
        raise ValueError("recurrent evidence slots differ from cognitive ingress")
    evidence_indices = [row["slot"] for row in evidence]
    if (
        evidence_indices != list(range(1, 1 + len(evidence_indices)))
        or any(
            row["context_index"] != index
            or not row["source"]
            or not _is_sha256(row["text_sha256"])
            for index, row in enumerate(evidence)
        )
    ):
        raise ValueError("recurrent evidence is not a canonical causal prefix")
    expected_hypothesis = [
        index
        for index in range(n_slots)
        if index != value["comm_slot"] and index not in evidence_indices
    ]
    if (
        value["hypothesis_slots"] != expected_hypothesis
        or not expected_hypothesis
        or value["evidence_precedes_hypothesis"] is not True
    ):
        raise ValueError("recurrent hypothesis topology is invalid")

    branches = value["branches"]
    if (
        not isinstance(branches, list)
        or len(branches) != n_branches
        or [row.get("branch_index") for row in branches] != list(range(n_branches))
    ):
        raise ValueError("recurrent grounding branch coverage is invalid")
    selected_row = None
    all_invariant = True
    for row in branches:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "branch_index",
                "role",
                "evidence_anchor_sha256",
                "initial_hypothesis_sha256",
                "transitions",
            }
            or not row["role"]
            or not _is_sha256(row["evidence_anchor_sha256"])
            or not _is_sha256(row["initial_hypothesis_sha256"])
            or not isinstance(row["transitions"], list)
            or not row["transitions"]
        ):
            raise ValueError("recurrent grounding branch row is invalid")
        if row["branch_index"] == selected_branch:
            selected_row = row
        for ordinal, transition in enumerate(row["transitions"]):
            if (
                not isinstance(transition, dict)
                or set(transition)
                != {
                    "ordinal",
                    "branch_step",
                    "window_start",
                    "window_end",
                    "evidence_pre_sha256",
                    "evidence_post_sha256",
                    "hypothesis_pre_sha256",
                    "hypothesis_post_sha256",
                    "evidence_unchanged",
                    "hypothesis_changed",
                }
                or transition["ordinal"] != ordinal
                or type(transition["branch_step"]) is not int
                or transition["branch_step"] < 0
                or type(transition["window_start"]) is not int
                or type(transition["window_end"]) is not int
                or not 0 <= transition["window_start"] < transition["window_end"]
                or any(
                    not _is_sha256(transition[key])
                    for key in (
                        "evidence_pre_sha256",
                        "evidence_post_sha256",
                        "hypothesis_pre_sha256",
                        "hypothesis_post_sha256",
                    )
                )
            ):
                raise ValueError("recurrent grounding transition is invalid")
            evidence_unchanged = (
                transition["evidence_pre_sha256"]
                == transition["evidence_post_sha256"]
                == row["evidence_anchor_sha256"]
            )
            hypothesis_changed = (
                transition["hypothesis_pre_sha256"]
                != transition["hypothesis_post_sha256"]
            )
            if (
                transition["evidence_unchanged"] is not evidence_unchanged
                or transition["hypothesis_changed"] is not hypothesis_changed
            ):
                raise ValueError("recurrent grounding transition verdict is false")
            if ordinal == 0 and (
                transition["hypothesis_pre_sha256"]
                != row["initial_hypothesis_sha256"]
            ):
                raise ValueError("recurrent hypothesis does not begin at its committed state")
            all_invariant = all_invariant and evidence_unchanged
    if selected_row is None:
        raise ValueError("selected recurrent branch is absent")
    selected_causal = any(
        transition["hypothesis_changed"]
        for transition in selected_row["transitions"]
    )
    if (
        value["selected_transition_count"] != len(selected_row["transitions"])
        or value["all_evidence_invariant"] is not all_invariant
        or value["selected_hypothesis_causal"] is not selected_causal
        or not all_invariant
        or not selected_causal
    ):
        raise ValueError("recurrent grounding aggregate verdict is invalid")
    return dict(value)


__all__ = [
    "RECURRENT_GROUNDING_SCHEMA",
    "build_recurrent_grounding_receipt",
    "canonical_sha256",
    "validate_recurrent_grounding_receipt",
]
