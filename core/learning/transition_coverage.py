"""Coverage admission for learned recurrent transition campaigns.

Identifiability proves that a public state/action pair has one target. It does
not prove that the training partition contains the primitives exercised by the
holdout partition. This module keeps those claims separate and makes missing
opcode, operand, state, structure, and length support fail closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Final

from core.learning.recurrence_curriculum import RecurrenceTrainingTask
from core.learning.recurrent_action_schema import (
    ACTION_SLOT_NAMES,
    action_targets_from_program,
)
from core.learning.recurrent_state_schema import (
    SEMANTIC_STATE_SLOT_NAMES,
    state_slot_names,
    state_targets_from_trace,
)

TRANSITION_COVERAGE_SCHEMA: Final = "aura.transition_primitive_coverage.v1"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _mask_key(mask: tuple[bool, ...]) -> str:
    return "".join("1" if value else "0" for value in mask)


def _partition_summary(
    tasks: tuple[RecurrenceTrainingTask, ...],
    *,
    state_slots: int,
) -> dict[str, Any]:
    action_support = {name: set() for name in ACTION_SLOT_NAMES}
    state_names = state_slot_names(state_slots)
    state_support = {name: set() for name in state_names}
    action_masks: set[str] = set()
    state_masks: set[str] = set()
    task_ids: set[str] = set()
    prompt_sha256s: set[str] = set()
    program_sha256s: set[str] = set()
    depths: set[int] = set()

    for task in tasks:
        if task.transition_trace is None or task.transition_program is None:
            raise ValueError("transition coverage requires exact transition programs")
        task_ids.add(task.task_id)
        prompt_sha256s.add(hashlib.sha256(task.prompt.encode("utf-8")).hexdigest())
        program_sha256s.add(task.transition_program.program_sha256)
        depths.add(task.depth)

        actions = action_targets_from_program(task.transition_program, task.depth)
        for values, mask in zip(actions.values, actions.masks, strict=True):
            action_masks.add(_mask_key(mask))
            for name, value, active in zip(
                ACTION_SLOT_NAMES,
                values,
                mask,
                strict=True,
            ):
                if active:
                    action_support[name].add(value)

        states = state_targets_from_trace(
            task.transition_trace,
            task.depth,
            state_slots=state_slots,
        )
        rows = ((states.initial_values, states.initial_masks),) + tuple(
            zip(states.values, states.masks, strict=True)
        )
        for values, mask in rows:
            state_masks.add(_mask_key(mask))
            for name, value, active in zip(state_names, values, mask, strict=True):
                if active:
                    state_support[name].add(value)

    return {
        "task_count": len(tasks),
        "task_ids": sorted(task_ids),
        "prompt_sha256s": sorted(prompt_sha256s),
        "program_sha256s": sorted(program_sha256s),
        "depths": sorted(depths),
        "opcode_support": sorted(action_support["opcode"]),
        "action_support": {
            name: sorted(action_support[name]) for name in ACTION_SLOT_NAMES
        },
        "state_support": {name: sorted(state_support[name]) for name in state_names},
        "action_mask_patterns": sorted(action_masks),
        "state_mask_patterns": sorted(state_masks),
    }


def _missing_support(
    training: dict[str, list[int]],
    holdout: dict[str, list[int]],
) -> dict[str, list[int]]:
    return {
        name: sorted(set(values) - set(training.get(name, ())))
        for name, values in holdout.items()
        if set(values) - set(training.get(name, ()))
    }


def audit_transition_coverage(
    train_tasks: Sequence[RecurrenceTrainingTask],
    holdout_tasks: Sequence[RecurrenceTrainingTask],
    *,
    state_slots: int = len(SEMANTIC_STATE_SLOT_NAMES),
) -> dict[str, Any]:
    """Return a canonical admission report for a frozen transition cohort."""

    if (
        isinstance(train_tasks, (str, bytes))
        or isinstance(holdout_tasks, (str, bytes))
        or not train_tasks
        or not holdout_tasks
        or type(state_slots) is not int
    ):
        raise ValueError("transition coverage cohort is invalid")
    state_slot_names(state_slots)
    train = tuple(train_tasks)
    holdout = tuple(holdout_tasks)
    if any(not isinstance(task, RecurrenceTrainingTask) for task in train + holdout):
        raise TypeError("transition coverage received a non-training task")

    train_ids = [task.task_id for task in train]
    holdout_ids = [task.task_id for task in holdout]
    if len(set(train_ids)) != len(train_ids) or len(set(holdout_ids)) != len(holdout_ids):
        raise ValueError("transition coverage cohort contains duplicate task identities")

    train_families = {task.family for task in train}
    holdout_families = {task.family for task in holdout}
    family_reports: dict[str, Any] = {}
    for family in sorted(train_families | holdout_families):
        training = _partition_summary(
            tuple(task for task in train if task.family == family),
            state_slots=state_slots,
        )
        evaluation = _partition_summary(
            tuple(task for task in holdout if task.family == family),
            state_slots=state_slots,
        )
        missing_actions = _missing_support(
            training["action_support"], evaluation["action_support"]
        )
        missing_states = _missing_support(
            training["state_support"], evaluation["state_support"]
        )
        missing_action_masks = sorted(
            set(evaluation["action_mask_patterns"])
            - set(training["action_mask_patterns"])
        )
        missing_state_masks = sorted(
            set(evaluation["state_mask_patterns"])
            - set(training["state_mask_patterns"])
        )
        depth_extrapolation = sorted(set(evaluation["depths"]) - set(training["depths"]))
        program_overlap = sorted(
            set(training["program_sha256s"]) & set(evaluation["program_sha256s"])
        )
        family_reports[family] = {
            "training": training,
            "holdout": evaluation,
            "missing_action_support": missing_actions,
            "missing_state_support": missing_states,
            "missing_action_mask_patterns": missing_action_masks,
            "missing_state_mask_patterns": missing_state_masks,
            "depth_extrapolation": depth_extrapolation,
            "exact_program_overlap_count": len(program_overlap),
            "exact_program_overlap_sha256s": program_overlap,
            "in_distribution_primitive_coverage_admitted": bool(training["task_count"])
            and bool(evaluation["task_count"])
            and not missing_actions
            and not missing_states
            and not missing_action_masks
            and not missing_state_masks
            and not depth_extrapolation
            and not program_overlap,
        }

    task_overlap = sorted(set(train_ids) & set(holdout_ids))
    train_prompts = {hashlib.sha256(task.prompt.encode("utf-8")).hexdigest() for task in train}
    holdout_prompts = {
        hashlib.sha256(task.prompt.encode("utf-8")).hexdigest() for task in holdout
    }
    prompt_overlap = sorted(train_prompts & holdout_prompts)
    admitted = (
        train_families == holdout_families
        and not task_overlap
        and not prompt_overlap
        and all(
            report["in_distribution_primitive_coverage_admitted"]
            for report in family_reports.values()
        )
    )
    body = {
        "schema": TRANSITION_COVERAGE_SCHEMA,
        "state_slots": state_slots,
        "families": family_reports,
        "partition": {
            "training_count": len(train),
            "holdout_count": len(holdout),
            "family_sets_equal": train_families == holdout_families,
            "task_identity_overlap": task_overlap,
            "prompt_overlap_sha256s": prompt_overlap,
        },
        "admission": {
            "in_distribution_primitive_coverage_admitted": admitted,
            "claim_boundary": "fresh_instances_with_covered_primitives_structure_and_depth",
        },
        "claims_not_supported": [
            "unseen_opcode_generalization",
            "unseen_operand_or_state_value_generalization",
            "unseen_structural_grammar_generalization",
            "unseen_task_length_generalization",
            "broad_reasoning_gain",
            "wow_signal",
        ],
    }
    return {**body, "report_sha256": _canonical_sha256(body)}


__all__ = [
    "TRANSITION_COVERAGE_SCHEMA",
    "audit_transition_coverage",
]
