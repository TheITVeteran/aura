"""Bounded identifiability audit for Aura's public recurrent transition contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.learning.recurrence_curriculum import RecurrenceTrainingTask
from core.learning.recurrent_action_schema import action_targets_from_program
from core.learning.recurrent_state_schema import state_targets_from_trace

TRANSITION_IDENTIFIABILITY_SCHEMA: Final = (
    "aura.unified_intrinsic.transition_identifiability.v2"
)


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


@dataclass(frozen=True, slots=True)
class PublicTransitionObservation:
    family: str
    task_id: str
    step: int
    state: tuple[int, ...]
    action: tuple[int, ...]
    action_prefix: tuple[tuple[int, ...], ...]
    next_state: tuple[int, ...]


def public_transition_observations(
    tasks: Sequence[RecurrenceTrainingTask],
) -> tuple[PublicTransitionObservation, ...]:
    """Project verified traces onto the exact public inference inputs."""

    observations: list[PublicTransitionObservation] = []
    for task in tasks:
        trace = task.transition_trace
        program = task.transition_program
        if trace is None or program is None or program.state_trace is not trace:
            raise ValueError("transition identifiability requires aligned process evidence")
        actions = action_targets_from_program(program, trace.depth).values
        states = state_targets_from_trace(trace, trace.depth)
        if len(actions) != trace.depth or len(states.values) != trace.depth:
            raise ValueError("transition identifiability trajectory differs")
        current = tuple(int(value) for value in states.initial_values)
        prefix: list[tuple[int, ...]] = []
        for step, (action, next_state) in enumerate(
            zip(actions, states.values, strict=True),
            start=1,
        ):
            canonical_action = tuple(int(value) for value in action)
            prefix.append(canonical_action)
            canonical_next = tuple(int(value) for value in next_state)
            observations.append(
                PublicTransitionObservation(
                    family=str(trace.family),
                    task_id=str(task.task_id),
                    step=step,
                    state=current,
                    action=canonical_action,
                    action_prefix=tuple(prefix),
                    next_state=canonical_next,
                )
            )
            current = canonical_next
    if not observations:
        raise ValueError("transition identifiability cohort is empty")
    return tuple(observations)


def _key_statistics(
    observations: Sequence[PublicTransitionObservation],
    key_fn: Callable[[PublicTransitionObservation], Any],
) -> dict[str, Any]:
    targets: dict[Any, Counter[tuple[int, ...]]] = defaultdict(Counter)
    examples: dict[Any, list[PublicTransitionObservation]] = defaultdict(list)
    for observation in observations:
        key = key_fn(observation)
        targets[key][observation.next_state] += 1
        if len(examples[key]) < 4:
            examples[key].append(observation)
    ambiguous = {key: counts for key, counts in targets.items() if len(counts) > 1}
    correct_at_deterministic_ceiling = sum(max(counts.values()) for counts in targets.values())
    collision_examples = []
    for key in sorted(ambiguous, key=repr)[:8]:
        collision_examples.append(
            {
                "key_sha256": _canonical_sha256(key),
                "targets": [
                    {"state": list(target), "observations": count}
                    for target, count in sorted(ambiguous[key].items())
                ],
                "sources": [
                    {
                        "family": row.family,
                        "task_id": row.task_id,
                        "step": row.step,
                    }
                    for row in examples[key]
                ],
            }
        )
    return {
        "observations": len(observations),
        "unique_keys": len(targets),
        "reused_keys": sum(1 for counts in targets.values() if sum(counts.values()) > 1),
        "ambiguous_keys": len(ambiguous),
        "ambiguous_observations": sum(sum(counts.values()) for counts in ambiguous.values()),
        "empirical_deterministic_accuracy_ceiling": (
            correct_at_deterministic_ceiling / len(observations)
        ),
        "collision_examples": collision_examples,
    }


def _audit_cohort(
    observations: Sequence[PublicTransitionObservation],
) -> dict[str, Any]:
    families = sorted({row.family for row in observations})

    def summarize(rows: Sequence[PublicTransitionObservation]) -> dict[str, Any]:
        maximum_prefix = max(len(row.action_prefix) for row in rows)
        suffix_curve = {}
        for width in range(1, maximum_prefix + 1):
            suffix_curve[str(width)] = _key_statistics(
                rows,
                lambda row, width=width: (
                    row.family,
                    row.state,
                    row.action_prefix[-width:],
                ),
            )
        return {
            "state_current_action": _key_statistics(
                rows,
                lambda row: (row.family, row.state, row.action),
            ),
            "state_full_public_prefix": _key_statistics(
                rows,
                lambda row: (row.family, row.state, row.action_prefix),
            ),
            "public_prefix_only": _key_statistics(
                rows,
                lambda row: (row.family, row.action_prefix),
            ),
            "state_suffix_curve": suffix_curve,
        }

    return {
        "overall": summarize(observations),
        "families": {
            family: summarize([row for row in observations if row.family == family])
            for family in families
        },
    }


def audit_public_transition_identifiability(
    train_tasks: Sequence[RecurrenceTrainingTask],
    holdout_tasks: Sequence[RecurrenceTrainingTask],
) -> dict[str, Any]:
    """Measure whether public histories identify the observed next state.

    This is a finite-cohort feasibility certificate. Zero observed collisions
    supports training on the declared public contract; it is not a proof over
    every program the generators could ever emit.
    """

    train = public_transition_observations(train_tasks)
    holdout = public_transition_observations(holdout_tasks)
    combined = train + holdout
    train_full = {
        (row.family, row.state, row.action_prefix): row.next_state for row in train
    }
    holdout_full = {
        (row.family, row.state, row.action_prefix): row.next_state for row in holdout
    }
    train_local = {(row.family, row.state, row.action) for row in train}
    holdout_local = {(row.family, row.state, row.action) for row in holdout}
    train_local_targets: dict[
        tuple[str, tuple[int, ...], tuple[int, ...]], set[tuple[int, ...]]
    ] = defaultdict(set)
    holdout_local_targets: dict[
        tuple[str, tuple[int, ...], tuple[int, ...]], set[tuple[int, ...]]
    ] = defaultdict(set)
    for row in train:
        train_local_targets[(row.family, row.state, row.action)].add(row.next_state)
    for row in holdout:
        holdout_local_targets[(row.family, row.state, row.action)].add(row.next_state)
    full_overlap = set(train_full) & set(holdout_full)
    overlap_disagreements = sum(
        train_full[key] != holdout_full[key] for key in full_overlap
    )
    local_overlap = set(train_local_targets) & set(holdout_local_targets)
    local_overlap_disagreements = sum(
        train_local_targets[key].isdisjoint(holdout_local_targets[key])
        for key in local_overlap
    )
    audit = _audit_cohort(combined)
    local_stats = audit["overall"]["state_current_action"]
    full_stats = audit["overall"]["state_full_public_prefix"]
    family_admission = {
        family: {
            "state_recurrent_transition_admitted": (
                values["state_current_action"]["ambiguous_keys"] == 0
            ),
            "public_prefix_replay_admitted": (
                values["state_full_public_prefix"]["ambiguous_keys"] == 0
            ),
        }
        for family, values in audit["families"].items()
    }
    state_recurrent_transition_admitted = (
        local_stats["ambiguous_keys"] == 0 and local_overlap_disagreements == 0
    )
    public_prefix_replay_admitted = (
        full_stats["ambiguous_keys"] == 0 and overlap_disagreements == 0
    )
    body = {
        "schema": TRANSITION_IDENTIFIABILITY_SCHEMA,
        "scope": "bounded_empirical_public_transition_contract",
        "claim_not_supported": [
            "universal_markov_proof",
            "learned_history_memory_is_lossless",
            "general_reasoning_gain",
            "wow_signal",
        ],
        "claim_boundary": {
            "state_current_action": (
                "bounded recurrent transition; the committed state must be load-bearing"
            ),
            "state_full_public_prefix": (
                "bounded public-prefix replay; this may recompute from action history and "
                "does not establish recurrent-state computation"
            ),
        },
        "train_tasks": len(train_tasks),
        "holdout_tasks": len(holdout_tasks),
        "train_observations": len(train),
        "holdout_observations": len(holdout),
        "audit": audit,
        "train_holdout_overlap": {
            "state_current_action_keys": len(train_local & holdout_local),
            "state_current_action_target_disagreements": local_overlap_disagreements,
            "state_full_public_prefix_keys": len(full_overlap),
            "full_prefix_target_disagreements": overlap_disagreements,
        },
        "admission": {
            "families": family_admission,
            "state_current_action_has_no_observed_ambiguity": (
                local_stats["ambiguous_keys"] == 0
            ),
            "overlapping_state_current_action_targets_agree": (
                local_overlap_disagreements == 0
            ),
            "full_public_prefix_has_no_observed_ambiguity": (
                full_stats["ambiguous_keys"] == 0
            ),
            "overlapping_full_prefix_targets_agree": overlap_disagreements == 0,
            "state_recurrent_transition_admitted": (
                state_recurrent_transition_admitted
            ),
            "public_prefix_replay_admitted": public_prefix_replay_admitted,
            # The unqualified admission is deliberately the stronger contract.
            # A causal action-prefix replay is useful, but it is not evidence
            # that the recurrent state itself implements the transition.
            "admitted": state_recurrent_transition_admitted,
        },
    }
    return {**body, "report_sha256": _canonical_sha256(body)}


__all__ = [
    "TRANSITION_IDENTIFIABILITY_SCHEMA",
    "PublicTransitionObservation",
    "audit_public_transition_identifiability",
    "public_transition_observations",
]
