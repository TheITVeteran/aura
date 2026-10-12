"""Freeze and settle the obligations owned by one refined physical action.

Rules retains each procedure's records. These operations evaluate every bound
obligation before publishing receipts through the procedure's settlement path.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from core.cognition.action_refinement import BoundActionRefinement, occupancy_state
from core.runtime.skill_contract import (
    PredicateState,
    SemanticPredicate,
    semantic_predicate_from_mapping,
)

if TYPE_CHECKING:
    from core.cognition.reading_the_rules import (
        EvidenceSnapshot,
        Frame,
        Rules,
        StepAttempt,
        StepReceipt,
    )


def attempt_for(procedure: Rules, frame: Frame, move: str, before: EvidenceSnapshot,
                effects: Sequence[SemanticPredicate | Mapping[str, Any]], *,
                refinement_id: str = "", role: str = "") -> StepAttempt:
    """Freeze the declaration and its effects before input can be delivered."""
    from core.cognition.reading_the_rules import StepAttempt

    predicates = [*procedure.effects.get(frame.key, ()), *effects]
    encoded = json.dumps([(p.to_dict() if isinstance(p, SemanticPredicate) else
                           semantic_predicate_from_mapping(p).to_dict()) for p in predicates],
                         sort_keys=True, allow_nan=False)
    procedure._attempt_number += 1
    return StepAttempt(procedure._attempt_number, frame.key, str(move), before,
                       str(getattr(procedure, "expects", {}).get(frame.key) or ""), encoded,
                       optional=procedure.passed(frame), refinement_id=refinement_id, refinement_role=role)


def begin_refinement(procedure: Rules, move: str, before: Any, binding: BoundActionRefinement,
                     effects: Sequence[SemanticPredicate | Mapping[str, Any]],
                     step_key: str | None) -> StepAttempt:
    """Attach both frozen declarations to their validated physical transfer."""
    from core.cognition.reading_the_rules import EvidenceSnapshot, PendingActionRefinement

    if not isinstance(binding, BoundActionRefinement):
        raise TypeError("a refinement needs a prebound physical transfer")
    baseline = EvidenceSnapshot.of(before)
    action = procedure.refinement_of(move, baseline.evidence)
    if action is None or action != binding.action or step_key not in (None, action.choice_key):
        raise ValueError("the refined obligations no longer match this current source and procedure")
    if action.with_transfer(binding.attempt_id, source_at=binding.source_at,
                            destination_at=binding.destination_at) != binding:
        raise ValueError("the physical refinement binding is invalid")
    occupancy = SemanticPredicate("refined_source_used", "after.refinement.state", expected="verified")
    attempts = tuple(attempt_for(procedure, procedure.frames[key], move, baseline, (*effects, occupancy),
                                 refinement_id=binding.attempt_id, role=role)
                     for key, role in ((action.choice_key, "choose_value"), (action.carry_key, "place_value")))
    procedure.pending_refinement = PendingActionRefinement(binding, attempts)
    procedure.pending_step = attempts[0]
    return procedure.pending_step


def settle_refinement(procedure: Rules, move: str, changed: bool | None, after: Any) -> StepReceipt:
    """Publish equal outcomes only after every frozen obligation was evaluated."""
    from core.cognition.reading_the_rules import EvidenceSnapshot, StepReceipt, _step_state

    pending = procedure.pending_refinement
    if pending is None or move != pending.binding.action.move or procedure.pending_step is not pending.attempts[0]:
        raise RuntimeError("the physical action differs from its pending refined obligations")
    snapshot = EvidenceSnapshot.of(after)
    outcomes = [occupancy_state(pending.binding, snapshot.evidence)]
    for attempt in pending.attempts:
        outcomes.append(_step_state(attempt, snapshot, changed))
        if attempt.expected and attempt.predicates != "[]":
            # The shared occupancy predicate retains each frame's independent
            # textual expectation.
            outcomes.append(_step_state(replace(attempt, predicates="[]"), snapshot, changed))
    state = (PredicateState.UNSATISFIED if any(s is PredicateState.UNSATISFIED for s, _ in outcomes) else
             PredicateState.UNKNOWN if any(s is PredicateState.UNKNOWN for s, _ in outcomes) else
             PredicateState.SATISFIED)
    reasons = list(dict.fromkeys(reason for outcome, reason in outcomes if outcome is state))
    reason = "the prebound action bundle: " + "; ".join(reasons)
    receipts = tuple(StepReceipt(attempt, snapshot, state, changed, reason) for attempt in pending.attempts)
    # There is no await or dispatch between these two receipt publications.
    try:
        for receipt in receipts:
            procedure._settled(receipt)
    finally:
        procedure.pending_step = None
        procedure.pending_refinement = None
    return receipts[0]
