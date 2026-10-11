"""Refine declared choice and placement obligations into one source-use action.

This module proposes an explicit alias: an owned revealed object is the value
chosen when that same object is picked up for its declared placement. It does
not assign meanings to anonymous drawings, execute input, or certify pixels.
The caller must validate the current measured source and bind physical input
before a procedure accepts this proposal.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.cognition.procedure_binding import _choice_subject, choice_slot
from core.runtime.skill_contract import PredicateState
from core.verify.invariants import invariant

Point = tuple[float, float]
Box = tuple[float, float, float, float]
ObjectWitness = tuple[str, Box, Box]


def _point(value: Any) -> Point | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in value)):
        return None
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class ActionRefinement:
    """A current variable binding and the two obligations its one action may meet."""

    move: str
    choice_key: str
    carry_key: str
    entity_tokens: tuple[str, ...]
    source: str
    destination: str
    source_witness: ObjectWitness
    surface_id: str
    bounds: tuple[float, float, float, float]
    capture_epoch: str
    capture_at: float
    work_epoch: str = ""

    def with_transfer(self, attempt_id: str, *, source_at: Point,
                      destination_at: Point) -> BoundActionRefinement:
        """Attach the caller's validated immutable physical transfer identity."""
        source_point, destination_point = _point(source_at), _point(destination_at)
        if not isinstance(attempt_id, str) or not attempt_id or source_point is None or destination_point is None:
            raise ValueError("a refinement needs one physical attempt and two measured points")
        x, y, width, height = self.source_witness[1]
        if not (x <= source_point[0] <= x + width and y <= source_point[1] <= y + height):
            raise ValueError("the refined pickup point is outside the bound object")
        return BoundActionRefinement(self, attempt_id, source_point, destination_point)


@dataclass(frozen=True)
class BoundActionRefinement:
    """A proposal attached before delivery to one validated physical attempt."""

    action: ActionRefinement
    attempt_id: str
    source_at: Point
    destination_at: Point


def _entity_in(thing: str) -> set[str]:
    # Frame.thing is the instruction's declared entity, not a guessed drawing
    # identity. A following source/destination clause is not part of its name.
    head = re.split(r"\b(?:from|into|onto|in|on|at|with|using|to)\b", str(thing).casefold(), maxsplit=1)[0]
    generic = frozenset("the a an any each one your this that these those".split())
    return {word.rstrip("s") for word in re.findall(r"[a-z]+", head) if word not in generic}


def choose_and_carry(rules: Any, move: str,
                     context: Mapping[str, Any] | None = None) -> ActionRefinement | None:
    """Propose only the earliest choice and its next same-entity placement.

    Current owned object witnesses provide values for the choice variable.
    Literal controls and text-only category choices retain separate actions.
    There is no skipped-step completion and no historical coordinate reuse.
    """
    from core.agency.acts_on_two_places import CARRY, two_places_of
    from core.cognition.reading_the_rules import holds

    if getattr(rules, "pending_step", None) is not None:
        return None
    two = two_places_of(move)
    revealed = getattr(rules, "revealed_choices", None)
    choice = rules.frames.get(revealed.for_step) if revealed is not None else None
    if two is None or two.act != CARRY or choice is None or not choice_slot(choice) or revealed is None:
        return None
    unresolved = [frame for frame in rules.steps() if not rules.passed(frame)]
    if choice not in unresolved:
        return None
    at = unresolved.index(choice)
    if any(holds(rules, frame) for frame in unresolved[:at]) or len(unresolved) <= at + 1:
        return None
    carry = unresolved[at + 1]
    entity = _choice_subject(choice)
    if (carry.act != "carry" or not carry.where or not entity or entity != _entity_in(carry.thing)
            or carry not in rules._matching_steps(move)):
        return None
    current = dict(context) if context is not None else rules.binding_context.evidence
    choice_move = f'click "{two.one}"'
    witness = dict(revealed.object_witnesses).get(choice_move)
    if witness is None or not revealed.offers(choice_move, current, choice.key):
        return None
    # offers() validates the full current object witness and surface context.
    # The proposal still has no authority to dispatch: the private pixels and
    # destination must be validated by the physical-input owner.
    return ActionRefinement(str(move), choice.key, carry.key, tuple(sorted(entity)), two.one, two.other,
                            witness, str(current["surface_id"]), tuple(current["bounds"]),
                            str(current["capture_epoch"]), float(current["capture_at"]),
                            str(current.get("work_epoch") or ""))


def occupancy_state(binding: BoundActionRefinement, after: Mapping[str, Any]) -> tuple[PredicateState, str]:
    """An exact owned transfer receipt, not generic change, proves this alias."""
    action = binding.action
    receipt = after.get("refinement")
    captured = after.get("capture_at")
    bounds = after.get("bounds")
    if (not isinstance(receipt, Mapping) or after.get("surface_id") != action.surface_id
            or not isinstance(bounds, (list, tuple)) or tuple(bounds) != action.bounds
            or str(after.get("work_epoch") or "") != action.work_epoch
            or not after.get("capture_epoch") or after["capture_epoch"] == action.capture_epoch
            or type(captured) not in (int, float) or not math.isfinite(captured) or captured <= action.capture_at
            or receipt.get("attempt_id") != binding.attempt_id or receipt.get("move") != action.move
            or receipt.get("source") != action.source or receipt.get("destination") != action.destination
            or _point(receipt.get("source_at")) != binding.source_at
            or _point(receipt.get("destination_at")) != binding.destination_at):
        return PredicateState.UNKNOWN, "the exact prebound source-use occupancy is unmeasured"
    if receipt.get("state") == "verified":
        return PredicateState.SATISFIED, "the prebound chosen source newly occupies its declared destination"
    if receipt.get("state") == "absent":
        return PredicateState.UNSATISFIED, "the prebound source did not occupy its declared destination"
    return PredicateState.UNKNOWN, "the prebound source-use outcome is unmeasured"


@invariant("cognition.refined_obligations_require_their_bound_transfer", scope="cognition",
           owner="core/cognition/action_refinement.py", observational=False)
def _bound_refinement_invariant() -> tuple:
    action = ActionRefinement('drag "item" to "canvas"', "choose", "place", ("component",), "item", "canvas",
                              ("a" * 64, (.1, .2, .1, .1), (.05, .1, .4, .7)), "editor", (0., 0., 400., 300.), "a", 1.)
    bound = action.with_transfer("transfer", source_at=(.15, .25), destination_at=(.7, .6))
    after = {"surface_id": "editor", "bounds": [0, 0, 400, 300], "capture_epoch": "b", "capture_at": 2.,
             "refinement": {"state": "verified", "attempt_id": "transfer", "move": action.move,
                            "source": "item", "destination": "canvas", "source_at": [.15, .25],
                            "destination_at": [.7, .6]}}
    assert occupancy_state(bound, after)[0] is PredicateState.SATISFIED
    assert occupancy_state(bound, {**after, "refinement": {**after["refinement"], "attempt_id": "other"}})[0] is PredicateState.UNKNOWN
    assert occupancy_state(bound, {**after, "refinement": None, "placement": {"state": "verified"}})[0] is PredicateState.UNKNOWN
    return ()
