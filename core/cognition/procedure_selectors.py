"""Bind a procedure's choice values to its confirmed selector effects."""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.cognition.procedure_binding import RevealedChoices, selection_reveals, selector_for
from core.cognition.selector_view import UNKNOWN_SELECTOR_VIEW, SelectorView

if TYPE_CHECKING:
    from core.cognition.reading_the_rules import Frame, Rules, StepReceipt


def opens_needed_choices(procedure: Rules, frame: Frame) -> bool:
    """Whether this selector reaches the next held choice's unavailable values."""
    from core.cognition.reading_the_rules import the_step_that_holds

    needed = the_step_that_holds(procedure)
    return bool(needed is not None and selector_for(frame, needed)
                and (procedure.revealed_choices is None or not any(
                    procedure.revealed_choices.offers(move, procedure.binding_context.evidence, needed.key)
                    for move in procedure.revealed_choices.moves)))


def bind_revealed_controls(procedure: Rules, receipt: StepReceipt) -> None:
    """Retain values and visual identity only from this procedure's owned effect."""
    from core.cognition.reading_the_rules import the_step_that_holds

    frame = procedure.frames.get(receipt.step_key)
    needed = the_step_that_holds(procedure)
    if frame is not None and needed is not None and (selector_for(frame, needed) or selection_reveals(frame, needed)):
        procedure.revealed_choices = RevealedChoices.from_effect(
            receipt.before.evidence, receipt.after.evidence, frame.key, needed.key, opened_move=receipt.attempt.move)
        view = SelectorView.from_effect(receipt, procedure.revealed_choices)
        procedure.selector_view = (view if view is not None else
                                   UNKNOWN_SELECTOR_VIEW if procedure.selector_view is not None else None)
        procedure.observe_context(receipt.after.evidence)
