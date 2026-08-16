"""What each number in an imagination frame actually rests on.

The frame emits novelty, salience, memory pressure, verification pressure,
routing directives, a selected attractor, a recurrent depth, ablation
predictions, and a field named ``causal_effects``. Every one of them is
produced by regex matches, keyword counts, caller flags and fixed
coefficients. CP126 raised four criticals about it, and they are one
sentence: the names promise measurement and the code performs lexical
scoring.

Renaming the fields is not available — ``causal_effects`` is read by
``cognitive_engine``, ``task_decomposer`` and ``cognitive_situation_frame``,
and a rename would break three live readers to fix a wording problem. So
the fix is the other direction: every quantity carries the BASIS it was
produced on, the frame publishes the map, and a reader that wants to know
whether a number was measured can ask instead of assuming.

The basis is not decoration. ``learn_from_feedback`` refuses to reinforce
on anything below ``MEASURED``, so a lexical score cannot become durable
learning by being passed around long enough.
"""

from __future__ import annotations

import enum
from typing import Any

__all__ = ["Basis", "BASIS_RANK", "basis_of", "describe_bases", "meets"]


class Basis(str, enum.Enum):
    """How a quantity in a frame came to exist. Ordered, weakest first."""

    #: A regex matched, a keyword was counted, a fixed coefficient was
    #: applied. Useful for steering a prompt. Not a finding about the world.
    LEXICAL = "lexical"
    #: Fixed phrases assembled from the first few keywords. No world state
    #: was transitioned, no candidate executed, no consequence propagated.
    TEMPLATE = "template"
    #: The caller said so. Carries exactly the caller's authority.
    CALLER_ASSERTED = "caller_asserted"
    #: Read from a real monitor, counter or clock at a known time.
    MEASURED = "measured"
    #: Produced by a model fit to outcomes, with the outcomes on record.
    LEARNED = "learned"

    @property
    def rank(self) -> int:
        return BASIS_RANK[self]


BASIS_RANK: dict[Basis, int] = {
    Basis.LEXICAL: 0,
    Basis.TEMPLATE: 0,
    Basis.CALLER_ASSERTED: 1,
    Basis.MEASURED: 2,
    Basis.LEARNED: 3,
}


def meets(basis: Basis, floor: Basis) -> bool:
    """Whether ``basis`` is at least as strong as ``floor``."""
    return basis.rank >= floor.rank


def basis_of(bases: dict[str, str] | None, field: str) -> Basis:
    """The recorded basis for one field, defaulting to the weakest.

    Defaulting DOWN is the whole point: a field nobody labelled is a field
    nobody measured, and reading it as measured is the failure this exists
    to stop.
    """
    raw = (bases or {}).get(field)
    try:
        return Basis(str(raw))
    except (TypeError, ValueError):
        return Basis.LEXICAL


def describe_bases(bases: dict[str, str] | None) -> dict[str, Any]:
    """A summary a health surface can publish without reading every field."""
    values = [basis_of(bases, key) for key in (bases or {})]
    return {
        "fields": len(values),
        "measured_or_better": sum(1 for b in values if meets(b, Basis.MEASURED)),
        "lexical_or_template": sum(1 for b in values if not meets(b, Basis.CALLER_ASSERTED)),
    }
