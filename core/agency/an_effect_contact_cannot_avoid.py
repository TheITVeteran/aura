"""Known consequences can require an action despite a harmless-looking appearance."""
from __future__ import annotations

from typing import Any

from core.agency.what_meeting_things_does import CLICK, MEET
from core.verify import invariant


def needs_a_countermeasure(meeting: Any, kind: int, shot: Any) -> bool:
    """Keep an available emission applicable where avoiding contact cannot prevent the cost."""
    if shot is None or kind not in getattr(meeting, 'lands_anyway', ()):
        return False
    # Actual beneficial meetings can contradict the source reading or its visual binding.
    return not (meeting.known(kind) and meeting.stance(kind) in (MEET, CLICK))


@invariant('agency.nonlocal_cost_keeps_a_known_countermeasure_available', scope='agency',
           owner='core/agency/an_effect_contact_cannot_avoid.py', observational=False)
def _nonlocal_countermeasure_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.what_meeting_things_does import AVOID

    meeting = SimpleNamespace(lands_anyway={3}, known=lambda _kind: False, stance=lambda _kind: AVOID)
    assert needs_a_countermeasure(meeting, 3, object())
    assert not needs_a_countermeasure(meeting, 3, None)
    assert not needs_a_countermeasure(meeting, 4, object())
    meeting.known, meeting.stance = lambda _kind: True, lambda _kind: MEET
    assert not needs_a_countermeasure(meeting, 3, object())
    return ()
