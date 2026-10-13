"""A control is known only when the current motion model can predict it."""
from __future__ import annotations

import math
from typing import Any

from core.agency.response_in_the_body_frame import bound_facing_response, response_is_known
from core.verify import invariant


def has_a_measured_response(hers: Any, key: str) -> bool:
    """Settled pace for a direct control; repeated pulse and rest for an accelerated body."""
    if hers.kind is None:
        return False
    if getattr(hers, "facing_controls", None) is not None:
        faces = bound_facing_response(hers)
        if faces is not None and response_is_known(faces, key):
            return True
    if hers.carried:
        measured = (hers.push_of(key), hers.push_of(""))
    else:
        measured = (hers.way_of(key),)
    return all(response is not None and len(response) == 2 and all(math.isfinite(value) for value in response)
               for response in measured)


@invariant("agency.control_readiness_matches_its_motion_model", scope="agency",
           owner="core/agency/controls_ready_for_the_motion_model.py", observational=False)
def _model_readiness_invariant() -> tuple:
    from types import SimpleNamespace

    actor = SimpleNamespace(kind=1, carried=True, way_of=lambda _key: (90., 0.), push_of=lambda _key: None)
    assert not has_a_measured_response(actor, "a")
    actor.push_of = lambda key: (0., 0.) if key else (0., 30.)
    assert has_a_measured_response(actor, "a")
    actor.carried = False
    assert has_a_measured_response(actor, "a")
    actor.way_of = lambda _key: None
    assert not has_a_measured_response(actor, "a")
    return ()
