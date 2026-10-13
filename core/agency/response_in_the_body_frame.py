"""Keep a measured heading response with the body that supplied it."""
from __future__ import annotations

import math
from typing import Any

from core.verify import invariant


def body_epoch(hers: Any) -> tuple | None:
    if hers.number is None or hers.kind is None:
        return None
    return hers.number, hers.kind, hers.identification.epoch


def bound_facing_response(hers: Any) -> Any:
    faces = getattr(hers, "facing_controls", None)
    identity = body_epoch(hers)
    return faces if identity is not None and faces is not None and faces.body_epoch == identity else None


def response_is_known(faces: Any, key: str) -> bool:
    return faces.turns_her() and (key in faces.turning_keys() or faces.way_of(key) is not None)


def expected_travel(faces: Any, key: str, at: float) -> tuple[float, float] | None:
    """Predict from the preceding observed heading, before reading this picture.

    A moving turn predicts a tangent while travel continues. A stopped body,
    a missing reading or a long gap supplies no angular identity check.
    """
    turns = faces.travel_turns
    if turns.last_at is None or not 0. <= at - turns.last_at <= turns.gap_s:
        return None
    if key in turns.rates():
        if not turns.travelling or not turns.samples or turns.last_velocity is None:
            return None
        angle = turns.samples[-1][1] + turns.rates()[key] * (at - turns.last_at)
        speed = math.hypot(*turns.last_velocity)
        return speed * math.cos(angle), speed * math.sin(angle)
    return faces.way_of(key)


@invariant("agency.heading_response_belongs_to_its_body_and_experiment", scope="agency",
           owner="core/agency/response_in_the_body_frame.py", observational=False)
def _body_heading_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.turning_while_travelling import TravelTurn
    from core.agency.which_way_she_faces import WhichWaySheFaces

    faces = WhichWaySheFaces(facing=0., body_epoch=(1, 2, 3))
    faces.along['drive'] = [80., 80.]
    for key, rate in (('', 0.), ('bend', 1.4)):
        faces.travel_turns.trials.extend(TravelTurn(key, began, began + .6, rate) for began in (-2., -1.))
    faces.travel_turns.saw('drive', 0., .3, 80., 0., measured=True)
    hers = SimpleNamespace(number=1, kind=2, identification=SimpleNamespace(epoch=3), facing_controls=faces)
    assert bound_facing_response(hers) is faces and response_is_known(faces, 'bend')
    assert math.isclose(expected_travel(faces, 'bend', .4)[1], 80. * math.sin(.14))
    assert expected_travel(faces, 'bend', .7) is None
    hers.identification.epoch = 4
    assert bound_facing_response(hers) is None
    return ()
