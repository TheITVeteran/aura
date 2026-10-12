"""Frames do not substitute for a missing acceleration response or missing axis evidence."""
from types import SimpleNamespace

import pytest

from core.agency.choosing_where_to_be import _Choosing
from core.agency.controls_ready_for_the_motion_model import has_a_measured_response
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def _pulse(hers, key, began, push):
    for offset in (.2, .25, .3, .35, .4):
        hers._hers.add(key, push[0] * offset, push[1] * offset, press=began, when=began + offset)


def test_many_frames_from_one_press_leave_an_accelerating_control_unknown():
    hers = WhichIsHers()
    hers.kind, hers.carried = 1, True
    _pulse(hers, "a", .0, (0., -110.))
    _pulse(hers, "", .5, (0., 90.))
    assert hers.tried("a") >= 4
    assert hers.way_of("a") is not None
    assert not has_a_measured_response(hers, "a")
    assert not hers.keys_known(["a"])
    _pulse(hers, "a", 1., (0., -110.))
    assert not has_a_measured_response(hers, "a")
    _pulse(hers, "", 1.5, (0., 90.))
    assert has_a_measured_response(hers, "a")
    assert hers.keys_known(["a"])


def test_a_missing_control_does_not_prove_the_body_cannot_move_on_that_axis():
    hers = WhichIsHers()
    hers.kind, hers.carried = 1, True
    for began in (.0, 1.):
        _pulse(hers, "a", began, (0., -110.))
        _pulse(hers, "", began + .5, (0., 90.))
    assert hers.axes(["a", "b"]) == (None, True)
    choice = SimpleNamespace(across=None, updown=True)
    assert _Choosing.line(choice) is None
    for began in (2., 3.):
        _pulse(hers, "b", began, (0., 90.))
    assert hers.axes(["a", "b"]) == (False, True)
    choice.across = False
    assert _Choosing.line(choice) == 0


def test_a_measured_zero_response_is_known_but_a_nonfinite_response_is_not():
    hers = SimpleNamespace(kind=1, carried=True, push_of=lambda _key: (0., 0.))
    assert has_a_measured_response(hers, "a")
    hers.push_of = lambda key: (float("nan"), 0.) if key else (0., 30.)
    assert not has_a_measured_response(hers, "a")


def test_the_registered_readiness_invariant():
    from core.agency.controls_ready_for_the_motion_model import _model_readiness_invariant

    assert _model_readiness_invariant() == ()
