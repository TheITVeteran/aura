"""A trial needs enough headroom to wait for its response and then brake."""
from types import SimpleNamespace

import pytest

from core.agency.keeping_a_trial_observable import (
    control_before_the_next_trial,
    visible_path_clearance,
)

pytestmark = pytest.mark.unit


def _scene(axis=1, sign=1):
    position, velocity = [120., 120.], [0., 0.]
    position[axis] = 170. if sign > 0 else 70.
    velocity[axis] = sign * 100.
    body = SimpleNamespace(number=1, seen=1., x=position[0], y=position[1], vx=velocity[0], vy=velocity[1], w=14., h=14.)
    rest, brake = [0., 0.], [0., 0.]
    rest[axis], brake[axis] = sign * 90., -sign * 110.
    inputs = {"": tuple(rest), "q": tuple(brake)}
    hers = SimpleNamespace(carried=True, thing=lambda _moves: body, push_of=inputs.get,
                            identification=SimpleNamespace(epoch=2))
    return body, hers, SimpleNamespace(shape=(240, 240)), inputs


@pytest.mark.parametrize("axis,sign", [(0, 1), (0, -1), (1, 1), (1, -1)])
def test_a_known_control_brakes_before_waiting_would_use_its_remaining_headroom(axis, sign):
    body, hers, moves, _inputs = _scene(axis, sign)
    result = control_before_the_next_trial(hers, moves, 1., ["q", "unmeasured"], .65)
    assert result is not None and result.key == "q"
    assert result.number == body.number and result.epoch == 2 and result.axis == axis
    assert result.wait_s == .65


def test_trialing_resumes_when_the_measured_braking_path_has_headroom():
    body, hers, moves, _inputs = _scene()
    body.y, body.vy = 140., 10.
    assert control_before_the_next_trial(hers, moves, 1., ["q"], .65) is None


@pytest.mark.parametrize("missing", ["rest", "brake", "fresh", "carried"])
def test_unmeasured_or_unobserved_motion_supplies_no_viability_claim(missing):
    body, hers, moves, inputs = _scene()
    if missing == "rest":
        inputs.pop("")
    elif missing == "brake":
        inputs.pop("q")
    elif missing == "fresh":
        body.seen = .9
    else:
        hers.carried = False
    assert control_before_the_next_trial(hers, moves, 1., ["q"], .65) is None


def test_slowing_one_axis_cannot_hide_a_loss_of_view_on_the_other():
    body, hers, moves, _inputs = _scene()
    body.x, body.vx = 230., 50.
    assert control_before_the_next_trial(hers, moves, 1., ["q"], .65) is None
    assert visible_path_clearance(body, moves.shape, [(1., (0., -110.))]) < 0


def test_a_turning_point_between_segment_ends_remains_part_of_the_projection():
    body = SimpleNamespace(x=50., y=88., vx=0., vy=60., w=10., h=10.)
    assert visible_path_clearance(body, (100, 100), [(1., (0., -120.))]) < 0
    assert body.y == 88.


def test_the_registered_unclamped_projection_invariant():
    from core.agency.keeping_a_trial_observable import _unclamped_trial_path_invariant

    assert _unclamped_trial_path_invariant() == ()


def test_the_registered_viability_invariant():
    from core.agency.keeping_a_trial_observable import _observed_viability_invariant

    assert _observed_viability_invariant() == ()
