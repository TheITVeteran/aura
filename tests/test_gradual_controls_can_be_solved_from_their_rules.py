"""Stored response must survive release in the forward model and inverse plan."""
import pytest

from core.agency.solving_for_a_required_state import solve_for_a_required_state
from tools.reference_runners.controls_that_take_time_to_build import (
    ChannelRule,
    GradualControls,
    MotionState,
)

pytestmark = pytest.mark.unit


def test_commanded_build_up_and_released_decay_match_an_independent_recurrence():
    model = GradualControls(((ChannelRule(), ChannelRule(decay=.8)),
                            (ChannelRule(drift=.2), ChannelRule(target=2., change=.1))),
                           ((1., -1.),), ((0., 1.),))
    state = MotionState((0.,), (1., 0.))
    place, passive, actuator = 0., 1., 0.
    for control in [1] * 10 + [0] * 10 + [1] * 5:
        if control:
            passive += .2
            actuator = min(2., actuator + .1)
        else:
            actuator *= .8
        place += passive - actuator
        state = model.advance(state, control)
        assert state.position[0] == pytest.approx(place)
        assert state.channels == pytest.approx((passive, actuator))
        assert model.outputs(state.channels, model.properties) == (state.channels[1],)


def test_inverse_planning_can_use_the_complete_stored_response():
    model = GradualControls(((ChannelRule(target=0., change=1.),),
                            (ChannelRule(target=3., change=1.),)), ((1.,),))
    initial = MotionState((0.,), (0.,))
    plan = solve_for_a_required_state(initial, (0, 1), advance=model.advance,
                                     outcome=lambda s: None if s.position[0] > 6 else
                                     s.position == (6.,) and s.channels == (0.,),
                                     cost=lambda s: abs(6. - s.position[0]) + abs(s.channels[0]),
                                     identity=lambda s: s, steps=12)
    assert plan.reached and plan.states[-1] == MotionState((6.,), (0.,))
    # Replay without calling the forward model used to find the plan.
    position, response = 0., 0.
    for control in plan.actions:
        response = min(3., response + 1.) if control else max(0., response - 1.)
        position += response
        assert position <= 6.
    assert position == 6. and response == 0.


def test_several_channels_can_control_position_and_posture_together():
    model = GradualControls(((ChannelRule(target=0., change=1.), ChannelRule(decay=.5)),
                            (ChannelRule(target=2., change=1.), ChannelRule(target=1., change=.5))),
                           ((1., 0.),), ((0., 1.),))
    initial = MotionState((0.,), (0., 0.))
    plan = solve_for_a_required_state(initial, (0, 1), advance=model.advance,
                                     outcome=lambda s: None if s.position[0] > 4 else
                                     s.position == (4.,) and s.channels[0] == 0. and
                                     abs(model.outputs(s.channels, model.properties)[0]) < .2,
                                     cost=lambda s: abs(4. - s.position[0]) + sum(abs(v) for v in s.channels),
                                     identity=lambda s: s, steps=12)
    assert plan.reached
    assert plan.states[-1].position == (4.,)
    assert plan.states[-1].channels[0] == 0.
    assert plan.states[-1].channels[1] < .2


@pytest.mark.parametrize("change", ["rule_count", "matrix_count", "nonfinite", "negative_rate"])
def test_missing_or_invalid_equations_are_not_a_forward_model(change):
    rules = ((ChannelRule(),),)
    movement = ((1.,),)
    if change == "rule_count":
        rules += ((ChannelRule(), ChannelRule()),)
    elif change == "matrix_count":
        movement = ((1., 1.),)
    elif change == "nonfinite":
        movement = ((float('nan'),),)
    else:
        rules = ((ChannelRule(change=-1.),),)
    with pytest.raises(ValueError):
        GradualControls(rules, movement)


def test_sequential_channel_updates_keep_the_order_of_the_supplied_rules():
    rule = ChannelRule(target=1., change=.2, ordered=True)
    # The positive step crosses the target; the subsequent negative check
    # sees that new value and returns it to where it began.
    assert rule.next(.9) == pytest.approx(.9)
    assert rule.next(1.1) == pytest.approx(.9)


@pytest.mark.parametrize("control", [-1, 2, True, .5])
def test_unknown_inputs_do_not_borrow_another_controls_rule(control):
    model = GradualControls(((ChannelRule(),),), ((1.,),))
    with pytest.raises(ValueError):
        model.advance(MotionState((0.,), (1.,)), control)
