"""An observed first response can buy another independent test without granting identity."""
from types import SimpleNamespace

import pytest

from core.agency.a_response_worth_measuring_again import response_worth_repeating
from core.agency.countering_motion_as_it_appears import CounteringMotion

pytestmark = pytest.mark.unit


def _scene(axis=1, sign=1, *, key="a", number=1):
    position = [120.0, 120.0]
    position[axis] = 190.0 if sign > 0 else 50.0
    going = [0.0, 0.0]
    going[axis] = sign * 40.0
    thing = SimpleNamespace(number=number, seen=0.9, born=-1.0, x=position[0], y=position[1],
                            w=12.0, h=12.0, vx=going[0], vy=going[1])

    def samples(began, acceleration):
        out = []
        for when in (began + .2, began + .25, began + .3, began + .35, began + .4):
            velocity = [0.0, 0.0]
            velocity[axis] = sign * (40.0 + acceleration * (when - began))
            out.append((round(when, 4), *velocity))
        return out

    record = SimpleNamespace(timed={(key, 0.0): samples(0.0, -110.0), ("", 0.5): samples(0.5, 90.0)})
    hers = SimpleNamespace(identification=SimpleNamespace(epoch=2, receipts=[]), _by_thing={number: record}, number=None)
    return thing, hers


def _response(thing, hers, keys=("a", "b", "c"), *, rest=0.5, at=0.9):
    return response_worth_repeating(keys, [thing], hers, (240, 240), at, rest,
                                   seconds=.45, steps=4, effect_over=40.0)


@pytest.mark.parametrize("axis,sign,key,keys", [(0, 1, "j", ("j", "k", "l")), (0, -1, "l", ("k", "j", "l")),
                                              (1, 1, "b", ("c", "b", "a")), (1, -1, "c", ("c", "a", "b"))])
def test_a_measured_trial_works_in_each_orientation_with_arbitrary_control_labels(axis, sign, key, keys):
    thing, hers = _scene(axis, sign, key=key)
    response = _response(thing, hers, keys)
    assert response is not None and response.key == key and response.number == 1
    assert response.epoch == 2
    assert hers.number is None and hers.identification.receipts == []


def test_an_independently_falling_thing_does_not_answer_the_same_pulse():
    thing, hers = _scene()
    stone, other = _scene(number=2)
    other._by_thing[2].timed[("a", 0.0)] = [(t, 0.0, 20.0 + 90.0 * t) for t, _x, _y in other._by_thing[2].timed[("a", 0.0)]]
    hers._by_thing[2] = other._by_thing[2]
    response = response_worth_repeating(["a", "b"], [thing, stone], hers, (240, 240), .9, .5,
                                       seconds=.45, steps=4, effect_over=40)
    assert response is not None and response.number == 1


def test_competing_responding_tracks_leave_the_experiment_unselected():
    thing, hers = _scene()
    second, other = _scene(number=2)
    hers._by_thing.update(other._by_thing)
    assert response_worth_repeating(["a"], [thing, second], hers, (240, 240), .9, .5,
                                   seconds=.45, steps=4, effect_over=40) is None
    assert hers.number is None


@pytest.mark.parametrize("fault", ["stale", "no_rest", "mixed", "overlap", "noisy"])
def test_unobserved_or_mixed_delivery_samples_cannot_suggest_a_response(fault):
    thing, hers = _scene()
    record = hers._by_thing[1]
    if fault == "stale":
        thing.seen = .8
    elif fault == "no_rest":
        record.timed.pop(("", .5))
    elif fault == "mixed":
        record.timed[("", .4)] = record.timed.pop(("", .5))
    elif fault == "overlap":
        record.timed[("a", .0)].append((.55, .0, -10.0))
    else:
        record.timed[("a", .0)][2] = (.3, 100.0, -200.0)
    assert _response(thing, hers) is None


def test_changed_epoch_ends_a_pending_experiment_before_old_measurements_are_used():
    trial = CounteringMotion()
    options = dict(held="", held_since=None, settle_s=.2, step_s=.45, steps=4,
                   wait_s=3, against=lambda _keys, _things: "")
    assert trial.choose(["a"], [], .0, epoch=1, **options) is None
    assert trial.choose(["a"], [], .5, epoch=2, **options) is None
    assert trial.done and trial.response is None


def test_a_provisional_trial_ends_when_its_track_has_no_fresh_observation():
    trial = CounteringMotion()
    witness = SimpleNamespace(number=1, key="a", epoch=2)
    track = SimpleNamespace(number=1, seen=1.0, born=.0, vx=.0, vy=40., w=12., h=12.,
                            path=[(.8, 1., 1.), (1., 1., 4.)])
    options = dict(held_since=.0, epoch=2, settle_s=.2, step_s=.45, steps=4,
                   wait_s=3, against=lambda _keys, _things: "", measured=lambda _keys, _things: witness)
    assert trial.choose(["a"], [track], 1., held="", **options) == "a"
    options["held_since"] = 1.
    assert trial.choose(["a"], [track], 1.5, held="a", **options) is None
    assert trial.done


def test_the_registered_invariant_runs_real_measurements_without_identity_authority():
    from core.agency.a_response_worth_measuring_again import _provisional_response_invariant

    assert _provisional_response_invariant() == ()
