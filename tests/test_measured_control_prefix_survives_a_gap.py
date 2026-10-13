"""A missing observation closes measured work without creating another trial."""
import math

import numpy as np
import pytest

from core.agency.controls_measured_on_one_thing import own_control_witness
from core.agency.which_one_answers_to_her import WhichIsHers, _measured_prefix_invariant, _Speeds
from core.perception.what_moves_in_the_picture import Thing, WhatMoves

pytestmark = pytest.mark.unit


def _prefix(samples, key='a', began=1., speed=30.):
    for at in (began + .25, began + .3):
        samples.add(key, speed, 0., press=began, when=at)


def test_gap_keeps_only_the_first_complete_continuous_interval():
    samples = _Speeds()
    _prefix(samples)
    before = list(samples.timed[('a', 1.)])
    samples.unseen('a', 1.)
    for at in (1.6, 1.7):
        samples.add('a', -300., 0., press=1., when=at)
    samples.unseen('a', 1.)
    assert samples.timed[('a', 1.)] == before
    assert samples.by_press[('a', 1.)] == [(30., 0.), (30., 0.)]
    assert samples.ratio() == (0., 0.)
    assert own_control_witness(7, samples, epoch=1) is None


def test_contradiction_after_a_gap_retires_the_trial_before_input_completion():
    samples = _Speeds()
    _prefix(samples)
    samples.unseen('a', 1.)
    samples.unseen('a', 1., contradicted=True)
    samples.add('a', 30., 0., press=1., when=1.6)
    assert not samples.by_press and not samples.timed and not samples.pinned_in
    assert ('a', 1.) in samples.interrupted


def test_later_inputs_cannot_revoke_a_completed_measurement():
    samples = _Speeds()
    _prefix(samples)
    samples.unseen('a', 1.)
    samples.end('a', 1.)
    samples.unseen('a', 1., contradicted=True)
    assert samples.by_press[('a', 1.)] == [(30., 0.), (30., 0.)]


def test_repeated_independent_press_prefixes_still_establish_a_measured_response():
    samples = _Speeds()
    for key, began, speed in [('a', 1., 30.), ('b', 3., -30.), ('a', 5., 30.), ('b', 7., -30.)]:
        _prefix(samples, key, began, speed)
        samples.unseen(key, began)
        samples.add(key, 300., 0., press=began, when=began + 1.)
    witness = own_control_witness(7, samples, epoch=2)
    assert witness is not None and witness.identity == 7
    assert witness.independent_trials == (('a', 2), ('b', 2))
    assert witness.statistic > 20 and witness.effect == 60.
    assert own_control_witness(8, _Speeds(), epoch=2) is None


def test_normal_body_discovery_keeps_valid_trials_when_later_motion_is_unavailable():
    hers, moves = WhichIsHers(), WhatMoves()
    moves.shape = (100, 100)
    body = Thing(7, 50., 50., 10., 10., np.ones(3), (100, 150, 200),
                 np.zeros((10, 10, 3), np.uint8), 0., 1., kind=0, moved=True)
    moves.things[7] = body
    for key, began, velocity in [('go', 1., 80.), ('back', 3., -80.),
                                  ('go', 5., 80.), ('back', 7., -80.)]:
        hers.holding(key, began, trying=True)
        for after in (.25, .35, .45):
            body.seen = began + after
            body.motion_evidence = 'measured' if after < .45 else 'unavailable'
            body.motion_measured_at = body.seen if after < .45 else None
            body.vx = velocity
            body.x += velocity * .1
            body.path.append((body.seen, body.x, body.y))
            hers.saw(moves, [], body.seen)
        hers.holding('', began + .8, trying=False)
    assert hers.number == 7
    assert all(len(values) == 2 for values in hers._by_thing[7].by_press.values())


@pytest.mark.parametrize('failure', ['untimed', 'duplicate', 'backward', 'before_press', 'nonfinite', 'mismatch'])
def test_incomplete_or_invalid_samples_do_not_get_sealed(failure):
    samples = _Speeds()
    _prefix(samples)
    if failure == 'untimed':
        samples.timed.clear()
    elif failure == 'duplicate':
        samples.timed[('a', 1.)][1] = samples.timed[('a', 1.)][0]
    elif failure == 'backward':
        samples.timed[('a', 1.)].reverse()
    elif failure == 'before_press':
        samples.timed[('a', 1.)][0] = (.5, 30., 0.)
    elif failure == 'nonfinite':
        samples.timed[('a', 1.)][0] = (math.nan, 30., 0.)
    else:
        samples.timed[('a', 1.)][0] = (1.25, 300., 0.)
    samples.unseen('a', 1.)
    assert not samples.by_press and not samples.timed
    assert ('a', 1.) not in samples.ended
    samples.add('a', 10., 0., press=1., when=1.6)
    assert samples.by_press[('a', 1.)] == [(10., 0.)]


def test_gap_bookkeeping_is_bounded_with_the_input_intervals():
    samples = _Speeds()
    for began in range(100):
        _prefix(samples, began=float(began))
        samples.unseen('a', float(began))
    assert len(samples.ended) == len(samples.sealed_by_gap) == 80
    assert samples.sealed_by_gap == samples.ended
    assert len(samples.by_press) == len(samples.timed) == 80


def test_pinned_prefix_keeps_its_original_constraints():
    samples = _Speeds()
    for at in (1.25, 1.3):
        samples.add('a', 0., 0., press=1., when=at, pinned=True)
    samples.unseen('a', 1.)
    assert samples.pinned_in[('a', 1.)] == 2
    assert samples.typical('a') is None


def test_gap_prefix_invariant_is_registered():
    assert _measured_prefix_invariant() == ()
