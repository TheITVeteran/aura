"""Missing pictures split measured arcs without inventing another input trial."""
import math

import pytest

from core.agency.turning_while_travelling import TurningWhileTravelling

pytestmark = pytest.mark.unit


def _observer():
    return TurningWhileTravelling(moving=20., settle_s=.2, gap_s=.35, turns=.5)


def _segment(turns, key, began, start, rate, *, count=5, phase=0.):
    for index in range(count):
        at = began + start + index * .1
        angle = phase + rate * index * .1
        turns.saw(key, began, at, 80 * math.cos(angle), 80 * math.sin(angle), measured=True)


def _close(turns, at):
    turns.saw('other', at, at, 0., 0., measured=False)


def _learn(turns, before):
    for index, key in enumerate(('', 'bend', '', 'bend', '')):
        began, rate = index * 2., 1.7 if key else 0.
        before(turns, key, began, rate)
        _segment(turns, key, began, .8, rate)
    _close(turns, 10.)


@pytest.mark.parametrize('failure', ['missing', 'nonfinite', 'short_prefix', 'gap', 'stopped'])
def test_a_later_continuing_arc_can_measure_the_same_delivered_press(failure):
    turns = _observer()

    def before(turns, key, began, rate):
        if failure in ('short_prefix', 'gap'):
            _segment(turns, key, began, .25, rate, count=2)
        if failure != 'gap':
            vx = float('nan') if failure == 'nonfinite' else 0.
            turns.saw(key, began, began + .5, vx, 0., measured=failure == 'stopped')

    _learn(turns, before)
    assert turns.rates() == {'bend': pytest.approx(1.7)}
    assert sum(t.key == 'bend' for t in turns.trials) == 2


def test_two_short_segments_do_not_join_across_an_unseen_picture():
    turns = _observer()
    for index, key in enumerate(('', 'bend', '', 'bend', '')):
        began, rate = index * 2., 1.7 if key else 0.
        _segment(turns, key, began, .25, rate, count=2)
        turns.saw(key, began, began + .5, 0., 0., measured=False)
        _segment(turns, key, began, .6, rate, count=2)
    _close(turns, 10.)
    assert not turns.rates()
    assert all(t.samples == 2 for t in turns.trials)


def test_a_complete_prefix_survives_a_missing_tail():
    turns = _observer()

    def before(turns, key, began, rate):
        _segment(turns, key, began, .25, rate)
        turns.saw(key, began, began + .7, 0., 0., measured=False)

    _learn(turns, before)
    assert turns.rates() == {'bend': pytest.approx(1.7)}
    assert len({(t.key, t.began) for t in turns.trials}) == len(turns.trials)


def test_separated_arcs_from_one_press_cannot_supply_independent_repeats():
    turns = _observer()
    _segment(turns, '', 0., .25, 0.)
    _segment(turns, 'bend', 2., .25, 1.7)
    turns.saw('bend', 2., 2.7, 0., 0., measured=False)
    _segment(turns, 'bend', 2., .8, 1.7)
    _segment(turns, '', 4., .25, 0.)
    _close(turns, 6.)
    assert not turns.rates()
    assert sum(t.key == 'bend' for t in turns.trials) == 1


def test_a_contradictory_continuing_suffix_retires_its_press():
    turns = _observer()
    _learn(turns, lambda *args: None)
    assert turns.rates()
    _segment(turns, 'bend', 12., .25, 1.7)
    turns.saw('bend', 12., 12.7, 0., 0., measured=False)
    _segment(turns, 'bend', 12., .8, -1.7)
    _close(turns, 14.)
    assert not turns.rates()
    assert not any(t.began == 12. for t in turns.trials)
    assert ('bend', 12.) in turns.retired


def test_a_stop_breaks_the_arc_but_a_later_arc_remains_observable():
    turns = _observer()
    _segment(turns, 'bend', 0., .25, 1.7, count=2)
    turns.saw('bend', 0., .5, 0., 0., measured=True)
    _segment(turns, 'bend', 0., .6, 1.7)
    _close(turns, 2.)
    assert len(turns.trials) == 1 and turns.trials[0].samples == 5


def test_two_new_independent_presses_are_required_after_a_contradiction():
    turns = _observer()
    _learn(turns, lambda *args: None)
    _segment(turns, 'bend', 12., .25, 1.7)
    turns.saw('bend', 12., 12.7, 0., 0., measured=False)
    _segment(turns, 'bend', 12., .8, -1.7)
    _close(turns, 14.)
    _segment(turns, 'bend', 16., .25, 1.7)
    _close(turns, 18.)
    assert not turns.rates()
    _segment(turns, 'bend', 20., .25, 1.7)
    _close(turns, 22.)
    assert turns.rates() == {'bend': pytest.approx(1.7)}


def test_retiring_other_presses_cannot_resurrect_old_contradicted_evidence():
    turns = _observer()
    _learn(turns, lambda *args: None)
    for index in range(66):
        key, began = ('bend' if index == 0 else 'other'), 12. + index * 2.
        _segment(turns, key, began, .25, 1.7)
        turns.saw(key, began, began + .7, 0., 0., measured=False)
        _segment(turns, key, began, .8, -1.7)
        _close(turns, began + 1.5)
    assert len(turns.retired) == 64 and not turns.rates()
    assert not any(t.key == 'bend' for t in turns.trials)


def test_a_reversal_cannot_bridge_two_short_arcs():
    turns = _observer()
    _segment(turns, 'bend', 0., .25, 1.7, count=2)
    _segment(turns, 'bend', 0., .45, 1.7, count=2, phase=math.pi)
    _close(turns, 2.)
    assert not turns.rates()
    assert all(t.samples < 4 for t in turns.trials)


def test_every_qualified_arc_must_agree_with_every_other_one_under_that_input():
    turns = _observer()
    _segment(turns, '', 0., .25, 0.)
    _segment(turns, 'bend', 2., .25, 1.)
    _segment(turns, '', 4., .25, 0.)
    for start, rate in ((.25, 1.), (.9, 1.29), (1.55, .71)):
        _segment(turns, 'bend', 6., start, rate)
        turns.saw('bend', 6., 6. + start + .45, 0., 0., measured=False)
    _close(turns, 9.)
    assert ('bend', 6.) in turns.retired and not turns.rates()


def test_a_retired_rolling_control_still_requests_new_experiments():
    from core.agency.which_way_she_faces import WhichWaySheFaces

    faces = WhichWaySheFaces(facing=0.)
    faces.along.update(drive=[80., 80.], bend=[120., 120.])
    faces.ways['drive'] = [0., 2.1, -2.1]
    faces.pressed.update(('drive', 'bend'))
    faces.last_tried = 'drive'
    turns = faces.travel_turns
    _learn(turns, lambda *args: None)
    _segment(turns, 'bend', 12., .25, 1.7)
    turns.saw('bend', 12., 12.7, 0., 0., measured=False)
    _segment(turns, 'bend', 12., .8, -1.7)
    _close(turns, 14.)
    assert not turns.rates() and turns.repeats('bend') == 0
    assert faces.ahead_key() == 'drive' and faces.trial(['drive', 'bend']) == 'bend'
    _segment(turns, 'bend', 16., .25, 0.)
    _close(turns, 18.)
    assert turns.needs_reprobe('bend')
    _segment(turns, 'bend', 20., .25, 0.)
    _close(turns, 22.)
    assert not turns.needs_reprobe('bend') and not turns.rates()


def test_separate_repeats_must_compare_all_their_observed_arcs():
    turns = _observer()
    _segment(turns, '', 0., .25, 0.)
    _segment(turns, '', 2., .25, 0.)
    for began, rate in ((4., 1.29), (6., .71)):
        _segment(turns, 'bend', began, .25, 1.)
        turns.saw('bend', began, began + .7, 0., 0., measured=False)
        _segment(turns, 'bend', began, .8, rate)
    _close(turns, 8.)
    assert not turns.retired and not turns.rates()


def test_the_segment_boundary_has_a_registered_measured_claim():
    from core.agency.turning_while_travelling import _travel_segments_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _travel_segments_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'travel_segments_keep_one_input_trial')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
