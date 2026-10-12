"""Input receipts distinguish a returning path from coincident autonomous motion."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.agency.controls_measured_on_one_thing import own_trajectory_witness
from core.agency.what_a_press_does import PressTrajectory, WhatAPressDoes

pytestmark = pytest.mark.unit


def _receipt(key, began, *, lift=True):
    samples = tuple((i / 10, 0., -40 * (1 - abs(i - 5) / 5) if lift else 0.) for i in range(1, 11))
    return PressTrajectory(key, 7, 2, began, began + 1., -1., samples)


def _trials():
    return [_receipt('a', 1), _receipt('', 3, lift=False), _receipt('a', 5), _receipt('', 7, lift=False)]


def _body():
    return SimpleNamespace(number=7, born=-1., seen=9.)


def test_returning_paths_identify_only_the_body_measured_in_repeated_trials():
    witness = own_trajectory_witness(_body(), _trials(), epoch=2)
    assert witness and witness.effect > 30 and witness.statistic > 20
    assert own_trajectory_witness(replace_body(number=8), _trials(), epoch=2) is None
    assert own_trajectory_witness(replace_body(born=0.), _trials(), epoch=2) is None
    assert own_trajectory_witness(_body(), _trials(), epoch=3) is None


def replace_body(**kwargs):
    return SimpleNamespace(**{**vars(_body()), **kwargs})


def test_repeated_autonomous_paths_do_not_supply_an_input_effect():
    trials = [replace(trial, samples=_receipt('a', 1).samples) for trial in _trials()]
    witness = own_trajectory_witness(_body(), trials, epoch=2)
    assert witness and witness.effect == 0 and witness.statistic == 0


def test_duplicate_or_overlapping_intervals_cannot_count_as_repeats():
    assert own_trajectory_witness(_body(), [_trials()[0]] * 2 + [_trials()[1]] * 2, epoch=2) is None


def test_changed_input_ends_the_entity_receipt_before_other_actions():
    presses = WhatAPressDoes()
    body = SimpleNamespace(number=7, kind=3, born=-1., seen=1., x=50., y=100., vx=0., vy=0., w=10., h=10., path=[])
    presses.pressed('a', 1., [body], epoch=2, independent=True)
    for i in range(1, 5):
        body.seen, body.y = 1 + i / 10, 100 - i * 8
        presses.saw({7: body}, body.seen)
    presses.pressed('b', 1.5, [body], epoch=2, independent=False)
    assert len(presses.entity_presses) == 1
    receipt = presses.entity_presses[0]
    assert receipt.key == 'a' and receipt.ended == pytest.approx(1.4)
    assert all(not watch.independent for watch in presses.watching)


def test_a_later_input_contradiction_cannot_revoke_an_ended_measured_receipt():
    presses = WhatAPressDoes()
    body = SimpleNamespace(number=7, kind=3, born=-1., seen=1., x=50., y=100.,
                           vx=0., vy=0., w=10., h=10., path=[],
                           motion_measured_at=1., motion_evidence='measured')
    presses.pressed('a', 1., [body], epoch=2, independent=True)
    for index in range(1, 5):
        body.seen = body.motion_measured_at = 1. + index * .1
        body.y -= 8.
        presses.saw({7: body}, body.seen)
    presses.pressed('b', 1.41, [body], epoch=2, independent=True)
    receipt = presses.entity_presses[0]
    assert receipt.key == 'a' and receipt.ended == pytest.approx(1.4)
    body.seen, body.motion_measured_at, body.motion_evidence = 1.5, None, 'contradicted'
    presses.saw({7: body}, body.seen)
    assert list(presses.entity_presses) == [receipt]


def test_unobserved_or_planned_inputs_cannot_be_identity_receipts():
    for independent, seen in ((False, 1.), (True, 0.)):
        presses = WhatAPressDoes()
        body = SimpleNamespace(number=7, kind=3, born=-1., seen=seen, x=50., y=100., vx=0., vy=0., w=10., h=10., path=[])
        presses.pressed('a', 1., [body], epoch=2, independent=independent)
        for i in range(1, 8):
            body.seen = 1 + i / 10
            presses.saw({7: body}, body.seen)
        assert not presses.entity_presses


def test_registered_trajectory_invariant():
    from core.agency.controls_measured_on_one_thing import _trajectory_control_invariant

    assert _trajectory_control_invariant() == ()


def test_registered_current_motion_invariant():
    from core.agency.finding_her_another_way import _current_body_motion_invariant

    assert _current_body_motion_invariant() == ()


def test_a_late_measured_response_revises_the_same_trial_curve():
    presses = WhatAPressDoes()
    body = SimpleNamespace(number=7, kind=3, born=-1., seen=1., x=50., y=100.,
                           vx=0., vy=0., w=10., h=10., path=[])
    heights = [0.] * 7 + [8., 16., 24., 24., 16., 8., 0.] + [0.] * 10
    for trial in range(3):
        began = 1. + trial * 2.
        body.seen, body.y = began, 100.
        presses.pressed('a', began, [body], epoch=2, independent=True)
        for index, height in enumerate(heights, 1):
            body.seen, body.y = began + index * .05, 100. - height
            presses.saw({7: body}, body.seen)
        presses.released('a', body.seen)

    curves = list(presses.by_number[('a', 7)])
    assert len(curves) == len(presses.curves[('a', 3)]) == 3
    assert len({id(curve) for curve in curves}) == 3
    assert all(min(dy for _at, _dx, dy in curve) == -24. for curve in curves)
    assert all(curve[-1][2] == 0. for curve in curves)
    assert presses.numbers_it_moves(4.) == {7}
    assert presses.lifts(3, 10., 7) == ['a']
    assert len(presses.entity_presses) == 3
    assert len({receipt.began for receipt in presses.entity_presses}) == 3
    assert all(min(dy for _at, _dx, dy in receipt.samples) == -24.
               for receipt in presses.entity_presses)


def test_an_unobserved_late_response_cannot_reopen_a_settled_curve():
    presses = WhatAPressDoes()
    body = SimpleNamespace(number=7, kind=3, born=-1., seen=1., x=50., y=100.,
                           vx=0., vy=0., w=10., h=10., path=[],
                           motion_measured_at=1., motion_evidence='measured')
    presses.pressed('a', 1., [body], epoch=2, independent=True)
    for index in range(1, 6):
        body.seen = body.motion_measured_at = 1. + index * .05
        presses.saw({7: body}, body.seen)
    settled = tuple(presses.by_number[('a', 7)][0])
    body.motion_measured_at, body.motion_evidence = None, 'unavailable'
    for index in range(6, 20):
        body.seen, body.y = 1. + index * .05, 76.
        presses.saw({7: body}, body.seen)

    assert tuple(presses.by_number[('a', 7)][0]) == settled
    assert len(presses.by_number[('a', 7)]) == 1
    assert not presses.numbers_it_moves(4.) and not presses.lifts(3, 10., 7)
    assert len(presses.entity_presses) == 1
    receipt = presses.entity_presses[0]
    assert receipt.ended == pytest.approx(1.25)
    assert all(dx == dy == 0. for _at, dx, dy in receipt.samples)


@pytest.mark.parametrize('selection', ['speeds', 'kind', 'trajectory', 'discovery'])
@pytest.mark.parametrize('missing', ['unavailable', 'contradicted', 'stale'])
def test_historical_control_evidence_cannot_establish_a_current_unmeasured_body(selection, missing):
    from core.agency.finding_her_another_way import found_by_press_curves
    from core.agency.which_one_answers_to_her import WhichIsHers

    hers = WhichIsHers()
    hers.identification.epoch = 2
    body = SimpleNamespace(number=7, kind=3, born=-1., seen=9., x=50., y=100., size=100.,
                           vx=0., vy=0., w=10., h=10., moved=True,
                           motion_measured_at=None, motion_evidence=missing)
    if missing == 'stale':
        body.seen = body.motion_measured_at = 8.9
        body.motion_evidence = 'measured'
    for key, began, speed in (('a', 1., 60.), ('b', 3., -60.), ('a', 5., 60.), ('b', 7., -60.)):
        for index in range(2):
            hers._by_thing[7].add(key, speed, 0., press=began, position=(index * 30., 100.),
                                 when=began + .1 + index * .1)
    hers._by_kind[3] = hers._by_thing[7]
    presses = WhatAPressDoes()
    presses.entity_presses.extend(_trials())
    presses.curves[('a', 3)].extend([list(_receipt('a', 1.).samples) for _ in range(3)])
    presses.by_number[('a', 7)].extend(presses.curves[('a', 3)])
    moves, run = SimpleNamespace(things={7: body}), SimpleNamespace(presses=presses)
    if selection in ('trajectory', 'discovery'):
        hers._by_thing.clear()

    def select():
        if selection == 'kind':
            return hers._the_one_of_a_kind_that_answers(moves, 9.)
        if selection == 'speeds':
            hers._decide(moves, 9.)
        elif selection == 'trajectory':
            hers.moved_by_her_presses(body, 9., receipts=presses.entity_presses)
        else:
            found_by_press_curves(run, moves, hers, 9., old_for=.5)
        return hers.number

    assert select() is None
    assert not hers.identification.receipts
    body.seen = body.motion_measured_at = 9.
    body.motion_evidence = 'measured'
    assert select() == 7


@pytest.mark.asyncio
async def test_delivered_trials_record_neutral_rest_but_planned_holds_do_not_claim_independence():
    from collections import Counter, deque
    from unittest.mock import AsyncMock, Mock

    from core.agency.playing_as_it_happens import _hold

    presses = Mock()
    run = SimpleNamespace(held="", held_trying=False, presses=presses, things={},
                           input_key_downs=Counter(), responses=deque())
    hers = SimpleNamespace(number=None, identification=SimpleNamespace(epoch=8), tapped=Mock(), holding=Mock())
    hands = SimpleNamespace(up=AsyncMock(), down=AsyncMock())
    await _hold(hands, run, hers, "a", 1., trying=True)
    assert presses.pressed.call_args.kwargs == {"epoch": 8, "independent": True}
    await _hold(hands, run, hers, "", 2., trying=True)
    assert presses.pressed.call_args.args[0] == ""
    assert presses.pressed.call_args.kwargs["independent"] is True
    await _hold(hands, run, hers, "a", 3., trying=False)
    assert presses.pressed.call_args.kwargs["independent"] is False
