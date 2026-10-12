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
