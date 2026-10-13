"""Prediction inputs have a physical deadline independent of picture delivery."""
import asyncio
import time

import pytest

from core.agency.inputs_while_measuring_control import hold_control, release_control_if_owned
from core.agency.playing_as_it_happens import _Run
from core.agency.steering_from_a_model import _release_after_prediction
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def _setup():
    run, hers = _Run(keys=['a', 'b'], began=time.monotonic()), WhichIsHers()
    actions = []

    class Hands:
        async def down(self, key):
            actions.append(('down', key))
        async def up(self, key):
            actions.append(('up', key))

    return run, hers, actions, Hands()


@pytest.mark.asyncio
async def test_a_press_expires_when_no_other_picture_arrives():
    run, hers, actions, hands = _setup()
    await hold_control(hands, run, hers, 'a', time.monotonic(), trying=False)
    since = run.held_since
    await asyncio.wait_for(_release_after_prediction(hands, run, hers, 'a', since, .01), timeout=1.)
    assert actions == [('down', 'a'), ('up', 'a')] and not run.held
    assert hers._held[-1][1:] == ('', False)


@pytest.mark.asyncio
async def test_an_old_deadline_cannot_release_a_new_press_of_the_same_key():
    run, hers, actions, hands = _setup()
    await hold_control(hands, run, hers, 'a', time.monotonic())
    since = run.held_since
    await hold_control(hands, run, hers, '', time.monotonic())
    await hold_control(hands, run, hers, 'a', time.monotonic())
    before = list(actions)
    assert not await release_control_if_owned(hands, run, hers, 'a', since)
    assert actions == before and run.held == 'a'


@pytest.mark.asyncio
async def test_an_old_deadline_cannot_release_another_key():
    run, hers, actions, hands = _setup()
    await hold_control(hands, run, hers, 'a', time.monotonic())
    since = run.held_since
    await hold_control(hands, run, hers, 'b', time.monotonic())
    before = list(actions)
    assert not await release_control_if_owned(hands, run, hers, 'a', since)
    assert actions == before and run.held == 'b'


@pytest.mark.asyncio
async def test_expiry_and_new_input_delivery_share_one_serial_boundary():
    run, hers, actions, _hands = _setup()
    entered, released = asyncio.Event(), asyncio.Event()

    class Hands:
        async def down(self, key):
            actions.append(('down', key))
        async def up(self, key):
            entered.set()
            await released.wait()
            actions.append(('up', key))

    hands = Hands()
    await hold_control(hands, run, hers, 'a', time.monotonic())
    expiry = asyncio.create_task(release_control_if_owned(hands, run, hers, 'a', run.held_since))
    await entered.wait()
    next_input = asyncio.create_task(hold_control(hands, run, hers, 'b', time.monotonic()))
    await asyncio.sleep(0)
    assert actions == [('down', 'a')]
    released.set()
    await asyncio.wait_for(asyncio.gather(expiry, next_input), timeout=1.)
    assert actions == [('down', 'a'), ('up', 'a'), ('down', 'b')]
    assert run.held == 'b'
