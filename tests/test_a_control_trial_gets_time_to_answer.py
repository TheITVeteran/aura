"""A planned control trial holds its input until there has been time to measure it."""
from types import SimpleNamespace

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def setup(monkeypatch, *, trials=0):
    now = [10.0]
    monkeypatch.setattr(play.time, "monotonic", lambda: now[0])
    keys = iter(["drive", "yaw", "drive"])
    faces = SimpleNamespace(trial=lambda _keys: next(keys))
    run = play._Run(keys=["drive", "yaw"], began=0, faces=faces, facing_trials=trials)
    hers = WhichIsHers()
    hers.number = 1
    choosing = SimpleNamespace(mine=SimpleNamespace(number=1))
    actions = []

    class Hands:
        async def down(self, key):
            actions.append(("down", key))

        async def up(self, key):
            actions.append(("up", key))

    return now, run, hers, choosing, actions, Hands()


@pytest.mark.asyncio
async def test_a_trial_owns_each_picture_until_its_measurement_time(monkeypatch):
    now, run, hers, choosing, actions, hands = setup(monkeypatch)
    assert await play._a_facing_trial(hands, run, hers, choosing, now[0])
    for _ in range(6):
        now[0] += .05
        assert await play._a_facing_trial(hands, run, hers, choosing, now[0])
    assert actions == [("down", "drive")]
    now[0] += play.TRY_A_KEY_S
    assert await play._a_facing_trial(hands, run, hers, choosing, now[0])
    assert actions == [("down", "drive"), ("up", "drive"), ("down", "yaw")]


@pytest.mark.asyncio
async def test_the_last_trial_keeps_its_time_after_the_count_limit(monkeypatch):
    now, run, hers, choosing, actions, hands = setup(monkeypatch, trials=play.FACING_TRIALS - 1)
    assert await play._a_facing_trial(hands, run, hers, choosing, now[0])
    now[0] += .05
    assert await play._a_facing_trial(hands, run, hers, choosing, now[0])
    assert actions == [("down", "drive")]
    now[0] += play.TRY_A_KEY_S
    assert not await play._a_facing_trial(hands, run, hers, choosing, now[0])


@pytest.mark.asyncio
async def test_a_changed_identity_ends_the_old_control_trial(monkeypatch):
    now, run, hers, choosing, _actions, hands = setup(monkeypatch)
    assert await play._a_facing_trial(hands, run, hers, choosing, now[0])
    hers.identification.reset("the measured body disappeared")
    now[0] += .05
    assert not await play._a_facing_trial(hands, run, hers, choosing, now[0])


def test_the_registered_control_trial_time_invariant():
    from core.agency.giving_a_trial_time import _control_trial_time_invariant

    assert _control_trial_time_invariant() == ()


@pytest.mark.asyncio
async def test_the_actor_does_not_plan_over_an_active_trial(monkeypatch):
    now, run, hers, choosing, actions, hands = setup(monkeypatch)
    run.trying = 2 * len(run.keys)
    choosing.ways = {key: (90, 0) for key in run.keys}
    choosing.lifts = []
    moves = SimpleNamespace(seeing=True, things={})

    def planning(*_args):
        raise AssertionError("planning interrupted a control trial")

    monkeypatch.setattr(play, "_where_to_meet_things", planning)
    await play._act(hands, run, moves, hers, None, choosing, now[0])
    now[0] += .05
    await play._act(hands, run, moves, hers, None, choosing, now[0])
    assert actions == [("down", "drive")]


@pytest.mark.asyncio
async def test_a_new_direction_trial_physically_represses_the_requested_held_input(monkeypatch):
    now, run, hers, choosing, actions, hands = setup(monkeypatch)
    now[0] = 8.
    await play._hold(hands, run, hers, 'drive', now[0], trying=False)
    old_began = run.held_since
    now[0] = 10.
    assert await play._a_facing_trial(hands, run, hers, choosing, now[0])
    assert actions == [('down', 'drive'), ('up', 'drive'), ('down', 'drive')]
    assert run.facing_trial.began == run.held_since > old_began
    assert run.facing_trial.until == pytest.approx(now[0] + play.TRY_A_KEY_S)
    assert hers._held[-1] == (run.held_since, 'drive', True)
    now[0] += .3
    assert await play._a_facing_trial(hands, run, hers, choosing, now[0])
    assert len(actions) == 3
