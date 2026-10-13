"""Calibration counts requests for one current body, without borrowing its predecessor's time."""
import math
from types import SimpleNamespace

import pytest

from core.agency.calibration_for_a_current_body import (
    MAX_CALIBRATION_CONTROLS,
    MAX_CALIBRATION_OWNERS,
    _current_body_calibration_invariant,
    calibration_for_current_body,
    calibration_owner,
)
from core.agency.giving_a_trial_time import try_a_facing_control
from core.agency.which_way_she_faces import WhichWaySheFaces

pytestmark = pytest.mark.unit


def _setup(*, keys=("drive", "bend"), key="bend"):
    faces = SimpleNamespace(trial=lambda _keys: key)
    run = SimpleNamespace(keys=list(keys), faces=faces, facing_trials=0, facing_trial=None,
                          held="", held_since=0., held_trying=False, tried_at=-math.inf,
                          intervals=(), longest_control_trial_s=0.)
    hers = SimpleNamespace(number=7, identification=SimpleNamespace(epoch=1), follows_pointer=False)
    choosing = SimpleNamespace(mine=SimpleNamespace(number=7, born=0.))
    actions = []

    async def hold(_hands, run, _hers, key, at, *, trying):
        actions.append((key, at))
        run.held, run.held_since, run.held_trying = key, at, trying

    return run, hers, choosing, actions, hold


async def _ask(run, hers, choosing, hold, at):
    return await try_a_facing_control(None, run, hers, choosing, at,
                                     hold=hold, seconds=.7, limit=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ("epoch", "number", "born"))
async def test_an_exhausted_predecessor_cannot_spend_the_current_bodys_programme(changed):
    run, hers, choosing, actions, hold = _setup()
    for index in range(10):
        assert await _ask(run, hers, choosing, hold, 10. + index)
    assert not await _ask(run, hers, choosing, hold, 20.)
    if changed == "epoch":
        hers.identification.epoch += 1
    elif changed == "number":
        hers.number = choosing.mine.number = 8
    else:
        choosing.mine.born = 5.
    assert await _ask(run, hers, choosing, hold, 21.)
    assert len(actions) == run.facing_trials == 11
    programme = calibration_for_current_body(run, hers, choosing.mine, minimum=10)
    assert programme.issued == 1


@pytest.mark.asyncio
async def test_the_final_request_keeps_its_full_time_but_the_same_owner_stays_exhausted():
    run, hers, choosing, actions, hold = _setup()
    for index in range(10):
        assert await _ask(run, hers, choosing, hold, 10. + index)
    final = run.facing_trial
    assert await _ask(run, hers, choosing, hold, final.began + .1)
    assert run.facing_trial is final and len(actions) == 10
    assert not await _ask(run, hers, choosing, hold, final.until + .1)
    assert not await _ask(run, hers, choosing, hold, final.until + 2.)
    assert len(actions) == run.facing_trials == 10


@pytest.mark.asyncio
async def test_neutral_requests_without_measurements_still_spend_the_finite_programme():
    run, hers, choosing, actions, hold = _setup(key="")
    choosing.mine.motion_measured = False
    for index in range(10):
        assert await _ask(run, hers, choosing, hold, 10. + index)
    assert not await _ask(run, hers, choosing, hold, 20.)
    assert len(actions) == 10 and all(key == "" for key, _at in actions)


def test_key_permutations_duplicates_and_later_changes_cannot_replenish_one_owner():
    run, hers, choosing, _actions, _hold = _setup(keys=("drive", "bend", "bend", ""))
    first = calibration_for_current_body(run, hers, choosing.mine, minimum=10)
    assert first.limit == 10
    assert all(first.take_trial() for _ in range(10))
    for keys in (["bend", "drive"], ["drive"] * 30 + ["bend"] * 30 + [""],
                 ["drive", "bend", "another", "one_more"]):
        run.keys = keys
        same = calibration_for_current_body(run, hers, choosing.mine, minimum=50)
        assert same is first and same.limit == 10 and not same.take_trial()


def test_missing_ownership_cannot_renew_a_previously_known_budget():
    run, hers, choosing, _actions, _hold = _setup()
    first = calibration_for_current_body(run, hers, choosing.mine, minimum=10)
    assert all(first.take_trial() for _ in range(first.limit))
    assert calibration_for_current_body(run, hers, SimpleNamespace(number=7), minimum=10) is None
    assert calibration_for_current_body(run, hers, choosing.mine, minimum=10) is first
    assert not first.take_trial()
    old_owner = first.owner
    hers.identification.epoch += 1
    assert calibration_for_current_body(run, hers, choosing.mine, minimum=10).take_trial()
    hers.identification.epoch = old_owner[0]
    assert calibration_for_current_body(run, hers, choosing.mine, minimum=10) is first
    assert not first.take_trial()


@pytest.mark.asyncio
async def test_full_owner_storage_declines_new_bodies_without_renewing_retained_ones():
    run, hers, choosing, actions, hold = _setup()
    first = calibration_for_current_body(run, hers, choosing.mine, minimum=10)
    for index in range(1, MAX_CALIBRATION_OWNERS):
        hers.identification.epoch = 1 + index
        assert calibration_for_current_body(run, hers, choosing.mine, minimum=10) is not None
    hers.identification.epoch += 1
    assert calibration_for_current_body(run, hers, choosing.mine, minimum=10) is None
    assert not await _ask(run, hers, choosing, hold, 10.)
    assert not actions and run.facing_trials == 0
    assert len(run._facing_calibration_programmes) == MAX_CALIBRATION_OWNERS
    assert all(first.take_trial() for _ in range(first.limit))
    hers.identification.epoch = first.owner[0]
    assert calibration_for_current_body(run, hers, choosing.mine, minimum=10) is first
    assert not await _ask(run, hers, choosing, hold, 11.)
    assert not first.take_trial() and not actions


@pytest.mark.asyncio
async def test_an_oversized_control_set_cannot_create_or_renew_a_programme():
    run, hers, choosing, actions, hold = _setup()
    run.keys = [f"control_{index}" for index in range(MAX_CALIBRATION_CONTROLS + 1)]
    assert calibration_for_current_body(run, hers, choosing.mine, minimum=10) is None
    assert not await _ask(run, hers, choosing, hold, 10.)
    assert not actions and not hasattr(run, "_facing_calibration_programmes")
    run.keys = ["drive", "bend"]
    first = calibration_for_current_body(run, hers, choosing.mine, minimum=10)
    assert first.take_trial()
    run.keys = [f"control_{index}" for index in range(MAX_CALIBRATION_CONTROLS + 1)]
    assert calibration_for_current_body(run, hers, choosing.mine, minimum=10) is None
    run.keys = ["bend", "drive", "bend"]
    assert calibration_for_current_body(run, hers, choosing.mine, minimum=10) is first
    assert first.issued == 1 and first.limit == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ("missing_number", "mismatch", "nan", "infinite"))
async def test_invalid_current_ownership_cannot_request_an_experiment(invalid):
    run, hers, choosing, actions, hold = _setup()
    if invalid == "missing_number":
        hers.number = None
    elif invalid == "mismatch":
        choosing.mine.number += 1
    else:
        choosing.mine.born = float("nan") if invalid == "nan" else math.inf
    assert calibration_owner(hers, choosing.mine) is None
    assert not await _ask(run, hers, choosing, hold, 10.)
    assert not actions and run.facing_trials == 0


@pytest.mark.asyncio
async def test_a_missing_birth_keeps_only_the_original_total_cap():
    run, hers, choosing, actions, hold = _setup()
    choosing.mine = SimpleNamespace(number=7)
    run.facing_trials = 9
    assert await _ask(run, hers, choosing, hold, 10.)
    hers.identification.epoch += 1
    assert not await _ask(run, hers, choosing, hold, 11.)
    assert run.facing_trials == 10 and len(actions) == 1
    assert not hasattr(run, "_facing_calibration_programmes")


@pytest.mark.asyncio
async def test_a_four_control_programme_can_observe_two_turns_with_their_anchors_and_neutral_travel():
    keys = ("drive", "bend_a", "bend_b", "bend_c")
    run, hers, choosing, actions, hold = _setup(keys=keys)
    faces = run.faces = WhichWaySheFaces(facing=0.)
    faces.along["drive"] = [80.]
    faces.ways["drive"] = [0., 2.1, -2.1]
    faces.pressed.update(keys)
    rates = {"": 0., "drive": 0., "bend_a": 1.1, "bend_b": -1.2, "bend_c": .9}
    for index in range(25):
        at = 10. + index
        if not await _ask(run, hers, choosing, hold, at):
            break
        key = run.held
        for offset in (.25, .35, .45, .55, .65):
            angle = rates[key] * offset
            faces.saw(80. * math.cos(angle), 80. * math.sin(angle), key, at + offset,
                      measured=True, began=at)
    assert faces.travel_turns.rates() == {key: pytest.approx(rates[key]) for key in keys if key != "drive"}
    assert all(faces.travel_turns.repeats(key) == 2 for key in keys if key != "drive")
    assert sum(key == "" for key, _at in actions) == 2
    programme = calibration_for_current_body(run, hers, choosing.mine, minimum=10)
    assert len(actions) == programme.issued == run.facing_trials and programme.issued <= programme.limit


def test_the_current_body_budget_has_a_measured_invariant():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _current_body_calibration_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == "calibration_belongs_to_the_current_owned_body")
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())
