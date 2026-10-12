"""A silent first picture leaves a finite later motion experiment available."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.countering_motion_as_it_appears import CounteringMotion

pytestmark = pytest.mark.unit


def _thing(at: float, *, born: float = 0.0, path: tuple | None = None) -> SimpleNamespace:
    return SimpleNamespace(seen=at, born=born, vx=0.0, vy=30.0, w=10.0, h=10.0,
                           path=path if path is not None else [(at - 0.1, 5.0, 8.0), (at, 5.0, 11.0)])


def _choose(trial: CounteringMotion, at: float, things=(), *, held="", since=0.0, epoch=2):
    return trial.choose(["a", "b"], things, at, held=held, held_since=since, epoch=epoch,
                        settle_s=0.2, step_s=0.5, steps=4, wait_s=3.0,
                        against=lambda _keys, observed: "b" if list(observed) else "")


def test_unmeasured_first_picture_does_not_retire_the_later_trial() -> None:
    trial = CounteringMotion()
    assert _choose(trial, 0.0) is None
    assert not trial.done
    assert _choose(trial, 0.5, [_thing(0.5)], held="a", since=0.0) is None
    assert _choose(trial, 1.0, [_thing(1.0)], since=0.5) == "b"
    assert _choose(trial, 1.5, held="b", since=1.0) == ""
    assert _choose(trial, 2.0, since=1.5) == "b"
    assert _choose(trial, 2.5, held="b", since=2.0) == ""
    assert _choose(trial, 3.0, since=2.5) is None
    assert trial.done


@pytest.mark.parametrize("thing", [_thing(0.5), _thing(1.0, born=0.6), _thing(1.0, born=0.9),
                                   _thing(1.0, path=[(1.0, 1.0, 1.0)]),
                                   _thing(1.0, path=[(0.1, 1.0, 1.0), (0.2, 2.0, 2.0)])])
def test_predicted_or_unsettled_tracks_do_not_start_a_trial(thing) -> None:
    trial = CounteringMotion()
    assert _choose(trial, 1.0, [thing], since=0.5) is None
    assert not trial.key
    assert not trial.done


def test_search_and_active_experiment_have_finite_budgets() -> None:
    unseen = CounteringMotion()
    assert _choose(unseen, 0.0) is None
    assert _choose(unseen, 3.0, [_thing(3.0)]) is None
    assert unseen.done
    stalled = CounteringMotion()
    assert _choose(stalled, 1.0, [_thing(1.0)]) == "b"
    assert _choose(stalled, 5.0, held="b", since=1.0) is None
    assert stalled.done


@pytest.mark.parametrize("held,epoch", [("a", 2), ("b", 3)])
def test_changed_control_or_identity_ends_the_experiment(held: str, epoch: int) -> None:
    trial = CounteringMotion()
    assert _choose(trial, 1.0, [_thing(1.0)]) == "b"
    assert _choose(trial, 1.5, held=held, since=1.0, epoch=epoch) is None
    assert trial.done


def test_each_delivered_control_gets_its_measurement_time() -> None:
    trial = CounteringMotion()
    assert _choose(trial, 1.0, [_thing(1.0)]) == "b"
    assert _choose(trial, 1.6, held="b", since=1.3) == "b"
    assert trial.phase == 0
    assert _choose(trial, 1.8, held="b", since=1.3) == ""


def test_invalid_measurements_establish_nothing() -> None:
    trial = CounteringMotion()
    malformed = _thing(1.0)
    malformed.vy = float("nan")
    assert _choose(trial, 1.0, [malformed]) is None
    assert _choose(trial, float("nan"), [_thing(1.0)]) is None
    assert trial.first_at == 1.0
    assert not trial.key


def test_an_unmeasured_neutral_start_is_not_a_measured_rest() -> None:
    trial = CounteringMotion()
    assert _choose(trial, 1.0, [_thing(1.0)], since=None) is None
    assert not trial.done
    assert _choose(trial, 2.0, [_thing(2.0)], since=1.0) == "b"
