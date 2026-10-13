"""A retry changes the application before another completion can count."""
import json
import math
import time
from types import SimpleNamespace

import pytest

from core.skills import sovereign_browser_drawing as drawing
from core.skills.a_new_attempt_must_be_seen import (
    AttemptObservation,
    _new_attempt_invariant,
    attempt_activity_was_seen,
    note_attempt_activity,
)
from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens


@pytest.mark.parametrize('at', [None, True, math.nan, math.inf, 9., 11.])
def test_an_unmeasured_old_or_future_capture_cannot_open_the_new_attempt(at):
    receipt = AttemptObservation(10.)
    receipt.observe(at, now=10.5, active=True, ended=False)
    assert receipt.activity_at is None


def test_a_new_page_reading_of_the_same_ending_cannot_count_as_another_run(monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 10.5)
    receipt = AttemptObservation(10.)
    reflexes = SimpleNamespace(keep={'attempt_observation': receipt}, stretches=[], play_in_sight=True,
                               ending_parts=['GAME OVER'], over_because='a way to start again appeared')
    ended = {'ended': 'lost', 'words': 'GAME OVER'}
    for capture in (9., 10.1, 10.2):
        note_attempt_activity(reflexes, {'text': 'GAME OVER', 'capture_at': capture, 'on_her_page': True})
        assert not drawing._run_reached_an_end(reflexes, ended, {'last_seen': 'GAME OVER'})
    note_attempt_activity(reflexes, {'text': 'Time: 59 Score: 0', 'capture_at': 10.3, 'on_her_page': True})
    assert receipt.activity_at == 10.3
    assert drawing._run_reached_an_end(reflexes, ended, {'last_seen': 'GAME OVER'})


def test_reading_completion_time_without_a_capture_time_cannot_establish_activity(monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 10.5)
    reflexes = SimpleNamespace(keep={'attempt_observation': AttemptObservation(10.)}, stretches=[], play_in_sight=True)
    note_attempt_activity(reflexes, {'text': 'Time: 59', 'at': 10.4, 'on_her_page': True})
    assert not attempt_activity_was_seen(reflexes)
    assert attempt_activity_was_seen(SimpleNamespace(keep={}, stretches=[])) is None


def test_the_actual_reader_marks_observed_activity_after_reset(monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 10.5)
    keep = {'attempt_observation': AttemptObservation(10.)}
    reflexes = PlayingAsItHappens(SimpleNamespace(), (0., 0., 1., 1.), 'play', 100., keep=keep)
    reflexes.play_in_sight = True
    reflexes.read({'text': 'GAME OVER', 'capture_at': 10.1, 'on_her_page': True})
    assert not attempt_activity_was_seen(reflexes)
    reflexes.read({'text': 'Time: 59 Score: 0', 'capture_at': 10.2, 'on_her_page': True})
    assert attempt_activity_was_seen(reflexes)


@pytest.mark.asyncio
@pytest.mark.parametrize('fails', [False, True])
async def test_memory_changes_only_after_the_physical_page_restart(monkeypatch, fails):
    events = []
    keep = {'learned': 'previously observed controls'}

    class Page:
        url = 'https://example.test/interactive-world'

        async def reload(self, **_kwargs):
            events.append('reload')
            if fails:
                raise RuntimeError('reload failed')

        async def evaluate(self, expression):
            events.append('geometry' if expression == drawing._WHERE_IT_DRAWS else 'capture support')
            return json.dumps({'left': .1, 'top': .2, 'right': .9, 'bottom': .8})

    async def start(page, band):
        events.append('start')
        return band

    def begin(state, _game):
        assert state is keep
        events.append('new memory')

    monkeypatch.setattr(drawing, '_start_what_is_covered', start)
    monkeypatch.setattr('core.skills.screen_pursuit_as_it_happens.begin_run', begin)
    result = await drawing._restart_attempt(Page(), (0., 0., 1., 1.), keep)
    assert keep['learned'] == 'previously observed controls'
    if fails:
        assert result is None and events == ['reload'] and 'attempt_observation' not in keep
    else:
        assert result == (.1, .2, .9, .8)
        assert events == ['reload', 'geometry', 'start', 'capture support', 'new memory']
        assert isinstance(keep['attempt_observation'], AttemptObservation)
        assert keep['attempt_observation'].activity_at is None


def test_the_new_attempt_boundary_has_a_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _new_attempt_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'an_ending_after_reset_needs_new_activity')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.asyncio
async def test_the_outer_loop_cannot_count_an_old_ending_after_a_physical_reload(monkeypatch):
    calls, reloads = [], []

    async def one_run(page, band, goal, _url, deadline, keep):
        calls.append(len(calls))
        reflexes = PlayingAsItHappens(page, band, goal, deadline, keep=keep)
        reflexes.play_in_sight = True
        if len(calls) == 1:
            reflexes.read({'text': 'Time: 59 Score: 0', 'capture_at': time.time(), 'on_her_page': True})
        reflexes.read({'text': 'GAME OVER YOUR SCORE: 10', 'capture_at': time.time(), 'on_her_page': True})
        reflexes.over_because = 'a way to start again appeared'
        reflexes.ending_words = 'GAME OVER YOUR SCORE: 10'
        reflexes.ending_parts = [reflexes.ending_words]
        return {'moves': [], 'last_seen': reflexes.ending_words}, reflexes

    async def restart(_page, band):
        reloads.append('reload')
        return band

    async def stock(_goal, _page):
        return None

    monkeypatch.setattr(drawing, '_one_run', one_run)
    monkeypatch.setattr(drawing, '_from_the_top', restart)
    monkeypatch.setattr(drawing, '_a_stocktaking', stock)
    monkeypatch.setattr(drawing, '_tell', lambda _line: None)
    monkeypatch.setattr(drawing, '_the_share_ends', lambda *_args: time.monotonic() + 5.)
    result = await drawing._played(SimpleNamespace(url='https://example.test/world'), (0., 0., 1., 1.),
                                   'Play three runs and report the best score.', 'https://example.test/world', {})
    assert len(calls) == 2 and reloads == ['reload']
    assert result['completed_runs'] == 1 and not result['completed'] and not result['won']
    assert result['stopped_because'] == 'an ending was shown without new activity being observed'
