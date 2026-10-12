"""Questions belong to an operation through admission, generation and publication."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.cognition.guidance_lifetime import (
    _completed_guidance_invariant,
    guidance_work,
    publication_stamp,
)
from core.cognition.what_things_are import asked_patiently
from core.runtime.what_stops_it import AnExecutionContext, under

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_natural_completion_cancels_the_question_and_preserves_parent_and_sibling():
    started = asyncio.Event()
    parent = AnExecutionContext()
    sibling = parent.under("sibling")

    async def ask(*_args):
        started.set()
        await asyncio.Event().wait()

    with under(parent):
        async with guidance_work() as owner:
            question = asyncio.create_task(asked_patiently(ask, "question", None, 1, matters=1))
            await asyncio.wait_for(started.wait(), 1)
        assert owner.closed and owner.context.stopping.stopped
        assert question.cancelled() and not owner.tasks
    assert not parent.stopping.stopped and not sibling.stopping.stopped


@pytest.mark.asyncio
async def test_swallowing_cancellation_cannot_return_a_result_or_start_another_generation():
    started = asyncio.Event()
    calls = []

    async def ask(*_args):
        calls.append("called")
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "late result"

    async with guidance_work():
        question = asyncio.create_task(asked_patiently(ask, "question", None, 1, matters=1))
        await asyncio.wait_for(started.wait(), 1)
    assert question.cancelled() and calls == ["called"]


@pytest.mark.asyncio
async def test_retirement_is_bounded_while_a_stubborn_observer_releases(monkeypatch):
    from core.cognition import guidance_lifetime

    monkeypatch.setattr(guidance_lifetime, "TEARDOWN_SECONDS", .01)
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def ask(*_args):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return "late result"

    async with guidance_work() as owner:
        stamp = publication_stamp()
        question = asyncio.create_task(asked_patiently(ask, "question", None, 1, matters=1))
        await asyncio.wait_for(started.wait(), 1)
    assert cancelled.is_set() and not question.done() and owner.closed
    with pytest.raises(asyncio.CancelledError):
        stamp.check()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(question, 1)
    assert not owner.tasks


@pytest.mark.asyncio
async def test_a_waiting_question_is_removed_without_interrupting_the_question_in_another_owner():
    from core.cognition.asking_in_turn import RULES, in_turn

    held, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def unrelated():
        async with in_turn(RULES):
            held.set()
            await release.wait()

    async def ask(*_args):
        calls.append("called")

    first = asyncio.create_task(unrelated())
    await asyncio.wait_for(held.wait(), 1)
    async with guidance_work() as owner:
        queued = asyncio.create_task(asked_patiently(ask, "question", None, 1, matters=1))
        await asyncio.sleep(0)
        assert queued in owner.tasks
    assert queued.cancelled() and not first.done() and not calls
    release.set()
    await asyncio.wait_for(first, 1)


@pytest.mark.asyncio
async def test_a_late_plan_cannot_replace_the_guide_or_narrate(monkeypatch):
    from core.cognition import guidance_lifetime
    from core.cognition import a_plan_to_an_end as plans
    from core.cognition.a_guide_to_a_place import Guide
    from core.cognition.reading_the_rules import Frame

    started, release = asyncio.Event(), asyncio.Event()
    guide = Guide(place="editor", goals=["finish"])
    frame = Frame("Open the palette", act="click things", thing="palette")
    plan = plans.Plan()
    plan.took([frame])
    said = []
    monkeypatch.setattr(guidance_lifetime, "TEARDOWN_SECONDS", .01)

    async def delayed(*_args):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return plan

    monkeypatch.setattr(plans, "_a_plan", delayed)
    async with guidance_work():
        assert plans.ask_for_a_plan(guide, lambda *_: None, on_screen=["palette"], tell=said.append)
        question = guide.planning
        await asyncio.wait_for(started.wait(), 1)
    assert guide.plan is None and guide.planning is None
    newer = plans.Plan()
    newer.took([Frame("Choose the palette item", act="click things", thing="palette item")])

    async def fresh(*_args):
        return newer

    monkeypatch.setattr(plans, "_a_plan", fresh)
    async with guidance_work():
        assert plans.ask_for_a_plan(guide, lambda *_: None, on_screen=["palette"])
        await asyncio.wait_for(guide.planning, 1)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(question, 1)
        assert guide.plan is newer and said == []


@pytest.mark.asyncio
async def test_late_counsel_cannot_publish_or_launch_another_question(monkeypatch):
    from core.cognition import guidance_lifetime
    from core.cognition import what_i_know_of_a_place as manuals
    from core.cognition import what_this_place_is
    from core.cognition.a_guide_to_a_place import Guide

    started, release = asyncio.Event(), asyncio.Event()
    guide = Guide(place="editor")
    said, next_questions = [], []
    monkeypatch.setattr(guidance_lifetime, "TEARDOWN_SECONDS", .01)

    async def delayed(*_args):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return {"goal": "a late invented goal"}

    monkeypatch.setattr(manuals, "what_i_know_of", delayed)
    monkeypatch.setattr(what_this_place_is, "ask_for_a_reading", lambda *_a, **_k: next_questions.append("asked"))
    async with guidance_work():
        assert manuals.ask_in_the_background(guide, task="use it", said="", ask=lambda *_: None, tell=said.append)
        question = guide.asking_what_i_know
        await asyncio.wait_for(started.wait(), 1)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(question, 1)
    assert not guide.supposed_goal and not said and not next_questions
    assert not guide.asked_what_i_know and guide.asking_what_i_know is None


@pytest.mark.asyncio
async def test_scoped_name_queues_release_only_the_retired_owners_names(monkeypatch):
    from core.cognition.what_things_are import WhatThingsAre

    started = asyncio.Event()
    store = WhatThingsAre()
    monkeypatch.setattr(store, "keep", lambda: None)
    retired_names = [f"old name {i}" for i in range(9)]
    callbacks = []

    async def old_ask(*_args):
        started.set()
        await asyncio.Event().wait()

    async def sibling_ask(_prompt, schema, _most):
        return schema(things=[{"name": "a different object", "bears": "to use"}])

    async with guidance_work() as old:
        assert store.ask_about(retired_names, old_ask, then=lambda _learned: callbacks.append("old"))
        await asyncio.wait_for(started.wait(), 1)
        async with guidance_work() as sibling:
            assert store.ask_about(["a different object"], sibling_ask, then=lambda _learned: callbacks.append("sibling"))
            sibling_task = store.asker
            await asyncio.sleep(0)
            await old.close()
            assert not sibling.closed
            await asyncio.wait_for(sibling_task, 1)
            assert store.of("a different object").bears == "to use"
    assert not store.asking and not store._owned_queues and callbacks == ["sibling"]


@pytest.mark.asyncio
async def test_delayed_vision_cannot_publish_coordinates_after_natural_completion(monkeypatch):
    import numpy as np

    from core.perception import where_i_am_on_screen
    from core.perception.where_the_words_point import PlacesNamed

    started, cancelled = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(where_i_am_on_screen, "_as_jpeg", lambda _picture: "image")

    async def see(*_args):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            return '{"bbox_2d": [100, 200, 300, 400]}'

    places = PlacesNamed(see=see)
    async with guidance_work():
        places.remember(np.zeros((8, 12, 3), np.uint8), surface="editor", episode="first")
        assert places.where("the highlighted destination") is None
        await asyncio.wait_for(started.wait(), 1)
    assert cancelled.is_set() and not places.found


@pytest.mark.asyncio
async def test_delayed_avatar_publication_is_retired_and_can_be_requested_in_a_new_run(monkeypatch):
    import numpy as np

    from core.agency.naming_what_she_sees import SeeingInPlay, _who_i_am_to_my_eyes
    from core.cognition.a_guide_to_a_place import Guide
    from core.perception import where_i_am_on_screen

    started = asyncio.Event()
    seeing, guide = SeeingInPlay(began=0.), Guide(place="editor")

    async def look(*_args):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return SimpleNamespace(what="late avatar")

    monkeypatch.setattr(where_i_am_on_screen, "where_the_avatar_is", look)
    async with guidance_work():
        _who_i_am_to_my_eyes(None, seeing, guide, SimpleNamespace(things={1: object()}, seeing=True, _last_at=3.), None, None,
                            np.zeros((8, 12, 3), np.uint8), 3., lambda *_: None)
        task = seeing.avatar_asking
        await asyncio.wait_for(started.wait(), 1)
    assert task.cancelled() and seeing.avatar is None and seeing.avatar_asking is None

    async def fresh(*_args):
        return SimpleNamespace(what="current avatar")

    monkeypatch.setattr(where_i_am_on_screen, "where_the_avatar_is", fresh)
    async with guidance_work():
        _who_i_am_to_my_eyes(None, seeing, guide, SimpleNamespace(things={1: object()}, seeing=True, _last_at=4.), None, None,
                            np.zeros((8, 12, 3), np.uint8), 4., lambda *_: None)
        await asyncio.wait_for(seeing.avatar_asking, 1)
        assert seeing.avatar.what == "current avatar"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["complete", "exception", "stop"])
async def test_browser_run_tears_down_guidance_on_every_exit(monkeypatch, ending):
    from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
    from core.skills import screen_pursuit, screen_pursuit_as_it_happens, screen_pursuit_on_a_page
    from core.skills import sovereign_browser_drawing as drawing

    started = asyncio.Event()
    observed = {}
    guide = Guide(place="editor")
    original = THE_GUIDE.get()

    async def ask(*_args):
        started.set()
        await asyncio.Event().wait()

    def background(*_args):
        observed["question"] = asyncio.create_task(asked_patiently(ask, "question", None, 1, matters=1))

    async def get_guide(*_args):
        return guide

    async def pursue(**_kwargs):
        await asyncio.wait_for(started.wait(), 1)
        if ending == "exception":
            raise ValueError("observed failure")
        if ending == "stop":
            raise asyncio.CancelledError("user stop")
        return {"completed": True}

    class Reflexes:
        def __init__(self, **_kwargs):
            observed["reflexes"] = self
            self.closed = False

        async def close(self):
            self.closed = True

    class Page:
        def __init__(self, **_kwargs):
            observed["page"] = self
            self.closed = False

        async def stop_watching(self):
            self.closed = True

    monkeypatch.setattr(drawing, "_the_guide", get_guide)
    monkeypatch.setattr(drawing, "_what_the_guide_says", background)
    monkeypatch.setattr(drawing, "_keep_the_guide", lambda *_: None)
    monkeypatch.setattr(screen_pursuit, "pursue_on_screen", pursue)
    monkeypatch.setattr(screen_pursuit_as_it_happens, "PlayingAsItHappens", Reflexes)
    monkeypatch.setattr(screen_pursuit_on_a_page, "OnAPage", Page)
    if ending == "complete":
        result, _ = await drawing._one_run(None, (0., 0., 1., 1.), "use it", "https://example.com", 1., {})
        assert result["completed"]
    else:
        expected = ValueError if ending == "exception" else asyncio.CancelledError
        with pytest.raises(expected):
            await drawing._one_run(None, (0., 0., 1., 1.), "use it", "https://example.com", 1., {})
    assert observed["question"].cancelled()
    assert observed["reflexes"].closed and observed["page"].closed
    assert THE_GUIDE.get() is original


def test_the_publication_invariant_has_an_executable_probe():
    assert _completed_guidance_invariant() == ()
