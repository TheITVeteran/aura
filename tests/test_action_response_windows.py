"""Input ownership and capture coverage constrain causal response learning."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.what_i_can_do_here import WhatWorksHere
from core.perception import watching_it_happen
from core.perception.watching_it_happen import Watching, _response_window_keeps_its_evidence
from core.perception.where_it_responds import Responsive, noticed
from core.runtime.skill_contract import PredicateState
from core.skills import screen_pursuit_decision_branches, screen_pursuit_surface
from core.skills.screen_pursuit_decision_branches import _it_did_nothing_from_here
from core.skills.screen_pursuit_on_a_page import OnAPage, _what_the_stream_shows, it_was_answered


async def _nothing():
    return None


def _picture(value=0):
    pixels = np.full((120, 160), 50, dtype=np.uint8)
    pixels[40:70, 40:80] = value + 50
    return pixels


def _baseline():
    watching = Watching(take=_nothing)
    for n in range(17):
        watching.saw(_picture(), 10.0 + n / 8)
    return watching


def _stills(watching, start, end, value=0):
    for at in np.arange(start, end + 0.01, 0.125):
        watching.saw(_picture(value), float(at))


def test_registered_response_canary_uses_owned_evidence():
    assert _response_window_keeps_its_evidence()


def test_repeated_owned_panel_changes_are_not_autonomous_background():
    watching = _baseline()
    first = watching.begin_act("select one", 12.05)
    watching.saw(_picture(190), 12.125)
    watching.finish_act(first, True, 12.15)
    _stills(watching, 12.25, 13.875, 190)
    second = watching.begin_act("select another", 13.95)
    watching.saw(_picture(), 14.0)
    watching.finish_act(second, True, 14.05)
    assert watching.answered(first.started) is True
    assert watching.answered(second.started) is True
    assert not watching._always().any()


def test_an_autonomous_change_in_the_same_place_remains_unanswered():
    watching = Watching(take=_nothing)
    for n in range(41):
        watching.saw(_picture(190 if (n // 4) % 2 else 0), 10.0 + n / 8)
    window = watching.begin_act("irrelevant click", 15.05)
    watching.finish_act(window, True, 15.075)
    for n in range(1, 19):
        watching.saw(_picture(190 if (n // 4) % 2 else 0), 15.0 + n / 8)
    assert watching.answered(window.started) is False


def test_a_large_autonomous_scene_is_not_credited_as_a_new_screen():
    watching = Watching(take=_nothing)
    for n in range(41):
        watching.saw(np.full((120, 160), 50 if n % 2 else 240, dtype=np.uint8), 10 + n / 8)
    window = watching.begin_act("irrelevant click", 15.05)
    watching.finish_act(window, True, 15.075)
    for n in range(1, 19):
        watching.saw(np.full((120, 160), 50 if n % 2 else 240, dtype=np.uint8), 15 + n / 8)
    assert watching.answered(window.started) is False


def _color_panel(selected=False):
    pixels = np.full((120, 160, 3), 50, dtype=np.uint8)
    pixels[40:70, 40:80] = (40, 220, 40) if selected else (220, 40, 40)
    return pixels


def test_a_same_luminance_color_selection_is_an_owned_visual_response():
    before, after = _color_panel(), _color_panel(True)
    assert np.array_equal(before.mean(axis=2), after.mean(axis=2))
    watching = Watching(take=_nothing)
    for n in range(17):
        watching.saw(before, 10 + n / 8)
    window = watching.begin_act("select the indicator", 12.05)
    watching.saw(after, 12.125)
    watching.finish_act(window, True, 12.15)
    assert watching.answered(window.started) is True


def test_same_luminance_autonomous_color_changes_are_still_unanswered():
    watching = Watching(take=_nothing)
    for n in range(41):
        watching.saw(_color_panel(bool(n % 2)), 10 + n / 8)
    window = watching.begin_act("irrelevant click", 15.05)
    watching.finish_act(window, True, 15.075)
    for n in range(1, 19):
        watching.saw(_color_panel(bool(n % 2)), 15 + n / 8)
    assert watching.answered(window.started) is False


def test_a_negative_needs_the_whole_response_interval():
    watching = _baseline()
    window = watching.begin_act("click", 12.05)
    watching.finish_act(window, True, 12.075)
    watching.saw(_picture(), 12.125)
    assert watching.answered(window.started) is None
    _stills(watching, 12.25, 14.25)
    assert watching.answered(window.started) is False


@pytest.mark.parametrize("fault", ["gap", "late", "insufficient_baseline", "refused", "unfinished", "overlap"])
def test_missing_or_ambiguous_coverage_is_unknown(fault):
    watching = _baseline()
    if fault == "insufficient_baseline":
        watching.moments.clear()
    started = 15.0 if fault == "late" else 12.05
    window = watching.begin_act("click", started)
    if fault != "unfinished":
        watching.finish_act(window, fault != "refused", started + .025)
    if fault == "gap":
        watching.saw(_picture(190), 13.0)
    elif fault == "overlap":
        watching.saw(_picture(), 12.125)
        other = watching.begin_act("later click", 12.15)
        watching.finish_act(other, True, 12.175)
        _stills(watching, 12.25, 14.25)
    else:
        watching.saw(_picture(190), started + .125)
    assert watching.answered(window.started) is None


def test_old_and_out_of_order_captures_cannot_supply_new_response():
    watching = _baseline()
    window = watching.begin_act("click", 12.05)
    watching.finish_act(window, True, 12.075)
    watching.saw(_picture(190), 11.9)
    watching.saw(_picture(190), float("nan"))
    assert watching._last_at == 12.0
    assert watching.answered(window.started) is None


@pytest.mark.asyncio
async def test_the_stream_preserves_capture_time_through_decode_delay():
    watching = _baseline()
    captures = iter([(_picture(190), 12.125)])

    async def take():
        try:
            return next(captures)
        except StopIteration:
            raise asyncio.CancelledError from None

    watching.take = take
    watching.per_second = 10000
    with pytest.raises(asyncio.CancelledError):
        await watching._run()
    assert watching._last_at == 12.125
    assert watching.moments[-1].at == 12.125


@pytest.mark.asyncio
@pytest.mark.parametrize("input_kind", ["click", "press", "carry"])
async def test_a_fast_response_during_input_is_owned(input_kind, monkeypatch):
    watching = _baseline()
    clock = SimpleNamespace(at=12.05)
    monkeypatch.setattr(watching_it_happen.time, "monotonic", lambda: clock.at)

    async def responds(*_args):
        assert watching.answer_window()["finished"] is None
        clock.at += .075
        watching.saw(_picture(190), clock.at)
        await asyncio.sleep(0)
        clock.at += .075

    async def move(*_args):
        await asyncio.sleep(0)

    page = SimpleNamespace(mouse=SimpleNamespace(click=responds, move=move, down=responds, up=move),
                           keyboard=SimpleNamespace(press=responds))
    surface = OnAPage(page, "editor")
    surface.watching, surface._focused = watching, True
    surface.CARRY_STEP_S = 0
    if input_kind == "click":
        delivered = await surface.click(.5, .5, [0, 0, 160, 120])
    elif input_kind == "press":
        delivered = await surface.press("space")
    else:
        delivered = await surface.carry((.2, .2), (.5, .5), [0, 0, 160, 120])
    assert delivered
    seen = {"text": "unchanged words", "layout": []}
    _what_the_stream_shows(seen, watching)
    assert seen["answer_window"]["started"] == 12.05
    assert seen["answer_window"]["finished"] > 12.125
    assert seen["answered"] is True
    assert it_was_answered(False, seen) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("input_kind", ["click", "carry"])
async def test_cancelled_input_closes_its_window_without_response_credit(input_kind):
    watching = _baseline()
    released = []

    async def cancel(*_args):
        raise asyncio.CancelledError

    async def move(*_args):
        return None

    async def up():
        released.append(True)

    surface = OnAPage(SimpleNamespace(mouse=SimpleNamespace(click=cancel, move=move, down=cancel, up=up)), "editor")
    surface.watching = watching
    with pytest.raises(asyncio.CancelledError):
        if input_kind == "click":
            await surface.click(.5, .5, [0, 0, 160, 120])
        else:
            await surface.carry((.2, .2), (.5, .5), [0, 0, 160, 120])
    window = watching.answer_window()
    assert window["finished"] is not None and window["delivered"] is False
    assert watching.answered(window["started"]) is None
    if input_kind == "carry":
        assert released == [True]


def test_unknown_response_never_learns_failure_or_success():
    can_do = WhatWorksHere()
    before = asdict(can_do)
    can_do.tried('click "panel"', None)
    assert asdict(can_do) == before
    responsive = Responsive()
    before = asdict(responsive)
    noticed(responsive, {"layout": []}, {"layout": []}, worked=None, acting="click panel")
    assert asdict(responsive) == before
    pending = {"did_nothing_from": {"view": "panel", "acts": {"left"}}}
    _it_did_nothing_from_here(pending, SimpleNamespace(as_text=lambda: "panel"), "right", None)
    assert pending["did_nothing_from"]["acts"] == {"left"}
    old = dict(screen_pursuit_surface._UNANSWERED)
    screen_pursuit_surface.it_answered("response-unknown-test", "left", None)
    assert screen_pursuit_surface._UNANSWERED == old
    assert it_was_answered(True, {"answer_window": {"delivered": True}, "answered": None}) is None
    assert it_was_answered(True, {"answer_window": {"delivered": True}, "answered": False}) is False


@pytest.mark.parametrize("reading", [{"layout": []}, {"layout": [{"text": "Library", "center_x": .2, "center_y": .2}]}])
def test_owned_visual_response_resets_unanswered_without_text_change(reading):
    responsive = Responsive(unanswered=3, unanswered_by={"one", "two"})
    noticed(responsive, reading, reading, worked=None, acting="select", response_state=PredicateState.SATISFIED)
    assert responsive.unanswered == 0 and responsive.unanswered_by == set()
    assert responsive.acts == responsive.effective == 1 and responsive.idle == 0
    assert responsive.answered == {} and responsive.regardless == {}
    assert "select" in responsive.tried


def test_owned_unanswered_input_keeps_autonomous_text_as_idle_control():
    responsive = Responsive()
    before = {"layout": [{"text": "Clock 10", "center_x": .2, "center_y": .2}]}
    after = {"layout": [{"text": "Clock 11", "center_x": .2, "center_y": .2}]}
    noticed(responsive, before, after, worked=True, acting="left", response_state=PredicateState.UNSATISFIED)
    assert responsive.unanswered == 1 and responsive.unanswered_by == {"left"}
    assert responsive.idle == 1 and responsive.effective == 0
    assert responsive.regardless == {(20, 20): 1} and responsive.answered == {}


def test_owned_unknown_skips_learning_even_when_words_changed():
    responsive = Responsive(unanswered=3, unanswered_by={"one", "two"})
    before = asdict(responsive)
    noticed(responsive, {"layout": []}, {"layout": [{"text": "new", "center_x": .2, "center_y": .2}]},
            worked=True, acting="select", response_state=PredicateState.UNKNOWN)
    assert asdict(responsive) == before


def test_legacy_response_discovery_preserves_its_text_comparison():
    reading = {"layout": [{"text": "Library", "center_x": .2, "center_y": .2}]}
    responsive = Responsive()
    noticed(responsive, reading, reading, worked=True, acting="select")
    assert responsive.unanswered == 1 and responsive.unanswered_by == {"select"}


def test_owned_response_rejects_an_untyped_boolean():
    with pytest.raises(TypeError, match="predicate state"):
        noticed(Responsive(), {}, {}, response_state=True)


@pytest.mark.parametrize("response_state", [PredicateState.SATISFIED, PredicateState.UNKNOWN])
def test_the_live_learning_seam_uses_the_validated_response(response_state, monkeypatch):
    monkeypatch.setattr(screen_pursuit_decision_branches, "_both_of_the_thing", lambda *_args: True)
    reading = {"layout": [{"text": "Library", "center_x": .2, "center_y": .2}]}
    responsive = Responsive(unanswered=3, unanswered_by={"one", "two"})
    original = asdict(responsive)
    pending = {"watched": reading, "response_state": response_state}
    attempt = SimpleNamespace(verdict=SimpleNamespace(observed_change=(True if response_state is PredicateState.SATISFIED else None)))
    screen_pursuit_decision_branches._decide_the_next_move_learned_same_measurement(
        {"app": "editor"}, attempt, reading, pending, SimpleNamespace(chosen=SimpleNamespace(name="select")),
        {"state": responsive, "moving": SimpleNamespace(saw=lambda *_args: None)}, "editor",
    )
    if response_state is PredicateState.SATISFIED:
        assert responsive.unanswered == 0 and responsive.effective == 1
    else:
        assert asdict(responsive) == original


def _idling(at, acts, panel=0):
    """A picture whose corner flickers by itself half a second after each of her acts, whatever the act (an idle
    animation that happens to fall there), and whose panel shows ``panel``."""
    pixels = _picture(panel)
    if any(0.5 <= at - act < 0.75 for act in acts):
        pixels[5:30, 110:150] = 220
    return pixels


def test_a_change_that_follows_every_act_alike_is_no_acts_answer():
    """LIVE 2026-10-11 a game's characters idled while she acted every few seconds; no picture was free of her acts to
    show it, and each arrow key, which the game does not listen for, was taken to have done something."""
    watching = Watching(take=_nothing)
    acts: list[float] = []
    for n in range(17):
        watching.saw(_idling(10.0 + n / 8, acts), 10.0 + n / 8)
    at = 12.25
    for name in ("click one", "click two", "up", "down", "click three", "space", "left"):
        window = watching.begin_act(name, at + 0.01)
        watching.finish_act(window, True, at + 0.02)
        acts.append(at + 0.01)
        for step in range(1, 25):
            watching.saw(_idling(at + step / 8, acts), at + step / 8)
        last = window
        at += 3.0
    assert watching.answered(last.started) is not True
    clicked = watching.begin_act("click the panel", at + 0.01)
    watching.finish_act(clicked, True, at + 0.02)
    acts.append(at + 0.01)
    for step in range(1, 25):
        watching.saw(_idling(at + step / 8, acts, panel=190), at + step / 8)
    assert watching.answered(clicked.started) is True
