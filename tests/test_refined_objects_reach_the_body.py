"""Owned selector values reach the body and prove their prebound obligations."""
from __future__ import annotations

import asyncio
import base64
import io
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.agency.putting_things_in_place import what_is_carried
from core.agency.what_i_can_do_here import WhatWorksHere
from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
from core.cognition.reading_the_rules import Frame, Rules, the_step_that_holds
from core.perception.observed_transfer import verify_transfer
from core.perception.the_lattice_she_holds import TheLatticeSheHolds
from core.runtime.skill_contract import PredicateState
from core.skills.screen_pursuit_on_a_page import HER_OWN_PAGE, OnAPage
from core.skills.screen_step_evidence import screen_evidence, settle_step


class _SelectorWorld:
    """The receiver decides whether the pictured object was actually picked up."""

    def __init__(self, entity, background, foreground):
        self.library = f"{entity} library"
        self.background, self.foreground = background, foreground
        self.source = (40, 70, 28, 24)
        self.pointer = (0., 0.)
        self.open = self.held = False
        self.extra_label = ""
        self.accept_pickup = True
        self.clicks, self.pickups, self.deliveries, self.bindings_at_pickup = [], [], [], []
        self.bound_obligations = lambda: None
        self.render()

    def render(self):
        image = Image.new("RGB", (320, 240), self.background)
        draw = ImageDraw.Draw(image)
        draw.rectangle((15, 15, 175, 120), outline=(150, 70, 180), width=2)
        draw.rectangle((205, 135, 300, 220), outline=(80, 170, 60), width=2)
        self.layout = []
        labels = [(self.library, (30, 30)), ("Drop here", (220, 170))]
        if self.extra_label:
            labels.append((self.extra_label, (30, 195)))
        for label, point in labels:
            left, top, right, bottom = draw.textbbox(point, label)
            draw.text(point, label, fill=(15, 15, 15))
            self.layout.append({"text": label, "x": left / 320, "y": top / 240,
                                "width": (right - left) / 320, "height": (bottom - top) / 240,
                                "center_x": (left + right) / 640, "center_y": (top + bottom) / 480})
        if self.open:
            x, y, width, height = self.source
            draw.rectangle((x, y, x + width - 1, y + height - 1), fill=self.foreground)
            draw.rectangle((x + 4, y + 5, x + width - 5, y + 10), fill=(50, 190, 170))
        self.image = np.asarray(image).copy()

    async def click(self, x, y):
        self.clicks.append((x, y))
        if 20 <= x <= 170 and 20 <= y <= 55:
            self.open = True
            self.render()

    async def move(self, x, y):
        self.pointer = x, y

    async def down(self):
        self.bindings_at_pickup.append(self.bound_obligations())
        x, y, width, height = self.source
        self.held = (self.accept_pickup and self.open
                     and x <= self.pointer[0] < x + width and y <= self.pointer[1] < y + height)
        self.pickups.append((self.pointer, self.held))

    async def up(self):
        if self.held:
            _, _, width, height = self.source
            self.source = (round(self.pointer[0] - width / 2), round(self.pointer[1] - height / 2),
                           width, height)
            self.deliveries.append(self.source)
            self.render()
        self.held = False

    async def capture(self, *_):
        out = io.BytesIO()
        Image.fromarray(self.image).save(out, format="PNG")
        return {"data": base64.b64encode(out.getvalue()).decode("ascii")}


class _Page:
    viewport_size = {"width": 320, "height": 240}

    def __init__(self, world, surface):
        self.url = f"https://unrelated.example/{surface}"
        self.world = world
        self.mouse = SimpleNamespace(click=world.click, move=world.move, down=world.down, up=world.up)
        self.context = SimpleNamespace(new_cdp_session=self.session)

    async def session(self, _page):
        return SimpleNamespace(send=self.world.capture)


def _run(move, observation, can_do, surface):
    return SimpleNamespace(
        key=move, observation=observation, can_do=can_do, about_to={"key": move}, anchor={"app": surface},
        at_rest={}, busy=Mock(), expected={"took": 1, "after": None}, follow_on=[], goal=surface,
        in_flight=Mock(), laid_out=SimpleNamespace(occupied=lambda: 1, rows=1, columns=1, as_text=lambda: surface),
        lattice=TheLatticeSheHolds(), made=None, moves=[], narrate=False, pacing={"brief": True, "choice": ""},
        pending={"watched": observation,
                 "observed_before": screen_evidence(observation, can_do.seen_before),
                 "deliberation": SimpleNamespace(chosen=SimpleNamespace(name=move))},
        responds={"state": None, "lattice": None}, target_app=surface,
        world=SimpleNamespace(acts_with_arrivals=0),
    )


@pytest.fixture
def setup_world(monkeypatch):
    from core.perception import what_the_pixels_show, where_the_words_point
    from core.skills import screen_pursuit_acting as acting
    from core.skills.screen_pursuit_looking import MOVES_SAID

    @contextmanager
    def setup(surface="diagram-editor", entity="component", background=(235, 235, 235), foreground=(170, 50, 80)):
        world = _SelectorWorld(entity, background, foreground)
        body = OnAPage(_Page(world, surface), surface)
        body.CARRY_STEP_S = 0

        async def no_background_stream(*_):
            return None

        # OCR and the frame stream are fixtures. Objects and their witnesses
        # come from actual rendered captures, and input uses the real body.
        looker = SimpleNamespace(last_looks=(), would_not_read=set(),
                                 read=lambda *_, **__: {"text": "", "layout": list(world.layout)},
                                 places_still=lambda *_: True, learn_what_was_read=lambda: None)
        monkeypatch.setattr(body, "_watch", no_background_stream)
        monkeypatch.setattr(what_the_pixels_show, "looker_for", lambda _: looker)
        monkeypatch.setattr(where_the_words_point, "remember_the_still_picture", lambda *_, **__: None)
        places = where_the_words_point.PlacesNamed()
        # A current named destination must resolve from its measured OCR
        # extent before scheduling a background localization model.
        localize = Mock(side_effect=AssertionError("known current destination requested background localization"))
        monkeypatch.setattr(places, "where", localize)
        monkeypatch.setattr(where_the_words_point, "PLACES", places)
        monkeypatch.setattr("core.agency.what_she_is_doing.a_step_taken", Mock())
        monkeypatch.setattr(acting, "_say_intent", Mock())
        monkeypatch.setattr(acting, "_say_it_did_not_land", Mock())
        monkeypatch.setattr(acting, "_looks_like", lambda *_: False)
        from core.skills import screen_pursuit_decision as decision

        monkeypatch.setattr(decision, "_decide_the_next_move_how_long_whole", AsyncMock(return_value=(None, "")))
        monkeypatch.setattr(decision, "_decide_the_next_move_blocker", AsyncMock(return_value=decision._FALL_THROUGH))
        monkeypatch.setattr(decision, "_decide_the_next_move_what_she_looking", AsyncMock(return_value=((0, 0, 1, 1), True)))
        # Stop after real settlement: later deliberation has no bearing on
        # whether this exact caller binds the newly revealed sources.
        monkeypatch.setattr(decision, "_the_screen_taken_in_while_a_lesson_plays", AsyncMock(return_value=True))

        async def observe_after(*_, **__):
            return await body.read(), None

        monkeypatch.setattr(acting, "_settled_after", observe_after)
        choice = Frame(f"Choose an item of {entity}", act="click things", thing=f"an item of {entity}")
        opener = Frame(f"Open the {entity} library", act="click things", thing=world.library, order=1)
        carry = Frame(f"Place the {entity} in the workspace", act="carry", thing=entity, where="Drop here", order=2)
        rules = Rules(frames={frame.key: frame for frame in (choice, opener, carry)})
        world.bound_obligations = lambda: rules.pending_refinement
        guide = Guide(rules=rules)
        tokens = [(HER_OWN_PAGE, HER_OWN_PAGE.set(body)), (THE_GUIDE, THE_GUIDE.set(guide)),
                  (MOVES_SAID, MOVES_SAID.set({"episode": surface}))]
        try:
            yield SimpleNamespace(world=world, body=body, surface=surface, rules=rules,
                                  choice=choice, opener=opener, carry=carry, can_do=WhatWorksHere(carrying_said=True))
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)
            localize.assert_not_called()

    return setup


async def _open_and_find_source(case):
    from core.skills import screen_pursuit_acting as acting
    from core.skills.screen_pursuit_looking import _take_in_the_screen

    await case.body.read()
    before = await case.body.read()
    _take_in_the_screen(case.can_do, before, (0, 0, 1, 1), False)
    assert the_step_that_holds(case.rules) is case.choice
    assert not case.rules.revealed_choices
    run = _run(f'click "{case.world.library}"', before, case.can_do, case.surface)
    assert await acting.carry_out_the_move(run)
    opened = run.at_rest["reading"]
    # Exactly the normal action receipt, rather than a fabricated reveal,
    # makes the anonymous measured object eligible as a choice value.
    from core.perception.drawn_objects import measured_objects
    from core.skills.screen_pursuit_decision import WAITING, decide_the_next_move

    class DecisionContext(SimpleNamespace):
        def __getattr__(self, _name):
            return None

    deciding = DecisionContext(**vars(run), drawn={"where": (0, 0, 1, 1)},
                               costs={}, in_the_way={"last": ""}, no_move={})
    assert await decide_the_next_move(opened, deciding) is WAITING
    assert run.pending["step_settled"]
    assert case.rules.passed(case.opener)
    _take_in_the_screen(case.can_do, opened, (0, 0, 1, 1), False)
    current = await case.body.read()
    _take_in_the_screen(case.can_do, current, (0, 0, 1, 1), False)
    assert case.rules.revealed_choices is not None and the_step_that_holds(case.rules) is case.choice
    candidates = case.can_do.available()
    carries = [move for move in candidates
               if what_is_carried(move) and case.rules.refinement_of(move) is not None]
    assert carries, (candidates, case.rules.revealed_choices, case.rules.binding_context.evidence)
    return _run(carries[0], current, case.can_do, case.surface)


@pytest.mark.asyncio
@pytest.mark.parametrize(("surface", "entity", "background", "foreground"), [
    ("diagram-editor", "component", (235, 235, 235), (170, 50, 80)),
    ("inventory", "asset", (25, 35, 50), (220, 140, 60)),
    ("document-layout", "card", (245, 240, 220), (60, 100, 200)),
])
async def test_one_actual_pickup_and_delivery_proves_the_two_prebound_obligations(
        setup_world, surface, entity, background, foreground):
    from core.skills import screen_pursuit_acting as acting
    from core.skills.screen_pursuit_decision_lesson import _the_lesson_first

    with setup_world(surface, entity, background, foreground) as case:
        run = await _open_and_find_source(case)
        source, _ = what_is_carried(run.key)
        valued = _the_lesson_first({run.key: 1., f'click "{source}"': 1.}, can_do=case.can_do)
        assert valued[run.key] > valued[f'click "{source}"']
        assert await acting.carry_out_the_move(run)
        held = case.world.bindings_at_pickup[0]
        assert held is not None and len(held.attempts) == 2
        assert {attempt.step_key for attempt in held.attempts} == {case.choice.key, case.carry.key}
        assert {attempt.refinement_role for attempt in held.attempts} == {"choose_value", "place_value"}
        assert {attempt.refinement_id for attempt in held.attempts} == {held.binding.attempt_id}
        assert len(case.world.clicks) == len(case.world.pickups) == len(case.world.deliveries) == 1
        assert case.world.pickups[0][1] and not case.world.held
        assert not case.rules.passed(case.choice) and not case.rules.passed(case.carry)
        intent = run.pending["transfer_intent"]
        assert intent.attempt_id == held.binding.attempt_id
        assert intent.before is run.pending["prepared_transfer"].before
        after = run.at_rest["reading"]
        receipt = verify_transfer(intent, after)
        assert receipt.state is PredicateState.SATISFIED, receipt.reason
        assert settle_step(run.pending, after, (), case.can_do)
        assert case.rules.passed(case.choice) and case.rules.passed(case.carry)
        assert case.rules.pending_step is None and case.rules.pending_refinement is None
        assert len(case.rules.receipts[-2:]) == 2
        assert all(item.state is PredicateState.SATISFIED for item in case.rules.receipts[-2:])
        assert len(run.moves) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["receiver_refuses", "source_moved_after_capture"])
async def test_input_delivery_and_unrelated_change_cannot_complete_a_failed_pickup(setup_world, failure):
    from core.skills import screen_pursuit_acting as acting

    with setup_world() as case:
        run = await _open_and_find_source(case)
        if failure == "receiver_refuses":
            case.world.accept_pickup = False
        else:
            _, y, width, height = case.world.source
            case.world.source = (95, y, width, height)
            case.world.render()
        assert await acting.carry_out_the_move(run)
        assert len(case.world.pickups) == 1 and case.world.pickups[0][1] is False
        assert not case.world.deliveries
        after = run.at_rest["reading"]
        receipt = verify_transfer(run.pending["transfer_intent"], after)
        assert receipt.state is not PredicateState.SATISFIED
        run.pending["verification_started"] = time.monotonic() - 31
        assert settle_step(run.pending, after, (), case.can_do)
        assert not case.rules.passed(case.choice) and not case.rules.passed(case.carry)
        assert case.rules.pending_step is None and case.rules.pending_refinement is None
        assert all(item.state is not PredicateState.SATISFIED for item in case.rules.receipts[-2:])
        assert "refinement" not in run.pending["step_after"]


@pytest.mark.asyncio
async def test_forged_current_object_witness_is_refused_before_any_body_input(setup_world):
    from core.skills import screen_pursuit_acting as acting

    with setup_world() as case:
        run = await _open_and_find_source(case)
        source, _ = what_is_carried(run.key)
        region = next(item for item in run.observation["objects"] if item["text"] == source)
        region["candidate_id"] = "f" * 64
        with pytest.raises(ValueError, match="current measured pixels"):
            await acting.carry_out_the_move(run)
        assert not case.world.pickups and not case.world.deliveries
        assert len(case.world.clicks) == 1
        assert not case.rules.passed(case.choice) and not case.rules.passed(case.carry)
        assert case.rules.pending_step is None and case.rules.pending_refinement is None


@pytest.mark.asyncio
async def test_cancellation_releases_the_actual_held_pointer_and_both_obligations(setup_world):
    from core.skills import screen_pursuit_acting as acting

    with setup_world() as case:
        run = await _open_and_find_source(case)
        original = case.body.page.mouse.move

        async def canceled_while_held(x, y):
            if case.world.held:
                raise asyncio.CancelledError
            await original(x, y)

        case.body.page.mouse.move = canceled_while_held
        with pytest.raises(asyncio.CancelledError):
            await acting.carry_out_the_move(run)
        assert len(case.world.pickups) == 1 and case.world.pickups[0][1] is True
        assert not case.world.held
        assert not case.rules.passed(case.choice) and not case.rules.passed(case.carry)
        assert case.rules.pending_step is None and case.rules.pending_refinement is None
        assert all(item.state is PredicateState.UNKNOWN for item in case.rules.receipts[-2:])
        assert run.pending["deliberation"] is None and "prepared_transfer" not in run.pending
        assert not run.moves


@pytest.mark.asyncio
async def test_a_missing_prior_step_does_not_discard_the_ranked_owned_transfer(setup_world):
    from core.skills import screen_pursuit_acting as acting
    from core.skills.screen_pursuit_bearings import things_to_click
    from core.skills.screen_pursuit_decision_lesson import _the_lesson_first
    from core.skills.screen_pursuit_looking import _take_in_the_screen

    with setup_world() as case:
        unavailable = Frame("Click the introduction", act="click things", thing="introduction", order=-1)
        case.rules.frames[unavailable.key] = unavailable
        # The receiver changes two visible screens without ever offering the
        # earlier control. Ranking observes those actual current inventories.
        for label in ("Alpha", "Beta"):
            case.world.extra_label = label
            case.world.render()
            observation = await case.body.read()
            _take_in_the_screen(case.can_do, observation, (0, 0, 1, 1), False)
            case.rules.the_step_to_do(things_to_click(observation, (0, 0, 1, 1)))
        assert len(case.rules.missing_on[unavailable.key]) >= 2
        assert case.rules.next_step() is unavailable and not case.rules.passed(unavailable)
        run = await _open_and_find_source(case)
        valued = _the_lesson_first({run.key: 1.}, can_do=case.can_do)
        assert run.key in valued and valued[run.key] > 1.
        assert await acting.carry_out_the_move(run)
        held = case.world.bindings_at_pickup[0]
        assert held is not None and {attempt.step_key for attempt in held.attempts} == {
            case.choice.key, case.carry.key}
        after = run.at_rest["reading"]
        receipt = verify_transfer(run.pending["transfer_intent"], after)
        assert receipt.state is PredicateState.SATISFIED, receipt.reason
        assert settle_step(run.pending, after, (), case.can_do)
        assert case.rules.passed(case.choice) and case.rules.passed(case.carry)
        assert len(case.world.pickups) == len(case.world.deliveries) == 1
        assert not case.rules.passed(unavailable) and case.rules.next_step() is unavailable
        assert all(item.step_key != unavailable.key for item in case.rules.receipts)
