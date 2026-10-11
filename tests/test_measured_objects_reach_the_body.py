"""Pixel discovery supplies the actual source of a delivered, independently checked carry."""
from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.agency.putting_things_in_place import what_is_carried
from core.agency.what_i_can_do_here import WhatWorksHere
from core.perception.observed_transfer import verify_transfer
from core.perception.the_lattice_she_holds import TheLatticeSheHolds
from core.runtime.skill_contract import PredicateState
from core.skills.screen_pursuit_on_a_page import HER_OWN_PAGE, OnAPage
from core.skills.screen_step_evidence import screen_evidence


class _World:
    """A renderable drag surface: the receiver decides whether the pickup was real."""

    def __init__(self, background, foreground):
        self.background, self.foreground = background, foreground
        self.source = (40, 40, 28, 24)
        self.pointer = (0., 0.)
        self.held = False
        self.pickups = []
        self.deliveries = []
        self.render()

    def render(self):
        image = Image.new("RGB", (320, 240), self.background)
        draw = ImageDraw.Draw(image)
        draw.rectangle((205, 135, 300, 220), outline=(80, 170, 60), width=2)
        label = "Drop here"
        box = draw.textbbox((220, 170), label)
        draw.text((220, 170), label, fill=(15, 15, 15))
        x, y, w, h = self.source
        draw.rectangle((x, y, x + w - 1, y + h - 1), fill=self.foreground)
        draw.rectangle((x + 4, y + 5, x + w - 5, y + 10), fill=(50, 190, 170))
        self.image = np.asarray(image).copy()
        left, top, right, bottom = box
        self.layout = [{"text": label, "x": left / 320, "y": top / 240,
                        "width": (right - left) / 320, "height": (bottom - top) / 240,
                        "center_x": (left + right) / 640, "center_y": (top + bottom) / 480}]

    async def move(self, x, y):
        self.pointer = x, y

    async def down(self):
        x, y, w, h = self.source
        self.held = x <= self.pointer[0] < x + w and y <= self.pointer[1] < y + h
        self.pickups.append((self.pointer, self.held))

    async def up(self):
        if self.held:
            _, _, w, h = self.source
            self.source = round(self.pointer[0] - w / 2), round(self.pointer[1] - h / 2), w, h
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
        self.mouse = SimpleNamespace(move=world.move, down=world.down, up=world.up)
        self.context = SimpleNamespace(new_cdp_session=self.session)

    async def session(self, _page):
        return SimpleNamespace(send=self.world.capture)


@pytest.mark.asyncio
@pytest.mark.parametrize(("surface", "background", "foreground"), [
    ("diagram-editor", (235, 235, 235), (170, 50, 80)),
    ("inventory", (25, 35, 50), (220, 140, 60)),
    ("document-layout", (245, 240, 220), (60, 100, 200)),
])
async def test_observation_inventory_and_dispatch_ground_the_source_before_transfer_proof(
        monkeypatch, surface, background, foreground):
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.perception import what_the_pixels_show, where_the_words_point
    from core.skills import screen_pursuit_acting as acting
    from core.skills.screen_pursuit_looking import MOVES_SAID, _take_in_the_screen

    world = _World(background, foreground)
    page = _Page(world, surface)
    body = OnAPage(page, surface)
    body.CARRY_STEP_S = 0

    async def no_background_stream(*_):
        return None

    # This offline fixture substitutes OCR and the frame stream, not object
    # geometry or the input method. Source points come from rendered pixels.
    looker = SimpleNamespace(last_looks=(), would_not_read=set(),
                             read=lambda *_, **__: {"text": "", "layout": list(world.layout)},
                             places_still=lambda *_: True, learn_what_was_read=lambda: None)
    monkeypatch.setattr(body, "_watch", no_background_stream)
    monkeypatch.setattr(what_the_pixels_show, "looker_for", lambda _: looker)
    monkeypatch.setattr(where_the_words_point, "remember_the_still_picture", lambda *_, **__: None)
    monkeypatch.setattr("core.agency.what_she_is_doing.a_step_taken", Mock())
    monkeypatch.setattr(acting, "_say_intent", Mock())
    monkeypatch.setattr(acting, "_say_it_did_not_land", Mock())
    monkeypatch.setattr(acting, "_looks_like", lambda *_: False)
    contexts = HER_OWN_PAGE.set(body), THE_GUIDE.set(None), MOVES_SAID.set({"episode": surface})
    try:
        can_do = WhatWorksHere(carrying_said=True)
        first = await body.read()
        _take_in_the_screen(can_do, first, (0, 0, 1, 1), False)
        before = await body.read()
        _take_in_the_screen(can_do, before, (0, 0, 1, 1), False)
        candidates = can_do.available()
        carries = [move for move in candidates if (parts := what_is_carried(move)) and parts[1] == "Drop here"]
        assert carries and all(what_is_carried(move)[0] in can_do.manipulation_sources for move in carries)
        chosen = carries[0]
        run = SimpleNamespace(
            key=chosen, observation=before, can_do=can_do, about_to={"key": chosen}, anchor={"app": surface},
            at_rest={}, busy=Mock(), expected={"took": 1, "after": None}, follow_on=[], goal=surface,
            in_flight=Mock(), laid_out=SimpleNamespace(occupied=lambda: 1, rows=1, columns=1, as_text=lambda: surface),
            lattice=TheLatticeSheHolds(), made=None, moves=[], narrate=False, pacing={"brief": True, "choice": ""},
            pending={"watched": before, "observed_before": screen_evidence(before),
                     "deliberation": SimpleNamespace(chosen=SimpleNamespace(name=chosen))},
            responds={"state": None, "lattice": None}, target_app=surface,
            world=SimpleNamespace(acts_with_arrivals=0),
        )

        async def observe_after(*_, **__):
            return await body.read(), None

        monkeypatch.setattr(acting, "_settled_after", observe_after)
        assert await acting.carry_out_the_move(run)
        assert world.pickups and all(held for _, held in world.pickups)
        assert len(world.deliveries) == 1 and not world.held
        after = run.at_rest["reading"]
        receipt = verify_transfer(run.pending["transfer_intent"], after)
        assert receipt.state is PredicateState.SATISFIED, receipt.reason
        assert receipt.source == what_is_carried(chosen)[0].casefold()
        assert before["_capture_epoch"] != after["_capture_epoch"]
        assert len(run.moves) == 1
    finally:
        MOVES_SAID.reset(contexts[2])
        THE_GUIDE.reset(contexts[1])
        HER_OWN_PAGE.reset(contexts[0])
