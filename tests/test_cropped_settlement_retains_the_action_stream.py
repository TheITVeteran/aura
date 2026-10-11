"""Post-input settlement keeps the crop and capture owner used for dispatch."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.runtime.skill_contract import PredicateState
from core.skills.screen_pursuit_looking import _settled_after
from core.skills.screen_pursuit_on_a_page import OnAPage
from core.skills.screen_step_evidence import owned_response_state, screen_evidence


@pytest.mark.asyncio
async def test_cropped_page_settlement_retains_its_real_watcher_and_owned_input(monkeypatch):
    from core.perception import what_her_page_shows
    from core.skills import screen_pursuit, screen_pursuit_on_a_page

    crop = (.1, .1, .9, .9)
    bounds = [16, 12, 128, 96]
    before_pixels = np.full((96, 128), 50, dtype=np.uint8)
    after_pixels = before_pixels.copy()
    after_pixels[30:50, 40:65] = 240
    changed = False
    providers, looks = [], []

    class Pictures:
        async def __call__(self):
            return None

        async def close(self):
            pass

    async def pictures(_page, over):
        providers.append(over)
        return Pictures()

    async def look(_page, over, *, name):
        looks.append(over)
        # The fake browser supplies photographs; the actual page adapter and
        # stream own, classify and bind their capture intervals.
        if changed:
            surface.watching.saw(after_pixels, time.monotonic())
        return {"text": "unchanged words", "layout": [], "surface_id": "page:editor",
                "bounds": bounds if over == crop else [0, 0, 160, 120],
                "_capture_epoch": str(len(looks)), "capture_at": time.time(),
                "settled": True if changed else None}

    async def click(*_args):
        nonlocal changed
        changed = True
        # A response during mouse delivery must remain owned through a later
        # cropped reading, even when its OCR stays exactly the same.
        surface.watching.saw(after_pixels, time.monotonic())

    surface = OnAPage(SimpleNamespace(mouse=SimpleNamespace(click=click)), "editor")
    monkeypatch.setattr(screen_pursuit_on_a_page, "_pictures_of", pictures)
    monkeypatch.setattr(what_her_page_shows, "look_at_a_page", look)

    async def read(_app, *, over=None):
        return await surface.read(over)

    monkeypatch.setattr(screen_pursuit, "read_screen", read)
    try:
        before = await surface.read(crop)
        watcher = surface.watching
        # Prime autonomous coverage with real still-frame comparisons. The
        # provider stays quiescent to make this causal counterfactual exact.
        baseline_end = time.monotonic() - .005
        for index in range(17):
            watcher.saw(before_pixels, baseline_end - 2 + index / 8)
        pending = {"observed_before": screen_evidence(before), "dispatch_sequence_before": 0,
                   "dispatch_started_monotonic": time.monotonic()}
        assert await surface.click(.5, .5, bounds)
        pending.update(dispatch_completed=time.time(), dispatch_completed_monotonic=time.monotonic())
        window = watcher.answer_window()
        assert window["delivered"] is True and window["sequence"] == 1
        after, moved = await _settled_after(
            before, "editor", patience=.8, over=crop,
            response_state=lambda observation: owned_response_state(pending, observation),
        )
        assert moved is True
        assert owned_response_state(pending, after) is PredicateState.SATISFIED
        assert surface.watching is watcher
        assert after["answer_window"]["sequence"] == window["sequence"]
        assert providers == [crop] and looks == [crop, crop]
        assert before["text"] == after["text"]
    finally:
        await surface.stop_watching()
