"""The real browser run begins while advice is pending and refuses advice after retirement."""
from __future__ import annotations

import asyncio
import time

import pytest

from core.cognition import guidance_lifetime, taking_stock
from core.cognition.a_guide_to_a_place import Guide
from core.skills import sovereign_browser_drawing as drawing
from core.skills import sovereign_browser_taking_stock as stocks

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_pending_optional_research_does_not_prevent_the_real_body_entry(monkeypatch):
    entered, release, acting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    admitted = []

    class PendingStock:
        taken = 0
        counsel = None

        def remembered(self):
            return False

        async def take(self, *args):
            self.taken += 1
            entered.set()
            stamp = guidance_lifetime.publication_stamp()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            stamp.check()
            admitted.append("late")

    async def guide(page, keep):
        return Guide(place="An editor")

    async def pursue(**kwargs):
        acting.set()
        await entered.wait()
        return {"moves": [], "completed": False}

    from core.skills import screen_pursuit

    monkeypatch.setattr(drawing, "_the_guide", guide)
    monkeypatch.setattr(drawing, "_what_the_guide_says", lambda *args: None)
    monkeypatch.setattr(drawing, "_keep_the_guide", lambda *args: None)
    monkeypatch.setattr(screen_pursuit, "pursue_on_screen", pursue)
    monkeypatch.setattr(guidance_lifetime, "TEARDOWN_SECONDS", .01)
    task = asyncio.create_task(drawing._one_run(None, (0, 0, 1, 1), "Make a card.", "https://example.org/editor",
                                              time.monotonic() + 60, {"stock": PendingStock()}))
    try:
        done, _ = await asyncio.wait({task}, timeout=.5)
        assert acting.is_set(), "optional research prevented entry into the body"
        assert task in done, "pending advice held the body after its own work finished"
        result, _reflexes = task.result()
        assert result["completed"] is False
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not admitted
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_actual_stocktaking_rejects_late_publication_after_its_parent_run_ends(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    said, keep = [], {}
    monkeypatch.setattr(stocks, "_tell", said.append)
    monkeypatch.setattr(guidance_lifetime, "TEARDOWN_SECONDS", .01)

    async def delayed(*args, **kwargs):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return taking_stock.Counsel((), (taking_stock.Heard("current instructions", "", "Drag the object into the destination."),))

    monkeypatch.setattr(taking_stock, "take_stock", delayed)
    async with guidance_lifetime.guidance_work():
        pending = guidance_lifetime.create_guidance_task(stocks.Stocktaking("An editor").take(
            taking_stock.BEFORE, "Make a card.", [], "", [], keep, time.monotonic() + 60))
        await asyncio.wait_for(entered.wait(), timeout=2)
    try:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert "counsel" not in keep
        assert len(said) == 1  # only the announcement; no retired advice
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
