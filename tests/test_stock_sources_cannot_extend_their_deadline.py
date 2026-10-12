"""A delayed source loses publication authority without holding up usable evidence."""
from __future__ import annotations

import asyncio

import pytest

from core.cognition import guidance_lifetime, taking_stock

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_a_source_that_resists_cancellation_does_not_extend_preparation(monkeypatch):
    release, cancelled = asyncio.Event(), asyncio.Event()
    late_allowed = []
    monkeypatch.setattr(guidance_lifetime, "TEARDOWN_SECONDS", .01)
    monkeypatch.setattr(taking_stock, "what_to_take", lambda heard, *_args: heard)

    async def fast(*args):
        return [taking_stock.Heard("current instructions", "", "Drag the object into the destination.")]

    async def slow(*args):
        stamp = guidance_lifetime.publication_stamp()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        stamp.check()
        late_allowed.append(True)
        return [taking_stock.Heard("late", "", "An obsolete instruction.")]

    situation = taking_stock.Situation("An editor", "Make a card.", (), "", taking_stock.BEFORE)
    operation = asyncio.create_task(taking_stock.take_stock(situation, {"current": fast, "late": slow}, seconds=.01))
    try:
        done, _pending = await asyncio.wait({operation}, timeout=.5)
        assert operation in done, "a source extended the caller's deadline by resisting cancellation"
        counsel = operation.result()
        assert [item.source for item in counsel.kept] == ["current instructions"]
        assert cancelled.is_set()
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not late_allowed
    finally:
        release.set()
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_empty_sources_need_no_wait_and_no_claim_of_evidence():
    situation = taking_stock.Situation("A form", "Fill the form.", (), "", taking_stock.BEFORE)
    counsel = await taking_stock.take_stock(situation, {}, seconds=.01)
    assert not counsel


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -1.])
async def test_invalid_budgets_do_not_start_sources(seconds):
    async def source(*args):
        pytest.fail("invalid preparation must not start")

    situation = taking_stock.Situation("A form", "Fill the form.", (), "", taking_stock.BEFORE)
    with pytest.raises(ValueError):
        await taking_stock.take_stock(situation, {"source": source}, seconds=seconds)
