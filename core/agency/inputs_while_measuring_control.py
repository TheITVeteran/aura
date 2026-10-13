"""Keep isolated input experiments separate from actions performed together.

Offline 2026-10-12 attacks were delivered during movement trials. Changing
poses then answered a key that was never tried alone. A second input ends
the first input's measured interval; it cannot extend its causal receipts.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

from core.verify import invariant

logger = logging.getLogger(__name__)
_PRIMARY: ContextVar[frozenset[tuple[int, int]]] = ContextVar('primary_control_delivery', default=frozenset())
_INPUTS = frozenset({'down', 'up', 'tap', 'press', 'click', 'point'})


def body_probe_is_active(run: Any, hers: Any, *, pointer_steps: int) -> bool:
    """Reserve a finite discovery pass for one input at a time, including its rests."""
    if getattr(run, 'pointer_first', False):
        return not hers.follows_pointer and 0 <= getattr(run, 'pointed', pointer_steps) < pointer_steps
    keys = getattr(run, 'keys', ())
    trying = getattr(run, 'trying', None)
    return bool(keys and type(trying) is int and 0 <= trying < 4 * len(keys)
                and (hers.number is None or not hers.keys_known(keys)))


class HandsWithControlReceipts:
    """Observe delivered input boundaries without changing what the hands do."""

    def __init__(self, hands: Any, run: Any, hers: Any) -> None:
        self.hands, self.run, self.hers = hands, run, hers

    def __getattr__(self, name: str) -> Any:
        original = getattr(self.hands, name)
        if name not in _INPUTS or not callable(original):
            return original

        async def delivered(*args: Any, **kwargs: Any) -> Any:
            if (id(self), id(asyncio.current_task())) not in _PRIMARY.get():
                self._other_input(name, time.monotonic())
            return await original(*args, **kwargs)

        return delivered

    def _other_input(self, name: str, at: float) -> None:
        if hasattr(self.run, 'previous_control'):
            self.run.previous_control = None
        held = self.hers._held_at(at)
        if held is not None:
            began, key, _trying = held
            for record in (*self.hers._by_thing.values(), self.hers._hers):
                record.end(key, began)
        self.run.presses.interrupted()
        if name != 'point':
            self.hers._followed.clear()
            self.hers._sighted = None
        logger.debug("single-input observation ended before %s", name)


@contextmanager
def _primary_delivery(hands: HandsWithControlReceipts):
    hands.hers._followed.clear()
    hands.hers._sighted = None
    token = _PRIMARY.set(_PRIMARY.get() | {(id(hands), id(asyncio.current_task()))})
    try:
        yield
    finally:
        _PRIMARY.reset(token)


def primary_delivery(hands: Any):
    """Only the trial's own input delivery may continue its independent interval."""
    return _primary_delivery(hands) if isinstance(hands, HandsWithControlReceipts) else nullcontext()


async def hold_control(hands: Any, run: Any, hers: Any, key: str, at: float, *, trying: bool = False,
                       now: Any = None) -> None:
    """A changed physical input starts one trial; a changed policy cannot."""
    if key == run.held:
        return
    now = now or time.monotonic
    began = now()
    with primary_delivery(hands):
        if run.held:
            await hands.up(run.held)
            if run.presses is not None:
                run.presses.released(run.held, now())
        dispatched = now()
        if key:
            await hands.down(key)
            run.input_key_downs[key] += 1
        if run.presses is not None:
            run.presses.pressed(key, now(), run.things.values(), hers.number,
                                epoch=hers.identification.epoch, independent=trying)
    delivered = now()
    run.responses.append(max(0.0, delivered - at))
    timing = getattr(run, 'control_timing', None)
    if timing is not None:
        run.control_delivery = timing.delivered(at, began, delivered)
    if key:
        hers.tapped(key, delivered, began=dispatched)
    run.held, run.held_since, run.held_trying = key, delivered, trying
    hers.holding(key, delivered, trying=trying)


@invariant("agency.a_second_input_ends_an_isolated_body_trial", scope="agency",
           owner="core/agency/inputs_while_measuring_control.py", observational=False)
def _single_input_interval_invariant() -> tuple:
    from core.agency.which_one_answers_to_her import _Speeds

    record = _Speeds()
    for at in (1.1, 1.2):
        record.add('a', 30., 0., press=1., when=at)
    before = list(record.by_press[('a', 1.)])
    record.end('a', 1.)
    record.add('a', -80., 0., press=1., when=1.3)
    record.unseen('a', 1., contradicted=True)
    assert record.by_press[('a', 1.)] == before
    assert record.timed[('a', 1.)][-1][0] == 1.2
    return ()
