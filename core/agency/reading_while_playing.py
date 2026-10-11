"""What her fast way of playing reads as it plays: the counters, the bars and the words on the screen.

Moved out of core/agency/playing_as_it_happens.py, which re-exports every name here. A counter's rise or a bar's fall
is a gain or a loss as its words say; a loss with nothing touching her is put down to what was near her then; and the
things that are writing are for reading, not for touching.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.agency.playing_as_it_happens import _Run
    from core.agency.what_meeting_things_does import WhatMeetingDoes
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.what_moves_in_the_picture import WhatMoves


async def _keep_reading(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers, moves: WhatMoves,
                        picture: Any, at: float, read_words: Callable[[Any], list[dict[str, Any]]] | None,
                        *, say: Any = None) -> None:
    """Read the counters off to one side, and the bars every picture; take in the last reading when it is done."""
    from core.agency.playing_as_it_happens import READ_EVERY_S
    from core.perception.the_drawing_as_objects import words_in

    _read_the_bars(run, meeting, hers, moves, picture, at, say)
    rendered = words_in(picture)
    if rendered is not None:
        if run.reading is not None:
            run.reading.cancel()
            run.reading = None
        run.read_at = at
        _read_the_words(run, meeting, hers, moves, rendered, at, say)
        return
    if read_words is None:
        return
    if run.reading is not None and run.reading.done():
        regions, when = run.reading.result()
        run.reading = None
        _read_the_words(run, meeting, hers, moves, regions, when, say)
    if run.reading is None and at - run.read_at >= READ_EVERY_S:
        run.read_at = at
        bgr = picture[:, :, ::-1].copy()

        async def _read() -> tuple[list[dict[str, Any]], float]:
            return await asyncio.to_thread(read_words, bgr), at

        run.reading = asyncio.ensure_future(_read())


def _read_the_words(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers,
                    moves: WhatMoves, regions: list[dict[str, Any]], when: float, say: Any = None) -> None:
    """The same counter semantics for renderer text and text read from pixels."""
    from core.agency.playing_as_it_happens import _say
    from core.cognition.a_guide_to_a_place import heard_in_play

    run.regions_read = list(regions)
    for region in regions:
        said = " ".join(str(region.get("text") or "").lower().split())
        if said and len(run.words_read) < 400:
            run.words_read.append((when, said))
    # What play says as it goes may change how it is played (a new power, a new stage): the guide hears it.
    for line in heard_in_play([str(region.get("text") or "") for region in regions])[:1]:
        _say(run, say, line, when, once=line)
    mine = hers.thing(moves)
    her_x = moves.share(mine.x, mine.y)[0] if mine is not None else None
    for verdict in meeting.read(regions, when, her_x):
        if verdict["what"] == "gain":
            run.gains += 1
            run.blows.gained(when)
            run.charging.gained(when)
        else:
            run.losses += 1
            hers.lost_a_life(when)
            _lost_from_a_distance(run, meeting, mine, moves, verdict)
            # A loss sets her back and changes the place: where she seemed not to get to before it is not known to be
            # out of her reach (offline 2026-10-09, set back to the start each time, the way to a coin was marked out
            # of her reach and she waited in a corner while it stood there).
            run.barred.clear()
    run.contest.heard(" ".join(str(region.get("text") or "") for region in regions))
    run.contest.counted(meeting.readouts.current, meeting.readouts.where, her_x, when, meeting.readouts.out_of)
    meeting.writing = _what_is_writing(moves, regions)


def _read_the_bars(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers, moves: WhatMoves, picture: Any, at: float,
                   say: Any) -> None:
    """Every bar's rise or fall, as a gain or a loss as its words say; and how much she has left, for how careful to be.

    A bar that empties as she is hit is the cost of what hit her, learned as a counter's fall is: LIVE 2026-10-09 three
    heroes shared a health bar, and she played as if nothing could hurt her.
    """
    from core.agency.naming_what_she_sees import where_on_screen
    from core.agency.playing_as_it_happens import _say
    from core.agency.what_she_has_left import bars_read, what_is_left
    from core.perception.how_full_a_bar_is import BarsOnTheScreen

    if run.bars is None:
        run.bars = BarsOnTheScreen()

    def place(x: float, y: float) -> str:
        return where_on_screen(moves, x * moves.shape[1], y * moves.shape[0]) if moves.shape[0] else "middle"

    # A bar with no name going down before anything on the screen answers her may be a story playing out or a thing
    # loading: counted, but not said to be what she has left (LIVE 2026-10-10 "The bar at the bottom left goes down: it's
    # what I have left." over an opening scene).
    answering = (getattr(hers, "kind", None) is not None or getattr(hers, "follows_pointer", False)
                 or (run.presses is not None and run.presses.moves_anything(4.0)))
    for verdict in bars_read(run.bars, picture, at, run.regions_read, place):
        meeting._verdict(verdict)
        if verdict["what"] == "gain":
            run.gains += 1
            continue
        run.losses += 1
        _lost_from_a_distance(run, meeting, hers.thing(moves), moves, verdict)
        hit = any(0.0 <= at - when <= 1.0 for when in meeting._met.values())
        named = verdict["counter"]
        if answering or not named.startswith("the bar"):
            _say(run, say, f"{named[0].upper()}{named[1:]} goes down{' as I am hit' if hit else ''}: it's what I have left.",
                 at, once=f"bar {named}")
    run.vitals = meeting.vitals = what_is_left(run.bars)


def _lost_from_a_distance(run: _Run, meeting: WhatMeetingDoes, mine: Any, moves: WhatMoves, verdict: dict[str, Any]) -> None:
    """A loss with nothing having touched her lately is put down to the things near her when it came, as they stood then."""
    from core.agency.playing_as_it_happens import TOUCHED_WITHIN_S

    if run.reach is None or mine is None:
        return
    then = (float(verdict.get("from", verdict["at"])) + float(verdict["at"])) / 2
    if any(abs(when - then) <= TOUCHED_WITHIN_S for when in meeting._met.values()):
        return
    run.reach.lost_untouched(_as_it_stood(mine, then), [_as_it_stood(thing, then) for thing in moves.things.values()
                                                         if thing.number != mine.number and thing.number not in meeting.writing])


def _as_it_stood(thing: Any, then: float) -> Any:
    """A thing as it stood at ``then``, from where its way had it nearest that time."""
    from types import SimpleNamespace

    path = list(getattr(thing, "path", ()) or ())
    at, x, y = min(path, key=lambda step: abs(step[0] - then)) if path else (then, thing.x, thing.y)
    return SimpleNamespace(number=thing.number, kind=thing.kind, x=x, y=y, vx=thing.vx, vy=thing.vy, w=thing.w, h=thing.h)


def _what_is_writing(moves: WhatMoves, regions: list[dict[str, Any]]) -> set[int]:
    """The things that are writing on the screen: words are for reading, not for touching."""
    from core.agency.playing_as_it_happens import _boxes_meet, _moved_box

    tall, wide = moves.shape
    boxes = [
        (float(r["x"]) * wide, float(r["y"]) * tall, (float(r["x"]) + float(r["width"])) * wide, (float(r["y"]) + float(r["height"])) * tall)
        for r in regions
    ]
    return {t.number for t in moves.things.values() if not t.moved and any(_boxes_meet(t.box(), _moved_box(b, 0, 0, 1.0)) for b in boxes)}
