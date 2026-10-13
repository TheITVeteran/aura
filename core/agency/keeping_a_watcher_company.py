"""Keeping a watcher company: in a stretch of play where nothing has been said for a while, how it is going is said.

A person playing a game for someone talks as they go. When nothing new happens to remark on, they still say what
they are doing and how it is going: "I've noticed the pot lights up when I click it", "that's twelve for me and nine
against so far", "still at it: thirty clicks in the last half minute". Her remarks while playing were each said once,
the first time: once she had said what each thing was and what each click did, she played on in silence. LIVE
2026-10-13 a point-and-click game was played for 106 seconds with nothing said, and then for 86 more, before an
audience.

So where a stretch has gone ``KEEP_COMPANY_S`` with nothing said, one line is offered of how it is going, made from
what the stretch itself has kept, in turn: what she has noticed lately and not said, and what she is doing (her moves
in it). Each is offered only where it is true and has changed since it was last offered, so nothing is made up and
nothing is said twice; and they take turns, so that a watcher who has heard one kind of line hears the other next.
How the game stands is said where it is read (the score, what is left), not here: what she counts as gained and lost
is what she sees happen, not the game's own count, and "159 for me" would tell a watcher nothing true.

Nothing here knows a game.
"""
from __future__ import annotations

from typing import Any

__all__ = ["KEEP_COMPANY_S", "keeping_company"]

#: How long a stretch of play may go with nothing said before she says how it is going, in seconds.
KEEP_COMPANY_S = 15.0
#: The fewest moves in a while for "what I am doing" to be worth saying.
MOVES_WORTH_SAYING = 3


def keeping_company(run: Any, at: float, notebook: Any = None) -> str:
    """The line to keep a watcher company with now, or '' where something was said lately or nothing new is true.

    ``run`` is the stretch being played (``began``, ``said_at``, ``clicks``, ``taps``,
    ``input_key_downs`` and a dict ``company`` this keeps its turns in); ``notebook`` what she has noticed
    (core/cognition/what_she_notices.py), if any."""
    if at - float(run.said_at) < KEEP_COMPANY_S or at - float(run.began) < KEEP_COMPANY_S:
        return ""
    if notebook is None:
        from core.cognition.a_guide_to_a_place import THE_GUIDE

        notebook = getattr(THE_GUIDE.get(), "notes", None)
    company = run.company
    ways = (_noticed_lately, _what_i_am_doing)
    for _ in ways:
        turn = int(company.get("turn", 0)) % len(ways)
        company["turn"] = turn + 1
        line = ways[turn](run, at, notebook, company)
        if line:
            return line
    return ""


def _noticed_lately(run: Any, at: float, notebook: Any, company: dict[str, Any]) -> str:
    """What she has noticed lately and not said: the note that holds her interest most."""
    if notebook is None or not hasattr(notebook, "lately"):
        return ""
    told = company.setdefault("noticed", set())
    said_already = set(getattr(run, "lines", ()) or ())
    for note in notebook.lately(at, most=5):
        said = " ".join(str(getattr(note, "said", "") or "").split()).rstrip(".")
        if not said or said in told or any(said.lower() in line.lower() for line in said_already):
            continue
        told.add(said)
        # A note that is already a remark ("that's new: ...") is said as it is; the rest are what she has noticed.
        if said.lower().startswith(("that's", "that is")):
            return f"{said[:1].upper()}{said[1:]}."
        return f"I've noticed {said[:1].lower()}{said[1:]}."
    return ""


def _what_i_am_doing(run: Any, at: float, notebook: Any, company: dict[str, Any]) -> str:
    """How many moves she has made since this was last said, and of what kind, where there have been enough."""
    keys = getattr(run, "input_key_downs", None) or {}
    made = {"clicks": int(getattr(run, "clicks", 0) or 0), "taps": int(getattr(run, "taps", 0) or 0),
            "presses": int(sum(keys.values()) if hasattr(keys, "values") else 0)}
    before = company.get("made") or {}
    since = {kind: made[kind] - int(before.get(kind, 0)) for kind in made}
    since_at = float(company.get("made_at", run.began))
    if sum(since.values()) < MOVES_WORTH_SAYING:
        return ""
    company["made"], company["made_at"] = made, at
    most = max(since, key=lambda kind: since[kind])
    one = {"clicks": "click", "taps": "tap", "presses": "press"}[most]
    return f"Still at it: {since[most]} {most if since[most] != 1 else one} in the last {round(at - since_at)} seconds."
