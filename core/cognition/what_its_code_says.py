"""What a place's code says of how it is played, taken into the guide to it as the surest thing known of it.

The code is what the game does: a key it moves its hero by moves the hero, whatever its screens say; a level it passes
once coverage reaches the quota is passed then and no sooner. So what its code was read to say
(core/perception/how_a_program_plays.py) goes into the guide (core/cognition/a_guide_to_a_place.py) above every
other source for what each key does and for what passes, fails and wins: the keys play is worked by and what each
does; the keys that are not part of play (a pause, the sound, a cheat that ends the level outright), kept out of it;
and how a level is passed, failed and the game won, said in plain words. What the code does not say (a key whose use
could not be read) is left to the other sources.

Nothing here knows a game.
"""
from __future__ import annotations

import re
from typing import Any

from core.cognition.a_guide_to_a_place import CODE
from core.verify import invariant

__all__ = ["CODE", "in_plain_words", "take_in_how_it_plays"]

#: What a key does, as the guide names a control's act.
_ACTS = {"moves": "move", "jumps": "jump", "ducks": "duck", "acts": "act", "pauses": "pause", "sounds": "mute",
         "quits": "quit", "ends the level": "cheat"}
#: The acts that are not part of play.
_NOT_PLAY = {"pause", "mute", "quit", "cheat"}
#: The words for what a kind of ending is, said of it.
_SAID = {"passed": "a level is passed when", "failed": "a level is failed when", "won": "the game is won when",
         "over": "the game is over when"}


def in_plain_words(condition: str) -> str:
    """A condition from code in plain words: names without their marks and with their words apart, calls as what they
    get. ``getCoverage() >= _quota`` is "coverage ≥ quota"; ``_paintBucket < 0`` is "paint bucket < 0"."""
    said = re.sub(r"\b(?:this|_root|_parent|_global|r\d+)\.", "", condition)
    said = re.sub(r"\b(?:get|calc|calculate|compute)_?(?=[A-Z_])", "", said)
    said = re.sub(r"\(\s*\)", "", said)
    said = re.sub(r"(?<![\w$.])[A-Za-z_$][\w$]*\.(?=[A-Za-z_$])", "", said)  # board._paintBucket: the last name says it
    said = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", said).replace("_", " ")
    for code, plain in ((">=", "≥"), ("<=", "≤"), ("!=", "≠"), ("===", "="), ("==", "="), ("&&", "and"), ("||", "or")):
        said = said.replace(code, plain)
    return " ".join(said.lower().split())


def take_in_how_it_plays(guide: Any, plays: Any, at: float) -> list[str]:
    """What a place's code says of how it is played, into its guide: the controls it was read to have, the controls
    that are not part of play, and how a level is passed and failed and the game won. The lines newly said."""
    if guide is None or not plays:
        return []
    from core.cognition.a_guide_to_a_place import Control

    for key, what in getattr(plays, "keys", {}).items():
        act = _ACTS.get(getattr(what, "does", ""))
        if not act:
            continue
        guide.controls[key] = Control(key, act, CODE, at)
        if act in _NOT_PLAY:
            named = f"{'a cheat' if act == 'cheat' else act} ({key})"
            if named not in guide.not_for_play:
                guide.not_for_play.append(named)
    news: list[str] = []
    for end in getattr(plays, "ends", []) or []:
        if end.kind not in _SAID or not end.when:
            continue
        said = f"{_SAID[end.kind]} {in_plain_words(end.when)}"
        for name, made in (end.made_of or {}).items():
            said += f" ({in_plain_words(name)} is {in_plain_words(made)})"
        kept = guide.lose if end.kind in ("failed", "over") else guide.win
        if said not in kept:
            kept.append(said)
            news.append(said)
    guide.code_says = plays
    guide.sources.add(CODE)
    return news


@invariant("cognition.parsed_controls_keep_their_source_priority", scope="cognition",
           owner="core/cognition/what_its_code_says.py", observational=False)
def _parsed_control_priority_invariant() -> tuple:
    from types import SimpleNamespace

    from core.cognition.a_guide_to_a_place import PROGRAM, Guide

    guide = Guide()
    plays = SimpleNamespace(keys={'j': SimpleNamespace(does='moves')}, ends=[])
    take_in_how_it_plays(guide, plays, 1.)
    guide.take_in(PROGRAM, ['Press J to quit.'], 2.)
    assert guide.controls['j'].act == 'move' and guide.controls['j'].source == CODE
    return ()
