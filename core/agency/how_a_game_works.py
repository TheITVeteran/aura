"""How a game works, from first principles: the few things anyone asks of a game they have never seen, each held as an
answer, how sure she is of it, and what told her.

A person who walks up to a game asks the same few things of it, whatever it is, and plays from the answers:

- what am I in it: a body I steer, a body that follows the mouse, someone aiming from this side of the picture, a body
  I steer while the mouse aims, a body that goes on by itself and does what I press when I press it, someone who sends
  a thing by pulling back and letting go, a hand that puts things in place, a hand among things, or someone making
  something;
- what does each of my inputs do (core/cognition/a_guide_to_a_place.py holds what is said of each);
- does it wait for me, go on without me, wait while I line a shot up, or take turns with me: which is how long I may
  think before each act, and whether a moment spent looking costs me;
- what counts: a win, a score, or a thing made, held against what the person asked of her.

These were answered in many places, each from the words afresh, and the places did not always agree: offline
2026-10-11 a food fight said "take aim with your mouse and click to throw; dodge food by pressing the space bar", and
one part of her pressed each arrow to see what it did, another took the opponent for her body, another waited for the
picture to come to rest. Here they are answered once, from all she has been told and all she has seen, for every part
of her to act on and for her to say.

What her own acts have shown outweighs what she was told, and what the screens say outweighs what the program alone
shows. Nothing here knows a game: every answer is a kind of thing a game can be, and every kind is told by what anyone
would go by.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.verify.invariants import invariant

__all__ = [
    "AIMS_FROM_A_BODY",
    "AGAINST_SOMEONE",
    "AIMS_FROM_HERE",
    "FOLLOWS",
    "MAKES",
    "PLACES",
    "POINTS",
    "RUNS",
    "SENDS",
    "SHOTS",
    "STEERED",
    "TIMES",
    "TURNS",
    "UNKNOWN",
    "WAITS",
    "Answer",
    "HowItWorks",
    "how_it_works",
    "seen_going_on_by_itself",
    "seen_to_answer",
    "seen_to_send",
    "understood",
]

# What she is in it.
STEERED = "a body I steer with keys"
FOLLOWS = "a body that follows the pointer"
AIMS_FROM_HERE = "someone aiming from this side of the picture"
AIMS_FROM_A_BODY = "a body I steer with keys while the pointer aims"
TIMES = "a body that goes on by itself and does what I press when I press it"
SENDS = "someone who sends a thing by pulling back and letting go"
PLACES = "a hand that puts things in place"
POINTS = "a hand among things"
MAKES = "someone making something"
UNKNOWN = ""

# Who else is there.
AGAINST_SOMEONE = "someone on the other side"

# Whether it waits for her.
WAITS = "it waits for me"
RUNS = "it goes on without me"
SHOTS = "it waits while I line up a shot, then goes"
TURNS = "we take turns"

#: The keys that move a body about.
_MOVING = frozenset({"up", "down", "left", "right"})
_WASD = frozenset("wasd")

#: How sure an answer is, by where it came from: her own acts, a screen's words, the program alone, a default.
SEEN_SURE = 0.9
SAID_SURE = 0.7
CODED_SURE = 0.4
GUESSED_SURE = 0.2

#: The situations a place's words put her in that are played as things happen (core/agency/situations_she_knows.py).
_IN_TIME = frozenset({"catching what falls", "keeping clear of what comes", "keeping something away from whoever is after it",
                      "holding on until the end", "getting them all against a clock", "shooting what comes",
                      "pressing at the right moment", "a fight at close quarters", "steering something somewhere"})


@dataclass
class Answer:
    """One answer: what it is, how sure she is of it (0 to 1), and what told her, in plain words."""

    value: str = UNKNOWN
    sure: float = 0.0
    why: list[str] = field(default_factory=list)

    def known(self, at_least: float = SAID_SURE) -> bool:
        return bool(self.value) and self.sure >= at_least

    def held_to(self, value: str, sure: float, why: str) -> None:
        """Taken to be ``value`` where that is as sure as what it was, or surer."""
        if sure >= self.sure:
            self.value, self.sure, self.why = value, sure, [why, *[w for w in self.why if w != why]][:4]


@dataclass
class HowItWorks:
    """What she understands of how a game works, axis by axis."""

    me: Answer = field(default_factory=Answer)
    tempo: Answer = field(default_factory=Answer)
    counts: Answer = field(default_factory=Answer)
    #: Who else is there: someone on the other side, played against, or no one.
    against: Answer = field(default_factory=Answer)
    #: What each input does, as said: "the pointer" -> "throw", "space" -> "dodge food".
    inputs: dict[str, str] = field(default_factory=dict)

    # -- what follows from it ---------------------------------------------------------------------------------------

    def has_a_body(self) -> bool | None:
        """Whether there is a body of hers in the picture to be found; None where that is not known."""
        if not self.me.known():
            return None
        return self.me.value not in (AIMS_FROM_HERE, SENDS, PLACES, POINTS, MAKES)

    def arrows_are_a_guess_worth_trying(self) -> bool:
        """Whether the arrows, which nothing named, are worth pressing to see what they do: not where she is known to
        be worked some other way."""
        return not self.me.known() or self.me.value in (STEERED, AIMS_FROM_A_BODY, TIMES)

    def try_pointer_first(self, otherwise: bool) -> bool:
        """Try the input known to move her before incidental controls on the interface.

        Offline 2026-10-12 a body named for the keys was never found while its
        menu's pointer controls kept the loop sweeping for a following body.
        """
        if self.me.known() and self.me.value in (STEERED, AIMS_FROM_A_BODY):
            return False
        if self.me.known() and self.me.value == FOLLOWS:
            return True
        return otherwise

    def against_someone(self) -> bool:
        """Whether the place is played against someone on the other side, as far as she understands it."""
        return self.against.value == AGAINST_SOMEONE and self.against.sure >= SAID_SURE

    def goes_on_without_her(self) -> bool:
        """Whether every moment she spends looking or thinking is a moment the game goes on: a move is not waited on to
        come to rest, nor a screen of play looked over at leisure."""
        return self.tempo.known(SAID_SURE) and self.tempo.value == RUNS

    # -- said -------------------------------------------------------------------------------------------------------

    def said(self) -> str:
        """In a sentence, as she understands it; "" where she understands nothing yet."""
        parts = []
        if self.me.value:
            parts.append(f"I'm {self.me.value}")
        if self.inputs:
            parts.append("; ".join(_a_control_said(key, act) for key, act in list(self.inputs.items())[:3]))
        if self.tempo.value:
            parts.append(self.tempo.value)
        if self.against.value:
            parts.append("there's someone on the other side")
        if self.counts.value:
            parts.append({"win": "there's a win to be had", "score": "it keeps a score",
                          "make": "it's for making something"}.get(self.counts.value, self.counts.value))
        return ("As I understand it: " + "; ".join(parts) + ".") if parts else ""

    def unknown(self) -> list[str]:
        """What she does not know yet of how it works, as questions her play is to answer."""
        asked = {"me": "what I am in it", "tempo": "whether it waits for me", "counts": "what counts"}
        return [question for axis, question in asked.items() if not getattr(self, axis).known(SAID_SURE)]


def _a_control_said(key: str, act: str) -> str:
    """A control and what it does, as said: "space to jump", and where the place said what it does ("increases forward
    speed"), "right increases forward speed", not "right to increases forward speed"."""
    first = act.split()[0] if act.split() else ""
    does = first.endswith("s") and not first.endswith(("ss", "us", "is")) and len(first) > 3
    return f"{key} {act}" if does else f"{key} to {act}"


@invariant("agency.known_body_controls_set_trial_priority", scope="agency",
           owner="core/agency/how_a_game_works.py", observational=False)
def _known_body_controls_invariant() -> tuple:
    assert not HowItWorks(me=Answer(STEERED, SEEN_SURE)).try_pointer_first(True)
    assert not HowItWorks(me=Answer(AIMS_FROM_A_BODY, SAID_SURE)).try_pointer_first(True)
    assert HowItWorks(me=Answer(FOLLOWS, SEEN_SURE)).try_pointer_first(False)
    assert HowItWorks(me=Answer(STEERED, CODED_SURE)).try_pointer_first(True)
    assert not HowItWorks().try_pointer_first(False)
    return ()


def understood(guide: Any, also: str = "") -> HowItWorks:
    """What she understands now of the place ``guide`` is to: from everything it has read, ``also`` (what is on the
    screen in front of her and not taken in yet), and what her play there has shown, which outweighs what was said.
    Worked out again only when there is more to go on."""
    read = getattr(guide, "_read", ()) or ()
    controls = getattr(guide, "controls", None) or {}
    key = (len(read), also, tuple((k, getattr(c, "act", "")) for k, c in controls.items()),
           tuple((getattr(guide, "pointer", None) or {}).items()))
    kept = getattr(guide, "how", None)
    if not (isinstance(kept, tuple) and kept[0] == key):
        said = " ".join([*sorted(read), also]).strip()
        kept = (key, how_it_works(said, guide if guide is not None and hasattr(guide, "controls") else None))
        if guide is not None and hasattr(guide, "how"):
            guide.how = kept
    how = kept[1]
    for axis, answer in (getattr(guide, "how_seen", None) or {}).items():
        getattr(how, axis).held_to(answer.value, answer.sure, answer.why[0] if answer.why else "")
    return how


def seen_going_on_by_itself(guide: Any, going: bool) -> None:
    """What a screen of play did while she did nothing, kept on the guide for every part of her to go by."""
    seen = _seen(guide)
    if seen is not None:
        answer = seen.setdefault("tempo", Answer())
        answer.held_to(RUNS if going else WAITS, SEEN_SURE,
                       "I saw it go on while I did nothing" if going else "nothing moved until I did something")


def seen_to_answer(guide: Any, by: str) -> None:
    """What her trials found her body answers to, "keys" or "the pointer", kept on the guide."""
    seen = _seen(guide)
    if seen is None or by not in ("keys", "the pointer"):
        return
    answer = seen.setdefault("me", Answer())
    if by == "the pointer":
        answer.held_to(FOLLOWS, SEEN_SURE, "something followed my pointer")
    else:
        answer.held_to(STEERED, SEEN_SURE, "something moved when I pressed the keys")


def seen_to_send(guide: Any) -> None:
    """That a thing went when she pressed it, pulled and let go, kept on the guide: she is someone who sends."""
    seen = _seen(guide)
    if seen is not None:
        seen.setdefault("me", Answer()).held_to(SENDS, SEEN_SURE, "a thing went when I pressed it, pulled back and let go")


def _seen(guide: Any) -> dict[str, Answer] | None:
    return getattr(guide, "how_seen", None) if guide is not None else None


def how_it_works(said: str, guide: Any = None, rules: Any = None) -> HowItWorks:
    """What she understands of how a place works from what it has said (its screens' words, what she was told, what
    she looked up), what its guide holds (core/cognition/a_guide_to_a_place.py) and its rules as read
    (core/agency/what_the_rules_said.py)."""
    from core.agency.what_the_rules_said import WhatTheRulesSaid

    rules = rules if rules is not None else WhatTheRulesSaid.read(said)
    how = HowItWorks()
    how.inputs = _inputs(guide)
    _what_counts(how, said)
    _who_else(how, said)
    _what_i_am(how, said, guide, rules)
    _whether_it_waits(how, said)
    return how


#: Keys said together as one control where they do one thing.
_TOGETHER = (("the arrows", ("up", "down", "left", "right")), ("WASD", ("w", "a", "s", "d")))


def _inputs(guide: Any) -> dict[str, str]:
    controls = getattr(guide, "controls", None) or {}
    out = {}
    for key, control in controls.items():
        act = " ".join(part for part in (getattr(control, "act", ""), getattr(control, "on", "")) if part)
        if act and act not in ("play", "start", "begin", "continue"):
            out[key] = act
    for name, keys in _TOGETHER:
        acts = {out.get(key) for key in keys}
        if len(acts) == 1 and None not in acts:
            out = {name: acts.pop(), **{key: act for key, act in out.items() if key not in keys}}
    return out


#: Words that put someone on the other side: an opponent, a rival, "you against", "beat him", "the first to".
#: Not enemies, which are what comes at her, nor the other side of a place, which is where she goes: offline 2026-10-12 a
#: painting game with enemies about and a far side to reach was said to be played against someone.
_AGAINST = re.compile(r"\b(?:you against|against (?:your|him|her|them)\b|opponents?|rivals?|versus|vs\.?|"
                      r"the other (?:player|team)|the first to|beat (?:him|her|them)\b|"
                      r"before (?:he|she|they) (?:does|do|can))", re.I)


def _who_else(how: HowItWorks, said: str) -> None:
    against = _AGAINST.search(said)
    if against:
        how.against.held_to(AGAINST_SOMEONE, SAID_SURE, f"its words say {against.group(0)!r}")


def _what_counts(how: HowItWorks, said: str) -> None:
    from core.language.how_a_game_ended import what_it_asks_of_a_player

    asks = what_it_asks_of_a_player(said)
    if asks:
        how.counts.held_to(asks, SAID_SURE, f"its words set a player to {asks} it" if asks != "make" else
                           "its words say it is for making something")


def _what_i_am(how: HowItWorks, said: str, guide: Any, rules: Any) -> None:
    from core.agency.mechanics_she_knows import mechanics_in
    from core.agency.playing_by_shots import sends_by_letting_go
    from core.agency.the_controls_a_game_names import controls_named_in
    from core.cognition.a_guide_to_a_place import PROGRAM, pointer_acts

    named, pointer_said = controls_named_in(said, keys_without_words=(), during_play=True)
    controls = getattr(guide, "controls", None) or {}
    worded = {key for key, control in controls.items() if getattr(control, "source", "") != PROGRAM}
    keys = set(named) | {key for key in worded if key != "the pointer"}
    coded_keys = {key for key in controls if key not in worded and key != "the pointer"}
    moving = bool(keys & _MOVING) or len(keys & _WASD) >= 3
    coded_moving = not moving and (bool(coded_keys & _MOVING) or len(coded_keys & _WASD) >= 3)
    pointer = pointer_said or "the pointer" in worded
    moves_with_pointer = (getattr(guide, "pointer", None) or {}).get("moves")
    sure = SAID_SURE if (moving or pointer or keys) else CODED_SURE
    if how.counts.value == "make":
        how.me.held_to(MAKES, SAID_SURE, "its words say it is for making something")
    elif moves_with_pointer:
        how.me.held_to(FOLLOWS, SAID_SURE, f"its words say the pointer moves {moves_with_pointer}")
    elif sends_by_letting_go(said) and not moving:
        how.me.held_to(SENDS, SAID_SURE, "its words speak of pulling back and letting go")
    elif moving and pointer and (rules.a_click_is_a_shot or bool(getattr(guide, "pointer_aims", lambda: False)())):
        how.me.held_to(AIMS_FROM_A_BODY, SAID_SURE, "its words give keys to move and the pointer to aim")
    elif moving:
        how.me.held_to(STEERED, SAID_SURE, "its words give keys that move")
    elif rules.a_click_is_a_shot:
        # Less sure where its program listens for keys that move: a body may be steered that the words leave unsaid.
        how.me.held_to(AIMS_FROM_HERE, CODED_SURE if coded_moving else SAID_SURE,
                       "its words say to aim with the pointer and click, and no key moves")
    elif guide is not None and pointer_acts(guide):
        how.me.held_to(TIMES, SAID_SURE, f"its words say a click is to {pointer_acts(guide)}")
    elif "dragging" in mechanics_in(said):
        how.me.held_to(PLACES, sure, "its words speak of dragging things")
    elif pointer:
        how.me.held_to(POINTS, CODED_SURE if coded_moving else sure, "its words name the pointer and no key")
    elif coded_moving:
        how.me.held_to(STEERED, CODED_SURE, "its program listens for keys that move")
    elif keys:
        # Keys named only for doing something ("space to jump"): a body, its moving not yet known.
        how.me.held_to(STEERED, GUESSED_SURE, "its words give keys, none of them for moving")


def _whether_it_waits(how: HowItWorks, said: str) -> None:
    from core.agency.situations_she_knows import situations_in

    if how.me.value == SENDS:
        how.tempo.held_to(SHOTS, SAID_SURE, "a thing sent waits to be sent")
        return
    if how.me.value in (MAKES, PLACES):
        how.tempo.held_to(WAITS, CODED_SURE, "what is made or put in place waits for the hand")
        return
    in_time = [situation.name for situation in situations_in(said) if situation.name in _IN_TIME]
    if in_time:
        how.tempo.held_to(RUNS, SAID_SURE, f"its words put me in {in_time[0]}")
        return
    hurried = _HURRIED.search(said)
    if hurried:
        how.tempo.held_to(RUNS, SAID_SURE, f"its words hurry me: {hurried.group(0)!r}")


#: Words that say a thing will not wait for her: what falls, lands or gets away before she acts, a clock running out.
_HURRIED = re.compile(r"\bbefore (?:they|it|the \w+|time) (?:\w+ away|falls?|lands?|hits? the|reach(?:es)? the|escapes?|"
                      r"runs? out|crash(?:es)?|explodes?|disappears?|vanish(?:es)?)\b|\bas (?:fast|quickly) as (?:you|possible)\b|\bhurry\b|"
                      r"\bin (?:\d+|sixty|thirty|ninety) seconds\b|\bout of time\b|\btime'?s up\b|\btime runs out\b|"
                      r"\b(?:beat|against) the clock\b", re.I)
