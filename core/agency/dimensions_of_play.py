"""The dimensions of a place to play: every way such places differ, each value a place can take, and what does each.

A game is not one kind of thing. It is a point in a space of independent dimensions: which screen of it she is on,
whose turn it is, who else is in it, how it is seen, what her controls are, what each does to what is hers, what the
place does to her when she does nothing, how the other things in it behave, what meeting each does, what the end is,
how she hears how she is doing, what she must remember, what she has to plan and keep to, what she carries and builds,
what has to be searched. A way of playing (core/agency/ways_of_playing.py) answers one value of one dimension, and
works whatever the other dimensions are; so the ways needed are the sum of the values, not the product of them
(docs/design-docs/THE_DIMENSIONS_OF_A_PLACE_TO_PLAY.md).

Each value names what does it: ways of playing by name, or ``module:function``; an empty ``done_by`` is a value she
has no answer for yet. ``signs`` are what shows a place has that value, in the terms its readings are kept in: the
mechanics its words and code show (core/agency/mechanics_she_knows.py), the ways of playing the map gives it
(docs/design-docs/THE_56_GAMES_AND_WHAT_THEY_ASK.md), and the situations its words put her in
(core/agency/situations_she_knows.py). tests/test_every_dimension_of_play_is_answered.py holds each answer to a
function that exists and is called from her live play.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DIMENSIONS", "Value", "values_of"]


@dataclass(frozen=True)
class Value:
    """One value of one dimension: what it asks of a player, what does it, and the signs that a place has it."""

    dimension: str
    name: str
    asks: str
    done_by: tuple[str, ...]
    signs: tuple[str, ...]


def _values(dimension: str, *rows: tuple[str, str, tuple[str, ...], tuple[str, ...]]) -> tuple[Value, ...]:
    return tuple(Value(dimension, name, asks, done_by, signs) for name, asks, done_by, signs in rows)


DIMENSIONS: dict[str, tuple[Value, ...]] = {
    "phase": _values(
        "phase",
        ("a way on", "a title, a menu or a next button, gone on from by its words",
         ("play as told", "core.skills.screen_pursuit_as_it_happens:PlayingAsItHappens"), ("menus", "choosing from options")),
        ("rules", "a screen of how it is played, read for what it asks", ("follow the lesson", "a legend", "a guide"),
         ("information shown", "story")),
        ("paused", "a screen that says it is paused, gone on from by what paused it", ("pause",), ("pausing",)),
        ("levels", "one stretch of play after another, each a new screen", ("a guide",), ("levels",)),
        ("a notice over it", "a dialog, a banner or a tab over the play, closed or gone on from",
         ("core.skills.sovereign_browser_drawing:played_on_the_drawing",), ("dialogs", "notices")),
        ("the end", "an end screen read for how it ended, and begun again where it is to be",
         ("core.language.how_a_game_ended:asks_to_win",), ("winning and losing",)),
    ),
    "turns": _values(
        "turns",
        ("as it happens", "a world that does not wait: looking and acting are one loop", ("steer",),
         ("steering", "pointer steering", "real-time control by imagining ahead")),
        ("waits for her", "a world that waits: each act chosen, then looked at", ("a plan to an end", "take stock"),
         ("waiting", "questions")),
        ("turns with a side", "a board, moves in turns, someone on the other side", ("a board in turns",), ("a board",)),
        ("shown, then done", "the screen shows, then waits for it done again", ("copy a sequence", "what a press lights"),
         ("following a sequence", "copy a sequence", "keys shown")),
        ("sent, then watched", "one act sent, then watched until it stops", ("send",),
         ("sending by strength", "aiming", "send")),
        ("timed windows", "a press counts by when it is made", ("time a press",), ("timing a press", "time a press")),
    ),
    "sides": _values(
        "sides",
        ("alone against the place", "nobody on the other side: what comes, comes", ("steer",), ("hazards", "enemies")),
        ("an opponent", "someone playing against her, whose moves she answers", ("a board in turns",
                                                                                 "core.agency.how_the_contest_stands:ContestStands"),
         ("a board",)),
        ("more than one of hers", "several of hers, one answering at a time", ("switch",), ("switching", "switch")),
        ("allies to keep", "someone on her side to keep safe or bring back", (), ("rescue and allies",)),
    ),
    "seeing": _values(
        "seeing",
        ("pixels", "what moves in the picture, and which thing is which", ("core.perception.what_moves_in_the_picture:WhatMoves",),
         ()),
        ("the drawing", "the drawing calls of a canvas, where they can be read", (
            "core.perception.the_drawing_as_objects:described",), ()),
        ("words", "the words on the screen, read where they stand", ("counters",
                                                                      "core.perception.what_the_pixels_show:recognize_text"),
         ("score", "lives", "a clock")),
        ("the program", "what its program says of its keys and its words, read before play", ("read the program",), ()),
        ("sound", "what it plays when something happens", (), ("sound controls",)),
    ),
    "space": _values(
        "space",
        ("one screen", "everything on one still screen", ("steer",), ()),
        ("a view going by", "a world that scrolls past", ("a view going by",), ("a world going by", "a view going by")),
        ("bigger than the view", "a world beyond the edges, and where in it she is", ("a world bigger than the view",),
         ("a world bigger than the screen", "a world bigger than the view")),
        ("a grid", "places in rows and columns, moved between", ("a grid's rule",), ("a grid's rule",)),
    ),
    "controls": _values(
        "controls",
        ("keys held", "a key held for as long as it is wanted", ("steer",), ("steering",)),
        ("the pointer followed", "a thing that goes where the pointer goes", ("steer",), ("pointer steering",)),
        ("clicks", "a click on a thing", ("click things",), ("clicking things", "click things")),
        ("drags", "a thing pressed on and carried to a place", ("carry",), ("dragging", "carry")),
        ("press, pull and let go", "a press pulled or held, let go to send", ("send",), ("sending by strength", "send")),
        ("held to charge", "a key held to build something up and let go", ("charge",), ("charging", "charge")),
        ("typed", "words typed into what asks for them", ("type",), ("typing", "type", "forms")),
        ("keys shown", "the keys a screen shows, pressed while it shows them", ("keys shown",), ("keys shown",)),
    ),
    "what a control does": _values(
        "what a control does",
        ("moves her one way", "each key moves her the same way on the screen, whichever way she faces", ("steer",),
         ("steering",)),
        ("pushes her", "each key adds to her going, which carries on", ("carried",), ("thrust", "carried")),
        ("plays out over a moment", "a press whose effect plays out after it: a jump, a dash", ("jump",),
         ("jumping", "jump")),
        ("turns her", "keys that turn her and keys that take her the way she faces", (), ("parking",)),
        ("lights a place", "a press that lights its place, where nothing of hers moves", ("what a press lights",),
         ("following a sequence",)),
        ("sends something out", "a press that sends a shot or a throw from her", ("shoot",), ("shooting", "shoot")),
        ("strikes close by", "a press that acts on what is close, sending nothing", ("strike",), ("striking", "strike")),
    ),
    "what the place does": _values(
        "what the place does",
        ("nothing", "she stays where she is left", ("steer",), ()),
        ("a pull", "something pulls her when she does nothing: down, along, aside", (), ("pull", "bouncing")),
        ("the view moves her", "the world going by carries her with it", ("a view going by",), ("a world going by",)),
        ("a clock runs down", "time running out whatever she does", (), ("a clock",)),
        ("what she has drains", "a bar or a supply that empties with time or use", ("bars",), ("fuel", "health")),
        ("more and faster", "what comes, coming more and faster as she goes on", (), ("levels",)),
    ),
    "how others behave": _values(
        "how others behave",
        ("go their way", "things that move along a line or a curve of their own", ("core.perception.what_moves_in_the_picture:WhatMoves",),
         ()),
        ("bounce", "things that come off walls and each other", ("core.perception.how_things_move_here:HowThingsMoveHere",),
         ("bouncing",)),
        ("watch for her", "things that see or reach her from a distance", ("unseen", "read the signs"), ("being seen",)),
        ("come for her", "things that turn toward her and follow her", (), ("enemies",)),
        ("send things at her", "things that throw or shoot at her", ("read the signs",), ("hazards",)),
    ),
    "meeting": _values(
        "meeting",
        ("worth meeting or not", "what meeting each kind does, learned from what counted", ("counters",),
         ("collecting", "hazards")),
        ("said by a legend", "what a thing is, from a picture beside words", ("a legend",), ()),
        ("by what she is now", "what meeting a thing does depends on her own state: a colour, a power held", (),
         ("matching", "power-ups")),
        ("pushed to a place", "a thing that moves when she runs into it, to be got somewhere", (), ("parking",)),
        ("used on", "a control used on the thing it is for, beside it", ("core.agency.using_a_control_on_a_thing:use_controls_on_things",),
         ("using things",)),
    ),
    "the end": _values(
        "the end",
        ("reach a place", "getting somewhere", ("a plan to an end",), ("reaching a place",)),
        ("collect", "getting the things worth getting", ("steer",), ("collecting",)),
        ("last", "holding on until the end", ("steer",), ("holding on until the end",)),
        ("take them out", "shooting or striking what is against her", ("shoot", "strike"), ("shooting", "enemies")),
        ("do it again", "doing again what was shown", ("copy a sequence",), ("following a sequence",)),
        ("build it", "putting parts where they go", ("stack", "a chain"), ("stacking", "chains")),
        ("cover it", "going over all of a place", ("steer",), ("covering ground",)),
        ("solve it", "a board, a rule, a puzzle", ("a board in turns", "a grid's rule"), ("a board", "matching")),
        ("make something", "nothing to win; something made", ("make",), ("making",)),
    ),
    "how it is going": _values(
        "how it is going",
        ("counters", "score, lives and time read off the screen", ("counters",), ("score", "lives")),
        ("bars", "bars that fill and empty", ("bars",), ("health", "fuel")),
        ("end screens", "words at the end of a run", ("core.language.how_a_game_ended:asks_to_win",), ("winning and losing",)),
        ("the screen answering", "a place lighting back, a thing going, a new screen", ("what a press lights",), ()),
    ),
    "memory": _values(
        "memory",
        ("what was shown, then hidden", "what each place showed when turned, kept until its pair turns up",
         ("remember what was shown",), ("matching",)),
        ("an order shown", "a sequence kept until it is done again", ("copy a sequence",), ("following a sequence",)),
        ("where things are out of sight", "what went out of view, and where", ("a world bigger than the view",),
         ("a world bigger than the screen",)),
        ("what she learned before", "what worked in this place last time", ("core.agency.what_she_keeps_of_a_game:kept_from",), ()),
    ),
    "keeping to it": _values(
        "keeping to it",
        ("the way it is played", "a way of playing taken up and kept while it pays", ("play as told",), ()),
        ("a plan to an end", "steps toward an end, each held to what it was to show", ("a plan to an end",),
         ("reaching a place",)),
        ("a thing gone for", "a thing chosen to go to, kept to until it is got, gone or plainly worse than another", (),
         ("collecting", "catching what falls")),
        ("when it is urgent", "less time left, fewer detours", (), ("a clock",)),
    ),
    "logistics": _values(
        "logistics",
        ("take and use", "a thing taken, then used on another", ("use things",), ("using things",)),
        ("carry to a place", "a thing carried and put down where it goes", ("carry",), ("dragging",)),
        ("give each what it asks", "things served to what asks for them", ("serve",), ("serving",)),
        ("spend a supply", "a meter or a count spent where it pays most", (), ("fuel", "holding to sustain")),
    ),
    "searching": _values(
        "searching",
        ("a board", "moves looked ahead against the other side", ("a board in turns",), ("a board",)),
        ("how hard and which way", "a shot's settings searched by where each went", ("send",), ("sending by strength",)),
        ("where a piece goes", "where a falling piece lands best", (), ("stacking", "matching")),
        ("a way through", "a way from here to there round what is in the way", (), ("reaching a place", "locked ways")),
    ),
}


def values_of(dimension: str) -> tuple[Value, ...]:
    return DIMENSIONS[dimension]
