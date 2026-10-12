"""The settings of her ways of playing, learned from what each paid, kind of situation by kind of situation.

Each way of playing has settings someone chose by hand: how often she throws at someone across the way, how often she
may duck, how soon a thing coming at her must be arriving for her to get out of its way. A person playing adjusts such
things as they go, and keeps what they learned for the next game like it: they throw faster when slow throws lose, duck
less when ducking leaves them no time to throw, and a player of many games knows that a game where someone across the
way throws at them wants quick hands, whatever the game is called.

So for each setting she has a few choices, and for each kind of situation, each choice keeps how well the stretches she
played with it went: a share of good and bad, as a Beta distribution. At the start of every stretch one value is drawn
from each choice's distribution and the choice whose draw is highest is played (Thompson sampling: Thompson, 1933; the
bandit problem). What has paid most is played most, and what has not been tried enough still gets its turns, in
proportion to the chance it is the best. The value chosen by hand starts a little ahead, so another must earn its way
past it.

How a stretch went is what play itself shows: what she gained against what she lost in it (her counters, her bars,
what she met), nothing given her from outside. The kind of situation is how she understands the game
(core/agency/how_a_game_works.py): what she is in it, whether it waits for her, what counts, and whether someone is
against her. Never a game's name: what is learned in one game of a kind is where she starts in the next.

Kept across games (core/runtime/what_she_learned.py), and brought back discounted like everything she carries over,
so a handful of stretches in a new game can overturn it.

Nothing here knows a game.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SETTINGS", "Setting", "SettingsThatPaid", "kind_of_situation"]

logger = logging.getLogger("Aura.SettingsThatPaid")


@dataclass(frozen=True)
class Setting:
    """A setting of a way of playing: its choices, the one chosen by hand, and what it is, said plainly."""

    name: str
    choices: tuple[float, ...]
    by_hand: float
    said: str


#: The settings learned in play. Each is read where it is used, with the value chosen by hand where none was learned.
SETTINGS: dict[str, Setting] = {setting.name: setting for setting in (
    Setting("throw every", (0.15, 0.2, 0.35, 0.6), 0.2, "how often I throw at someone across the way, in seconds"),
    Setting("duck every", (0.3, 0.5, 1.0), 0.5, "how often I may get out of the way, in seconds"),
    Setting("arriving within", (0.4, 0.7, 1.0), 1.0, "how soon a thing coming at me must arrive for me to get out of its way"),
)}

#: How far ahead the choice made by hand starts, in stretches gone well; and how much a stretch carried over from another
#: game counts, against one played in this.
BY_HAND_AHEAD = 1.0
CARRIED_OVER = 0.5
#: The fewest gains and losses in a stretch for it to say how a setting went.
SAYS_HOW_IT_WENT = 2


def kind_of_situation(how: Any) -> str:
    """The kind of situation a game puts her in, as she understands it: what she is in it, its tempo, what counts, and
    whether someone is against her; '' where she does not know what she is in it."""
    me = getattr(getattr(how, "me", None), "value", "")
    if not me:
        return ""
    parts = [me, getattr(how.tempo, "value", "") or "?", getattr(how.counts, "value", "") or "?",
             "against someone" if how.against_someone() else "no one against"]
    return " | ".join(parts)


@dataclass
class SettingsThatPaid:
    """For one kind of situation: how stretches went with each choice of each setting, and the choices of this stretch."""

    kind: str
    #: setting name -> str(choice) -> [went well, went badly], in stretches (fractions of one).
    went: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    chosen: dict[str, float] = field(default_factory=dict)

    def choose(self, rng: random.Random | None = None) -> dict[str, float]:
        """This stretch's choice of every setting, each drawn by how it has paid."""
        rng = rng or random
        self.chosen = {}
        for name, setting in SETTINGS.items():
            best, highest = setting.by_hand, -1.0
            for choice in setting.choices:
                good, bad = self.went.get(name, {}).get(str(choice), [0.0, 0.0])
                ahead = BY_HAND_AHEAD if choice == setting.by_hand else 0.0
                draw = rng.betavariate(1.0 + good + ahead, 1.0 + bad)
                if draw > highest:
                    best, highest = choice, draw
            self.chosen[name] = best
        return dict(self.chosen)

    def get(self, name: str) -> float:
        """The value of a setting for this stretch: the one chosen, else the one chosen by hand."""
        return self.chosen.get(name, SETTINGS[name].by_hand)

    def paid(self, gains: int, losses: int) -> bool:
        """How the stretch went, by what she gained against what she lost in it, for each setting's choice; False
        where too little happened in it to say."""
        if not self.chosen or gains + losses < SAYS_HOW_IT_WENT:
            return False
        well = gains / (gains + losses)
        for name, choice in self.chosen.items():
            counts = self.went.setdefault(name, {}).setdefault(str(choice), [0.0, 0.0])
            counts[0] += well
            counts[1] += 1.0 - well
        return True

    def best(self, name: str) -> tuple[float, float] | None:
        """The choice of a setting that has paid best so far, and the share of stretches that went well with it; None
        where none has been played enough to say."""
        tried = [(good / (good + bad), float(choice)) for choice, (good, bad) in self.went.get(name, {}).items()
                 if good + bad >= 3.0]
        if not tried:
            return None
        share, choice = max(tried)
        return choice, share

    def says(self) -> str:
        """What she has learned of her settings in situations like this one, as she would say it."""
        lines = []
        for name, setting in SETTINGS.items():
            best = self.best(name)
            if best is not None and best[0] != setting.by_hand:
                lines.append(f"{setting.said}: {best[0]:g} has gone best ({best[1]:.0%} of the time)")
        return "; ".join(lines)

    # -- kept across games ------------------------------------------------------------------------------------------

    def keep(self) -> bool:
        from core.runtime.what_she_learned import named, remember

        return bool(self.kind) and remember(named("settings that paid", self.kind), {"went": self.went})

    @classmethod
    def for_the_kind(cls, kind: str) -> SettingsThatPaid:
        """What was learned before in situations of this kind, discounted, to start from."""
        if not kind:
            return cls(kind)
        from core.runtime.what_she_learned import named, recall

        held = recall(named("settings that paid", kind)).get("went")
        went: dict[str, dict[str, list[float]]] = {}
        if isinstance(held, dict):
            for name, choices in held.items():
                if name in SETTINGS and isinstance(choices, dict):
                    went[name] = {str(choice): [CARRIED_OVER * float(counts[0]), CARRIED_OVER * float(counts[1])]
                                  for choice, counts in choices.items()
                                  if isinstance(counts, list) and len(counts) == 2}
        return cls(kind, went)


def for_this_stretch(keep: dict[str, Any]) -> SettingsThatPaid:
    """The settings of a stretch about to be played: what was learned in this game's kind of situation, kept from the
    stretch before where the kind is the same, each setting chosen afresh by how it has paid."""
    from core.agency.how_a_game_works import understood
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    kind = kind_of_situation(understood(guide)) if guide is not None and hasattr(guide, "controls") else ""
    held = keep.get("settings")
    settings = held if isinstance(held, SettingsThatPaid) and held.kind == kind else SettingsThatPaid.for_the_kind(kind)
    # Where the situation is not understood yet, nothing learned in it could be kept: the settings chosen by hand.
    chosen = settings.choose() if kind else {}
    logger.info("settings for a stretch of %r: %s", kind or "a situation not yet understood", chosen)
    return settings


def after_the_stretch(settings: SettingsThatPaid | None, gains: int, losses: int) -> None:
    """How the stretch went, for the choices played in it; kept for the next game of its kind."""
    if settings is not None and settings.paid(gains, losses):
        settings.keep()


def setting(run: Any, name: str, otherwise: float) -> float:
    """The value of a setting for the stretch ``run`` is playing, else ``otherwise``."""
    settings = getattr(run, "settings", None)
    return settings.get(name) if isinstance(settings, SettingsThatPaid) else otherwise
