"""What she wants, formed from what the things in her life have been worth to her.

Bryan, 13 October: she has to direct herself. "I want to do XYZ, I am going to
figure out how to do XYZ", from her own inner pull, and choosing among options
she gives herself. Things have to matter to her, and she has to be able to
know what they are.

Measured the same day, on her last 500 self-started choices (10 to 13 October):
she chose 11 distinct things, one of them, "Running a self-integrity scan", 379
times. Every option on every slate came from a budget draining at a fixed rate
or from a loop with a fixed menu, and the topic of a curiosity goal was drawn
with `random.choice`. The valuation in this package was registered and had
been asked to appraise nothing, because nothing in her life was handed to it.

So a want had no object. This is the record that gives it one.

    thing      whatever a turn was about that has a name: a person, a subject,
               something she made. Things she cannot go to (herself, the
               source that won her attention, how an act came out) are kept
               too, because they share the credit for a turn
    liking     the thing's own share of what its turns were worth to her. A
               turn's worth is read at its end against what she expected
               (core/affect/what_it_was_worth.py), signed and ranked among
               her recent turns, so it lies in [-1, 1]. The things present
               predict it together and divide the error between them by how
               unsure she is of each, which is the Kalman form of the
               Rescorla-Wagner rule (Dayan and Kakade 2000; Gershman 2015).
               A new thing beside a familiar one takes most of a surprise,
               and a thing that only ever arrives with something she already
               likes earns nothing of its own
    unsure     the variance of that share. It starts at the variance of a
               number that could be anywhere in [-1, 1], falls with contact,
               and stops falling at what the last 256 contacts would leave,
               the window every other reading of her history uses, so an old
               association keeps learning
    known      whether she can say she likes it or dislikes it: its turns were
               worth more than she expected more often, or less often, than
               her turns in general, by a two-sided sign test at the
               conventional level. Until then the answer is that she has not
               had enough of it to say
    ways       how she has gone to it, each with what going that way was
               worth. A way is an act she has actually taken, so the options
               she gives herself are her own
    open       what she started with it and did not finish, in the words she
               started it in

A want is read off the record each time it is asked for. Every thing she can
go to is appraised by `core/conation/engine.py`: what it has come to be worth
pulls, what one more contact would still teach her about it pulls (expected
information gain, from `unsure`), and having just had it holds the pull down
until it recovers. Whatever comes out above nothing is a want, with the thing,
the way she would go, and the measurements behind it.

What a turn was about is read from its words as counts and never as meaning.
A term is one of a turn's things when two measurements agree. The exchange
uses it more than her exchanges have before, by Dunning's log-likelihood ratio
(1993), the standard measure of which words are key to a text against a
reference, corrected for the number of terms the exchange offers. And it is
dwelt on where it comes up: it is found in fewer than half the exchanges its
count would reach if it were spread by chance, which is a residual inverse
document frequency above one bit (Church and Gale 1995). The second is what
tells a subject from a word that is merely frequent in one kind of talk.
Measured on her 351 stored exchanges on 13 October, "limpet" and "scar" read
2.4 and 2.6 bits and "its", "home" and "watch" 0.6, 0.5 and 0.3. Her
reference is her own exchanges, the stored ones included.

Nothing here reads what a sentence says. Being told she loves something adds
a thing she has met once and knows nothing about.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from statistics import NormalDist
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.conation.state import Incentive
from core.runtime.atomic_writer import atomic_write_text_behind
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Conation.Wants")

__all__ = [
    "FIND_OUT",
    "REACHABLE_KINDS",
    "read_turn",
    "Thing",
    "Want",
    "Way",
    "WhatSheWants",
    "get_what_she_wants",
    "liking_of",
    "terms_of",
    "reset_for_test",
]

#: How many contacts a reading of her own history is taken over.
_WINDOW = 256

#: The chance of calling a like or a dislike that is not there; the level
#: core/agency/what_she_is_like.py reads her choices at.
_WRONG = 0.05

#: The variance of a share that could be anywhere in [-1, 1].
_UNKNOWN = 1.0 / 3.0

#: The kinds of thing a turn names that she can go to herself. The rest of
#: what a turn was about (herself, the source that won her attention, an act
#: and how it came out) shares the credit and is nowhere to go.
REACHABLE_KINDS: frozenset[str] = frozenset({"topic", "person"})

#: The way any named thing affords before she has gone to it by another:
#: finding out about it.
FIND_OUT = "find_out"

MAX_THINGS = 4096
#: How many distinct terms her reference of past exchanges keeps.
MAX_TERMS = 65536

#: A word: letters only, in any script, long enough to carry content.
_WORD = re.compile(r"[^\W\d_]{3,}")
#: Where a run of words ends, so a pair never spans two clauses.
_BREAK = re.compile(r"[.!?;:,()\[\]\n\"]+")


def terms_of(*texts: str) -> Counter[str]:
    """The words of an exchange and the pairs that stand side by side in it."""
    found: Counter[str] = Counter()
    for text in texts:
        for run in _BREAK.split(str(text or "").lower()):
            words = _WORD.findall(run)
            found.update(words)
            found.update(f"{a} {b}" for a, b in zip(words, words[1:], strict=False))
    return found
MAX_OPEN = 8
SCHEMA = 2


def liking_of(reading: Any) -> float | None:
    """A turn's worth as liking at contact, in [-1, 1], or nothing to read.

    `size` is where the turn's worth sits among her recent turns, so the
    scale is her own and no channel's units reach it.
    """
    if reading is None or not getattr(reading, "measured", False):
        return None
    try:
        worth = float(getattr(reading, "worth", 0.0) or 0.0)
        size = float(getattr(reading, "size", 0.0) or 0.0)
    # not a failure: a reading that is not numbers is no reading.
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(worth) and math.isfinite(size)) or worth == 0.0:
        return 0.0
    return math.copysign(max(0.0, min(1.0, size)), worth)


def _two_sided(above: int, of: int, chance: float) -> float:
    """The chance of a count at least this far from chance, either way."""
    if of <= 0 or not 0.0 < chance < 1.0:
        return 1.0

    def mass(k: int) -> float:
        return math.exp(
            math.lgamma(of + 1) - math.lgamma(k + 1) - math.lgamma(of - k + 1)
            + k * math.log(chance) + (of - k) * math.log(1.0 - chance)
        )

    below = sum(mass(k) for k in range(0, above + 1))
    over = sum(mass(k) for k in range(above, of + 1))
    return min(1.0, 2.0 * min(below, over))


@dataclass
class Way:
    """One way she has gone to a thing, and what going that way was worth."""

    times: int = 0
    liking: float = 0.0

    def went(self, liking: float) -> None:
        self.times += 1
        self.liking += (liking - self.liking) / min(self.times, _WINDOW)


@dataclass
class Thing:
    key: str
    label: str = ""
    reachable: bool = False
    contacts: int = 0
    good: int = 0
    liking: float = 0.0
    unsure: float = _UNKNOWN
    channels: dict[str, float] = field(default_factory=dict)
    first: int = 0
    last: int = 0
    ways: dict[str, Way] = field(default_factory=dict)
    open: dict[str, str] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.key.split(":", 1)[0] if ":" in self.key else self.key

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "reachable": self.reachable,
            "contacts": self.contacts,
            "good": self.good,
            "liking": self.liking,
            "unsure": self.unsure,
            "channels": dict(self.channels),
            "first": self.first,
            "last": self.last,
            "ways": {name: [way.times, way.liking] for name, way in self.ways.items()},
            "open": dict(self.open),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Thing:
        thing = cls(key=str(data["key"]))
        thing.label = str(data.get("label", ""))
        thing.reachable = bool(data.get("reachable", False))
        thing.contacts = max(0, int(data.get("contacts", 0)))
        thing.good = max(0, min(thing.contacts, int(data.get("good", 0))))
        thing.liking = max(-1.0, min(1.0, float(data.get("liking", 0.0))))
        thing.unsure = max(0.0, min(_UNKNOWN, float(data.get("unsure", _UNKNOWN))))
        thing.channels = {str(k): float(v) for k, v in dict(data.get("channels", {})).items()}
        thing.first = int(data.get("first", 0))
        thing.last = int(data.get("last", 0))
        for name, (times, liking) in dict(data.get("ways", {})).items():
            thing.ways[str(name)] = Way(times=int(times), liking=float(liking))
        thing.open = {str(k): str(v) for k, v in dict(data.get("open", {})).items()}
        return thing


@dataclass(frozen=True)
class Want:
    """One thing she wants now, how she would go to it, and why."""

    thing: str
    label: str
    way: str
    #: "finish" when she left something undone with it, "more" when what it
    #: has been worth is the larger pull, "find out" when what she does not
    #: yet know about it is.
    kind: str
    wanting: float
    utility: float
    #: How much of her attention and planning it should claim, in [0, 1].
    priority: float
    origin: str
    why: str
    liking: float
    contacts: int
    #: True when she likes it, False when she dislikes it, None while she has
    #: not had enough of it to say.
    known: bool | None
    unfinished: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "thing": self.thing,
            "label": self.label,
            "way": self.way,
            "kind": self.kind,
            "wanting": round(self.wanting, 6),
            "utility": round(self.utility, 6),
            "priority": round(self.priority, 6),
            "origin": self.origin,
            "why": self.why,
            "liking": round(self.liking, 6),
            "contacts": self.contacts,
            "known": self.known,
            "unfinished": self.unfinished,
        }


class WhatSheWants:
    """The things she has met, what each has been worth, and what follows."""

    def __init__(self, path: str | Path | None = None, *, engine: Any = None) -> None:
        self._lock = checked_lock("conation.what_she_wants", reentrant=True)
        self._things: dict[str, Thing] = {}
        self._turns = 0
        self._good_turns = 0
        #: Running mean of her squared prediction errors: how far a turn's
        #: liking usually lands from what the things in it predicted.
        self._noise = _UNKNOWN
        self._read = 0
        #: Where she went since the last turn was read, and how. The turn that
        #: is read next is the one that going is part of.
        self._went: list[tuple[str, str]] = []
        #: How often each term has been used across her exchanges so far, and
        #: how many terms those exchanges held: what a new exchange is set
        #: against to say which of its terms are particular to it.
        self._said: Counter[str] = Counter()
        self._said_total = 0
        #: How many of those exchanges each term was in, and how many there
        #: have been: whether a term is dwelt on or spread thin.
        self._said_in: Counter[str] = Counter()
        self._exchanges = 0
        self._standing: list[Want] = []
        self._engine = engine
        self._path = Path(path) if path is not None else None
        self._load()

    # ── where it is kept ─────────────────────────────────────────────────

    def _file(self) -> Path:
        if self._path is None:
            self._path = state_root() / "data" / "conation" / "what_she_wants.json"
        return self._path

    def _load(self) -> None:
        try:
            stored = json.loads(self._file().read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            record_degradation(
                "what_she_wants", exc, severity="warning",
                action="started with nothing met; the stored record did not read back",
            )
            return
        try:
            if int(stored.get("schema", 0)) != SCHEMA:
                return
            self._turns = max(0, int(stored.get("turns", 0)))
            self._good_turns = max(0, min(self._turns, int(stored.get("good_turns", 0))))
            self._noise = max(1e-9, float(stored.get("noise", _UNKNOWN)))
            self._read = max(0, int(stored.get("read", 0)))
            for row in stored.get("things", []):
                thing = Thing.from_dict(row)
                self._things[thing.key] = thing
            self._said = Counter({str(k): int(v) for k, v in dict(stored.get("said", {})).items()})
            self._said_total = max(0, int(stored.get("said_total", 0)))
            self._said_in = Counter({str(k): int(v) for k, v in dict(stored.get("said_in", {})).items()})
            self._exchanges = max(0, int(stored.get("exchanges", 0)))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._things.clear()
            self._turns = self._good_turns = self._read = 0
            self._noise = _UNKNOWN
            record_degradation(
                "what_she_wants", exc, severity="warning",
                action="started with nothing met; the stored record had another shape",
            )

    def _held(self) -> dict[str, Any]:
        """Everything kept, copied while the lock is held."""
        return {
            "schema": SCHEMA,
            "turns": self._turns,
            "good_turns": self._good_turns,
            "noise": self._noise,
            "read": self._read,
            "saved_at": time.time(),
            "things": [thing.to_dict() for thing in self._things.values()],
            "said": dict(self._said),
            "said_total": self._said_total,
            "said_in": dict(self._said_in),
            "exchanges": self._exchanges,
        }

    def _save(self, payload: dict[str, Any]) -> None:
        """Written after the lock is let go: a write can wait on the disk."""
        try:
            path = self._file()
            path.parent.mkdir(parents=True, exist_ok=True)
            # Behind the loop: a turn ends inside the kernel's async tick.
            atomic_write_text_behind(path, json.dumps(payload), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            record_degradation(
                "what_she_wants", exc, severity="warning",
                action="kept what she met this turn in memory only",
            )

    def _conation(self) -> Any:
        if self._engine is None:
            from core.conation.engine import get_conation

            self._engine = get_conation()
        return self._engine

    # ── what a turn was about ────────────────────────────────────────────

    def about(self, *texts: str) -> list[str]:
        """The terms this exchange uses more than her exchanges have before.

        Most particular first. The exchange then joins what later ones are
        set against. With nothing yet to set it against, nothing is
        particular.
        """
        here = terms_of(*texts)
        if sum(here.values()) <= 0:
            return []
        with self._lock:
            found = self._subjects(here)
            self._count_exchange(here)
        return found

    def subjects_of(self, *texts: str) -> list[str]:
        """What `about` would say of this exchange, without counting it.

        For asking what something said is about when it is not one of her
        exchanges yet, or may never be.
        """
        here = terms_of(*texts)
        if sum(here.values()) <= 0:
            return []
        with self._lock:
            return self._subjects(here)

    def _subjects(self, here: Counter[str]) -> list[str]:
        """The subjects among an exchange's terms. Lock held."""
        total = sum(here.values())
        before, before_total = self._said, self._said_total
        scored: list[tuple[float, str]] = []
        if before_total > 0:
            # One test per term offered, so the level each is held to is the
            # conventional one divided by how many there are.
            critical = NormalDist().inv_cdf(1.0 - _WRONG / len(here) / 2.0) ** 2
            exchanges = self._exchanges + 1
            for term, count in here.items():
                past = before.get(term, 0)
                if count / total <= past / before_total:
                    continue
                pooled = (count + past) / (total + before_total)
                ratio = 2.0 * count * math.log(count / (total * pooled))
                if past:
                    ratio += 2.0 * past * math.log(past / (before_total * pooled))
                if ratio <= critical:
                    continue
                # Dwelt on: with this exchange counted, the exchanges it is in
                # are fewer than half of those a term used this often would be
                # in by chance.
                by_chance = exchanges * (1.0 - math.exp(-(past + count) / exchanges))
                if math.log2(by_chance / (self._said_in.get(term, 0) + 1)) > 1.0:
                    scored.append((ratio, term))
        # A pair is one thing when neither word is there without the other.
        # Then the words are the pair, and a pair one of whose words also
        # stands alone is that word beside something else.
        pairs = {
            term
            for _ratio, term in scored
            if " " in term and all(here[word] <= here[term] for word in term.split())
        }
        inside = {word for term in pairs for word in term.split()}
        return [
            term
            for _ratio, term in sorted(scored, key=lambda row: (-row[0], row[1]))
            if (term in pairs) or (" " not in term and term not in inside)
        ]

    def _count_exchange(self, here: Counter[str]) -> None:
        """Add one exchange to what later ones are set against. Lock held."""
        self._said.update(here)
        self._said_total += sum(here.values())
        self._said_in.update(here.keys())
        self._exchanges += 1
        if len(self._said) > MAX_TERMS:
            self._said = Counter(dict(self._said.most_common(MAX_TERMS // 2)))
            self._said_in = Counter({term: self._said_in[term] for term in self._said})

    def talked_before(self) -> bool:
        """Whether there is anything yet to set an exchange against."""
        with self._lock:
            return self._said_total > 0

    def remember_talk(self, exchanges: Iterable[Iterable[str]]) -> int:
        """Exchanges she had before this record was kept, as reference only.

        Nothing in them is read as a thing she met: what they were worth was
        never measured. They are what a new exchange is set against.
        """
        added = 0
        with self._lock:
            for exchange in exchanges:
                self._count_exchange(terms_of(*exchange))
                added += 1
            held = self._held()
        self._save(held)
        return added

    # ── a turn ───────────────────────────────────────────────────────────

    def _touch(self, key: str, label: str, reachable: bool) -> Thing:
        thing = self._things.get(key)
        if thing is None:
            if len(self._things) >= MAX_THINGS:
                oldest = min(self._things.values(), key=lambda t: (t.contacts, t.last))
                self._things.pop(oldest.key, None)
            thing = Thing(key=key, first=self._turns)
            self._things[key] = thing
        if label and not thing.label:
            thing.label = label[:160]
        thing.reachable = thing.reachable or reachable
        return thing

    def met(
        self,
        present: Iterable[str],
        liking: float | None,
        *,
        reachable: Iterable[str] = (),
        labels: Mapping[str, str] | None = None,
        payoff: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """One turn's contact with everything it was about.

        `reachable` names the present things she could go to herself. A thing
        she went to since the last turn was read is part of this one, with the
        way she went (see `went`).
        """
        if liking is None:
            return {"read": False}
        liking = max(-1.0, min(1.0, float(liking)))
        labels = dict(labels or {})
        can_reach = set(reachable)
        keys = list(dict.fromkeys(str(key) for key in present if str(key).strip()))
        with self._lock:
            went, self._went = dict(self._went), []
            for key in went:
                if key not in keys:
                    keys.append(key)
                can_reach.add(key)
            if not keys:
                return {"read": False}
            self._turns += 1
            self._read += 1
            self._good_turns += liking > 0.0
            things = [self._touch(key, labels.get(key, ""), key in can_reach) for key in keys]
            predicted = max(-1.0, min(1.0, sum(thing.liking for thing in things)))
            error = liking - predicted
            total = sum(thing.unsure for thing in things) + self._noise
            floor = self._noise / _WINDOW
            shares: dict[str, float] = {}
            for thing in things:
                gain = thing.unsure / total
                shares[thing.key] = gain
                thing.liking = max(-1.0, min(1.0, thing.liking + gain * error))
                thing.unsure = max(floor, thing.unsure * (1.0 - gain))
                thing.contacts += 1
                thing.good += liking > 0.0
                thing.last = self._turns
                rate = 1.0 / min(thing.contacts, _WINDOW)
                for channel, paid in (payoff or {}).items():
                    try:
                        paid = float(paid)
                    # not a failure: a payoff that is not a number is one
                    # channel that says nothing about this thing this turn.
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(paid):
                        held = thing.channels.get(channel, 0.0)
                        thing.channels[channel] = held + (paid - held) * rate
                if went.get(thing.key):
                    thing.ways.setdefault(went[thing.key], Way()).went(liking)
            self._noise += (error * error - self._noise) / min(self._read, _WINDOW)
            self._noise = max(1e-9, self._noise)
            held = self._held()
        self._save(held)
        self._teach(things, liking, error, shares)
        return {"read": True, "predicted": predicted, "error": error, "shares": shares}

    def _teach(self, things: list[Thing], liking: float, error: float, shares: Mapping[str, float]) -> None:
        """Fold the contact into the valuation that prices her wants.

        Having had a thing holds the want for it down by as much as the
        contact paid, and what she expected against what she got is the error
        her curiosity about it learns from.
        """
        try:
            engine = self._conation()
            for thing in things:
                if not thing.reachable:
                    continue
                engine.learn(
                    thing.key,
                    experienced_liking=liking,
                    realised_pull=max(0.0, thing.liking),
                    completeness=max(0.0, liking),
                    prediction_error=min(1.0, abs(error) * shares.get(thing.key, 0.0)),
                )
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "what_she_wants", exc, severity="warning",
                action="kept the contact on her record; the valuation was not taught by it",
            )

    # ── where she went, and what she started and did not finish ─────────

    def went(self, key: str, way: str, *, label: str = "") -> None:
        """She went to a thing herself, this way. Credited when the turn is read."""
        if not key or not way:
            return
        with self._lock:
            self._touch(key, label, True)
            self._went.append((key, way))

    def began(self, key: str, item: str, words: str, *, label: str = "") -> None:
        """She started something with a thing; it stays open until she ends it."""
        if not key or not item or not words.strip():
            return
        with self._lock:
            thing = self._touch(key, label, True)
            if item not in thing.open and len(thing.open) >= MAX_OPEN:
                thing.open.pop(next(iter(thing.open)))
            thing.open[item] = " ".join(words.split())[:500]
            held = self._held()
        self._save(held)

    def ended(self, key: str, item: str, *, finished: bool) -> None:
        """What she started with a thing came to an end, done or given up."""
        with self._lock:
            thing = self._things.get(key)
            if thing is None or item not in thing.open:
                return
            thing.open.pop(item, None)
            held = self._held()
        self._save(held)
        if not finished:
            self.failed(key)

    def failed(self, key: str) -> None:
        """She went for a thing and did not get it.

        Failing at something she wants builds up, by how much she wanted it
        (core/conation/dynamics.py), and past the point that file sets she
        stops wanting it for now.
        """
        try:
            engine = self._conation()
            record = engine.salience._records.get(key)
            wanting = float(getattr(record, "last_wanting", 0.0) or 0.0)
            engine.dynamics.observe_attempt(key, wanting=wanting, succeeded=False)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "what_she_wants", exc, severity="debug",
                action="an attempt that failed left no frustration behind",
            )

    # ── what she can say about a thing ───────────────────────────────────

    def thing(self, key: str) -> Thing | None:
        return self._things.get(key)

    def known(self, key: str) -> bool | None:
        """Whether she likes it, dislikes it, or has not had enough of it to say."""
        with self._lock:
            thing = self._things.get(key)
            if thing is None or thing.contacts <= 0 or self._turns <= 0:
                return None
            chance = self._good_turns / self._turns
            if _two_sided(thing.good, thing.contacts, chance) >= _WRONG:
                return None
            return thing.good / thing.contacts > chance

    def likes(self) -> list[Thing]:
        """What she likes, most liked first; reachable or not."""
        with self._lock:
            found = [thing for thing in self._things.values() if self.known(thing.key) is True]
        return sorted(found, key=lambda thing: (-thing.liking, thing.key))

    def dislikes(self) -> list[Thing]:
        with self._lock:
            found = [thing for thing in self._things.values() if self.known(thing.key) is False]
        return sorted(found, key=lambda thing: (thing.liking, thing.key))

    def regard_for_text(self, text: str) -> float:
        """Her strongest stated like or dislike among the things a text names.

        Zero when it names nothing she can say she likes or dislikes. A name
        is a thing's own label, whole, so this reads which of her things an
        option is about and nothing about how the option is worded.
        """
        lowered = " ".join(str(text or "").lower().split())
        strongest = 0.0
        with self._lock:
            things = [thing for thing in self._things.values() if len(thing.label) > 2]
        for thing in things:
            if thing.label.lower() not in lowered or self.known(thing.key) is None:
                continue
            if abs(thing.liking) > abs(strongest):
                strongest = thing.liking
        return strongest

    # ── what she wants ───────────────────────────────────────────────────

    def _way_for(self, thing: Thing, offered: Iterable[str]) -> str:
        """The way to go to it, from the ways she has gone to things.

        Each way's worth is its mean with the optimism of UCB1 (Auer,
        Cesa-Bianchi and Fischer 2002) over the times she has gone to this
        thing, so a way never tried on it comes first. Untried ways are
        ordered by what they have been worth on other things of its kind.
        """
        ways = set(thing.ways) | set(offered)
        elsewhere: dict[str, list[float]] = {}
        for other in self._things.values():
            if other.kind != thing.kind or other.key == thing.key:
                continue
            for name, way in other.ways.items():
                elsewhere.setdefault(name, []).append(way.liking)
                ways.add(name)
        if not ways:
            return FIND_OUT
        went = sum(way.times for way in thing.ways.values())

        def score(name: str) -> tuple[float, float, str]:
            pooled = elsewhere.get(name, [])
            prior = sum(pooled) / len(pooled) if pooled else 0.0
            tried = thing.ways.get(name)
            if tried is None or tried.times <= 0 or went <= 0:
                return (math.inf, prior, name)
            # Liking spans two units, so the bound does.
            bound = tried.liking + 2.0 * math.sqrt(2.0 * math.log(went) / tried.times)
            return (bound, prior, name)

        return max(sorted(ways), key=score)

    def wants(self, *, ways: Iterable[str] = (), limit: int | None = None) -> list[Want]:
        """What she wants now, most wanted first."""
        offered = tuple(ways)
        with self._lock:
            candidates = [thing for thing in self._things.values() if thing.reachable and thing.label]
            noise = self._noise
        try:
            engine = self._conation()
        except (ImportError, AttributeError) as exc:
            record_degradation(
                "what_she_wants", exc, severity="warning",
                action="no want could be priced; nothing is reported as wanted",
            )
            return []
        cap = int(getattr(engine, "MAX_TRACKED_STATES", len(candidates)) or len(candidates))
        candidates.sort(key=lambda t: (-(max(0.0, t.liking) + math.sqrt(t.unsure)), t.key))
        found: list[Want] = []
        for thing in candidates[:cap]:
            gain = thing.unsure / (thing.unsure + noise)
            after = max(noise / _WINDOW, thing.unsure * (1.0 - gain))
            state = engine.appraise(
                Incentive(
                    key=thing.key,
                    description=thing.label,
                    cached_value=max(0.0, thing.liking) if thing.contacts else None,
                ),
                epistemic_affordance=1.0,
                prior_variance=[thing.unsure],
                posterior_variance=[after],
            )
            utility = float(engine.utility(state))
            known = self.known(thing.key)
            if utility <= 0.0 or known is False:
                continue
            if engine.dynamics.frustration(thing.key).should_disengage():
                continue
            origin = str(getattr(state.dominant_origin, "value", state.dominant_origin) or "")
            unfinished = next(iter(thing.open.values()), "")
            other_pull = state.magnitude_of(state.dominant_origin) if state.dominant_origin else 0.0
            if unfinished:
                kind = "finish"
            elif max(0.0, thing.liking) >= other_pull:
                kind = "more"
            else:
                kind = "find out"
            found.append(
                Want(
                    thing=thing.key,
                    label=thing.label,
                    way=self._way_for(thing, offered),
                    kind=kind,
                    wanting=float(state.wanting),
                    utility=utility,
                    priority=float(engine.attention_priority(state)),
                    origin=origin,
                    why=(
                        f"{thing.contacts} contacts, {thing.good} of them worth more than she expected; "
                        f"her share of their worth {thing.liking:+.3f}, variance {thing.unsure:.4f}; "
                        f"{state.why()}"
                    ),
                    liking=thing.liking,
                    contacts=thing.contacts,
                    known=known,
                    unfinished=unfinished,
                )
            )
        found.sort(key=lambda want: (-want.utility, want.thing))
        with self._lock:
            self._standing = list(found)
        return found if limit is None else found[: max(0, int(limit))]

    def standing(self) -> list[Want]:
        """What she wanted the last time it was asked, without asking again.

        Asking appraises every thing, and an appraisal is itself a look at
        it. A readout takes the last answer.
        """
        with self._lock:
            return list(self._standing)

    def status(self) -> dict[str, Any]:
        with self._lock:
            reachable = sum(1 for thing in self._things.values() if thing.reachable)
            return {
                "things": len(self._things),
                "reachable": reachable,
                "turns": self._turns,
                "noise": round(self._noise, 6),
                "likes": [thing.key for thing in self.likes()[:8]],
                "dislikes": [thing.key for thing in self.dislikes()[:8]],
                "wants": [want.as_dict() for want in self._standing[:8]],
            }


def read_turn(
    present: Iterable[str],
    reading: Any,
    record: WhatSheWants | None = None,
    *,
    said: Iterable[str] = (),
) -> dict[str, Any]:
    """Fold the turn that just ended into what she has met.

    `present` is what the turn was about, as core/affect/feelings_about.py
    names it, and `reading` is what the turn was worth. `said` is what was
    said in it, by anybody. The subjects are the terms the exchange dwells
    on, and the one topic label the turn arrived with is not used.
    """
    record = record or get_what_she_wants()
    keys = [str(key) for key in present if str(key).strip()]
    words = [str(text) for text in said if str(text or "").strip()]
    # The topic label a turn arrives with is one word for a whole conversation
    # and stays on her state after it: live, 13 October, it read "times"
    # through a morning about tide pools and on the ticks after it. Subjects
    # come from what was said, and a turn in which nothing was said has none.
    keys = [key for key in keys if not key.startswith("topic:")]
    if words:
        keys.extend(f"topic:{term}" for term in record.about(*words))
    reachable = [key for key in keys if ":" in key and key.split(":", 1)[0] in REACHABLE_KINDS]
    labels = {key: key.split(":", 1)[1].strip() for key in reachable}
    return record.met(
        keys,
        liking_of(reading),
        reachable=reachable,
        labels=labels,
        payoff=getattr(reading, "payoff", None),
    )


_RECORD: WhatSheWants | None = None
_RECORD_LOCK = checked_lock("conation.what_she_wants.singleton")


def get_what_she_wants() -> WhatSheWants:
    global _RECORD
    if _RECORD is None:
        with _RECORD_LOCK:
            if _RECORD is None:
                _RECORD = WhatSheWants()
    return _RECORD


def reset_for_test(path: str | Path | None = None, *, engine: Any = None) -> WhatSheWants:
    global _RECORD
    with _RECORD_LOCK:
        _RECORD = WhatSheWants(path, engine=engine)
    return _RECORD
