"""What an exchange is about, read from its words as counts and never as meaning.

A term is one of an exchange's subjects when two measurements agree. The
exchange uses it more than her exchanges have before, by Dunning's
log-likelihood ratio (1993), the standard measure of which words are key to a
text against a reference, corrected for the number of terms the exchange
offers. And it is dwelt on where it comes up: it is found in fewer than half
the exchanges its count would reach if it were spread by chance, which is a
residual inverse document frequency above one bit (Church and Gale 1995). The
second is what tells a subject from a word that is merely frequent in one kind
of talk. Measured on her 351 stored exchanges on 13 October 2026, "limpet" and
"scar" read 2.4 and 2.6 bits and "its", "home" and "watch" 0.6, 0.5 and 0.3.

Her reference is her own exchanges, the stored ones included. It is kept with
her record of what she has met (core/conation/what_she_wants.py), which holds
one of these and saves it.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from statistics import NormalDist
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["MAX_TERMS", "WhatHerTalkIsAbout", "terms_of"]

#: The chance of calling a term a subject that is not one; the conventional level.
_WRONG = 0.05

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


class WhatHerTalkIsAbout:
    """How often each term has been used across her exchanges, and what that makes of a new one."""

    def __init__(self, stored: Mapping[str, Any] | None = None, *, changed: Callable[[], None] | None = None) -> None:
        self._lock = checked_lock("conation.what_her_talk_is_about", reentrant=True)
        held = dict(stored or {})
        #: How often each term has been used, and how many terms her exchanges held.
        self._said: Counter[str] = Counter({str(k): int(v) for k, v in dict(held.get("said", {})).items()})
        self._said_total = max(0, int(held.get("said_total", 0)))
        #: How many of those exchanges each term was in, and how many there have been.
        self._said_in: Counter[str] = Counter({str(k): int(v) for k, v in dict(held.get("said_in", {})).items()})
        self._exchanges = max(0, int(held.get("exchanges", 0)))
        self._shares: dict[str, float] = {}
        self._changed = changed

    def held(self) -> dict[str, Any]:
        """What is kept of her talk, for whoever saves it."""
        with self._lock:
            return {
                "said": dict(self._said),
                "said_total": self._said_total,
                "said_in": dict(self._said_in),
                "exchanges": self._exchanges,
            }

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
            found = _subjects(self, here)
            _count_exchange(self, here)
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
            return _subjects(self, here)


    def shares(self) -> dict[str, float]:
        """Each subject's share of what the last exchange read dwelt on."""
        with self._lock:
            return dict(self._shares)


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
                _count_exchange(self, terms_of(*exchange))
                added += 1
        if self._changed is not None:
            self._changed()
        return added


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
    kept = [
        (ratio, term)
        for ratio, term in sorted(scored, key=lambda row: (-row[0], row[1]))
        if (term in pairs) or (" " not in term and term not in inside)
    ]
    whole = sum(ratio for ratio, _term in kept)
    self._shares = {term: ratio / whole for ratio, term in kept} if whole > 0.0 else {}
    return [term for _ratio, term in kept]


def _count_exchange(self, here: Counter[str]) -> None:
    """Add one exchange to what later ones are set against. Lock held."""
    self._said.update(here)
    self._said_total += sum(here.values())
    self._said_in.update(here.keys())
    self._exchanges += 1
    if len(self._said) > MAX_TERMS:
        self._said = Counter(dict(self._said.most_common(MAX_TERMS // 2)))
        self._said_in = Counter({term: self._said_in[term] for term in self._said})
