"""What she believes about a person, in a shape that cannot lose the qualifiers.

The obvious way to let an agent accumulate a view of someone is a free-text
block she rewrites over time — "notes on Bryan" — periodically consolidated by
a model to keep it inside its budget. That design has a failure built into it,
and the failure is not carelessness.

In prose, the load-bearing parts of knowing a person are *adjectives*:
"seemed", "once", "when a build was failing", "though usually not". Compressing
prose drops adjectives before it drops nouns — that is what compression is. So
consolidation walks a note from

    "Bryan seemed frustrated once, during a failing deploy"

to "Bryan gets frustrated" to "Bryan is easily frustrated", and no individual
step is unreasonable and nobody ever decided it. What is lost at each step —
frequency, conditions, hedging, the times it did not happen — is precisely what
distinguishes understanding someone from having a caricature of them.

So person-knowledge is not stored as prose here. An observation is a record
with **fields**: what was noticed, under what conditions, how many times, when
last, and every occasion it did *not* hold. Dropping a qualifier stops being a
stylistic choice and becomes a schema violation.

Two consequences fall out:

* **Consolidation is aggregation, not summarisation.** Merging two sightings
  increments a count and appends an episode id. There is no operation in this
  module that hands a note to a language model to be rewritten, and
  ``test_interpersonal_model.py`` asserts that no such path exists.
* **The render states evidence, not a verdict.** It says "noticed 3 times, most
  recently 2 days ago, 1 occasion it did not hold" rather than a confidence
  score. A manufactured number would be a summary of the evidence — the same
  lossy move one level up, and one nobody could audit.

Counter-examples are first-class because prose summarisation never keeps them.
A view of someone that can only accumulate confirmations is not a model, it is
a grudge.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger("Aura.InterpersonalModel")

__all__ = [
    "Occurrence",
    "Observation",
    "PersonModel",
    "DAY_SECONDS",
]

DAY_SECONDS = 86400.0


def _normalize(claim: str) -> str:
    """Claims match on normalised text. Deliberately exact-ish: fuzzy merging
    is how two distinct observations become one blurred one."""
    return " ".join(claim.lower().split())


@dataclass(frozen=True)
class Occurrence:
    """One time something was actually observed.

    ``episode_id`` is what makes a claim auditable — it points at the memory
    that justifies it. A claim with no occurrences cannot exist here, which is
    the structural version of "no assertion without evidence".
    """

    episode_id: str
    at: float = field(default_factory=time.time)
    note: str = ""


@dataclass
class Observation:
    """One thing she has noticed about someone, with its evidence attached."""

    claim: str
    conditions: str = ""
    occurrences: list[Occurrence] = field(default_factory=list)
    counter_examples: list[Occurrence] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize(self.claim), _normalize(self.conditions))

    @property
    def support(self) -> int:
        return len(self.occurrences)

    @property
    def contradictions(self) -> int:
        return len(self.counter_examples)

    def last_seen(self) -> float | None:
        stamps = [o.at for o in self.occurrences]
        return max(stamps) if stamps else None

    def first_seen(self) -> float | None:
        stamps = [o.at for o in self.occurrences]
        return min(stamps) if stamps else None

    def episodes(self) -> list[str]:
        """Every memory that justifies this, so it can be checked."""
        return [o.episode_id for o in self.occurrences]

    def render(self, *, now: float | None = None) -> str:
        """The observation as context text, stating evidence rather than verdict.

        Frequency and recency are rendered as words because the model reads
        words — but they are *derived from the counts*, never authored, so they
        cannot drift away from what actually happened.
        """
        now = time.time() if now is None else now
        parts = [self.claim.strip()]
        if self.conditions:
            parts.append(f"({self.conditions.strip()})")

        evidence = [f"noticed {_times(self.support)}"]
        last = self.last_seen()
        if last is not None:
            evidence.append(f"most recently {_ago(now - last)}")
        if self.contradictions:
            evidence.append(
                f"{_times(self.contradictions)} it did not hold"
            )
        parts.append("— " + ", ".join(evidence))
        return " ".join(parts)


def _times(count: int) -> str:
    if count == 1:
        return "once"
    if count == 2:
        return "twice"
    return f"{count} times"


def _ago(seconds: float) -> str:
    if seconds < DAY_SECONDS:
        return "today"
    days = int(seconds // DAY_SECONDS)
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days} days ago"
    months = days // 30
    return "a month ago" if months == 1 else f"{months} months ago"


class PersonModel:
    """Her structured view of one person.

    Bounded by ``max_observations`` so it cannot grow without limit, but the
    eviction rule is deliberately *weakest evidence first* rather than oldest:
    a thing noticed once eighteen months ago and never since is a worse thing
    to keep than a standing pattern that happens to be old.
    """

    def __init__(self, person: str, *, max_observations: int = 64) -> None:
        if not person or not person.strip():
            raise ValueError("a person model needs someone to be about")
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        self.person = person.strip()
        self.max_observations = max_observations
        self._observations: dict[tuple[str, str], Observation] = {}

    def __len__(self) -> int:
        return len(self._observations)

    def __iter__(self):
        return iter(self._observations.values())

    # -- recording ---------------------------------------------------------

    def observe(
        self,
        claim: str,
        *,
        episode_id: str,
        conditions: str = "",
        note: str = "",
        at: float | None = None,
    ) -> Observation:
        """Record a sighting. Repeats increment; they never rewrite.

        This is the whole mechanism. Seeing the same thing a second time makes
        the count 2 — it does not make the claim stronger in wording, because
        the wording is not where strength lives.
        """
        if not claim or not claim.strip():
            raise ValueError("an observation needs a claim")
        if not episode_id:
            raise ValueError(
                "an observation needs the episode that justifies it; a claim "
                "with no evidence is exactly what this module exists to prevent"
            )
        occurrence = Occurrence(
            episode_id=episode_id,
            at=time.time() if at is None else at,
            note=note,
        )
        observation = Observation(claim=claim.strip(), conditions=conditions.strip())
        existing = self._observations.get(observation.key)
        if existing is None:
            self._observations[observation.key] = observation
            existing = observation
        existing.occurrences.append(occurrence)
        self._evict_if_needed()
        return existing

    def contradict(
        self,
        claim: str,
        *,
        episode_id: str,
        conditions: str = "",
        note: str = "",
        at: float | None = None,
    ) -> Observation | None:
        """Record an occasion the claim did *not* hold.

        A view that can only accumulate confirmations is not a model. Prose
        summarisation never keeps counter-evidence; here it is a field, so it
        survives every consolidation by construction.
        """
        key = (_normalize(claim), _normalize(conditions))
        observation = self._observations.get(key)
        if observation is None:
            return None
        observation.counter_examples.append(
            Occurrence(
                episode_id=episode_id,
                at=time.time() if at is None else at,
                note=note,
            )
        )
        return observation

    def forget(self, claim: str, *, conditions: str = "") -> bool:
        """Drop an observation outright — a correction, not a decay."""
        return self._observations.pop((_normalize(claim), _normalize(conditions)), None) is not None

    # -- consolidation is aggregation --------------------------------------

    def merge(self, other: "PersonModel") -> None:
        """Fold another model's observations in by aggregating the evidence.

        Note what this does not do: it never reconciles two claims into a
        third, better-worded one. Merging is set union on occurrences, so the
        counts stay true and nothing acquires confidence it did not earn.
        """
        if _normalize(other.person) != _normalize(self.person):
            raise ValueError(
                f"refusing to merge notes about {other.person!r} into "
                f"{self.person!r} — conflating two people is not a compression"
            )
        for observation in other:
            mine = self._observations.get(observation.key)
            if mine is None:
                self._observations[observation.key] = observation
                continue
            seen = {(o.episode_id, o.at) for o in mine.occurrences}
            mine.occurrences.extend(
                o for o in observation.occurrences if (o.episode_id, o.at) not in seen
            )
            seen_counter = {(o.episode_id, o.at) for o in mine.counter_examples}
            mine.counter_examples.extend(
                o for o in observation.counter_examples
                if (o.episode_id, o.at) not in seen_counter
            )
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if len(self._observations) <= self.max_observations:
            return
        # Weakest evidence first: fewest net sightings, then least recent.
        ranked = sorted(
            self._observations.values(),
            key=lambda o: (o.support - o.contradictions, o.last_seen() or 0.0),
        )
        for observation in ranked[: len(self._observations) - self.max_observations]:
            logger.debug(
                "evicting weakly-evidenced observation about %s: %s",
                self.person, observation.claim,
            )
            del self._observations[observation.key]

    # -- reading -----------------------------------------------------------

    def strongest(self, limit: int = 10, *, now: float | None = None) -> list[Observation]:
        """Best-evidenced observations first. Net of counter-examples."""
        return sorted(
            self._observations.values(),
            key=lambda o: (o.support - o.contradictions, o.last_seen() or 0.0),
            reverse=True,
        )[:limit]

    def render(self, *, limit: int = 10, now: float | None = None) -> str:
        """The block text. Every line carries its own evidence."""
        observations = self.strongest(limit, now=now)
        if not observations:
            return f"No observations about {self.person} yet."
        lines = [f"What I have noticed about {self.person}:"]
        lines.extend(f"- {o.render(now=now)}" for o in observations)
        return "\n".join(lines)

    def audit(self) -> list[dict[str, object]]:
        """Every claim with the episodes behind it, for a human to check."""
        return [
            {
                "claim": o.claim,
                "conditions": o.conditions,
                "support": o.support,
                "contradictions": o.contradictions,
                "episodes": o.episodes(),
                "counter_episodes": [c.episode_id for c in o.counter_examples],
            }
            for o in self.strongest(limit=len(self._observations))
        ]
