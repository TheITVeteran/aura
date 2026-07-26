"""Additional decision surfaces the organ watches.

The point of having built general machinery is that a new control point is a
registration and a resolver, not a new learner. Each one here declares what the
situation looks like, what the options are, which of them are safe to explore
with, and — the part that actually costs thought — how anyone finds out whether
the decision was any good.

That last part is where control points are won or lost, and it is worth being
blunt about the difference between the two here.

**memory.retrieval_breadth** is *self-grading*. Whether a retrieval came back
with anything usable is visible at the call site, immediately, with no
downstream cooperation. It will accumulate real evidence quickly.

**cognition.effort** is not. Whether a given depth of thinking was the right
amount is only answerable once the episode's answer has been graded by a
verifier, which happens elsewhere and sometimes not at all. Its resolver
therefore returns UNOBSERVED unless a grade was actually reported. That means
the control point may sit at OBSERVE for a long time, and that is the correct
outcome — a head that cannot be graded must never be promoted, and the
machinery is built so that it simply won't be, rather than being promoted on
whatever weak proxy was to hand.

Registering a control point that cannot yet be graded is still worth doing. It
starts the corpus, it makes the gap visible in the health report instead of
invisible in nobody's head, and when a grading path does appear the history is
already there.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from core.ontogeny.experience import Episode, Outcome, OutcomeKind
from core.ontogeny.features import FeatureSchema, register_schema
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Ontogeny.ControlPoints")

MEMORY_RETRIEVAL = "memory.retrieval_breadth"
COGNITION_EFFORT = "cognition.effort"


# ── memory.retrieval_breadth ────────────────────────────────────────────────

MEMORY_RETRIEVAL_SCHEMA = FeatureSchema(
    control_point=MEMORY_RETRIEVAL,
    version=1,
    names=(
        "limit",
        "risk_sensitive",
        "need_failures",
        "need_tools",
        "horizon_now",
        "horizon_session",
        "horizon_long",
        "kind_debug",
        "kind_converse",
        "kind_decide",
        "kind_recall_fact",
        "query_words",
        "stores_available",
        "novelty",
    ),
    sources={
        "limit": "core/memory/intentional_retrieval.py:RetrievalIntent",
        "risk_sensitive": "core/memory/intentional_retrieval.py:RetrievalIntent",
        "stores_available": "core/memory/intentional_retrieval.py:IntentionalRetriever",
        "novelty": "core/ontogeny/state.py",
    },
)

#: How far the plan's inclusion threshold moves for each breadth choice. A
#: lower threshold consults more stores; the cost is latency and dilution.
BREADTH_THRESHOLD_DELTA: dict[str, float] = {
    "narrow": 0.12,
    "balanced": 0.0,
    "broad": -0.10,
}


def retrieval_features(
    *,
    limit: int,
    kind: str,
    risk_sensitive: bool,
    need_failures: bool,
    need_tools: bool,
    time_horizon: str,
    query: str,
    stores_available: int,
    novelty: float,
) -> dict[str, float]:
    return {
        "limit": float(limit),
        "risk_sensitive": 1.0 if risk_sensitive else 0.0,
        "need_failures": 1.0 if need_failures else 0.0,
        "need_tools": 1.0 if need_tools else 0.0,
        "horizon_now": 1.0 if time_horizon == "now" else 0.0,
        "horizon_session": 1.0 if time_horizon == "session" else 0.0,
        "horizon_long": 1.0 if time_horizon == "long" else 0.0,
        "kind_debug": 1.0 if kind == "debug" else 0.0,
        "kind_converse": 1.0 if kind == "converse" else 0.0,
        "kind_decide": 1.0 if kind in ("decide", "act_irreversible") else 0.0,
        "kind_recall_fact": 1.0 if kind == "recall_fact" else 0.0,
        "query_words": float(min(40, len(query.split()))),
        "stores_available": float(stores_available),
        "novelty": float(novelty),
    }


class RetrievalResolver:
    """Grades a retrieval by what came back, at the call site, immediately.

    Deliberately modest about what it claims. Returning nothing is a real
    failure and is graded as one. Returning something *relevant enough to have
    been worth the fetch* is graded a success. Whether those memories then
    changed the answer is a better question that this resolver cannot see, so
    it does not pretend to — the honest signal available here is whether the
    retrieval did its own job.
    """

    control_point = MEMORY_RETRIEVAL
    horizon_s = 120.0

    #: A hit below this weighted score is noise the merge step would rather
    #: not have fetched.
    RELEVANCE_FLOOR = 0.15

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._outcomes: dict[str, tuple[int, float, float]] = {}

    def note_result(self, episode_id: str, *, hits: int, best_score: float) -> None:
        if not episode_id:
            return
        with self._lock:
            self._outcomes[episode_id] = (int(hits), float(best_score), time.time())
            if len(self._outcomes) > 8000:
                cutoff = time.time() - 4 * self.horizon_s
                for key in [k for k, (_, _, at) in self._outcomes.items() if at < cutoff]:
                    self._outcomes.pop(key, None)

    def resolve(self, episode: Episode) -> Outcome | None:
        with self._lock:
            recorded = self._outcomes.pop(episode.episode_id, None)
        if recorded is None:
            return None
        hits, best_score, _ = recorded
        if hits == 0:
            return Outcome(
                kind=OutcomeKind.FAILURE, utility=0.0, resolver="memory.no_hits",
                detail={"breadth": episode.decision},
            )
        useful = best_score >= self.RELEVANCE_FLOOR
        return Outcome(
            kind=OutcomeKind.SUCCESS if useful else OutcomeKind.FAILURE,
            utility=min(1.0, best_score) if useful else 0.0,
            resolver="memory.hit_quality",
            detail={"breadth": episode.decision, "hits": hits, "best_score": round(best_score, 4)},
        )

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {"pending": len(self._outcomes), "horizon_s": self.horizon_s}


# ── cognition.effort ────────────────────────────────────────────────────────

COGNITION_EFFORT_SCHEMA = FeatureSchema(
    control_point=COGNITION_EFFORT,
    version=1,
    names=(
        "stakes",
        "uncertainty",
        "novelty",
        "body_pressure",
        "foreground",
        "resident_scale",
        "hour_of_day_sin",
        "hour_of_day_cos",
    ),
    sources={
        "stakes": "core/brain/latent_cortex_service.py:allocate",
        "uncertainty": "core/brain/latent_cortex_service.py:allocate",
        "novelty": "core/ontogeny/state.py",
        "body_pressure": "core/brain/latent_cortex_service.py:_body_pressure",
    },
)

#: Multipliers applied to the allocator's depth. Bounded on both sides: the
#: organ may tune how hard she thinks, never remove the floor or the ceiling.
EFFORT_MULTIPLIER: dict[str, float] = {
    "lean": 0.75,
    "standard": 1.0,
    "deep": 1.3,
}


class EffortResolver:
    """Grades a thinking-depth choice only when the episode was actually graded.

    There is no proxy here on purpose. Latency is not quality; convergence is
    not correctness; "the answer looked confident" is the thing this whole
    layer exists to stop trusting. If no verifier graded the episode, the
    decision is UNOBSERVED and teaches nothing, and the control point stays at
    OBSERVE until a grading path exists.
    """

    control_point = COGNITION_EFFORT
    horizon_s = 600.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._grades: dict[str, tuple[float, float]] = {}

    def note_grade(self, episode_id: str, *, verified_score: float) -> None:
        """Called by whatever independently graded the episode's answer."""
        if not episode_id:
            return
        with self._lock:
            self._grades[episode_id] = (float(verified_score), time.time())
            if len(self._grades) > 4000:
                cutoff = time.time() - 4 * self.horizon_s
                for key in [k for k, (_, at) in self._grades.items() if at < cutoff]:
                    self._grades.pop(key, None)

    def resolve(self, episode: Episode) -> Outcome | None:
        with self._lock:
            recorded = self._grades.pop(episode.episode_id, None)
        if recorded is None:
            return None
        score, _ = recorded
        return Outcome.from_utility(
            max(0.0, min(1.0, score)), "cognition.verified_answer",
            effort=episode.decision,
        )

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {"pending_grades": len(self._grades), "horizon_s": self.horizon_s}


# ── registration ────────────────────────────────────────────────────────────

_retrieval_resolver: RetrievalResolver | None = None
_effort_resolver: EffortResolver | None = None
_registered = False


def get_retrieval_resolver() -> RetrievalResolver:
    global _retrieval_resolver
    if _retrieval_resolver is None:
        _retrieval_resolver = RetrievalResolver()
    return _retrieval_resolver


def get_effort_resolver() -> EffortResolver:
    global _effort_resolver
    if _effort_resolver is None:
        _effort_resolver = EffortResolver()
    return _effort_resolver


def register(core: Any) -> list[str]:
    """Register both control points on a running organ. Idempotent."""
    global _registered
    if _registered:
        return []
    from core.ontogeny.service import ControlPoint

    register_schema(MEMORY_RETRIEVAL_SCHEMA)
    register_schema(COGNITION_EFFORT_SCHEMA)

    core.register(ControlPoint(
        name=MEMORY_RETRIEVAL,
        schema=MEMORY_RETRIEVAL_SCHEMA,
        actions=tuple(BREADTH_THRESHOLD_DELTA),
        # All three are safe to explore with: the worst case is a retrieval
        # that fetches too much or too little, which costs latency and is
        # visible in the same breath.
        explorable=tuple(BREADTH_THRESHOLD_DELTA),
        horizon_s=RetrievalResolver.horizon_s,
    ))
    core.register(ControlPoint(
        name=COGNITION_EFFORT,
        schema=COGNITION_EFFORT_SCHEMA,
        actions=tuple(EFFORT_MULTIPLIER),
        # Never explores by thinking *less*. Randomly under-thinking a real
        # answer spends the user's question to buy evidence, which is not a
        # trade the organ gets to make.
        explorable=("standard", "deep"),
        horizon_s=EffortResolver.horizon_s,
    ))
    core.resolvers.register(get_retrieval_resolver())
    core.resolvers.register(get_effort_resolver())
    _registered = True
    logger.info("ontogeny: registered %s and %s", MEMORY_RETRIEVAL, COGNITION_EFFORT)
    return [MEMORY_RETRIEVAL, COGNITION_EFFORT]


def reset_for_test() -> None:
    global _registered, _retrieval_resolver, _effort_resolver
    _registered = False
    _retrieval_resolver = None
    _effort_resolver = None


def novelty_now() -> float:
    """Current novelty, for control points that want it as a feature."""
    try:
        from core.ontogeny.service import get_ontogeny

        return float(get_ontogeny().novelty())
    except (ImportError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("ontogeny_control_points", exc, severity="debug",
                           action="novelty unavailable as a feature")
        return 0.5


__all__ = [
    "BREADTH_THRESHOLD_DELTA",
    "COGNITION_EFFORT",
    "COGNITION_EFFORT_SCHEMA",
    "EFFORT_MULTIPLIER",
    "MEMORY_RETRIEVAL",
    "MEMORY_RETRIEVAL_SCHEMA",
    "EffortResolver",
    "RetrievalResolver",
    "get_effort_resolver",
    "get_retrieval_resolver",
    "novelty_now",
    "register",
    "reset_for_test",
    "retrieval_features",
]
