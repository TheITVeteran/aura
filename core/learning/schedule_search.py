"""Search the execution schedule, not just the loop count (CP232).

Anima Rationis line 103: the ordinary layer order (0,1,...,63) is a
convention of how the model was trained, not a mathematical requirement of
how it must be run. Each block becomes an instruction; a schedule becomes a
neural program; the search is for the program that suits the problem.

    h_{t+1} = B_{pi_t}(h_t)

This is the highest-ceiling and most speculative of the seven components,
and the document is explicit about how to keep it honest (line 138): do not
let a language model guess schedules -- search them with evolutionary
optimization or beam search against HELD-OUT VERIFIED tasks. A schedule
selected on the tasks it is scored on is a memorized answer key.

The measured obstacle this attacks: cos(pass 1, pass 2) = 0.9994. Running
the SAME window twice barely rotates the state, because the residual stream
is dominated by accumulated magnitude and each pass adds another increment
along the same ray. A different schedule composes DIFFERENT circuits, which
is a genuinely different function rather than the same one applied twice.

What this module refuses to do: report a schedule's training score as its
value. Every schedule carries the held-out score it earned, and the
selection reports the gap between the two, because a large gap is
overfitting to the search set and looks identical to a discovery.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

SCHEDULE_SEARCH_SCHEMA = "aura.schedule_search.v1"


@dataclass(frozen=True)
class LayerSchedule:
    """An execution program over a model's blocks.

    ``segments`` are (start, stop) half-open layer ranges, applied in
    order. The vanilla schedule is a single segment covering everything.
    """

    segments: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a schedule needs at least one segment")
        for start, stop in self.segments:
            if type(start) is not int or type(stop) is not int:
                raise ValueError("segment bounds must be integers")
            if start < 0 or stop <= start:
                raise ValueError(f"invalid segment ({start}, {stop})")

    @classmethod
    def vanilla(cls, total_layers: int) -> LayerSchedule:
        return cls(segments=((0, total_layers),))

    def layer_applications(self) -> int:
        """Total compute, so schedules can be compared at equal cost."""
        return sum(stop - start for start, stop in self.segments)

    def touched_layers(self) -> set[int]:
        touched: set[int] = set()
        for start, stop in self.segments:
            touched.update(range(start, stop))
        return touched

    def is_valid_for(self, total_layers: int) -> bool:
        return all(stop <= total_layers for _, stop in self.segments)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULE_SEARCH_SCHEMA,
            "segments": [list(s) for s in self.segments],
            "layer_applications": self.layer_applications(),
            "distinct_layers": len(self.touched_layers()),
        }


def run_schedule(model: Any, hidden: Any, schedule: LayerSchedule) -> Any:
    """Execute a schedule over a model's blocks.

    No caches: a schedule may revisit a layer, and a KV cache would append
    that layer's keys twice and corrupt the history -- the same hazard
    gradient checkpointing hit in intrinsic_recurrence.
    """
    from mlx_lm.models.base import create_attention_mask

    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        raise ValueError("model has no transformer layers")
    if not schedule.is_valid_for(len(layers)):
        raise ValueError("schedule references layers the model does not have")
    for start, stop in schedule.segments:
        for index in range(start, stop):
            hidden = layers[index](hidden, create_attention_mask(hidden, None), None)
    return hidden


# ── Mutation: the moves an evolutionary search may make ─────────────────


def mutate(
    schedule: LayerSchedule,
    total_layers: int,
    rng: random.Random,
    *,
    max_segments: int = 8,
) -> LayerSchedule:
    """One structural edit: repeat, drop, extend, shrink or reorder."""
    segments = list(schedule.segments)
    moves = ["repeat", "extend", "shrink", "shift"]
    if len(segments) > 1:
        moves += ["drop", "swap"]
    if len(segments) >= max_segments:
        moves = [m for m in moves if m != "repeat"]
    move = rng.choice(moves)
    index = rng.randrange(len(segments))
    start, stop = segments[index]

    if move == "repeat":
        segments.insert(index + 1, (start, stop))
    elif move == "drop":
        segments.pop(index)
    elif move == "extend":
        segments[index] = (start, min(total_layers, stop + rng.randint(1, 4)))
    elif move == "shrink":
        segments[index] = (start, max(start + 1, stop - rng.randint(1, 4)))
    elif move == "shift":
        delta = rng.randint(-4, 4)
        new_start = max(0, min(total_layers - 1, start + delta))
        new_stop = max(new_start + 1, min(total_layers, stop + delta))
        segments[index] = (new_start, new_stop)
    elif move == "swap" and len(segments) > 1:
        other = rng.randrange(len(segments))
        segments[index], segments[other] = segments[other], segments[index]
    return LayerSchedule(segments=tuple(segments))


@dataclass
class ScheduleCandidate:
    """A schedule with BOTH its scores, so overfitting stays visible."""

    schedule: LayerSchedule
    search_score: float
    holdout_score: float | None = None

    def generalization_gap(self) -> float | None:
        if self.holdout_score is None:
            return None
        return self.search_score - self.holdout_score


def evolve_schedules(
    *,
    total_layers: int,
    search_scorer: Callable[[LayerSchedule], float],
    holdout_scorer: Callable[[LayerSchedule], float],
    generations: int = 8,
    population: int = 12,
    seed: int = 0,
    compute_budget: int | None = None,
    initial: LayerSchedule | None = None,
) -> dict[str, Any]:
    """Evolutionary search, scored on held-out tasks before selection.

    The document is explicit (line 138) that schedules must be searched
    against held-out verified tasks rather than guessed. Two separate
    scorers are REQUIRED here so that cannot be quietly skipped: selecting
    on the same tasks a schedule is scored on produces a memorized answer
    key that is indistinguishable from a discovery in the final number.

    ``compute_budget`` caps layer applications so a "better" schedule
    cannot win simply by running more compute than the baseline.
    """
    if type(total_layers) is not int or total_layers < 2:
        raise ValueError("total_layers must be at least 2")
    if type(generations) is not int or not 1 <= generations <= 500:
        raise ValueError("generations must be inside [1, 500]")
    if type(population) is not int or not 4 <= population <= 200:
        raise ValueError("population must be inside [4, 200]")
    if search_scorer is holdout_scorer:
        raise ValueError(
            "search and held-out scorers must be different task sets: "
            "selecting on the tasks being scored produces an answer key"
        )

    rng = random.Random(seed)
    baseline = initial or LayerSchedule.vanilla(total_layers)
    budget = compute_budget or baseline.layer_applications() * 4

    def admissible(candidate: LayerSchedule) -> bool:
        return (
            candidate.is_valid_for(total_layers)
            and candidate.layer_applications() <= budget
        )

    pool = [baseline]
    while len(pool) < population:
        candidate = mutate(rng.choice(pool), total_layers, rng)
        if admissible(candidate):
            pool.append(candidate)

    history: list[dict[str, Any]] = []
    evaluated: dict[tuple, float] = {}
    for generation in range(generations):
        scored: list[ScheduleCandidate] = []
        for candidate in pool:
            key = candidate.segments
            if key not in evaluated:
                evaluated[key] = float(search_scorer(candidate))
            scored.append(ScheduleCandidate(candidate, evaluated[key]))
        scored.sort(key=lambda c: c.search_score, reverse=True)
        survivors = scored[: max(2, population // 3)]
        history.append(
            {
                "generation": generation,
                "best_search_score": round(survivors[0].search_score, 6),
                "population": len(pool),
            }
        )
        pool = [c.schedule for c in survivors]
        while len(pool) < population:
            candidate = mutate(rng.choice(pool), total_layers, rng)
            if admissible(candidate):
                pool.append(candidate)

    # Held-out scoring happens ONCE, after the search is finished, on the
    # finalists only. Scoring held-out during selection would make it part
    # of the search set.
    finalists = []
    for schedule in pool[: max(2, population // 3)]:
        finalists.append(
            ScheduleCandidate(
                schedule,
                evaluated[schedule.segments],
                float(holdout_scorer(schedule)),
            )
        )
    finalists.sort(key=lambda c: c.holdout_score or 0.0, reverse=True)
    best = finalists[0]
    baseline_holdout = float(holdout_scorer(baseline))

    return {
        "schema": SCHEDULE_SEARCH_SCHEMA,
        "best_schedule": best.schedule.to_receipt(),
        "best_search_score": round(best.search_score, 6),
        "best_holdout_score": round(best.holdout_score or 0.0, 6),
        "generalization_gap": round(best.generalization_gap() or 0.0, 6),
        "baseline_holdout_score": round(baseline_holdout, 6),
        "improvement": round((best.holdout_score or 0.0) - baseline_holdout, 6),
        # A schedule that beats the baseline only on the search set found
        # the search set, not a better program.
        "beats_baseline": bool((best.holdout_score or 0.0) > baseline_holdout),
        "overfit_warning": bool((best.generalization_gap() or 0.0) > 0.1),
        "compute_budget": budget,
        "baseline_applications": baseline.layer_applications(),
        "best_applications": best.schedule.layer_applications(),
        "generations": history,
        "unique_schedules_evaluated": len(evaluated),
    }


__all__ = [
    "SCHEDULE_SEARCH_SCHEMA",
    "LayerSchedule",
    "ScheduleCandidate",
    "evolve_schedules",
    "mutate",
    "run_schedule",
]
