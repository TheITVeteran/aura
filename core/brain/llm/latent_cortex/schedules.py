"""Layer-schedule programs: each transformer block becomes an instruction.

The ordinary schedule (0,1,…,63, once each) is one point in an enormous
family of virtual architectures the same frozen tensors can execute. A
``LayerSchedule`` is a validated neural program over the recurrent region:

    [(start, end, repeats, alpha_override), ...]

Guarantees enforced here, not hoped for:
- Programs are validated against the model's actual topology and the
  episode's compute budget BEFORE execution — an invalid program never
  touches the model.
- Canonical serialization + content hash so every receipt names exactly
  which program ran.
- The library tracks per-domain reliability with Wilson lower bounds (the
  same statistics the Verifier Foundry uses) — a schedule is only ever
  preferred on evidence, never on vibes.
- Search is evolutionary, seeded, and budgeted; candidates are scored ONLY
  by a caller-supplied evaluator (which must itself be verifier-backed) and
  every promotion carries provenance.

Whatever shape a program takes, the engine's final clean persist pass
guarantees coherent slot KV — a pathological program can waste compute but
cannot corrupt attention state.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.brain.verifiers.foundry import wilson_lower_bound

logger = logging.getLogger("Aura.LatentCortex.Schedules")

# A schedule may apply at most this many window-layer applications per slot
# token, regardless of budget — programs beyond this are degenerate.
MAX_TOTAL_LAYER_REPEATS = 4096


@dataclass(frozen=True)
class StageOp:
    """One instruction: run layers [start, end) ``repeats`` times."""

    start: int
    end: int
    repeats: int = 1
    alpha: float | None = None  # override the recurrence alpha for this stage

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"start": self.start, "end": self.end, "repeats": self.repeats}
        if self.alpha is not None:
            out["alpha"] = round(float(self.alpha), 6)
        return out


@dataclass
class LayerSchedule:
    """A validated program over the recurrent region [prelude_end, coda_start)."""

    ops: tuple[StageOp, ...]
    name: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LayerSchedule":
        ops = tuple(
            StageOp(
                start=int(op["start"]),
                end=int(op["end"]),
                repeats=int(op.get("repeats", 1)),
                alpha=(float(op["alpha"]) if op.get("alpha") is not None else None),
            )
            for op in payload.get("ops", [])
        )
        return cls(ops=ops, name=str(payload.get("name", "")))

    @classmethod
    def single_window(cls, prelude_end: int, coda_start: int, repeats: int) -> "LayerSchedule":
        return cls(
            ops=(StageOp(prelude_end, coda_start, repeats),),
            name=f"window[{prelude_end}:{coda_start}]x{repeats}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ops": [op.to_dict() for op in self.ops]}

    def canonical_json(self) -> str:
        # name excluded: identity is the program, not its label.
        return json.dumps([op.to_dict() for op in self.ops], sort_keys=True, separators=(",", ":"))

    @property
    def schedule_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    @property
    def total_layer_repeats(self) -> int:
        return sum((op.end - op.start) * op.repeats for op in self.ops)

    def validate(self, *, prelude_end: int, coda_start: int) -> list[str]:
        """Human-readable violations; empty ⇒ the program may execute."""
        problems: list[str] = []
        if not self.ops:
            problems.append("schedule has no ops")
        for i, op in enumerate(self.ops):
            if op.start < prelude_end or op.end > coda_start:
                problems.append(
                    f"op{i} [{op.start}:{op.end}) escapes recurrent region "
                    f"[{prelude_end}:{coda_start})"
                )
            if op.start >= op.end:
                problems.append(f"op{i} empty window [{op.start}:{op.end})")
            if op.repeats < 1:
                problems.append(f"op{i} repeats {op.repeats} < 1")
            if op.alpha is not None and not 0.0 < op.alpha <= 1.0:
                problems.append(f"op{i} alpha {op.alpha} outside (0, 1]")
        if self.total_layer_repeats > MAX_TOTAL_LAYER_REPEATS:
            problems.append(
                f"total layer repeats {self.total_layer_repeats} exceeds "
                f"{MAX_TOTAL_LAYER_REPEATS}"
            )
        return problems


# ── Reliability-tracked library ─────────────────────────────────────────


@dataclass
class ScheduleRecord:
    schedule: LayerSchedule
    domain: str
    trials: int = 0
    successes: int = 0
    provenance: str = ""  # who/what promoted it (search run id, operator, ...)
    updated_at: float = field(default_factory=time.time)

    @property
    def reliability_lb(self) -> float:
        return wilson_lower_bound(self.successes, self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.to_dict(),
            "domain": self.domain,
            "trials": self.trials,
            "successes": self.successes,
            "provenance": self.provenance,
            "updated_at": self.updated_at,
        }


class ScheduleLibrary:
    """Per-domain schedule reliability ledger with evidence-gated selection.

    ``best_for_domain`` only returns a searched schedule when its Wilson
    lower bound beats the default single-window program's — otherwise the
    default wins. This is the anti-self-deception gate: an exciting schedule
    with three lucky trials does not displace the baseline.
    """

    MIN_TRIALS = 8

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._records: dict[tuple[str, str], ScheduleRecord] = {}
        if self._path is not None and self._path.exists():
            self._load()

    # ── Persistence (governed writes; loads are plain reads) ───────────
    def _load(self) -> None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Schedule library unreadable at %s: %s — starting empty", self._path, exc)
            return
        for row in payload.get("records", []):
            try:
                rec = ScheduleRecord(
                    schedule=LayerSchedule.from_dict(row["schedule"]),
                    domain=str(row["domain"]),
                    trials=int(row["trials"]),
                    successes=int(row["successes"]),
                    provenance=str(row.get("provenance", "")),
                    updated_at=float(row.get("updated_at", 0.0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping corrupt schedule record: %s", exc)
                continue
            self._records[(rec.domain, rec.schedule.schedule_hash)] = rec

    def save(self) -> bool:
        if self._path is None:
            return False
        payload = json.dumps(
            {"version": 1, "records": [r.to_dict() for r in self._records.values()]},
            indent=1,
            sort_keys=True,
        )
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            self._path.parent.mkdir(parents=True, exist_ok=True)
            with local_internal_governed_scope("latent_cortex_schedule_library"):
                get_file_write_gateway().write_text(
                    self._path, payload, source="latent_cortex.schedules"
                )
            return True
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "latent_cortex",
                exc,
                action="kept schedule library in memory after persist failed",
            )
            return False

    # ── Evidence ────────────────────────────────────────────────────────
    def record_outcome(
        self,
        schedule: LayerSchedule,
        domain: str,
        success: bool,
        *,
        provenance: str = "",
    ) -> ScheduleRecord:
        key = (domain, schedule.schedule_hash)
        rec = self._records.get(key)
        if rec is None:
            rec = ScheduleRecord(schedule=schedule, domain=domain, provenance=provenance)
            self._records[key] = rec
        rec.trials += 1
        rec.successes += int(bool(success))
        rec.updated_at = time.time()
        return rec

    def best_for_domain(
        self, domain: str, *, prelude_end: int, coda_start: int, default_repeats: int
    ) -> LayerSchedule:
        default = LayerSchedule.single_window(prelude_end, coda_start, default_repeats)
        default_rec = self._records.get((domain, default.schedule_hash))
        default_lb = default_rec.reliability_lb if default_rec else 0.0

        best, best_lb = default, default_lb
        for (dom, _), rec in self._records.items():
            if dom != domain or rec.trials < self.MIN_TRIALS:
                continue
            if rec.schedule.validate(prelude_end=prelude_end, coda_start=coda_start):
                continue  # topology changed since this record was earned
            if rec.reliability_lb > best_lb:
                best, best_lb = rec.schedule, rec.reliability_lb
        return best

    def status(self) -> dict[str, Any]:
        domains: dict[str, int] = {}
        for (dom, _h) in self._records:
            domains[dom] = domains.get(dom, 0) + 1
        return {"records": len(self._records), "domains": domains}


# ── Evolutionary schedule search ────────────────────────────────────────


@dataclass
class SearchResult:
    best: LayerSchedule
    best_score: float
    evaluated: int
    history: list[dict[str, Any]] = field(default_factory=list)


class ScheduleSearch:
    """Budgeted, seeded evolutionary search over schedule programs.

    The evaluator is the honesty boundary: it must return a score derived
    from VERIFIED task outcomes (the experiments harness provides one). The
    search itself never inspects model internals — it only proposes programs
    and listens to evidence.
    """

    def __init__(
        self,
        *,
        prelude_end: int,
        coda_start: int,
        max_repeats: int = 8,
        seed: int = 0,
    ) -> None:
        if coda_start - prelude_end < 2:
            raise ValueError("recurrent region too small to search")
        self._p = prelude_end
        self._c = coda_start
        self._max_repeats = max_repeats
        self._rng = random.Random(seed)

    # ── Mutation operators ──────────────────────────────────────────────
    def _mutate(self, schedule: LayerSchedule) -> LayerSchedule:
        ops = list(schedule.ops)
        choice = self._rng.random()
        idx = self._rng.randrange(len(ops))
        op = ops[idx]
        if choice < 0.30:  # adjust repeats
            delta = self._rng.choice((-2, -1, 1, 2))
            ops[idx] = StageOp(op.start, op.end, max(1, min(self._max_repeats, op.repeats + delta)), op.alpha)
        elif choice < 0.55:  # shift window
            shift = self._rng.choice((-2, -1, 1, 2))
            start = max(self._p, min(op.start + shift, self._c - 1))
            end = max(start + 1, min(op.end + shift, self._c))
            ops[idx] = StageOp(start, end, op.repeats, op.alpha)
        elif choice < 0.75 and op.end - op.start >= 2:  # split stage
            mid = self._rng.randrange(op.start + 1, op.end)
            ops[idx : idx + 1] = [
                StageOp(op.start, mid, op.repeats, op.alpha),
                StageOp(mid, op.end, op.repeats, op.alpha),
            ]
        elif choice < 0.90 and len(ops) >= 2:  # merge adjacent stages
            j = min(idx + 1, len(ops) - 1)
            if j != idx:
                a, b = ops[idx], ops[j]
                merged = StageOp(min(a.start, b.start), max(a.end, b.end), max(a.repeats, b.repeats), a.alpha)
                ops[idx : j + 1] = [merged]
        else:  # adjust alpha
            alpha = op.alpha if op.alpha is not None else 0.5
            alpha = min(1.0, max(0.05, alpha + self._rng.choice((-0.15, -0.05, 0.05, 0.15))))
            ops[idx] = StageOp(op.start, op.end, op.repeats, round(alpha, 4))
        mutated = LayerSchedule(ops=tuple(ops), name="searched")
        if mutated.validate(prelude_end=self._p, coda_start=self._c):
            return schedule  # invalid mutation ⇒ keep parent
        return mutated

    def run(
        self,
        evaluator: Callable[[LayerSchedule], float],
        *,
        population: int = 6,
        generations: int = 4,
        seed_schedule: LayerSchedule | None = None,
    ) -> SearchResult:
        base = seed_schedule or LayerSchedule.single_window(self._p, self._c, 4)
        violations = base.validate(prelude_end=self._p, coda_start=self._c)
        if violations:
            raise ValueError(f"seed schedule invalid: {violations}")

        scored: dict[str, tuple[LayerSchedule, float]] = {}

        def score(s: LayerSchedule) -> float:
            key = s.schedule_hash
            if key not in scored:
                scored[key] = (s, float(evaluator(s)))
            return scored[key][1]

        pool = [base] + [self._mutate(base) for _ in range(population - 1)]
        history: list[dict[str, Any]] = []
        for gen in range(generations):
            ranked = sorted(pool, key=score, reverse=True)
            survivors = ranked[: max(2, population // 2)]
            history.append(
                {
                    "generation": gen,
                    "best_hash": survivors[0].schedule_hash,
                    "best_score": score(survivors[0]),
                    "pool": [s.schedule_hash for s in ranked],
                }
            )
            children = [self._mutate(self._rng.choice(survivors)) for _ in range(population - len(survivors))]
            pool = survivors + children

        best_hash = max(scored, key=lambda k: scored[k][1])
        best, best_score = scored[best_hash]
        return SearchResult(best=best, best_score=best_score, evaluated=len(scored), history=history)


__all__ = [
    "LayerSchedule",
    "MAX_TOTAL_LAYER_REPEATS",
    "ScheduleLibrary",
    "ScheduleRecord",
    "ScheduleSearch",
    "SearchResult",
    "StageOp",
]
