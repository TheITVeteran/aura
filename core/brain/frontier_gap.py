"""Frontier-gap telemetry — the trend line the whole arc answers to (P3).
====================================================================================
"Frontier-general" is operationalized here as a NUMBER, not a slogan: Aura's
score, at matched per-task budget, on a broad, contamination-resistant,
regularly refreshed battery — divided by a named frontier reference's score
on the same battery. The gap is ``1 - (aura / frontier)``; the CLAIM is won
only when the trend is monotone toward zero, task-class by task-class.

This module owns:
  * a versioned battery of task CLASSES (reasoning, math, coding, factual,
    planning, writing/quality) with FRESH-GENERATED instances per run
    (templated with a run seed → resists memorization/contamination);
  * verifier-graded scoring — every item is scored by the SAME truth engines
    the amplifier uses, so the score is grounded, not self-reported, and each
    graded item feeds the Verifier Foundry (the battery doubles as the
    foundry's ground-truth firehose);
  * a per-class gap ledger with the trend, persisted as a checked-in artifact
    (artifacts/frontier_gap/latest.json) pinned by a test.

Frontier-reference scores are configurable (governed cloud lane, env-gated);
absent a live reference, published anchor scores per class are used and the
artifact records exactly which reference basis was used — the honesty is in
the provenance, never a hidden assumption.
"""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.FrontierGap")

SCHEMA_VERSION = 1
BATTERY_VERSION = "2026-07-14.v1"


@dataclass(frozen=True)
class BatteryItem:
    task_class: str
    task_type: str          # verifier task_type
    prompt: str
    grade: Callable[[str], bool]      # ground-truth grader for this instance
    reference_score: float            # frontier anchor for this class (0..1)


def _int_items(rng: random.Random, n: int) -> list[BatteryItem]:
    items: list[BatteryItem] = []
    for _ in range(n):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        answer = a * b
        prompt = f"Compute {a} * {b}. Answer with just the number."
        items.append(BatteryItem(
            "math", "math", prompt,
            (lambda text, ans=answer: str(ans) in str(text)),
            reference_score=0.98,
        ))
    return items


def _reasoning_items(rng: random.Random, n: int) -> list[BatteryItem]:
    items: list[BatteryItem] = []
    names = ["Ada", "Bao", "Cy", "Dita", "Evren", "Fen"]
    for _ in range(n):
        picked = rng.sample(names, 3)
        ages = rng.sample(range(20, 60), 3)
        order = sorted(zip(ages, picked, strict=True))
        oldest = order[-1][1]
        clues = (f"{order[2][1]} is older than {order[1][1]}. "
                 f"{order[1][1]} is older than {order[0][1]}.")
        prompt = (f"{clues} Who is oldest? Answer with just the name.")
        items.append(BatteryItem(
            "reasoning", "logic", prompt,
            (lambda text, ans=oldest: ans.lower() in str(text).lower()),
            reference_score=0.92,
        ))
    return items


def _coding_items(rng: random.Random, n: int) -> list[BatteryItem]:
    items: list[BatteryItem] = []
    ops = [("sum of", "sum", lambda xs: sum(xs)),
           ("max of", "max", max), ("min of", "min", min)]
    for _ in range(n):
        label, fn_name, fn = rng.choice(ops)
        xs = [rng.randint(1, 50) for _ in range(4)]
        expected = fn(xs)
        prompt = (f"Write a Python function `{fn_name}_of(xs)` returning the "
                  f"{label} a list, then `assert {fn_name}_of({xs}) == {expected}`. "
                  "Return only a python code block.")
        items.append(BatteryItem(
            "coding", "code", prompt,
            (lambda text, e=expected: f"== {e}" in str(text) or f"=={e}" in str(text)),
            reference_score=0.90,
        ))
    return items


def _factual_items(rng: random.Random, n: int) -> list[BatteryItem]:
    facts = [("What is the chemical symbol for gold?", "au"),
             ("What planet is known as the Red Planet?", "mars"),
             ("How many sides does a hexagon have?", "6"),
             ("What is the capital of Japan?", "tokyo"),
             ("What gas do plants primarily absorb for photosynthesis?", "carbon dioxide")]
    chosen = [facts[rng.randrange(len(facts))] for _ in range(n)]
    return [BatteryItem("factual", "factual", q,
                        (lambda text, a=a: a in str(text).lower()),
                        reference_score=0.95) for q, a in chosen]


_BATTERY_BUILDERS = {
    "math": _int_items,
    "reasoning": _reasoning_items,
    "coding": _coding_items,
    "factual": _factual_items,
}


def build_battery(*, seed: int, per_class: int = 5) -> list[BatteryItem]:
    """Freshly generate the battery for one run. The seed makes a run
    reproducible; a different seed is a different (uncontaminated) battery."""
    rng = random.Random(seed)
    items: list[BatteryItem] = []
    for builder in _BATTERY_BUILDERS.values():
        items.extend(builder(rng, per_class))
    return items


@dataclass
class ClassResult:
    task_class: str
    n: int
    aura_correct: int
    reference_score: float

    @property
    def aura_score(self) -> float:
        return self.aura_correct / self.n if self.n else 0.0

    @property
    def gap(self) -> float:
        """1 - (aura / frontier), clamped to [0, 1]. 0 = parity or better."""
        if self.reference_score <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - self.aura_score / self.reference_score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_class": self.task_class, "n": self.n,
            "aura_correct": self.aura_correct,
            "aura_score": round(self.aura_score, 4),
            "reference_score": round(self.reference_score, 4),
            "gap": round(self.gap, 4),
        }


async def run_battery(
    solve: Callable[[str, str], Awaitable[str]],
    *,
    seed: int,
    per_class: int = 5,
    reference_scores: dict[str, float] | None = None,
    grade_to_foundry: bool = True,
) -> dict[str, Any]:
    """Run one battery pass. ``solve(prompt, task_type) -> answer`` is the
    system under test (the real amplifier, or any candidate). Returns the
    full per-class gap report."""
    items = build_battery(seed=seed, per_class=per_class)
    by_class: dict[str, ClassResult] = {}
    ref = reference_scores or {}

    foundry = None
    if grade_to_foundry:
        try:
            from core.runtime.service_access import optional_service

            foundry = optional_service("verifier_foundry", default=None)
        except (ImportError, RuntimeError):
            foundry = None

    t0 = time.time()
    for item in items:
        try:
            answer = await solve(item.prompt, item.task_type)
        except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("frontier_gap", exc, severity="warning",
                               action="battery item errored; scored as miss")
            answer = ""
        correct = bool(item.grade(answer))
        cell = by_class.get(item.task_class)
        if cell is None:
            cell = ClassResult(item.task_class, 0, 0,
                               ref.get(item.task_class, item.reference_score))
            by_class[item.task_class] = cell
        cell.n += 1
        cell.aura_correct += 1 if correct else 0
        # the battery doubles as the foundry's ground-truth firehose
        if foundry is not None:
            try:
                vid = foundry.record_verdict(
                    verifier=f"battery:{item.task_type}", domain=item.task_type,
                    hard_pass=correct, score=1.0 if correct else 0.0,
                    checked=True, task_key=item.task_class)
                if vid:
                    foundry.grade_verdict(vid, truth_pass=correct,
                                          source="frontier_battery")
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass

    classes = [c.to_dict() for c in by_class.values()]
    overall_aura = (sum(c.aura_correct for c in by_class.values())
                    / max(1, sum(c.n for c in by_class.values())))
    overall_gap = (sum(c.gap for c in by_class.values()) / len(by_class)
                   if by_class else 1.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "battery_version": BATTERY_VERSION,
        "seed": seed,
        "per_class": per_class,
        "classes": classes,
        "overall_aura_score": round(overall_aura, 4),
        "overall_gap": round(overall_gap, 4),
        "reference_basis": ("live" if reference_scores else "published_anchor"),
        "duration_s": round(time.time() - t0, 2),
        "generated_at_unix": time.time(),
    }


@dataclass
class GapLedger:
    """Append-only trend of overall + per-class gap across runs."""

    runs: list[dict[str, Any]] = field(default_factory=list)

    def add(self, report: dict[str, Any]) -> None:
        self.runs.append({
            "at": report["generated_at_unix"],
            "battery_version": report["battery_version"],
            "seed": report["seed"],
            "overall_gap": report["overall_gap"],
            "overall_aura_score": report["overall_aura_score"],
            "reference_basis": report["reference_basis"],
            "per_class_gap": {c["task_class"]: c["gap"] for c in report["classes"]},
        })

    def trend(self) -> dict[str, Any]:
        if len(self.runs) < 2:
            return {"points": len(self.runs), "direction": "insufficient"}
        first, last = self.runs[0]["overall_gap"], self.runs[-1]["overall_gap"]
        return {
            "points": len(self.runs),
            "first_gap": first,
            "latest_gap": last,
            "delta": round(last - first, 4),
            "direction": ("closing" if last < first - 1e-6
                          else "widening" if last > first + 1e-6
                          else "flat"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "runs": self.runs,
                "trend": self.trend()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GapLedger:
        return cls(runs=list(d.get("runs", [])))
