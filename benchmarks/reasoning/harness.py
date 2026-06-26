"""Run the reasoning battery and compute honesty metrics."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .suites import ReasoningCase, default_suite

GenerateFn = Callable[[str, float], Awaitable[str]]


@dataclass
class CaseOutcome:
    case_id: str
    task_type: str
    should_pass: bool
    verified: bool
    confidence: float
    correct: bool                # verified == should_pass
    false_confidence: bool       # wrong yet asserted with high confidence
    latency_ms: float
    known_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_type": self.task_type,
            "should_pass": self.should_pass,
            "verified": self.verified,
            "confidence": round(self.confidence, 3),
            "correct": self.correct,
            "false_confidence": self.false_confidence,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class BenchmarkResult:
    n: int
    pass_rate: float                  # fraction graded correctly (verified == should_pass)
    verifier_catch_rate: float        # of should-fail cases, fraction actually caught
    false_confidence_rate: float      # of wrong gradings, fraction asserted confidently
    hallucination_catch_rate: float   # of fabrication cases, fraction caught
    mean_latency_ms: float
    by_task: dict[str, float] = field(default_factory=dict)
    outcomes: list[CaseOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "pass_rate": round(self.pass_rate, 3),
            "verifier_catch_rate": round(self.verifier_catch_rate, 3),
            "false_confidence_rate": round(self.false_confidence_rate, 3),
            "hallucination_catch_rate": round(self.hallucination_catch_rate, 3),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "by_task": {k: round(v, 3) for k, v in self.by_task.items()},
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def summary(self) -> str:
        return (
            f"reasoning benchmark: n={self.n} "
            f"pass={self.pass_rate:.0%} verifier_catch={self.verifier_catch_rate:.0%} "
            f"false_conf={self.false_confidence_rate:.0%} "
            f"hallucination_catch={self.hallucination_catch_rate:.0%} "
            f"lat={self.mean_latency_ms:.0f}ms"
        )


class ReasoningBenchmark:
    def __init__(self, cases: list[ReasoningCase] | None = None, *, confidence_floor: float = 0.7) -> None:
        self.cases = cases or default_suite()
        self.confidence_floor = confidence_floor

    def _canned_generator(self, case: ReasoningCase) -> GenerateFn:
        async def gen(prompt: str, temperature: float) -> str:
            return case.candidate

        return gen

    async def _run_case(self, case: ReasoningCase, generate: GenerateFn | None) -> CaseOutcome:
        from core.brain.reasoning_amplifier_v2 import amplify_turn

        gen = generate or self._canned_generator(case)
        t0 = time.monotonic()
        result = await amplify_turn(
            case.objective,
            gen,
            task_type=case.task_type,
            evidence=case.evidence,
            time_budget_s=20.0,
        )
        latency = (time.monotonic() - t0) * 1000.0
        verified = bool(result.verified)
        correct = verified == case.should_pass
        # False confidence = graded wrong AND asserted with high confidence.
        false_confidence = (not correct) and result.confidence >= self.confidence_floor
        return CaseOutcome(
            case_id=case.case_id,
            task_type=case.task_type,
            should_pass=case.should_pass,
            verified=verified,
            confidence=result.confidence,
            correct=correct,
            false_confidence=false_confidence,
            latency_ms=latency,
            known_failures=result.receipt.known_failures,
        )

    async def run(self, *, generate: GenerateFn | None = None) -> BenchmarkResult:
        outcomes = [await self._run_case(c, generate) for c in self.cases]
        n = len(outcomes)
        correct = [o for o in outcomes if o.correct]
        should_fail = [o for o in outcomes if not o.should_pass]
        caught = [o for o in should_fail if not o.verified]
        wrong = [o for o in outcomes if not o.correct]
        false_conf = [o for o in wrong if o.false_confidence]
        # Hallucination cases are the fabrication-style ones (repo/citation should-fail).
        halluc = [o for o in should_fail if o.task_type in {"repo_audit", "factual", "architecture"}]
        halluc_caught = [o for o in halluc if not o.verified]

        by_task: dict[str, list[CaseOutcome]] = {}
        for o in outcomes:
            by_task.setdefault(o.task_type, []).append(o)

        return BenchmarkResult(
            n=n,
            pass_rate=len(correct) / max(1, n),
            verifier_catch_rate=len(caught) / max(1, len(should_fail)),
            false_confidence_rate=len(false_conf) / max(1, len(wrong)) if wrong else 0.0,
            hallucination_catch_rate=len(halluc_caught) / max(1, len(halluc)) if halluc else 1.0,
            mean_latency_ms=sum(o.latency_ms for o in outcomes) / max(1, n),
            by_task={k: sum(1 for o in v if o.correct) / len(v) for k, v in by_task.items()},
            outcomes=outcomes,
        )


def run_benchmark(*, generate: GenerateFn | None = None, cases: list[ReasoningCase] | None = None) -> BenchmarkResult:
    return asyncio.run(ReasoningBenchmark(cases).run(generate=generate))


def write_results(result: BenchmarkResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
