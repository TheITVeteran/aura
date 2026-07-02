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
    correct: bool                # graded truth (verifier-grade or gold-match)
    false_confidence: bool       # wrong yet asserted with high confidence
    latency_ms: float
    mode: str = "deterministic"  # "deterministic" | "live"
    answer_gold_match: bool | None = None  # live only: did the answer contain gold?
    known_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_type": self.task_type,
            "mode": self.mode,
            "should_pass": self.should_pass,
            "verified": self.verified,
            "confidence": round(self.confidence, 3),
            "correct": self.correct,
            "gold_match": self.answer_gold_match,
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

    @staticmethod
    def _gold_match(answer: str, gold: str) -> bool:
        if not gold:
            return True
        a = " ".join(str(answer or "").lower().split())
        return gold.lower() in a

    async def _run_case(self, case: ReasoningCase, generate: GenerateFn | None) -> CaseOutcome:
        from core.brain.reasoning_amplifier_v2 import amplify_turn

        live = generate is not None
        gen = generate or self._canned_generator(case)
        t0 = time.monotonic()
        result = await amplify_turn(
            case.objective,
            gen,
            task_type=case.task_type,
            evidence=case.evidence,
            time_budget_s=30.0 if live else 20.0,
        )
        latency = (time.monotonic() - t0) * 1000.0
        verified = bool(result.verified)

        if live:
            # Grade the REAL answer against gold; "correct" means the model got it
            # right. A good system also makes verified track correctness.
            gold_match = self._gold_match(result.answer, case.gold)
            correct = gold_match
            false_confidence = (not correct) and result.confidence >= self.confidence_floor
            return CaseOutcome(
                case_id=case.case_id, task_type=case.task_type, should_pass=case.should_pass,
                verified=verified, confidence=result.confidence, correct=correct,
                false_confidence=false_confidence, latency_ms=latency, mode="live",
                answer_gold_match=gold_match, known_failures=result.receipt.known_failures,
            )

        # Deterministic: grade whether the verifier reached the expected verdict.
        # The amplifier may legitimately REPAIR a seeded error (the hardened math
        # verifier derives the exact answer from the question), so a should-fail
        # case counts as handled when it is either flagged unverified OR verifiably
        # repaired to the gold answer. Rubber-stamping the seeded wrong answer —
        # verified=True with a final answer that still misses gold — stays a miss.
        gold_match = self._gold_match(result.answer, case.gold) if case.gold else None
        if case.should_pass:
            correct = verified
        else:
            repaired = bool(verified and case.gold and gold_match)
            correct = (not verified) or repaired
        false_confidence = (not correct) and result.confidence >= self.confidence_floor
        return CaseOutcome(
            case_id=case.case_id, task_type=case.task_type, should_pass=case.should_pass,
            verified=verified, confidence=result.confidence, correct=correct,
            false_confidence=false_confidence, latency_ms=latency, mode="deterministic",
            answer_gold_match=gold_match, known_failures=result.receipt.known_failures,
        )

    async def run(self, *, generate: GenerateFn | None = None) -> BenchmarkResult:
        live = generate is not None
        cases = [c for c in self.cases if (c.run_live or not live)]
        outcomes = [await self._run_case(c, generate) for c in cases]
        n = len(outcomes)
        correct = [o for o in outcomes if o.correct]
        wrong = [o for o in outcomes if not o.correct]
        false_conf = [o for o in wrong if o.false_confidence]

        if live:
            # Verifier-catch = of answers the model actually got WRONG, how many did
            # the truth engines flag (verified=False)? i.e. does verification track
            # correctness. Hallucination = same, restricted to repo/factual answers.
            should_catch = wrong
            caught = [o for o in wrong if not o.verified]
            halluc = [o for o in wrong if o.task_type in {"repo_audit", "factual", "architecture"}]
        else:
            should_catch = [o for o in outcomes if not o.should_pass]
            # Caught = flagged unverified OR verifiably repaired to gold.
            caught = [o for o in should_catch if (not o.verified) or o.answer_gold_match]
            halluc = [o for o in should_catch if o.task_type in {"repo_audit", "factual", "architecture"}]
        halluc_caught = [o for o in halluc if not o.verified]

        by_task: dict[str, list[CaseOutcome]] = {}
        for o in outcomes:
            by_task.setdefault(o.task_type, []).append(o)

        return BenchmarkResult(
            n=n,
            pass_rate=len(correct) / max(1, n),
            verifier_catch_rate=(len(caught) / max(1, len(should_catch))) if should_catch else 1.0,
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
