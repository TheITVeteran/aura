"""core/evals/eval_arena.py — Massive Evaluation Arena."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("Aura.EvalArena")


@dataclass
class EvalReport:
    """A report of a specific benchmark family run."""
    family_name: str  # e.g., "software_engineering", "planning", "truthfulness", "tool_use"
    score: float  # 0.0 to 1.0
    passed_tests: int
    total_tests: int
    timestamp: float = field(default_factory=time.time)


class EvalArena:
    """Tracks and records Aura's capability scores across multiple dimensions over time."""

    def __init__(self) -> None:
        self.history: List[EvalReport] = []
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("Evaluation Arena fully online.")

    def record_run(self, family: str, passed: int, total: int) -> EvalReport:
        """Records a benchmark execution score."""
        score = passed / total if total > 0 else 1.0
        report = EvalReport(
            family_name=family,
            score=score,
            passed_tests=passed,
            total_tests=total,
        )
        self.history.append(report)
        logger.info("Recorded eval run: family=%s score=%.2f (%d/%d)", family, score, passed, total)
        return report

    def get_aggregate_stats(self) -> Dict[str, float]:
        """Calculates average scores for each capability family."""
        scores_by_family: Dict[str, List[float]] = {}
        for r in self.history:
            scores_by_family.setdefault(r.family_name, []).append(r.score)
        
        return {
            family: sum(scores) / len(scores)
            for family, scores in scores_by_family.items()
        }


_arena_instance: EvalArena | None = None


def get_eval_arena() -> EvalArena:
    global _arena_instance
    if _arena_instance is None:
        _arena_instance = EvalArena()
    return _arena_instance
