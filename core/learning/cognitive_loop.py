"""The conductor: one autonomous loop instead of a hand-driven pipeline (CP243).

Every result this session was produced by a pipeline I hand-designed. That
is the opposite of the thing being aimed at, which is a loop that runs
itself for any query in any domain:

    identify gap -> acquire -> deliberate -> act -> verify -> learn -> retain

This module is that loop, as one reusable conductor. It does NOT hard-code a
domain or a pipeline; it calls the same organ seams every cycle, so the same
code handles a science question, a coding task, or an ordinary one. It is
the answer to "without you hand-designing a pipeline each time": the loop IS
the pipeline, and it is data-driven.

It is deliberately organ-agnostic. Every stage is a seam -- a protocol or a
callback -- so the conductor is testable without the live 32B and wireable
to the live application without change. What plugs in:

* gap detector   -> does the model already know this? (skip acquisition if so)
* producers      -> the workspace spine (retrieval, imagination, ...)
* deliberator    -> the model reasoning over the material, in words
* verifier       -> a programmatic or organ check of the candidate answer
* learner        -> turns a verified outcome into a retained training signal

Two properties are load-bearing, both learned the hard way this session:

* **Every stage degrades honestly.** A missing organ makes its stage report
  "unavailable" and the loop continues without it; nothing is ever
  fabricated to keep a cycle looking successful.
* **Self-correction is real, not decorative.** If verification fails, the
  loop re-acquires and re-deliberates up to a bounded budget, and every
  attempt is recorded. A loop that could not show its retries would be
  indistinguishable from one that never retried.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

COGNITIVE_LOOP_SCHEMA = "aura.cognitive_loop.v1"


class GapDetector(Protocol):
    """Does the model already know the answer, or is knowledge missing?

    Returns True when a gap exists (acquisition is worth doing). The honest
    default when no detector is wired is 'assume a gap' -- acquiring
    knowledge you turned out to already have is cheap; skipping acquisition
    you needed is a wrong answer.
    """

    def has_gap(self, query: str) -> bool: ...


class Deliberator(Protocol):
    """The model reasoning, in words, over the query and gathered material.

    Returns a candidate answer string. This is the token-level deliberation
    the session found to WORK (versus silent latent looping, which did not).
    """

    def deliberate(self, query: str, material: list[str]) -> str: ...


class Verifier(Protocol):
    """A programmatic or organ check of a candidate answer.

    Returns a dict with at least ``{"correct": bool}`` and may include a
    bounded ``feedback`` or ``reason`` string for the next attempt. Reward
    comes from here and only here -- never from the model's own confidence,
    which would strengthen confident mistakes.
    """

    def check(self, query: str, candidate: str) -> dict[str, Any]: ...


@dataclass
class StageResult:
    name: str
    status: str  # "ok" | "unavailable" | "skipped" | "failed"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopResult:
    """Everything one cycle did, so it can be read rather than trusted."""

    query: str
    answer: str | None
    verified: bool
    attempts: int
    stages: list[StageResult]
    learned: bool

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": COGNITIVE_LOOP_SCHEMA,
            "query": self.query,
            "answer": self.answer,
            "verified": self.verified,
            "attempts": self.attempts,
            "learned": self.learned,
            "stages": [
                {"name": s.name, "status": s.status, **s.detail}
                for s in self.stages
            ],
        }


@dataclass
class CognitiveLoop:
    """One autonomous cycle of the full loop, wireable to live organs."""

    composer: Any = None            # WorkspaceComposer (the producer spine)
    deliberator: Deliberator | None = None
    verifier: Verifier | None = None
    gap_detector: GapDetector | None = None
    learner: Callable[[str, str, dict], bool] | None = None
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 16:
            raise ValueError("max_attempts must be inside [1, 16]")
        if self.deliberator is None:
            raise ValueError(
                "a loop with no deliberator cannot think; wire the model in"
            )

    def _identify_gap(self, query: str) -> tuple[bool, StageResult]:
        if self.gap_detector is None:
            # No detector -> assume a gap. Acquiring knowledge already held
            # is cheap; skipping acquisition you needed is a wrong answer.
            return True, StageResult("identify_gap", "unavailable",
                                     {"assumed_gap": True})
        try:
            gap = bool(self.gap_detector.has_gap(query))
        except Exception as exc:
            return True, StageResult(
                "identify_gap",
                "failed",
                {"assumed_gap": True, "error": type(exc).__name__},
            )
        return gap, StageResult("identify_gap", "ok", {"gap": gap})

    def _acquire(self, query: str, gap: bool) -> tuple[list[str], StageResult]:
        if not gap:
            return [], StageResult("acquire", "skipped", {"reason": "no_gap"})
        if self.composer is None:
            return [], StageResult("acquire", "unavailable", {})
        try:
            block = self.composer.compose(query)
        except Exception as exc:
            return [], StageResult(
                "acquire", "failed", {"error": type(exc).__name__}
            )
        if not isinstance(block, dict):
            return [], StageResult(
                "acquire", "failed", {"error": "invalid_composer_result"}
            )
        raw_lines = block.get("lines", [])
        if not isinstance(raw_lines, list):
            return [], StageResult(
                "acquire", "failed", {"error": "invalid_material_lines"}
            )
        lines = [str(line).strip() for line in raw_lines if str(line).strip()]
        return lines, StageResult("acquire", "ok", {
            "material": len(lines),
            "grounded": block.get("grounded", 0),
            "hypothetical": block.get("hypothetical", 0),
        })

    def _deliberate(self, query: str, material: list[str]) -> tuple[str, StageResult]:
        try:
            answer = self.deliberator.deliberate(query, material)
        except Exception as exc:
            return "", StageResult("deliberate", "failed", {"error": type(exc).__name__})
        answer_text = str(answer or "").strip()
        return answer_text, StageResult(
            "deliberate",
            "ok" if answer_text else "failed",
            {"answered": bool(answer_text), **({"error": "empty_answer"} if not answer_text else {})},
        )

    def _verify(self, query: str, candidate: str) -> tuple[bool, StageResult]:
        if not candidate.strip():
            return False, StageResult(
                "verify", "skipped", {"verified": False, "reason": "no_answer"}
            )
        if self.verifier is None:
            # No verifier -> the answer is UNVERIFIED, never assumed correct.
            return False, StageResult("verify", "unavailable", {"verified": False})
        try:
            verdict = self.verifier.check(query, candidate)
        except Exception as exc:
            return False, StageResult(
                "verify", "failed", {"verified": False, "error": type(exc).__name__}
            )
        if not isinstance(verdict, dict) or "correct" not in verdict:
            return False, StageResult(
                "verify",
                "failed",
                {"verified": False, "error": "invalid_verifier_result"},
            )
        ok = bool(verdict.get("correct"))
        detail: dict[str, Any] = {"verified": ok}
        for key in ("feedback", "reason"):
            value = str(verdict.get(key) or "").strip()
            if value:
                detail[key] = value[:500]
        return ok, StageResult("verify", "ok", detail)

    async def _adeliberate(self, query: str, material: list[str]) -> tuple[str, StageResult]:
        import inspect

        try:
            result = self.deliberator.deliberate(query, material)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            return "", StageResult("deliberate", "failed", {"error": type(exc).__name__})
        answer_text = str(result or "").strip()
        return answer_text, StageResult(
            "deliberate",
            "ok" if answer_text else "failed",
            {"answered": bool(answer_text), **({"error": "empty_answer"} if not answer_text else {})},
        )

    async def _averify(self, query: str, candidate: str) -> tuple[bool, StageResult]:
        import inspect

        if not candidate.strip():
            return False, StageResult(
                "verify", "skipped", {"verified": False, "reason": "no_answer"}
            )
        if self.verifier is None:
            return False, StageResult("verify", "unavailable", {"verified": False})
        try:
            verdict = self.verifier.check(query, candidate)
            if inspect.isawaitable(verdict):
                verdict = await verdict
        except Exception as exc:
            return False, StageResult(
                "verify", "failed", {"verified": False, "error": type(exc).__name__}
            )
        if not isinstance(verdict, dict) or "correct" not in verdict:
            return False, StageResult(
                "verify",
                "failed",
                {"verified": False, "error": "invalid_verifier_result"},
            )
        ok = bool(verdict.get("correct"))
        detail: dict[str, Any] = {"verified": ok}
        for key in ("feedback", "reason"):
            value = str(verdict.get(key) or "").strip()
            if value:
                detail[key] = value[:500]
        return ok, StageResult("verify", "ok", detail)

    def _attempt_budget(self, stages: list[StageResult]) -> int:
        """Retries require a verifier capable of judging the changed answer."""
        if self.verifier is not None or self.max_attempts == 1:
            return self.max_attempts
        stages.append(
            StageResult(
                "retry_control",
                "skipped",
                {
                    "reason": "verifier_unavailable",
                    "configured_attempts": self.max_attempts,
                    "effective_attempts": 1,
                },
            )
        )
        return 1

    @staticmethod
    def _correction_material(
        candidate: str,
        verify_stage: StageResult,
    ) -> str:
        """Turn adjudicated failure into bounded next-attempt context."""
        feedback = str(
            verify_stage.detail.get("feedback")
            or verify_stage.detail.get("reason")
            or "The verifier rejected the prior candidate."
        ).strip()[:500]
        prior = str(candidate or "").strip()[:500]
        return (
            "[verifier correction] The prior candidate was rejected. "
            f"Feedback: {feedback} Prior candidate: {prior}"
        )

    async def arun(self, query: str) -> LoopResult:
        """Async cycle for live organs (the LLM router's generate is async).

        Mirrors ``run`` exactly -- same stages, same self-correction, same
        honest-degradation and never-retain-unverified rules -- but awaits
        the deliberator and verifier when they are coroutines. The sync
        ``run`` stays the tested reference; this is the live path.
        """
        if not str(query).strip():
            raise ValueError("query must be non-empty")
        stages: list[StageResult] = []
        gap, gap_stage = self._identify_gap(query)
        stages.append(gap_stage)
        answer: str | None = None
        verified = False
        attempts = 0
        correction_material = ""
        for attempt in range(self._attempt_budget(stages)):
            attempts = attempt + 1
            material, acq_stage = self._acquire(query, gap)
            if correction_material:
                material = [*material, correction_material]
            deliberated, del_stage = await self._adeliberate(query, material)
            verified, ver_stage = await self._averify(query, deliberated)
            for stage in (acq_stage, del_stage, ver_stage):
                stage.detail["attempt"] = attempts
                stages.append(stage)
            answer = deliberated or answer
            if verified:
                break
            gap = True
            if self.verifier is not None and deliberated:
                correction_material = self._correction_material(deliberated, ver_stage)
        learned = False
        if verified and self.learner is not None and answer is not None:
            import inspect

            try:
                outcome = self.learner(query, answer, {"verified": True})
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                learned = bool(outcome)
            except Exception as exc:
                learned = False
                stages.append(
                    StageResult(
                        "learn",
                        "failed",
                        {"retained": False, "error": type(exc).__name__},
                    )
                )
            else:
                stages.append(StageResult("learn", "ok" if learned else "skipped",
                                          {"retained": learned}))
        elif self.learner is not None:
            stages.append(StageResult("learn", "skipped",
                                      {"reason": "unverified" if not verified else "no_answer"}))
        return LoopResult(query=query, answer=answer, verified=verified,
                          attempts=attempts, stages=stages, learned=learned)

    def run(self, query: str) -> LoopResult:
        """Run one cycle, with bounded self-correction on verification failure."""
        if not str(query).strip():
            raise ValueError("query must be non-empty")
        stages: list[StageResult] = []
        gap, gap_stage = self._identify_gap(query)
        stages.append(gap_stage)

        answer: str | None = None
        verified = False
        attempts = 0
        correction_material = ""
        for attempt in range(self._attempt_budget(stages)):
            attempts = attempt + 1
            material, acq_stage = self._acquire(query, gap)
            if correction_material:
                material = [*material, correction_material]
            deliberated, del_stage = self._deliberate(query, material)
            verified, ver_stage = self._verify(query, deliberated)
            # Tag each stage with the attempt so retries are legible, not
            # silently collapsed into one line.
            for stage in (acq_stage, del_stage, ver_stage):
                stage.detail["attempt"] = attempts
                stages.append(stage)
            answer = deliberated or answer
            if verified:
                break
            # Self-correction: a failed check means try again -- widening
            # acquisition next round -- until the budget is spent.
            gap = True
            if self.verifier is not None and deliberated:
                correction_material = self._correction_material(deliberated, ver_stage)

        # Learn: a verified outcome becomes a retained training signal. An
        # unverified one never does -- retaining unverified answers is how a
        # system trains on its own mistakes.
        learned = False
        if verified and self.learner is not None and answer is not None:
            try:
                learned = bool(self.learner(query, answer, {"verified": True}))
            except Exception as exc:
                learned = False
                stages.append(
                    StageResult(
                        "learn",
                        "failed",
                        {"retained": False, "error": type(exc).__name__},
                    )
                )
            else:
                stages.append(StageResult("learn", "ok" if learned else "skipped",
                                          {"retained": learned}))
        elif verified and self.learner is not None:
            stages.append(StageResult("learn", "skipped", {"reason": "no_answer"}))
        elif self.learner is not None:
            stages.append(StageResult("learn", "skipped",
                                      {"reason": "unverified"}))

        return LoopResult(
            query=query, answer=answer, verified=verified,
            attempts=attempts, stages=stages, learned=learned,
        )


def loop_health(results: list[LoopResult]) -> dict[str, Any]:
    """Is the loop actually working across a batch, or just running?

    Reports the numbers that decide whether the loop is real: how often it
    verified, how often self-correction rescued a first-attempt failure, and
    how often it retained a signal. A loop that never verifies is running,
    not working.
    """
    if not results:
        raise ValueError("no loop results to assess")
    verified = sum(1 for r in results if r.verified)
    rescued = sum(1 for r in results if r.verified and r.attempts > 1)
    learned = sum(1 for r in results if r.learned)
    return {
        "schema": COGNITIVE_LOOP_SCHEMA,
        "cycles": len(results),
        "verified_rate": round(verified / len(results), 4),
        "self_correction_rescues": rescued,
        "retained": learned,
        # The honest headline: a loop that verifies nothing is not a loop
        # that works, however smoothly it runs.
        "working": bool(verified > 0),
    }


__all__ = [
    "COGNITIVE_LOOP_SCHEMA",
    "CognitiveLoop",
    "Deliberator",
    "GapDetector",
    "LoopResult",
    "StageResult",
    "Verifier",
    "loop_health",
]
