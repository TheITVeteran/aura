"""Tests for the courtroom adversarial reasoning mode."""
from __future__ import annotations

import pytest

from core.brain.courtroom import Courtroom, CourtroomVerdict
from core.brain.verifiers import get_verifier_registry


def _make_generate(responses: dict[str, str], default: str = "Answer: 42"):
    """Return a generate(prompt, temp) that keys off role markers in the prompt."""

    async def generate(prompt: str, temperature: float) -> str:
        for key, val in responses.items():
            if key in prompt:
                return val
        return default

    return generate


@pytest.mark.asyncio
async def test_courtroom_runs_full_pipeline():
    gen = _make_generate(
        {
            "Solver": "Reasoning: 6*7=42.\nAnswer: 42",
            "Skeptic. WITHOUT": "Common error: arithmetic slips.",
            "Skeptic. Find": "The answer looks sound.",
            "Evidence Clerk": "- 6 times 7 is 42.",
            "Judge": "Reviewed all.\nVerdict: 42",
            "Simplifier": "Answer: 42",
        }
    )
    court = Courtroom(gen, verifier=get_verifier_registry())
    verdict = await court.deliberate("What is 6 times 7?", task_type="math", n_candidates=2)
    assert isinstance(verdict, CourtroomVerdict)
    assert "42" in verdict.answer
    assert verdict.verifier_ok
    assert verdict.confidence > 0.6
    assert len(verdict.candidates) == 2
    assert verdict.winning_role == "simplifier"


@pytest.mark.asyncio
async def test_courtroom_honors_failed_verifier():
    # Solver asserts a wrong arithmetic identity; the math truth engine must catch it,
    # dragging confidence down and surfacing the issue.
    gen = _make_generate(
        {
            "Solver": "Answer: 2 + 2 = 5",
            "Judge": "Verdict: 2 + 2 = 5",
            "Simplifier": "2 + 2 = 5",
        },
        default="2 + 2 = 5",
    )
    court = Courtroom(gen, verifier=get_verifier_registry())
    verdict = await court.deliberate("What is 2 + 2?", task_type="math", n_candidates=1)
    assert not verdict.verifier_ok
    assert verdict.unresolved
    assert verdict.confidence < 0.6


@pytest.mark.asyncio
async def test_courtroom_handles_empty_solver():
    async def gen(prompt: str, temperature: float) -> str:
        return "" if "Solver" in prompt else "x"

    court = Courtroom(gen)
    verdict = await court.deliberate("anything")
    assert verdict.answer == ""
    assert verdict.confidence == 0.0


@pytest.mark.asyncio
async def test_courtroom_reports_judge_when_simplifier_is_empty():
    async def gen(prompt: str, temperature: float) -> str:
        if "Simplifier" in prompt:
            return ""
        if "Judge" in prompt:
            return "Verdict: judge-authored"
        if "Solver" in prompt:
            return "Answer: solver-authored"
        return "review"

    verdict = await Courtroom(gen).deliberate("choose", n_candidates=1)

    assert verdict.answer == "judge-authored"
    assert verdict.winning_role == "judge"
    assert verdict.verifier_ok is False
    assert "verifier_unavailable" in verdict.verifier_issues


@pytest.mark.asyncio
async def test_courtroom_reports_solver_when_judge_and_simplifier_are_empty():
    async def gen(prompt: str, temperature: float) -> str:
        if "Judge" in prompt or "Simplifier" in prompt:
            return ""
        if "Solver" in prompt:
            return "Answer: solver-authored"
        return "review"

    verdict = await Courtroom(gen).deliberate("choose", n_candidates=1)

    assert verdict.answer == "solver-authored"
    assert verdict.winning_role == "solver"
