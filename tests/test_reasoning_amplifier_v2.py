"""Tests for the Reasoning Amplifier v2 orchestrator."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.calibration_gate import CalibrationGate
from core.brain.reasoning_amplifier_v2 import (
    AmplificationRequest,
    ReasoningAmplifierV2,
    ReasoningBudgetPolicy,
    ReasoningMode,
    classify_task_type,
    normalize_problem,
)
from core.brain.reasoning_memory import ReasoningMemory
from core.brain.verifiers import get_verifier_registry


def _gen(answer: str):
    async def generate(prompt: str, temperature: float) -> str:
        return answer

    return generate


def _amp(generate, tmp_path: Path) -> ReasoningAmplifierV2:
    return ReasoningAmplifierV2(
        generate,
        verifier=get_verifier_registry(),
        calibration=CalibrationGate(),
        memory=ReasoningMemory(path=tmp_path / "refl.jsonl"),
    )


def test_classify_task_type():
    assert classify_task_type("fix the bug in this function") == "code"
    assert classify_task_type("compute the factorial of 6") == "math"
    assert classify_task_type("which file implements the inference gate") == "repo_audit"
    assert classify_task_type("what is the capital of France") == "factual"
    assert classify_task_type("how would you plan the migration steps") == "planning"


def test_normalize_problem_attaches_verification_plan():
    p = normalize_problem("solve x for 2x = 4", task_type="math")
    assert p.task_type == "math"
    assert any("arithmetic" in s or "sympy" in s for s in p.verification_plan)


def test_budget_policy():
    assert ReasoningBudgetPolicy.choose_mode("code") is ReasoningMode.DEEP
    assert ReasoningBudgetPolicy.choose_mode("factual") is ReasoningMode.NORMAL
    assert ReasoningBudgetPolicy.choose_mode("architecture", risk_level="high") is ReasoningMode.EXTREME
    assert ReasoningBudgetPolicy.choose_mode("generic", explicit=ReasoningMode.FAST) is ReasoningMode.FAST


@pytest.mark.asyncio
async def test_fast_mode_returns_calibrated_answer(tmp_path):
    amp = _amp(_gen("Paris is definitely the capital of France."), tmp_path)
    req = AmplificationRequest(objective="capital of France?", mode=ReasoningMode.FAST)
    out = await amp.amplify(req)
    assert out.answer
    assert out.receipt.mode == "fast"
    assert out.receipt.budget_used["samples"] >= 1


@pytest.mark.asyncio
async def test_math_error_is_caught_and_lowers_confidence(tmp_path):
    amp = _amp(_gen("The result: 2 + 2 = 5"), tmp_path)
    req = AmplificationRequest(objective="what is 2 + 2", task_type="math", mode=ReasoningMode.NORMAL)
    out = await amp.amplify(req)
    assert not out.verified
    assert out.confidence < 0.6
    assert any("arithmetic" in f for f in out.receipt.known_failures)


@pytest.mark.asyncio
async def test_proof_mode_refuses_unverified(tmp_path):
    amp = _amp(_gen("3 * 3 = 10"), tmp_path)
    req = AmplificationRequest(objective="3 * 3", task_type="math", mode=ReasoningMode.PROOF)
    out = await amp.amplify(req)
    assert "can't assert" in out.answer.lower() or "did not survive" in out.answer.lower()
    assert "proof_refused_unverified" in out.receipt.fallbacks_used


@pytest.mark.asyncio
async def test_clean_code_verifies(tmp_path):
    amp = _amp(_gen("Here is the fix:\n```python\ndef add(a, b):\n    return a + b\n```"), tmp_path)
    req = AmplificationRequest(objective="write an add function", task_type="code", mode=ReasoningMode.NORMAL)
    out = await amp.amplify(req)
    assert out.verified


@pytest.mark.asyncio
async def test_receipt_is_complete(tmp_path):
    amp = _amp(_gen("The answer is 4."), tmp_path)
    req = AmplificationRequest(objective="2+2", task_type="math", mode=ReasoningMode.NORMAL)
    out = await amp.amplify(req)
    d = out.receipt.to_dict()
    for key in ("mode", "strategy_used", "task_type", "num_candidates", "confidence",
                "epistemic_status", "budget_used"):
        assert key in d


@pytest.mark.asyncio
async def test_memory_guard_applied_second_time(tmp_path):
    # First episode fails verification → a failure-mode is recorded.
    mem = ReasoningMemory(path=tmp_path / "refl.jsonl")
    amp1 = ReasoningAmplifierV2(_gen("2 + 2 = 5"), verifier=get_verifier_registry(),
                                calibration=CalibrationGate(), memory=mem)
    await amp1.amplify(AmplificationRequest(objective="add 2 and 2 carefully", task_type="math",
                                            mode=ReasoningMode.NORMAL))
    # Second, similar episode should surface the guard.
    captured = {}

    async def capture_gen(prompt: str, temperature: float) -> str:
        captured["prompt"] = prompt
        return "2 + 2 = 4"

    amp2 = ReasoningAmplifierV2(capture_gen, verifier=get_verifier_registry(),
                                calibration=CalibrationGate(), memory=mem)
    out = await amp2.amplify(AmplificationRequest(objective="add 2 and 2 again", task_type="math",
                                                  mode=ReasoningMode.NORMAL))
    assert "Lessons from similar past reasoning" in captured.get("prompt", "")
    assert out.receipt.guards_applied


@pytest.mark.asyncio
async def test_deep_mode_uses_courtroom(tmp_path):
    # A generate fn that always answers correctly so the courtroom + verifier pass.
    amp = _amp(_gen("Answer: 12 * 12 = 144"), tmp_path)
    req = AmplificationRequest(objective="what is 12 times 12", task_type="math", mode=ReasoningMode.DEEP,
                               time_budget_s=20.0)
    out = await amp.amplify(req)
    assert out.receipt.mode == "deep"
    assert out.receipt.strategy_used in {"courtroom", "self_consistency", "direct"}
