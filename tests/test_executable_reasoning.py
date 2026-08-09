from __future__ import annotations

import json
import time

import pytest

from core.brain.calibration_gate import CalibrationGate
from core.brain.executable_reasoning import (
    EXECUTABLE_REASONING_SCHEMA,
    derive_executable_candidate,
    should_use_executable_reasoning,
)
from core.brain.reasoning_amplifier_v2 import (
    AmplificationRequest,
    ReasoningAmplifierV2,
    ReasoningMode,
)
from core.brain.reasoning_memory import ReasoningMemory
from core.brain.verifiers.base import VerificationResult


class _Execution:
    def __init__(
        self,
        *,
        stdout: str,
        ok: bool = True,
        traceback: str = "",
    ) -> None:
        self.ok = ok
        self.refused = False
        self.timed_out = False
        self.stdout = stdout
        self.stderr = ""
        self.traceback = traceback
        self.final_code = "print('private source')"

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "refused": False,
            "timed_out": False,
            "rounds": 1,
            "stdout_total_bytes": len(self.stdout.encode()),
            "stderr_total_bytes": 0,
            "stdout_sha256": "evidence-digest",
            "final_code_sha256": "source-digest",
            "final_code_chars": len(self.final_code),
            "isolation": {
                "sandboxed": True,
                "network_denied": True,
                "user_data_denied": True,
            },
        }


class _Sandbox:
    def __init__(self, *results: _Execution) -> None:
        self.results = list(results)
        self.calls = 0
        self.sources: list[str] = []

    async def run(self, code):
        self.calls += 1
        self.sources.append(code)
        assert "private source" in code
        return self.results[min(self.calls - 1, len(self.results) - 1)]


def test_semantic_admission_is_general_and_honors_no_execution() -> None:
    assert should_use_executable_reasoning(
        "Count all subsets of [2, 4, 7] whose sum is 9.",
        task_type="logic",
    )
    assert should_use_executable_reasoning(
        "What is 23891 modulo 71?",
        task_type="math",
    )
    assert not should_use_executable_reasoning(
        "Evaluate this function without executing it: def f(): return 4",
        task_type="code",
        explicitly_enabled=True,
    )
    assert not should_use_executable_reasoning(
        "Explain why curiosity matters.",
        task_type="factual",
    )


@pytest.mark.asyncio
async def test_model_authored_program_becomes_bounded_candidate() -> None:
    prompts: list[str] = []

    async def generate(prompt: str, temperature: float) -> str:
        prompts.append(prompt)
        assert temperature == 0.2
        return "```python\nprint('private source')\n```"

    expected = 'FINAL_ANSWER: {"count":2}'
    result = await derive_executable_candidate(
        objective="Count the valid assignments in [1, 2, 3].",
        task_type="math",
        generate=generate,
        sandbox=_Sandbox(_Execution(stdout=expected)),
        deadline=time.monotonic() + 10.0,
        response_contract='{"count":int}',
    )

    assert result.succeeded is True
    assert result.candidate == expected
    assert result.receipt["schema"] == EXECUTABLE_REASONING_SCHEMA
    assert result.receipt["status"] == "candidate_ready"
    assert result.receipt["contract_valid"] is True
    assert result.receipt["program_chars"] > 0
    serialized = json.dumps(result.receipt)
    assert "private source" not in serialized
    assert "authoring a self-contained pure-Python scratch program" in prompts[0]


@pytest.mark.asyncio
async def test_failed_program_is_withheld_from_disjoint_restart() -> None:
    prompts: list[str] = []
    first_source = "print('private source')\nassert False, 'anchoring literal'"
    second_source = "print('private source')"

    async def generate(prompt: str, temperature: float) -> str:
        prompts.append(prompt)
        return f"```python\n{first_source if temperature == 0.2 else second_source}\n```"

    sandbox = _Sandbox(
        _Execution(stdout="", ok=False, traceback="AssertionError: anchoring literal"),
        _Execution(stdout='FINAL_ANSWER: {"answer":4}'),
    )
    result = await derive_executable_candidate(
        objective="Compute 2 + 2.",
        task_type="math",
        generate=generate,
        sandbox=sandbox,
        deadline=time.monotonic() + 10.0,
        response_contract='{"answer":int}',
    )

    assert result.succeeded is True
    assert result.receipt["generation_calls"] == 2
    assert result.receipt["attempts"][0]["status"] == "execution_failed"
    assert result.receipt["attempts"][1]["status"] == "executed"
    assert first_source not in prompts[1]
    assert "AssertionError: anchoring literal" not in prompts[1]
    assert "sandbox_execution_failed:AssertionError" in prompts[1]
    assert result.receipt["attempts"][0]["program_sha256"] in prompts[1]


class _ExactCandidateVerifier:
    async def verify(self, candidate, *, task_type=None, context=None):
        del task_type, context
        exact = str(candidate).strip() == 'FINAL_ANSWER: {"answer":4}'
        return VerificationResult(
            domain="math",
            ok=exact,
            checked=True,
            score=1.0 if exact else 0.0,
            engine="test_exact",
            issues=[] if exact else ["wrong answer"],
        )


@pytest.mark.asyncio
async def test_amplifier_promotes_verified_executable_result_over_wrong_seed(
    tmp_path,
) -> None:
    calls = 0

    async def generate(prompt: str, temperature: float) -> str:
        nonlocal calls
        calls += 1
        assert "pure-Python scratch program" in prompt
        return "```python\nprint('private source')\n```"

    sandbox = _Sandbox(_Execution(stdout='FINAL_ANSWER: {"answer":4}'))
    amplifier = ReasoningAmplifierV2(
        generate,
        verifier=_ExactCandidateVerifier(),
        sandbox=sandbox,
        calibration=CalibrationGate(),
        memory=ReasoningMemory(path=tmp_path / "memory.jsonl"),
    )
    result = await amplifier.amplify(
        AmplificationRequest(
            objective="Compute 2 + 2 and return the result.",
            task_type="math",
            mode=ReasoningMode.NORMAL,
            sample_budget=3,
            context={
                "seed_candidates": ['FINAL_ANSWER: {"answer":5}'],
                "response_contract": '{"answer":int}',
                "enable_executable_reasoning": True,
                "read_only_evaluation": True,
                "sealed_evaluation": True,
                "skip_evidence": True,
                "skip_cache": True,
                "disable_batched_candidates": True,
            },
        )
    )

    assert calls == 1
    assert sandbox.calls == 1
    assert result.source_answer == 'FINAL_ANSWER: {"answer":4}'
    assert result.verified is True
    assert result.receipt.strategy_used == "executable_reasoning"
    assert "executable_reasoning_verified" in result.receipt.fallbacks_used
    assert result.receipt.cognitive_operations[0]["status"] == "candidate_ready"


@pytest.mark.asyncio
async def test_no_execution_constraint_uses_ordinary_candidate(tmp_path) -> None:
    sandbox = _Sandbox(_Execution(stdout="must not run"))

    async def generate(_prompt: str, _temperature: float) -> str:
        return "The function returns 4."

    class _PassVerifier:
        async def verify(self, candidate, *, task_type=None, context=None):
            del candidate, task_type, context
            return VerificationResult(
                domain="code",
                ok=True,
                checked=True,
                score=1.0,
                engine="test",
            )

    amplifier = ReasoningAmplifierV2(
        generate,
        verifier=_PassVerifier(),
        sandbox=sandbox,
        calibration=CalibrationGate(),
        memory=ReasoningMemory(path=tmp_path / "memory.jsonl"),
    )
    result = await amplifier.amplify(
        AmplificationRequest(
            objective="Evaluate this function without executing it: def f(): return 4",
            task_type="code",
            mode=ReasoningMode.NORMAL,
            sample_budget=1,
            context={
                "read_only_evaluation": True,
                "sealed_evaluation": True,
                "skip_evidence": True,
                "skip_cache": True,
                "disable_batched_candidates": True,
                "enable_executable_reasoning": True,
            },
        )
    )

    assert result.source_answer == "The function returns 4."
    assert sandbox.calls == 0
    assert result.receipt.cognitive_operations[0]["status"] == "not_applicable"
