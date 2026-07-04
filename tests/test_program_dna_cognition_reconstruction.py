"""The cognition reconstruction path must verify honestly, not rubber-stamp.

These run offline by scripting the code generator with a fixed implementation
and letting the REAL sandbox judge it against held-out cases. The point is that
correct code earns ``supported``, wrong code earns ``refuted``, and the verdict
comes from actually executing the reconstruction — not from a self-report.
"""
from __future__ import annotations

import pytest

from core.self_improvement.program_dna import ProgramDNAReconstructionEngine


class _ScriptedGenerator:
    """Stands in for the live model by returning a pre-decided implementation,
    so the sandbox-verification wiring can be proven without the 32B lane."""

    def __init__(self, code: str) -> None:
        self._code = code

    async def generate_async(self, prompt: str, context: dict) -> str:
        return self._code


def _script(monkeypatch, code: str) -> None:
    import core.brain.llm.code_generator as code_generator

    monkeypatch.setattr(code_generator, "LLMCodeGenerator", lambda *a, **k: _ScriptedGenerator(code))


@pytest.mark.asyncio
async def test_correct_reconstruction_is_supported_by_held_out(monkeypatch):
    _script(
        monkeypatch,
        "```python\ndef reconstructed(case):\n    return case['a'] + case['b']\n```",
    )
    engine = ProgramDNAReconstructionEngine()
    result = await engine.reconstruct_executable_via_cognition(
        target="adder",
        spec_docs=["Returns the sum of integer fields a and b."],
        train_examples=[{"input": {"a": 1, "b": 2}, "output": 3}],
        held_out=[
            {"input": {"a": 4, "b": 5}, "expected": 9},
            {"input": {"a": 0, "b": 0}, "expected": 0},
        ],
    )
    assert result["status"] == "supported"
    assert result["epistemic_status"] == "supported"
    assert result["held_out_passed"] == result["held_out_total"] == 2
    assert result["equivalence"] == 1.0
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_wrong_reconstruction_is_refuted_not_rubber_stamped(monkeypatch):
    _script(
        monkeypatch,
        "```python\ndef reconstructed(case):\n    return case['a'] - case['b']\n```",
    )
    engine = ProgramDNAReconstructionEngine()
    result = await engine.reconstruct_executable_via_cognition(
        target="adder",
        spec_docs=["Returns the sum of integer fields a and b."],
        train_examples=[{"input": {"a": 1, "b": 2}, "output": 3}],
        held_out=[{"input": {"a": 4, "b": 5}, "expected": 9}],
    )
    assert result["status"] == "refuted"
    assert result["ok"] is False
    assert result["held_out_passed"] == 0


@pytest.mark.asyncio
async def test_empty_generation_is_conjecture_never_supported(monkeypatch):
    _script(monkeypatch, "   ")
    engine = ProgramDNAReconstructionEngine()
    result = await engine.reconstruct_executable_via_cognition(
        target="adder",
        spec_docs=["Returns the sum of a and b."],
        train_examples=[{"input": {"a": 1, "b": 2}, "output": 3}],
        held_out=[{"input": {"a": 4, "b": 5}, "expected": 9}],
    )
    assert result["status"] == "conjecture"
    assert result["ok"] is False


@pytest.mark.asyncio
async def test_unauthorized_reconstruction_is_blocked(monkeypatch):
    _script(monkeypatch, "```python\ndef reconstructed(case):\n    return 1\n```")
    engine = ProgramDNAReconstructionEngine()
    result = await engine.reconstruct_executable_via_cognition(
        target="whatever",
        spec_docs=[],
        train_examples=[],
        held_out=[],
        authorization="unspecified",
    )
    assert result["status"] == "blocked"
    assert "authorization_required_for_program_reconstruction" in result["blocked_reasons"]
