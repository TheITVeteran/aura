"""A NATURAL user request must reach the strong, verifiable reverse-engineering
path (not just a structural blueprint). Pins target resolution and the skill's
runnable reverse-engineering, verified against the real host binary."""
from __future__ import annotations

import asyncio

from core.discovery.reconstruction_sandbox import GeneralReconstructionEvaluator
from core.self_improvement.host_reconstruction import (
    KNOWN_TARGETS,
    resolve_target,
    reverse_engineer_host_binary,
)


def test_user_phrasings_resolve_to_known_targets():
    assert resolve_target("base64").name == "base64"
    assert resolve_target("the base64 tool").name == "base64"
    assert resolve_target("md5sum").name == "md5"
    assert resolve_target("reverse").name == "rev"
    assert resolve_target("some_unknown_program") is None


class _StubEngine:
    """Simulates the live 32B reconstructing base64 correctly."""

    async def reconstruct_executable_via_cognition(self, **kwargs):
        code = (
            "import base64\n"
            "def reconstructed(case):\n"
            "    return base64.b64encode(case['text'].encode()).decode() + '\\n'\n"
        )
        assert kwargs.get("sandbox_profile") == "general"
        ev = GeneralReconstructionEvaluator(timeout_seconds=5.0)
        held = kwargs["held_out"]
        passed = sum(
            1
            for c in held
            if ev.evaluate(code, "reconstructed", [((c["input"],), c["expected"])]).outcome == "passed"
        )
        total = len(held)
        return {
            "status": "supported" if passed == total else "refuted",
            "held_out_passed": passed,
            "held_out_total": total,
            "equivalence": passed / total if total else 0.0,
            "code": code,
        }


def test_reverse_engineer_host_binary_verifies_against_real_output():
    target = KNOWN_TARGETS["base64"]
    report = asyncio.run(reverse_engineer_host_binary(_StubEngine(), target))
    assert report["status"] == "supported"
    assert report["held_out_passed"] == report["held_out_total"]
    assert report["held_out_total"] >= 3
    assert "NO source" in report["policy"]


def test_skill_reverse_engineer_mode_returns_verified_result(monkeypatch):
    import core.skills.program_dna_reconstruct as mod

    monkeypatch.setattr(mod, "get_runtime_service", lambda *a, **k: _StubEngine())
    skill = mod.ProgramDNAReconstructSkill()
    result = asyncio.run(
        skill.execute({"target": "base64", "analysis_mode": "reverse_engineer"})
    )
    assert result["ok"] is True
    assert result["result"]["status"] == "supported"
    assert "held-out" in result["summary"]


def test_default_reconstruct_mode_prefers_runnable_for_known_binary(monkeypatch):
    # even the default mode gets the strong path when the target is a known binary
    import core.skills.program_dna_reconstruct as mod

    monkeypatch.setattr(mod, "get_runtime_service", lambda *a, **k: _StubEngine())
    skill = mod.ProgramDNAReconstructSkill()
    result = asyncio.run(skill.execute({"target": "base64", "analysis_mode": "reconstruct"}))
    assert result["result"]["status"] == "supported"
