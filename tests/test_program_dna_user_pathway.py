"""A NATURAL user request must reach the strong, verifiable reverse-engineering
path (not just a structural blueprint). Pins target resolution and the skill's
runnable reverse-engineering, verified against the real host binary."""
from __future__ import annotations

import asyncio

import pytest

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


@pytest.mark.asyncio
async def test_live_chat_program_dna_request_runs_governed_skill(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    async def _fake_governed_skill(skill_name, params, *, objective, extra_context=None):
        calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "objective": objective,
                "extra_context": dict(extra_context or {}),
            }
        )
        return {
            "ok": True,
            "summary": "Reverse-engineered base64 from behavior only (no source): 4/4 held-out cases reproduced.",
            "result": {
                "target": "base64",
                "status": "supported",
                "held_out_passed": 4,
                "held_out_total": 4,
            },
        }

    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "Reverse engineer base64 from its behavior only — no source — and prove your reconstruction matches the real command on held-out inputs."
    )

    assert result is not None
    assert result["ok"] is True
    assert result["status"] == "program_dna_reconstruct_completed"
    assert "4/4 held-out" in result["response"]
    assert calls == [
        {
            "skill_name": "program_dna_reconstruct",
            "params": {
                "target": "base64",
                "authorization": "user_owned",
                "analysis_mode": "reverse_engineer",
                "emit_scaffold": False,
                "observed_behaviors": [],
                "tests": [],
            },
            "objective": (
                "Reverse engineer base64 from its behavior only — no source — and prove your "
                "reconstruction matches the real command on held-out inputs."
            ),
            "extra_context": {
                "origin": "desktop_ui",
                "source": "desktop_ui",
                "route": "chat.program_dna_reconstruct",
                "program_dna_execution_contract": True,
                "foreground_request": True,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "verification_required": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_live_chat_program_dna_does_not_execute_conceptual_question(monkeypatch):
    from interface.routes import chat as chat_routes

    async def _forbidden_governed_skill(*_args, **_kwargs):
        raise AssertionError("conceptual Program DNA questions must stay conversational")

    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _forbidden_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "What is Program DNA and how would it help Aura understand software?"
    )

    assert result is None


@pytest.mark.asyncio
async def test_live_chat_rsi_median_request_runs_verified_lab(monkeypatch):
    from interface.routes import chat as chat_routes

    calls = []

    async def _fake_governed_skill(skill_name, params, *, objective, extra_context=None):
        calls.append(
            {
                "skill_name": skill_name,
                "params": dict(params),
                "extra_context": dict(extra_context or {}),
            }
        )
        if skill_name == "file_operation":
            return {"ok": True, "path": params["path"]}
        if skill_name == "improve_own_code":
            return {
                "ok": True,
                "summary": "Improved median: passed 5/5 checks (original passed 2/5); enacted=True.",
                "result": {
                    "original_passed": 2,
                    "improved_passed": 5,
                    "total_checks": 5,
                    "enacted": True,
                    "status": "verified_improvement",
                },
            }
        raise AssertionError(skill_name)

    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _fake_governed_skill)
    result = await chat_routes._execute_governed_capability_request_from_chat(
        "Here's a buggy median function: it returns the upper-middle element for even-length lists. Improve it and verify the fix passes."
    )

    assert result is not None
    assert result["ok"] is True
    assert result["status"] == "rsi_self_improvement_completed"
    assert "Original passed 2/5" in result["response"]
    assert "verified improvement passed 5/5" in result["response"]
    assert [call["skill_name"] for call in calls] == ["file_operation", "improve_own_code"]
    assert calls[1]["params"]["func_name"] == "median"
    assert calls[1]["extra_context"]["rsi_execution_contract"] is True
