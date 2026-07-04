from __future__ import annotations

from pathlib import Path

import pytest

from core.learning.autonomous_rsi import (
    generate_solver_source,
    solve_with_generated_code,
)
from core.promotion.dynamic_benchmark import Task


def test_generated_solver_rejects_filesystem_side_effects(tmp_path: Path):
    marker = tmp_path / "should_not_exist"
    source = f"""
import os

def solve(task):
    os.system("touch {marker}")
    return 999
"""
    task = Task("gcd", "", 6, {"a": 54, "b": 24})

    assert solve_with_generated_code(task, source) is None
    assert not marker.exists()


def test_generate_solver_source_rejects_unknown_handlers():
    with pytest.raises(ValueError, match="unsupported RSI solver handlers"):
        generate_solver_source({"gcd", "filesystem"}, generation_id="Aura-GX")


def test_generate_solver_source_uses_verified_deterministic_path_without_router(monkeypatch):
    from core.container import ServiceContainer

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *_args, **_kwargs: None))
    # No generation backend at all: on machines with local code weights,
    # get_local_code_model() would otherwise take the LLM path and generate
    # real code — this test pins the deterministic fallback contract.
    monkeypatch.setattr(
        "core.brain.llm.local_code_model.get_local_code_model", lambda: None
    )

    source, metadata = generate_solver_source({"gcd"}, generation_id="Aura-G1")
    task = Task("gcd", "", 6, {"a": 54, "b": 24})

    assert metadata["parse_result"] == "deterministic_verified"
    assert metadata["sandbox_result"]["pass"] is True
    assert metadata["safety_contract"]["static_validation"] is True
    assert solve_with_generated_code(task, source) == 6


def test_generate_solver_source_demotes_unsafe_llm_candidate(monkeypatch):
    import core.brain.llm.code_generator as code_generator
    from core.container import ServiceContainer

    class UnsafeGenerator:
        def __init__(self, **_kwargs):
            self.is_background = True

        def generate(self, *_args, **_kwargs):
            return "import os\n\ndef solve(task):\n    os.system('true')\n    return 1\n"

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *_args, **_kwargs: object()))
    monkeypatch.setattr(code_generator, "LLMCodeGenerator", UnsafeGenerator)
    # Environment independence: with real local code weights present,
    # get_local_code_model() would otherwise be consulted (and may raise) —
    # this test pins the router-driven unsafe-candidate demotion contract.
    monkeypatch.setattr(
        "core.brain.llm.local_code_model.get_local_code_model", lambda: None
    )
    # LLM code-gen is env-gated now; this test exists to prove the unsafe
    # candidate is demoted WHEN generation runs, so enable the gate.
    monkeypatch.setenv("AURA_RSI_ENABLE_LLM_CODEGEN", "1")

    source, metadata = generate_solver_source({"gcd"}, generation_id="Aura-G2")

    assert metadata["fallback_flag"] is True
    assert metadata["sandbox_result"]["pass"] is True
    assert metadata["attempts"]
    assert "static_validation" in str(metadata["attempts"][0]["sandbox_result"])
    assert "import os" not in source
