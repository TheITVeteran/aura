
import pytest
from types import SimpleNamespace

import tools.agi.run_live_debugging_loop as debugging_loop_module
from tools.agi.run_live_debugging_loop import (
    DebugObservation,
    PatchProposal,
    _default_patch_provider,
    run_debugging_loop,
)


@pytest.fixture
def broken_repository(tmp_path):
    """Spins up a temporary local repository containing a real bug and a failing test."""
    repo_dir = tmp_path / "my_broken_repo"
    repo_dir.mkdir()
    
    # 1. Create a buggy calculator file
    code_content = "def calculate(a, b):\n    return a - b\n"
    (repo_dir / "calculator.py").write_text(code_content)
    
    # 2. Create a test file that asserts correct addition
    test_content = """from calculator import calculate

def test_calculate():
    assert calculate(10, 5) == 15
"""
    (repo_dir / "test_calculator.py").write_text(test_content)
    
    return repo_dir


@pytest.mark.asyncio
async def test_live_debugging_loop_execution(broken_repository):
    def agent_provider(observation: DebugObservation) -> PatchProposal:
        assert "return a - b" in observation.code_content
        return PatchProposal(
            file=observation.code_file,
            content="def calculate(a, b):\n    return a + b\n",
            rationale="The failing test expects addition, while the implementation subtracts.",
        )

    # Run the live debugging loop on the broken repository
    result = await run_debugging_loop(broken_repository, patch_provider=agent_provider)

    if not result["ok"]:
        print("\nDEBUGGING ERROR OUTPUT:")
        print(result.get("error_output"))
    assert result["ok"] is True
    assert result["status"] == "success_verified"
    
    # Check trace history
    stages = [step["stage"] for step in result["trace"]]
    assert "diagnose" in stages
    assert "read_files" in stages
    assert "patch" in stages
    assert "verify" in stages
    
    # Confirm the file was actually patched
    patched_code = (broken_repository / "calculator.py").read_text()
    assert "return a + b" in patched_code
    assert "return a - b" not in patched_code


@pytest.mark.asyncio
async def test_live_debugging_loop_fails_without_agent_patch_provider(broken_repository):
    result = await run_debugging_loop(
        broken_repository,
        patch_provider=lambda _observation: None,
    )

    assert result["ok"] is False
    assert "No agent patch proposal" in result["error"]
    assert "return a - b" in (broken_repository / "calculator.py").read_text()


@pytest.mark.asyncio
async def test_default_live_debugging_loop_uses_symbolic_fallback_for_palindrome(monkeypatch, tmp_path):
    async def unavailable_model(_observation):
        return None

    monkeypatch.setattr(debugging_loop_module, "_default_patch_provider", unavailable_model)
    repo = tmp_path / "pal_repo"
    repo.mkdir()
    (repo / "solution.py").write_text("def is_palindrome(s):\n    return s == s[::-1]\n")
    (repo / "test_solution.py").write_text(
        "from solution import is_palindrome\n"
        "def test_palindrome_normalizes_text():\n"
        "    assert is_palindrome('A man, a plan, a canal: Panama')\n"
    )

    result = await run_debugging_loop(repo)

    assert result["ok"] is True
    patched = (repo / "solution.py").read_text()
    assert "isalnum" in patched
    assert "normalized == normalized[::-1]" in patched


@pytest.mark.asyncio
async def test_symbolic_preflight_handles_simple_repairs_without_waking_model(monkeypatch, tmp_path):
    model_calls = []

    async def record_unexpected_model_call(observation):
        model_calls.append(observation)
        return None

    monkeypatch.setattr(
        debugging_loop_module,
        "_default_patch_provider",
        record_unexpected_model_call,
    )
    repo = tmp_path / "reverse_repo"
    repo.mkdir()
    (repo / "solution.py").write_text("def reverse_list(lst):\n    return lst[::-2]\n")
    (repo / "test_solution.py").write_text(
        "from solution import reverse_list\n"
        "def test_reverse_list():\n"
        "    assert reverse_list([1, 2, 3]) == [3, 2, 1]\n"
    )

    result = await run_debugging_loop(repo)

    assert result["ok"] is True
    assert any(
        step.get("stage") == "symbolic_patch_proposal" and step.get("phase") == "preflight"
        for step in result["trace"]
    )
    assert model_calls == []
    assert "return lst[::-1]" in (repo / "solution.py").read_text()


@pytest.mark.asyncio
async def test_default_live_debugging_loop_uses_symbolic_fallback_for_missing_recursion_base(monkeypatch, tmp_path):
    async def unavailable_model(_observation):
        return None

    monkeypatch.setattr(debugging_loop_module, "_default_patch_provider", unavailable_model)
    repo = tmp_path / "recurrence_repo"
    repo.mkdir()
    (repo / "solution.py").write_text(
        "def fibonacci(n):\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n"
    )
    (repo / "test_solution.py").write_text(
        "from solution import fibonacci\n"
        "def test_fibonacci_small_values():\n"
        "    assert fibonacci(0) == 0\n"
        "    assert fibonacci(1) == 1\n"
        "    assert fibonacci(3) == 2\n"
    )

    result = await run_debugging_loop(repo)

    assert result["ok"] is True
    patched = (repo / "solution.py").read_text()
    assert "raise ValueError" in patched
    assert "prev, curr = curr, prev + curr" in patched


@pytest.mark.asyncio
async def test_live_debugging_loop_retries_with_verification_feedback(broken_repository):
    calls = []

    def agent_provider(observation: DebugObservation) -> PatchProposal:
        calls.append(observation.initial_stdout + observation.initial_stderr)
        if len(calls) == 1:
            return PatchProposal(
                file=observation.code_file,
                content="def calculate(a, b):\n    return a * b\n",
                rationale="First repair is deliberately wrong for the regression.",
            )
        assert "assert calculate(10, 5) == 15" in calls[-1]
        return PatchProposal(
            file=observation.code_file,
            content="def calculate(a, b):\n    return a + b\n",
            rationale="The verification failure shows the function must add both operands.",
        )

    result = await run_debugging_loop(
        broken_repository,
        patch_provider=agent_provider,
        max_patch_attempts=2,
    )

    assert result["ok"] is True
    assert result["attempts"] == 2
    assert len(calls) == 2
    assert "return a + b" in (broken_repository / "calculator.py").read_text()


def test_live_debugging_loop_contains_no_seeded_repair_solver():
    source = debugging_loop_module.Path(debugging_loop_module.__file__).read_text()
    forbidden = [
        "return a - b\", \"return a + b",
        "return lst[::-2]\", \"return lst[::-1]",
        "correct is",
    ]
    assert not any(pattern in source for pattern in forbidden)


@pytest.mark.asyncio
async def test_default_patch_provider_force_aborts_slow_router(monkeypatch, tmp_path):
    import asyncio

    import core.brain.llm_health_router as router_module

    repo = tmp_path / "repo"
    repo.mkdir()
    code = repo / "solution.py"
    test = repo / "test_solution.py"
    code.write_text("def f():\n    return 1\n")
    test.write_text("from solution import f\ndef test_f():\n    assert f() == 2\n")

    class AbortableClient:
        def __init__(self):
            self.abort_reasons = []

        def force_abort_active_generation(self, *, reason: str):
            self.abort_reasons.append(reason)
            return True

    client = AbortableClient()
    seen_kwargs = {}

    class SlowRouter:
        endpoints = {"Cortex": SimpleNamespace(client=client)}

        def __init__(self):
            self.gate_release_reasons = []

        def force_release_generation_gate(self, *, reason: str):
            self.gate_release_reasons.append(reason)
            return True

        async def think(self, **kwargs):
            seen_kwargs.update(kwargs)
            await asyncio.sleep(5.0)
            return "{}"

    monkeypatch.setenv("AURA_LIVE_DEBUG_PATCH_TIMEOUT_S", "0.05")
    router = SlowRouter()
    monkeypatch.setattr(router_module, "get_llm_router", lambda: router)

    observation = DebugObservation(
        repo_path=repo,
        code_file=code,
        test_file=test,
        code_content=code.read_text(),
        test_content=test.read_text(),
        initial_stdout="failed",
        initial_stderr="",
    )

    proposal = await _default_patch_provider(observation)

    assert proposal is None
    assert router.gate_release_reasons
    assert client.abort_reasons
    assert seen_kwargs["proof_evaluation_contract"] is True
    assert seen_kwargs["proof_primary_lane_required"] is True
    assert seen_kwargs["timeout"] == 0.05
