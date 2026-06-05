
import pytest

import tools.agi.run_live_debugging_loop as debugging_loop_module
from tools.agi.run_live_debugging_loop import (
    DebugObservation,
    PatchProposal,
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


def test_live_debugging_loop_contains_no_seeded_repair_solver():
    source = debugging_loop_module.Path(debugging_loop_module.__file__).read_text()
    forbidden = [
        "return a - b\", \"return a + b",
        "return lst[::-2]\", \"return lst[::-1]",
        "correct is",
    ]
    assert not any(pattern in source for pattern in forbidden)
