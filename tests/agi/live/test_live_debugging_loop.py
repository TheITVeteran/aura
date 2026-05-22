import pytest
from pathlib import Path

from tools.agi.run_live_debugging_loop import run_debugging_loop


@pytest.fixture
def broken_repository(tmp_path):
    """Spins up a temporary local repository containing a real bug and a failing test."""
    repo_dir = tmp_path / "my_broken_repo"
    repo_dir.mkdir()
    
    # 1. Create a buggy calculator file
    code_content = """def calculate(a, b):
    # BUG: correct is return a + b
    return a - b
"""
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
    # Run the live debugging loop on the broken repository
    result = await run_debugging_loop(broken_repository)

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
