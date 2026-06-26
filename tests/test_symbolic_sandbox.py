"""Tests for the Symbolic Sandbox cognitive scratchpad."""
from __future__ import annotations

import pytest

from core.brain.symbolic_sandbox import SymbolicSandbox, get_symbolic_sandbox


@pytest.mark.asyncio
async def test_runs_pure_computation():
    sbx = SymbolicSandbox(timeout=15.0)
    res = await sbx.run("print(sum(range(10)))")
    assert res.ok
    assert res.stdout.strip() == "45"


@pytest.mark.asyncio
async def test_refuses_dangerous_import():
    sbx = SymbolicSandbox()
    res = await sbx.run("import os\nprint(os.getcwd())")
    assert not res.ok and res.refused
    assert any("os" in w for w in res.warnings)


@pytest.mark.asyncio
async def test_refuses_syntax_error():
    sbx = SymbolicSandbox()
    res = await sbx.run("def broken(:\n  pass")
    assert not res.ok and res.refused


@pytest.mark.asyncio
async def test_captures_runtime_traceback():
    sbx = SymbolicSandbox(timeout=15.0)
    res = await sbx.run("print(1/0)")
    assert not res.ok and not res.refused
    assert "ZeroDivisionError" in (res.traceback + res.stderr)


@pytest.mark.asyncio
async def test_self_correction_loop_fixes_code():
    sbx = SymbolicSandbox(timeout=15.0)

    async def repair(code: str, traceback: str) -> str:
        # The "generator" notices the division-by-zero and fixes it.
        assert "ZeroDivisionError" in traceback
        return "print(10 // 2)"

    res = await sbx.run_with_self_correction("print(10 // 0)", repair, max_rounds=3)
    assert res.ok
    assert res.stdout.strip() == "5"
    assert res.rounds == 2


@pytest.mark.asyncio
async def test_self_correction_gives_up_after_rounds():
    sbx = SymbolicSandbox(timeout=15.0)

    calls = {"n": 0}

    async def repair(code: str, traceback: str) -> str:
        calls["n"] += 1
        # Distinct-but-still-broken each time, so the loop actually retries.
        return f"x = {calls['n']}\nprint(1/0)"

    res = await sbx.run_with_self_correction("print(1/0)", repair, max_rounds=2)
    assert not res.ok
    assert res.rounds == 2


def test_singleton():
    assert get_symbolic_sandbox() is get_symbolic_sandbox()
