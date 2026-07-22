"""CP126 hardening contracts for core/brain/symbolic_sandbox.py.

Covers timeout/round validation, the shared repair deadline, capture bounds and
truncation-omission proof, multi-fence concatenation, and sanitization of
untrusted execution output before it reaches a repair generator. The subprocess
path is faked — no real code is executed here.
"""
from __future__ import annotations

import math

import pytest

import core.brain.symbolic_sandbox as ss
from core.brain.symbolic_sandbox import (
    SandboxResult,
    SymbolicSandbox,
    _bound_capture,
    _clamp_timeout,
    _safe_diagnostic,
    _strip_fence,
)

# ── fa63fa56: timeout / round validation ───────────────────────────────────


@pytest.mark.parametrize("value,expected", [(math.nan, 12.0), (9999, 300.0), (-5, 0.1), (8.0, 8.0), ("x", 12.0)])
def test_clamp_timeout(value, expected):
    assert _clamp_timeout(value) == expected


@pytest.mark.asyncio
async def test_zero_rounds_runs_nothing():
    sbx = SymbolicSandbox()
    calls = {"repair": 0}

    async def repair(code, tb):
        calls["repair"] += 1
        return "print(1)"

    async def _fail_run(code, **kw):
        calls["run"] = calls.get("run", 0) + 1
        return SandboxResult(ok=False, traceback="boom")

    sbx.run = _fail_run  # type: ignore[assignment]
    res = await sbx.run_with_self_correction("print(1/0)", repair, max_rounds=0)
    assert res.refused is True
    assert calls["repair"] == 0 and "run" not in calls


# ── 368450d3: multiple code fences are concatenated, not dropped ───────────


def test_multiple_fences_concatenated():
    code = "```python\na = 1\n```\nchatter\n```py\nb = 2\n```"
    out = _strip_fence(code)
    assert "a = 1" in out and "b = 2" in out


def test_no_fence_returns_raw():
    assert _strip_fence("print(1)") == "print(1)"


# ── 25836389: untrusted diagnostics are sanitized before repair ────────────


def test_safe_diagnostic_wraps_bounds_and_strips():
    dirty = "ZeroDivisionError\x00\x07: division" + ("Z" * 20000)
    out = _safe_diagnostic(dirty)
    assert out.startswith("[UNTRUSTED SANDBOX DIAGNOSTIC")
    assert "\x00" not in out and "\x07" not in out
    assert "ZeroDivisionError" in out  # content preserved
    assert len(out) < 20000  # bounded


@pytest.mark.asyncio
async def test_repair_receives_sanitized_diagnostic():
    sbx = SymbolicSandbox()
    seen = {}

    async def repair(code, tb):
        seen["tb"] = tb
        return "print(1)"  # still runs (faked) as failing below

    async def _fail_run(code, **kw):
        return SandboxResult(ok=False, traceback="RuntimeError\x00: bad")

    sbx.run = _fail_run  # type: ignore[assignment]
    await sbx.run_with_self_correction("x", repair, max_rounds=2)
    assert seen["tb"].startswith("[UNTRUSTED SANDBOX DIAGNOSTIC")
    assert "\x00" not in seen["tb"]


# ── a560942d + 6a5123fc: capture bound + omission proof ────────────────────


def test_bound_capture_truncates_and_reports_length():
    text = "A" * 200000
    bounded, original = _bound_capture(text)
    assert original == 200000
    assert len(bounded) <= ss._MAX_CAPTURE


def test_to_dict_reports_truncation_and_digest():
    res = SandboxResult(ok=True, stdout="X" * 2000, stdout_bytes=2000)
    d = res.to_dict()
    assert d["stdout_truncated"] is True
    assert d["stdout_total_bytes"] == 2000
    assert len(d["stdout_sha256"]) == 64


# ── b427cc4d: rounds share one absolute deadline ───────────────────────────


@pytest.mark.asyncio
async def test_shared_deadline_stops_the_loop(monkeypatch):
    sbx = SymbolicSandbox(timeout=1.0)

    async def repair(code, tb):
        return f"different-{tb[:3]}"

    async def _fail_run(code, **kw):
        return SandboxResult(ok=False, traceback="fail")

    sbx.run = _fail_run  # type: ignore[assignment]

    # Clock calls: deadline calc, round0 top, round0 mid-check (all t=0), then
    # round1 top jumps past the shared deadline so the loop stops as timed_out.
    ticks = iter([0.0, 0.0, 0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(ss.time, "monotonic", lambda: next(ticks, 100.0))
    res = await sbx.run_with_self_correction("x", repair, max_rounds=5)
    assert res.timed_out is True
    assert any("deadline" in w for w in res.warnings)
