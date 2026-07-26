"""CP126 hardening contracts for core/actuators/sandbox_operator.py.

The sandbox runs synthesized Python, so the rails matter: an AST safety gate
BEFORE anything is written or run, a confined private-mode trust root, bounded
timeout/code/output, reaped failure artifacts, no leaked local paths, and
sanitized affect-grounding evidence. A fake subprocess gateway and a fake
Heartstone keep tests hermetic — no real affect state is moved.
"""
from __future__ import annotations

import os
import time

import pytest

import core.actuators.sandbox_operator as so
from core.actuators.sandbox_operator import (
    SandboxOperator,
    _clamp_timeout,
    _safe_evidence,
)


class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeProcess:
    """Minimal Popen stand-in matching what the operator uses."""

    def __init__(self, result, timeout_exc=None):
        self._result = result
        self._timeout_exc = timeout_exc
        self.pid = 424242
        self.returncode = None
        self._communicated = False

    def communicate(self, timeout=None):
        if self._timeout_exc is not None and not self._communicated:
            self._communicated = True
            raise self._timeout_exc
        self.returncode = self._result.returncode
        return self._result.stdout, self._result.stderr

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self._result.returncode
        return self.returncode


class _FakeGateway:
    def __init__(self, result=None, timeout_exc=None):
        self.runs: list[list[str]] = []
        self.result = result or _FakeResult(0, "hello\n", "")
        self.timeout_exc = timeout_exc

    def run(self, cmd, **kwargs):
        self.runs.append(list(cmd))
        return self.result

    def spawn(self, cmd, **kwargs):
        # CP126 84fc4f9d: the operator now spawns into its own process group
        # so a timeout can reap descendants; the stub follows that contract.
        self.runs.append(list(cmd))
        return _FakeProcess(self.result, self.timeout_exc)


class _FakeHeart:
    def __init__(self):
        self.success = 0
        self.failures: list[tuple[int, str]] = []

    def on_sandbox_success(self):
        self.success += 1

    def on_sandbox_failure(self, exit_code, stderr):
        self.failures.append((exit_code, stderr))


@pytest.fixture
def op(tmp_path, monkeypatch):
    gw = _FakeGateway()
    heart = _FakeHeart()
    monkeypatch.setattr(so, "get_subprocess_gateway", lambda: gw)
    monkeypatch.setattr(so, "get_runtime_service", lambda name, default=None: default)
    monkeypatch.setattr("core.affect.heartstone_values.get_heartstone_values", lambda: heart)
    operator = SandboxOperator(sandbox_dir=str(tmp_path / "sbx"))
    operator._gw = gw  # type: ignore[attr-defined]
    operator._heart = heart  # type: ignore[attr-defined]
    return operator


# ── 1e0d7a7e: AST validation before execution ──────────────────────────────


@pytest.mark.parametrize("code", ["import os\nos.system('x')", "open('/etc/passwd')", "eval('1')", "import subprocess"])
def test_unsafe_code_is_refused_without_running(op, code):
    res = op.execute_synthesized_tool(code)
    assert res["success"] is False and res.get("refused") is True
    assert op._gw.runs == []  # never reached a subprocess
    assert op._heart.success == 0 and op._heart.failures == []  # refusal moves no affect


def test_safe_code_runs(op):
    res = op.execute_synthesized_tool("print('hello world')")
    assert res["success"] is True
    assert op._gw.runs  # executed


# ── 0a9cff03: code and output size limits ──────────────────────────────────


def test_oversize_code_is_refused(op, monkeypatch):
    monkeypatch.setattr(so, "_MAX_CODE_BYTES", 20)
    res = op.execute_synthesized_tool("print('x' * 1000000000)  # long line to exceed limit")
    assert res["success"] is False and res.get("refused") is True


def test_output_is_bounded(op):
    op._gw.result = _FakeResult(0, "A" * 500000, "")
    res = op.execute_synthesized_tool("print('ok')")
    assert len(res["stdout"]) < 500000


# ── c43940a6: timeout is finite and bounded ────────────────────────────────


@pytest.mark.parametrize("value,expected", [(float("nan"), 10.0), (99999, 300.0), (-5, 1.0), (12.0, 12.0)])
def test_timeout_is_clamped(value, expected):
    assert _clamp_timeout(value) == expected


# ── 14a0bc46: no absolute local path in the result ─────────────────────────


def test_result_exposes_basename_not_abspath(op):
    res = op.execute_synthesized_tool("print('ok')")
    assert "sandbox_file" in res
    assert os.sep not in res["sandbox_file"]
    assert "file_path" not in res


# ── 2bac9d46: confined, private-mode trust root ────────────────────────────


def test_sandbox_root_is_private_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(so, "get_subprocess_gateway", lambda: _FakeGateway())
    operator = SandboxOperator(sandbox_dir=str(tmp_path / "priv"))
    mode = os.stat(operator.sandbox_dir).st_mode & 0o777
    assert mode == 0o700


# ── a2ecc66d: failed artifacts are reaped ──────────────────────────────────


def test_prune_removes_stale_artifacts(op, tmp_path, monkeypatch):
    stale = os.path.join(op.sandbox_dir, "old_fail.py")
    with open(stale, "w") as f:
        f.write("x")
    old = time.time() - (so._SANDBOX_RETENTION_S + 100)
    os.utime(stale, (old, old))
    op._prune_sandbox()
    assert not os.path.exists(stale)


# ── f96149d2: untrusted stderr is sanitized before affect ──────────────────


def test_evidence_is_sanitized_and_bounded():
    dirty = "line1\x00\x07evil" + ("Z" * 5000)
    clean = _safe_evidence(dirty)
    assert "\x00" not in clean and "\x07" not in clean
    assert len(clean) <= so._MAX_EVIDENCE_CHARS


def test_failure_grounds_sanitized_evidence(op):
    """Grounding needs a checked postcondition (CP126 0f681b67).

    This previously passed no expected_output, so it asserted that a non-zero
    EXIT CODE alone moved affect — the defect itself. The sanitization property
    it exists to protect is unchanged; it is now exercised through a verified
    outcome.
    """
    op._gw.result = _FakeResult(1, "", "boom\x00danger")
    op.execute_synthesized_tool("print('x')", expected_output="never-appears")
    assert op._heart.failures
    _code, evidence = op._heart.failures[0]
    assert "\x00" not in evidence


def test_an_exit_code_alone_does_not_ground_affect(op):
    """CP126 0f681b67: no postcondition means no evidence of task outcome."""
    op._gw.result = _FakeResult(1, "", "boom")
    result = op.execute_synthesized_tool("print('x')")

    assert op._heart.failures == []
    assert result["outcome_verified"] is False
    assert result["affect_grounded"] is False


# ── fb9c3ae3: exit-zero is not success without the expected output ─────────


def test_expected_output_postcondition(op):
    op._gw.result = _FakeResult(0, "wrong output\n", "")
    res = op.execute_synthesized_tool("print('wrong output')", expected_output="the-right-answer")
    assert res["success"] is False
    assert "postcondition" in res["stderr"]
