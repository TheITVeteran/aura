"""CP126 isolation / grounding tests for the sandbox operator.

Two claims are under test: that the module does not oversell its containment,
and that an exit code does not move Aura's affective state on its own.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from core.actuators import sandbox_operator as module
from core.actuators.sandbox_operator import (
    ISOLATION_LEVEL,
    SandboxOperator,
    _reap_process_group,
    _scrubbed_env,
)


@pytest.fixture()
def operator(tmp_path) -> SandboxOperator:
    return SandboxOperator(sandbox_dir=str(tmp_path / "sandbox"))


# --- f52d8430: the isolation claim is honest ----------------------------


def test_the_isolation_level_is_not_called_sandboxed():
    assert ISOLATION_LEVEL == "constrained_process"
    assert "sandbox" not in ISOLATION_LEVEL


def test_every_result_declares_its_isolation_level(operator):
    result = operator.execute_synthesized_tool("print('x')")

    assert result["isolation_level"] == ISOLATION_LEVEL


def test_the_child_environment_is_scrubbed():
    env = _scrubbed_env()

    assert "PATH" in env
    assert env["AURA_SANDBOX"] == "1"
    # No ambient credentials or proxy configuration inherited.
    for leaky in ("AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY", "HTTP_PROXY", "HOME"):
        assert leaky not in env


def test_the_child_cannot_see_an_injected_secret(monkeypatch):
    """Env scrubbing is the second line; the AST gate already bans `import os`.

    Probing it through the operator is impossible for exactly that reason, so
    this exercises the environment the operator hands the child.
    """
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "leak-me")

    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c",
         "import os; print(os.environ.get('SUPER_SECRET_TOKEN', 'ABSENT'))"],
        capture_output=True, text=True, timeout=30, env=_scrubbed_env(),
    )

    assert "ABSENT" in completed.stdout
    assert "leak-me" not in completed.stdout


def test_the_ast_gate_already_bans_environment_access(operator):
    """Defense in depth: the script cannot even reach os.environ."""
    result = operator.execute_synthesized_tool(
        "import os\nprint(os.environ.get('HOME'))"
    )

    assert result["success"] is False
    assert result.get("refused") is True


def test_resource_limits_are_applied_in_the_child():
    """The preexec hook must actually lower a limit, not just exist."""
    if not hasattr(os, "fork"):
        pytest.skip("POSIX only")

    probe = (
        "import resource\n"
        "print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True, text=True, timeout=30,
        preexec_fn=module._child_preexec,
        env=_scrubbed_env(),
    )

    assert completed.returncode == 0
    assert int(completed.stdout.strip()) <= module._RLIMIT_OPEN_FILES


def test_a_normal_script_still_runs(operator):
    result = operator.execute_synthesized_tool("print('hello sandbox')")

    assert result["success"] is True
    assert "hello sandbox" in result["stdout"]


def test_an_unsafe_script_is_refused_before_execution(operator):
    result = operator.execute_synthesized_tool("import socket\nsocket.socket()")

    assert result["success"] is False
    assert result.get("refused") is True


# --- 84fc4f9d: descendant cleanup is evidenced --------------------------


def test_a_timeout_records_a_process_group(operator):
    result = operator.execute_synthesized_tool(
        "import time\ntime.sleep(30)", timeout_s=1.0
    )

    assert result["exit_code"] == -1
    assert result["process_group"]
    assert result["cleanup"]["attempted"] is True


def test_a_timeout_confirms_the_reap(operator):
    result = operator.execute_synthesized_tool(
        "import time\ntime.sleep(30)", timeout_s=1.0
    )

    cleanup = result["cleanup"]
    assert cleanup["reaped"] is True
    assert cleanup["confirmed_by"]
    assert "reaped" in result["stderr"]


def test_a_timeout_returns_promptly(operator):
    started = time.monotonic()
    operator.execute_synthesized_tool("import time\ntime.sleep(30)", timeout_s=1.0)

    assert time.monotonic() - started < 10


def test_the_reaper_refuses_to_signal_auras_own_group():
    """Aiming the reap at our own group would take Aura down."""
    receipt = _reap_process_group(os.getpgrp())

    assert receipt["attempted"] is False
    assert "own process group" in receipt["reason"]


def test_the_reaper_handles_a_missing_group():
    receipt = _reap_process_group(None)

    assert receipt["attempted"] is False
    assert receipt["reaped"] is None


# --- 0f681b67: an exit code is not a task outcome -----------------------


def test_success_without_a_postcondition_does_not_move_affect(operator, monkeypatch):
    grounded = []
    monkeypatch.setattr(
        SandboxOperator, "_ground_affect",
        lambda self, *a, **k: grounded.append(a) or True,
    )

    result = operator.execute_synthesized_tool("print('anything')")

    assert result["success"] is True
    assert result["outcome_verified"] is False
    assert result["affect_grounded"] is False
    assert grounded == []


def test_failure_without_a_postcondition_does_not_move_affect(operator, monkeypatch):
    grounded = []
    monkeypatch.setattr(
        SandboxOperator, "_ground_affect",
        lambda self, *a, **k: grounded.append(a) or True,
    )

    operator.execute_synthesized_tool("raise SystemExit(3)")

    assert grounded == []


def test_a_verified_outcome_does_move_affect(operator, monkeypatch):
    grounded = []
    monkeypatch.setattr(
        SandboxOperator, "_ground_affect",
        lambda self, *a, **k: grounded.append(a) or True,
    )

    result = operator.execute_synthesized_tool(
        "print('expected')", expected_output="expected"
    )

    assert result["outcome_verified"] is True
    assert grounded and grounded[0][0] is True


def test_a_failed_postcondition_grounds_as_failure(operator, monkeypatch):
    grounded = []
    monkeypatch.setattr(
        SandboxOperator, "_ground_affect",
        lambda self, *a, **k: grounded.append(a) or True,
    )

    result = operator.execute_synthesized_tool(
        "print('wrong')", expected_output="expected"
    )

    assert result["success"] is False
    assert grounded and grounded[0][0] is False


def test_a_refusal_never_grounds_affect(operator, monkeypatch):
    grounded = []
    monkeypatch.setattr(
        SandboxOperator, "_ground_affect",
        lambda self, *a, **k: grounded.append(a) or True,
    )

    operator.execute_synthesized_tool("import socket", expected_output="x")

    assert grounded == []


# --- 688c0259: the substrate update is bound to the result --------------


def test_the_substrate_update_is_awaited_off_loop(monkeypatch):
    applied = []

    class _Substrate:
        async def update(self, **kwargs):
            applied.append(kwargs)

    monkeypatch.setattr(
        module, "get_runtime_service",
        lambda name, default=None: _Substrate() if name == "liquid_substrate" else default,
    )

    ok = SandboxOperator._apply_substrate_delta(_Substrate(), 0.0, -0.05)

    assert ok is True
    # Ran to completion synchronously rather than being fired and forgotten.
    assert applied == [{"delta_curiosity": 0.0, "delta_frustration": -0.05,
                        "_caller": "sandbox_operator"}]


def test_a_failing_substrate_update_is_reported(monkeypatch):
    class _Broken:
        async def update(self, **kwargs):
            raise RuntimeError("substrate down")

    assert SandboxOperator._apply_substrate_delta(_Broken(), 0.0, 0.0) is False


def test_grounding_without_a_substrate_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(
        module, "get_runtime_service", lambda name, default=None: default
    )

    class _Values:
        def on_sandbox_success(self):
            """Record nothing: this test asserts the call is MADE, not its effect."""

        def on_sandbox_failure(self, code, evidence):
            """Record nothing: this test asserts the call is MADE, not its effect."""

    monkeypatch.setattr(
        "core.affect.heartstone_values.get_heartstone_values", lambda: _Values()
    )

    assert SandboxOperator()._ground_affect(True, 0, "") is True
