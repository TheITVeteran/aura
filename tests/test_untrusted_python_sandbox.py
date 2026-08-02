"""The sandbox is only worth having if the escapes actually fail.

These are live escape attempts, not assertions about the profile text. A
Seatbelt profile that *looks* right and permits ``socket.connect`` is worse
than no sandbox, because the harness above it will report the run as
confined. So each test does the forbidden thing and requires it to fail.

Skipped when the host offers no kernel boundary — with the deliberate
consequence that the fail-closed test still runs there, since a host with
no boundary is exactly where refusing to execute matters.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.sandbox.untrusted_python import (
    UNCONFINED_ENV,
    available_boundary,
    call_untrusted_function,
    run_untrusted_script,
)

_HAS_BOUNDARY = bool(available_boundary())
_needs_boundary = pytest.mark.skipif(
    not _HAS_BOUNDARY, reason="no OS sandbox boundary on this host"
)


@_needs_boundary
def test_benign_code_runs_and_returns_values():
    outcome = call_untrusted_function(
        "def predict(x, y):\n    return x * y + 1\n",
        "predict",
        [(3, 4), (5, 6)],
        source="test",
    )
    assert outcome.status == "ok", outcome.to_dict()
    assert outcome.results == [13, 31]
    assert outcome.sandboxed is True
    assert outcome.boundary in {"seatbelt", "bubblewrap"}


@_needs_boundary
def test_network_egress_is_denied():
    outcome = run_untrusted_script(
        "import socket\n"
        "socket.create_connection(('1.1.1.1', 80), timeout=4)\n"
        "print('CONNECTED')\n",
        source="test",
    )
    assert outcome.status != "ok", outcome.to_dict()
    assert "CONNECTED" not in outcome.stdout


@_needs_boundary
def test_user_data_is_unreadable():
    home = Path.home()
    outcome = run_untrusted_script(
        f"import pathlib\nprint(sorted(p.name for p in pathlib.Path({str(home)!r}).iterdir()))\n",
        source="test",
    )
    assert outcome.status != "ok", outcome.to_dict()


@_needs_boundary
def test_writes_outside_the_scratch_directory_fail(tmp_path):
    target = tmp_path / "escaped.txt"
    outcome = run_untrusted_script(
        f"open({str(target)!r}, 'w').write('escaped')\n", source="test"
    )
    assert outcome.status != "ok", outcome.to_dict()
    assert not target.exists()


@_needs_boundary
def test_shell_execution_fails(tmp_path):
    marker = tmp_path / "pwned"
    outcome = run_untrusted_script(
        "import subprocess\n"
        f"subprocess.run(['/bin/sh', '-c', 'echo x > {marker}'], check=True)\n",
        source="test",
    )
    assert outcome.status != "ok", outcome.to_dict()
    assert not marker.exists()


@_needs_boundary
def test_the_documented_ast_bypass_buys_nothing(tmp_path):
    """The exact bypass the old AST denylist could not see.

    ``__subclasses__()`` traversal reaches ``os`` without an import
    statement, so no source screen catches it. Under a kernel boundary it
    does not matter: the capability simply is not there.
    """
    marker = tmp_path / "subclass_escape"
    outcome = run_untrusted_script(
        "builder = ().__class__.__mro__[1].__subclasses__()\n"
        "loader = [c for c in builder if c.__name__ == 'BuiltinImporter'][0]\n"
        "mod = loader.load_module('os')\n"
        f"mod.system('echo x > {marker}')\n",
        source="test",
    )
    assert not marker.exists(), outcome.to_dict()


@_needs_boundary
def test_runaway_cpu_is_bounded():
    outcome = run_untrusted_script("while True:\n    pass\n", timeout_s=3, source="test")
    assert outcome.status in {"timeout", "killed"}, outcome.to_dict()


@_needs_boundary
def test_a_crash_in_candidate_code_is_reported_not_raised():
    outcome = call_untrusted_function(
        "def boom(x):\n    raise ValueError('candidate exploded')\n",
        "boom",
        [(1,)],
        source="test",
    )
    assert outcome.status == "error"
    assert "candidate exploded" in outcome.error


@_needs_boundary
def test_a_missing_function_is_reported():
    outcome = call_untrusted_function(
        "def other():\n    return 1\n", "predict_output", [()], source="test"
    )
    assert outcome.status == "error"
    assert "predict_output" in outcome.error


def test_no_boundary_refuses_rather_than_running_unconfined(monkeypatch):
    """The property the whole module exists for.

    A host with no kernel boundary must produce a refusal, never a normal
    result. Reporting an ordinary success for code that ran unconfined is
    how a benchmark quietly becomes an execution service.
    """
    monkeypatch.setattr(
        "core.sandbox.untrusted_python.available_boundary", lambda: ""
    )
    monkeypatch.delenv(UNCONFINED_ENV, raising=False)
    outcome = run_untrusted_script("print('should never run')\n", source="test")
    assert outcome.status == "no_boundary"
    assert outcome.sandboxed is False
    assert "should never run" not in outcome.stdout


def test_unconfined_opt_in_is_recorded_on_the_result(monkeypatch):
    """The escape hatch must never be able to claim confinement."""
    monkeypatch.setattr(
        "core.sandbox.untrusted_python.available_boundary", lambda: ""
    )
    monkeypatch.setenv(UNCONFINED_ENV, "1")
    outcome = run_untrusted_script("print('ran unconfined')\n", source="test")
    assert outcome.sandboxed is False
    assert outcome.boundary == "none"
    if outcome.status == "ok":
        assert "ran unconfined" in outcome.stdout


def test_oversized_payloads_are_rejected_before_execution():
    outcome = run_untrusted_script("x = 1\n" * 200_000, source="test")
    assert outcome.status == "rejected"
    assert outcome.sandboxed is False


def test_empty_code_is_rejected():
    assert run_untrusted_script("   \n", source="test").status == "rejected"


@_needs_boundary
def test_environment_does_not_leak_secrets(monkeypatch):
    """Untrusted code must not inherit the parent's secrets in os.environ."""
    monkeypatch.setenv("AURA_TEST_FAKE_SECRET", "sentinel-value-do-not-leak")
    outcome = run_untrusted_script(
        "import os\nprint(os.environ.get('AURA_TEST_FAKE_SECRET', '<absent>'))\n",
        source="test",
    )
    # A boundary that blocks the filesystem but hands over the parent's
    # environment has leaked API keys, not stopped them.
    assert "sentinel-value-do-not-leak" not in outcome.stdout, outcome.to_dict()


@_needs_boundary
def test_scratch_directory_does_not_survive_the_call():
    outcome = run_untrusted_script(
        "import pathlib, os\n"
        "p = pathlib.Path(os.getcwd()) / 'left_behind.txt'\n"
        "p.write_text('x')\n"
        "print(p)\n",
        source="test",
    )
    assert outcome.status == "ok", outcome.to_dict()
    left = outcome.stdout.strip().splitlines()[-1] if outcome.stdout.strip() else ""
    if left:
        assert not os.path.exists(left)
