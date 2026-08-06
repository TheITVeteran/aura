"""CP126 tests for the shared constrained-execution primitives.

Two modules run model-generated Python and both drew the same findings, so
these pin the one shared solution: the containment claim is honest, the quotas
are real, and a timeout can prove it reaped the descendant tree.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from core.runtime import constrained_exec as module
from core.runtime.constrained_exec import (
    ISOLATION_LEVEL,
    RLIMIT_OPEN_FILES,
    SAFE_ENV_KEYS,
    child_preexec,
    isolation_receipt,
    reap_process_group,
    scrubbed_env,
)


# --- the containment claim is honest -------------------------------------


def test_the_isolation_level_is_never_called_sandboxed():
    assert ISOLATION_LEVEL == "constrained_process"
    assert "sandbox" not in ISOLATION_LEVEL


def test_the_receipt_states_it_is_not_an_os_sandbox():
    receipt = isolation_receipt()

    assert receipt["os_sandbox"] is False
    assert receipt["isolation_level"] == ISOLATION_LEVEL
    assert "no filesystem" in receipt["bound"]


def test_the_receipt_declares_the_static_gate_is_advisory():
    """CP126 64b318f6: passing an AST denylist is admission, not proof."""
    assert isolation_receipt()["static_gate"] == "ast_denylist_advisory"


def test_the_receipt_can_declare_an_unenforced_limit():
    receipt = isolation_receipt(resource_limits_enforced=False)

    assert receipt["resource_limits_enforced"] is False


def test_the_receipt_names_the_quotas():
    limits = isolation_receipt()["resource_limits"]

    assert limits["cpu_s"]["requested"] == module.RLIMIT_CPU_S
    assert limits["open_files"]["requested"] == RLIMIT_OPEN_FILES
    assert limits["processes"]["requested"] == module.RLIMIT_PROCESSES
    # Each entry says whether the platform actually honours it.
    assert all("enforced" in entry for entry in limits.values())


# --- environment scrubbing ------------------------------------------------


def test_only_safe_keys_are_inherited(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leak")
    monkeypatch.setenv("HTTP_PROXY", "http://evil")
    monkeypatch.setenv("HOME", str(tmp_path))

    env = scrubbed_env()

    for leaked in ("AWS_SECRET_ACCESS_KEY", "HTTP_PROXY", "HOME"):
        assert leaked not in env
    assert set(env) <= set(SAFE_ENV_KEYS) | {
        "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "AURA_SANDBOX", "PATH",
    }


def test_extra_keys_can_be_added_explicitly():
    env = scrubbed_env(HOME="/tmp/scratch", TMPDIR="/tmp/scratch")

    assert env["HOME"] == "/tmp/scratch"
    assert env["TMPDIR"] == "/tmp/scratch"


def test_a_child_really_cannot_see_a_secret(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "leak-me")

    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c",
         "import os; print(os.environ.get('SUPER_SECRET_TOKEN', 'ABSENT'))"],
        capture_output=True, text=True, timeout=30, env=scrubbed_env(),
    )

    assert "ABSENT" in completed.stdout
    assert "leak-me" not in completed.stdout


# --- kernel-enforced quotas ----------------------------------------------


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_the_descriptor_limit_is_actually_lowered():
    probe = "import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])"
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True, text=True, timeout=30,
        preexec_fn=child_preexec, env=scrubbed_env(),
    )

    assert completed.returncode == 0
    assert int(completed.stdout.strip()) <= RLIMIT_OPEN_FILES


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_the_address_space_limit_is_not_claimed_where_it_does_not_bind():
    """Darwin accepts RLIMIT_AS and ignores it.

    Publishing the requested 2 GiB ceiling as if it were enforced would be the
    very defect this campaign is about, so the receipt reports what actually
    binds and the child is asked directly.
    """
    probe = "import resource; print(resource.getrlimit(resource.RLIMIT_AS)[0])"
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True, text=True, timeout=30,
        preexec_fn=child_preexec, env=scrubbed_env(),
    )
    child_limit = int(completed.stdout.strip())
    # RLIM_INFINITY is a huge positive sentinel, not a negative number.
    binds = 0 < child_limit <= module.RLIMIT_ADDRESS_SPACE_BYTES

    claimed = isolation_receipt()["resource_limits"]["address_space_bytes"]["enforced"]

    # The receipt must agree with the platform, whichever way it goes.
    assert claimed == binds


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_an_enforced_limit_really_binds_in_the_child():
    """Whatever effective_limits() claims for descriptors must be true."""
    probe = "import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])"
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True, text=True, timeout=30,
        preexec_fn=child_preexec, env=scrubbed_env(),
    )

    if isolation_receipt()["resource_limits"]["open_files"]["enforced"]:
        assert int(completed.stdout.strip()) <= RLIMIT_OPEN_FILES


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_the_child_starts_its_own_session():
    probe = "import os; print(os.getpid() == os.getpgid(0))"
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True, text=True, timeout=30,
        preexec_fn=child_preexec, env=scrubbed_env(),
    )

    assert completed.stdout.strip() == "True"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_preexec_is_safe_when_already_a_session_leader():
    """start_new_session=True already calls setsid; a second call must not raise.

    Exercised in a CHILD, never in-process: child_preexec lowers RLIMIT_NPROC
    and RLIMIT_NOFILE irreversibly, so calling it here would cripple the test
    runner for every subsequent subprocess.
    """
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", "print('ok')"],
        capture_output=True, text=True, timeout=30,
        preexec_fn=child_preexec, start_new_session=True, env=scrubbed_env(),
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "ok"


# --- the reap produces evidence ------------------------------------------


def test_the_reaper_refuses_to_signal_our_own_group():
    receipt = reap_process_group(os.getpgrp())

    assert receipt["attempted"] is False
    assert "own process group" in receipt["reason"]


def test_a_missing_group_is_reported_not_guessed():
    receipt = reap_process_group(None)

    assert receipt["attempted"] is False
    assert receipt["reaped"] is None
    assert receipt["reason"]


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX only")
def test_a_real_child_tree_is_reaped_with_evidence():
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", "import time; time.sleep(30)"],
        start_new_session=True, env=scrubbed_env(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        receipt = reap_process_group(process.pid, process)

        assert receipt["attempted"] is True
        assert receipt["reaped"] is True
        assert receipt["confirmed_by"]
        assert process.poll() is not None
    finally:
        if process.poll() is None:  # pragma: no cover - cleanup guard
            process.kill()


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX only")
def test_the_reap_is_confirmed_by_exit_not_by_the_signal_result():
    """macOS answers EPERM for a zombie group; the signal cannot decide."""
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", "pass"],
        start_new_session=True, env=scrubbed_env(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    process.wait(timeout=30)

    receipt = reap_process_group(process.pid, process)

    assert receipt["reaped"] is True


# --- both callers use the shared solution --------------------------------


def test_the_sandbox_operator_uses_the_shared_primitives():
    from core.actuators import sandbox_operator

    assert sandbox_operator.ISOLATION_LEVEL is ISOLATION_LEVEL
    assert sandbox_operator._scrubbed_env is scrubbed_env
    assert sandbox_operator._reap_process_group is reap_process_group


def test_the_symbolic_sandbox_uses_the_shared_primitives():
    from core.brain import symbolic_sandbox

    assert symbolic_sandbox.ISOLATION_LEVEL is ISOLATION_LEVEL
    assert symbolic_sandbox.scrubbed_env is scrubbed_env


def test_the_symbolic_sandbox_result_carries_the_bound():
    from core.brain.symbolic_sandbox import SymbolicSandbox

    result = asyncio.run(SymbolicSandbox().run("print(2 + 2)"))
    payload = result.to_dict()

    assert payload["ok"] is True
    assert payload["isolation"]["os_sandbox"] is False
    assert payload["isolation"]["resource_limits_enforced"] is False


def test_the_symbolic_sandbox_result_omits_the_generated_code():
    """CP126 93229cf5: the serialized form carries a hash, not the source."""
    from core.brain.symbolic_sandbox import SymbolicSandbox

    # A comment: present in the SOURCE, absent from the program's output.
    payload = asyncio.run(
        SymbolicSandbox().run("# canary_in_the_source_only\nprint('ok')")
    ).to_dict()

    assert "final_code" not in payload
    assert "canary_in_the_source_only" not in str(payload)
    assert len(payload["final_code_sha256"]) == 64


def test_executing_python_is_not_labeled_read_only():
    """CP126 23199cb8: the authority label must describe the effect."""
    import inspect

    from core.brain.symbolic_sandbox import SymbolicSandbox

    source = inspect.getsource(SymbolicSandbox.run)
    code = "\n".join(
        line for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )

    assert "read_only=False" in code
    assert "read_only=True" not in code
