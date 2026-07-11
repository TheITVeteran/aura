from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from core.runtime import shutdown_coordinator, subprocess_gateway
from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_offline_tooling_run_requires_named_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    with pytest.raises(ValueError, match="offline subprocess tooling requires"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('bad-source')"],
            timeout=5,
            offline_tooling=True,
            source="adhoc:test",
        )


def test_read_only_run_requires_attributable_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    with pytest.raises(ValueError, match="read-only subprocess probes require"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('anonymous')"],
            timeout=5,
            read_only=True,
        )


def test_read_only_run_rejects_multiline_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    with pytest.raises(ValueError, match="single-line"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('bad-source')"],
            timeout=5,
            read_only=True,
            source="test\nspoof",
        )


def test_read_only_run_allows_named_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('named')"],
        timeout=5,
        read_only=True,
        source="test.subprocess_gateway.read_only",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "named"


def test_shutdown_latch_blocks_effectful_subprocess_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    clear_shutdown_request()
    try:
        request_shutdown("unit-test")
        with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
            subprocess_gateway.SubprocessGateway().run(
                [sys.executable, "-c", "print('must-not-run')"],
                timeout=5,
                source="test.subprocess_gateway.shutdown_effectful_run",
            )
    finally:
        clear_shutdown_request()


def test_shutdown_latch_blocks_implicit_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    request_shutdown("unit-test")
    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('must-not-run')"],
            timeout=5,
            read_only=True,
            source="test.subprocess_gateway.shutdown_implicit_read_only_probe",
        )


def test_shutdown_latch_allows_explicit_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    request_shutdown("unit-test")
    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('probe-ok')"],
        timeout=5,
        read_only=True,
        allow_during_shutdown=True,
        source="test.subprocess_gateway.shutdown_read_only_probe",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "probe-ok"


def test_shutdown_latch_never_allows_live_process_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    request_shutdown("unit-test")

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        subprocess_gateway.SubprocessGateway().spawn(
            [sys.executable, "-c", "print('must-not-run')"],
            read_only=True,
            allow_during_shutdown=True,
            source="test.subprocess_gateway.shutdown_live_handle",
        )


@pytest.mark.asyncio
async def test_global_resource_fence_allows_only_bounded_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.runtime.runtime_hygiene import RuntimeHygieneManager

    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())
    request_shutdown("unit-test")

    result = await subprocess_gateway.SubprocessGateway().run_async(
        [sys.executable, "-c", "print('bounded-probe-ok')"],
        timeout=5,
        read_only=True,
        allow_during_shutdown=True,
        source="test.subprocess_gateway.shutdown_bounded_probe",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "bounded-probe-ok"
    assert shutdown_coordinator.shutdown_admission_snapshot()["counts"][
        "allowed_read_only"
    ] >= 1
    await hygiene.stop()


def test_shutdown_latch_never_allows_effectful_offline_tooling_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    request_shutdown("unit-test")

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('must-not-run')"],
            timeout=5,
            offline_tooling=True,
            allow_during_shutdown=True,
            source="maintenance_tooling:test_shutdown_effectful_override",
        )


def test_shutdown_latch_blocks_shell_spawn_before_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _must_not_spawn(*_args, **_kwargs):
        raise AssertionError("shell subprocess creation reached after shutdown latch")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _must_not_spawn)
    request_shutdown("unit-test")

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_shell_async(
            "printf must-not-run",
            source="test.subprocess_gateway.shutdown_shell_spawn",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        asyncio.run(_attempt())


def test_async_spawn_terminates_child_when_shutdown_crosses_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    class _Process:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            return 0

    process = _Process()

    async def _spawn(*_args, **_kwargs):
        request_shutdown("crossed-create-subprocess-exec")
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('must-not-survive')"],
            source="test.subprocess_gateway.crossed_async_spawn",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="runtime shutdown"):
        asyncio.run(_attempt())
    assert process.terminated is True
    assert process.killed is False


def test_read_only_spawn_async_requires_attributable_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('anonymous')"],
            read_only=True,
        )

    with pytest.raises(ValueError, match="read-only subprocess probes require"):
        asyncio.run(_attempt())


def test_read_only_spawn_async_allows_named_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _attempt() -> str:
        proc = await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('named-async')"],
            stdout=asyncio.subprocess.PIPE,
            read_only=True,
            source="test.subprocess_gateway.read_only_async",
        )
        stdout, _stderr = await proc.communicate()
        return stdout.decode("utf-8").strip()

    assert asyncio.run(_attempt()) == "named-async"


def test_spawn_async_registers_gateway_process_with_runtime_hygiene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    registrations: list[dict[str, object]] = []

    def _register(proc, *, kind, source, command) -> None:
        registrations.append(
            {
                "pid": getattr(proc, "pid", None),
                "kind": kind,
                "source": source,
                "command": tuple(command),
            }
        )

    monkeypatch.setattr(subprocess_gateway, "_register_runtime_hygiene_process", _register)

    async def _attempt() -> str:
        proc = await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('registered-async')"],
            stdout=asyncio.subprocess.PIPE,
            read_only=True,
            source="test.subprocess_gateway.register_async",
        )
        stdout, _stderr = await proc.communicate()
        return stdout.decode("utf-8").strip()

    assert asyncio.run(_attempt()) == "registered-async"
    assert registrations
    assert registrations[0]["pid"]
    assert registrations[0]["kind"] == "subprocess"
    assert registrations[0]["source"] == "test.subprocess_gateway.register_async"


def test_spawn_shell_async_registers_gateway_process_with_runtime_hygiene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    registrations: list[dict[str, object]] = []

    def _register(proc, *, kind, source, command) -> None:
        registrations.append(
            {
                "pid": getattr(proc, "pid", None),
                "kind": kind,
                "source": source,
                "command": command,
            }
        )

    monkeypatch.setattr(subprocess_gateway, "_register_runtime_hygiene_process", _register)

    async def _attempt() -> str:
        proc = await subprocess_gateway.SubprocessGateway().spawn_shell_async(
            f"{sys.executable} -c \"print('registered-shell-async')\"",
            stdout=asyncio.subprocess.PIPE,
            source="test.subprocess_gateway.register_shell_async",
        )
        stdout, _stderr = await proc.communicate()
        return stdout.decode("utf-8").strip()

    assert asyncio.run(_attempt()) == "registered-shell-async"
    assert registrations
    assert registrations[0]["pid"]
    assert registrations[0]["kind"] == "subprocess"
    assert registrations[0]["source"] == "test.subprocess_gateway.register_shell_async"


def test_read_only_run_async_requires_attributable_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().run_async(
            [sys.executable, "-c", "print('anonymous')"],
            read_only=True,
        )

    with pytest.raises(ValueError, match="read-only subprocess probes require"):
        asyncio.run(_attempt())


def test_read_only_run_async_allows_named_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    async def _attempt() -> str:
        result = await subprocess_gateway.SubprocessGateway().run_async(
            [sys.executable, "-c", "print('named-run-async')"],
            read_only=True,
            source="test.subprocess_gateway.read_only_run_async",
        )
        return result.stdout.strip()

    assert asyncio.run(_attempt()) == "named-run-async"


def test_spawn_shell_async_denied_when_live_governance_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_shell_async(
            f"{sys.executable} -c \"print('strict-shell')\"",
            source="test.subprocess_gateway.shell",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        asyncio.run(_attempt())


def test_offline_tooling_run_denied_when_live_governance_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('strict')"],
            timeout=5,
            offline_tooling=True,
            source="proof_tooling:test",
        )


def test_offline_tooling_run_allowed_for_approved_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('ok')"],
        timeout=5,
        offline_tooling=True,
        source="proof_tooling:test",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_desktop_safe_run_blocks_proof_scale_environment_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_ALLOW_DESKTOP_LONGRUNS", raising=False)

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="desktop-safe long-run"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('nethack_challenge.py should-not-run')"],
            timeout=5,
            read_only=True,
            source="test.subprocess_gateway.desktop_guard",
        )


def test_desktop_safe_run_allows_explicit_operator_longrun_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_ALLOW_DESKTOP_LONGRUNS", "1")

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('nethack_challenge.py override-ok')"],
        timeout=5,
        read_only=True,
        source="test.subprocess_gateway.desktop_guard_override",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "nethack_challenge.py override-ok"


def test_desktop_safe_shell_spawn_blocks_proof_batteries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.delenv("AURA_ALLOW_DESKTOP_LONGRUNS", raising=False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_shell_async(
            f"{sys.executable} -c \"print('run_dnu_agi_proof_battery.py should-not-run')\"",
            source="test.subprocess_gateway.desktop_shell_guard",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation, match="desktop-safe long-run"):
        asyncio.run(_attempt())


def test_nethack_runner_rejects_env_only_longrun_bypass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    env = os.environ.copy()
    env.update(
        {
            "AURA_ALLOW_DESKTOP_NETHACK": "1",
            "AURA_ALLOW_DESKTOP_LONGRUNS": "1",
            "AURA_ALLOW_LONG_NETHACK_RUN": "1",
            "AURA_NETHACK_STEPS": "100000",
            "AURA_NETHACK_LONG_RUN_CONFIRM_FILE": str(tmp_path / "missing-confirmation"),
        }
    )
    env.pop("AURA_NETHACK_UNSAFE_RAM_CONFIRM", None)

    result = subprocess_gateway.SubprocessGateway().run(
        ["bash", "scripts/nethack_runner.sh"],
        cwd=PROJECT_ROOT,
        env=env,
        timeout=5,
        capture_output=True,
        source="test.subprocess_gateway.nethack_runner_guard",
    )

    assert result.returncode == 64
    runner_log = Path.home() / ".aura/logs/nethack/runner.log"
    assert "without one-shot confirmation file" in runner_log.read_text(encoding="utf-8")


def test_offline_tooling_spawn_denied_when_live_governance_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        subprocess_gateway.SubprocessGateway().spawn(
            [sys.executable, "-c", "print('strict-spawn')"],
            offline_tooling=True,
            source="training_tooling:test",
        )


def test_offline_tooling_spawn_async_denied_when_live_governance_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('strict-async-spawn')"],
            offline_tooling=True,
            source="maintenance_tooling:test",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        asyncio.run(_attempt())


def test_proof_tooling_run_allowed_in_test_mode_with_live_governance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.setenv("AURA_TEST_MODE", "1")

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('proof-ok')"],
        timeout=5,
        offline_tooling=True,
        source="proof_tooling:test",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "proof-ok"


def test_proof_tooling_run_allowed_with_explicit_child_test_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.delenv("AURA_TEST_MODE", raising=False)
    child_env = os.environ.copy()
    child_env["AURA_TEST_MODE"] = "1"

    result = subprocess_gateway.SubprocessGateway().run(
        [sys.executable, "-c", "print('proof-env-ok')"],
        timeout=5,
        env=child_env,
        offline_tooling=True,
        source="certification_tooling:test",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "proof-env-ok"


def test_non_proof_tooling_still_denied_in_test_mode_with_live_governance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)
    monkeypatch.setenv("AURA_TEST_MODE", "1")

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('training-denied')"],
            timeout=5,
            offline_tooling=True,
            source="training_tooling:test",
        )


def test_spawn_routes_stdout_and_stderr_to_gateway_owned_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    stdout_path = tmp_path / "child.stdout"
    stderr_path = tmp_path / "child.stderr"

    proc = subprocess_gateway.SubprocessGateway().spawn(
        [
            sys.executable,
            "-c",
            "import sys; print('gateway-out'); print('gateway-err', file=sys.stderr)",
        ],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        source="test.subprocess_gateway.path_streams",
    )
    assert proc.wait(timeout=5) == 0
    for stream in getattr(proc, "_aura_gateway_streams", ()):
        stream.close()

    assert stdout_path.read_text(encoding="utf-8").strip() == "gateway-out"
    assert stderr_path.read_text(encoding="utf-8").strip() == "gateway-err"


def test_spawn_accepts_preexec_fn_for_resource_fenced_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)
    proc = subprocess_gateway.SubprocessGateway().spawn(
        [sys.executable, "-c", "print('preexec-supported')"],
        preexec_fn=None,
        source="test.subprocess_gateway.preexec_none",
    )
    assert proc.wait(timeout=5) == 0
