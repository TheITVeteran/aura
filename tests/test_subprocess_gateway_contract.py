from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from core.runtime import subprocess_gateway

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
