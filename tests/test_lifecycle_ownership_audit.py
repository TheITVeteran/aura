from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from tools.closeout.audit_lifecycle_ownership import ROOT, audit


def _copy_contract_sources(tmp_path: Path) -> None:
    for relative in (
        "core/tasks/managed_command.py",
        "core/container.py",
        "core/runtime/service_registry.py",
        "core/state/state_repository.py",
        "core/runtime/process_identity.py",
        "scripts/one_off/aura_cleanup.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")


def test_repository_lifecycle_ownership_contract_is_complete() -> None:
    report = audit(ROOT)

    assert report["passed"] is True
    assert report["issues"] == []
    assert report["checked_functions"] >= 13
    assert report["natural_interpreter_exit_required_by_test"] is True


def test_audit_rejects_diagnostic_cold_construction(tmp_path: Path) -> None:
    _copy_contract_sources(tmp_path)
    path = tmp_path / "core/container.py"
    source = path.read_text(encoding="utf-8").replace(
        "ServiceContainer.peek(service_name, default=default)",
        "ServiceContainer.get(service_name, default=default)",
        1,
    )
    path.write_text(source, encoding="utf-8")

    report = audit(tmp_path)

    assert report["passed"] is False
    assert any("gained constructing calls" in issue for issue in report["issues"])


def test_audit_rejects_unclosed_repository_worker(tmp_path: Path) -> None:
    _copy_contract_sources(tmp_path)
    path = tmp_path / "core/state/state_repository.py"
    source = path.read_text(encoding="utf-8").replace(
        "await self._db.close()",
        "await asyncio.sleep(0)",
        1,
    )
    path.write_text(source, encoding="utf-8")

    report = audit(tmp_path)

    assert report["passed"] is False
    assert any(
        "StateRepository.close lost exact owner calls ['self._db.close']" in issue
        for issue in report["issues"]
    )


def test_audit_rejects_timeout_path_that_does_not_reap_child(tmp_path: Path) -> None:
    _copy_contract_sources(tmp_path)
    path = tmp_path / "core/tasks/managed_command.py"
    source = path.read_text(encoding="utf-8").replace(
        "stdout_bytes, stderr_bytes = await process.communicate()",
        "stdout_bytes, stderr_bytes = b'', b''",
        1,
    )
    path.write_text(source, encoding="utf-8")

    report = audit(tmp_path)

    assert report["passed"] is False
    assert "managed command timeout path no longer reaps the killed child" in report["issues"]


def test_audit_rejects_cleanup_without_pid_reuse_fence(tmp_path: Path) -> None:
    _copy_contract_sources(tmp_path)
    path = tmp_path / "scripts/one_off/aura_cleanup.py"
    source = path.read_text(encoding="utf-8").replace(
        "current_create_time",
        "unfenced_create_time",
    )
    path.write_text(source, encoding="utf-8")

    report = audit(tmp_path)

    assert report["passed"] is False
    assert "stale-runtime cleanup lost its PID-reuse creation-time fence" in report["issues"]


def test_aiosqlite_owner_closes_and_child_interpreter_exits_naturally(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "child-state.db"
    child = textwrap.dedent(
        f"""
        import asyncio
        import json
        from core.state.state_repository import StateRepository

        async def exercise():
            repo = StateRepository(db_path={str(db_path)!r}, is_vault_owner=True)
            first = await repo._ensure_db()
            second = await repo._ensure_db()
            assert first is second
            await repo.close()
            assert repo._db is None
            return {{"connection_reused": True, "repository_closed": True}}

        print(json.dumps(asyncio.run(exercise()), sort_keys=True))
        """
    )
    env = dict(os.environ)
    env.update(
        {
            "AURA_TEST_MODE": "1",
            "AURA_LOG_DIR": str(tmp_path / "logs"),
            "PYTHONPATH": str(ROOT),
        }
    )
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-c", child],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=15.0)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise AssertionError(
            "child interpreter did not exit naturally after StateRepository.close(); "
            "the test killed it only to prevent a leaked process"
        ) from exc

    assert process.returncode == 0, stderr
    assert time.monotonic() - started < 15.0
    assert json.loads(stdout.strip().splitlines()[-1]) == {
        "connection_reused": True,
        "repository_closed": True,
    }
