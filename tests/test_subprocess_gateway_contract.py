from __future__ import annotations

import asyncio
import sys

import pytest

from core.runtime import subprocess_gateway


def test_offline_tooling_run_requires_named_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: False)

    with pytest.raises(ValueError, match="offline subprocess tooling requires"):
        subprocess_gateway.SubprocessGateway().run(
            [sys.executable, "-c", "print('bad-source')"],
            timeout=5,
            offline_tooling=True,
            source="adhoc:test",
        )


def test_offline_tooling_run_denied_when_live_governance_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)

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


def test_offline_tooling_spawn_denied_when_live_governance_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess_gateway, "governance_runtime_active", lambda: True)

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

    async def _attempt() -> None:
        await subprocess_gateway.SubprocessGateway().spawn_async(
            [sys.executable, "-c", "print('strict-async-spawn')"],
            offline_tooling=True,
            source="maintenance_tooling:test",
        )

    with pytest.raises(subprocess_gateway.GovernanceViolation):
        asyncio.run(_attempt())
