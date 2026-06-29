from __future__ import annotations

import asyncio
from pathlib import Path

from core.self_modification.distributed_sandbox_gateway import (
    DistributedSandboxGateway,
    SandboxSweepRequest,
)


def _request(tmp_path: Path, **overrides):
    values = {
        "candidate_root": tmp_path,
        "test_targets": ("tests/test_candidate.py",),
        "risk_tier": 1,
        "requested_workers": 2,
    }
    values.update(overrides)
    return SandboxSweepRequest(**values)


def test_disabled_requested_provider_fails_closed(tmp_path: Path) -> None:
    result = asyncio.run(
        DistributedSandboxGateway(provider="disabled").validate(_request(tmp_path))
    )
    assert result.passed is False
    assert result.status == "provider_unavailable"
    assert result.attempts == 0


def test_local_provider_runs_bounded_independent_attempts(tmp_path: Path) -> None:
    calls: list[tuple[Path, tuple[str, ...], int]] = []

    async def runner(root, targets, timeout_s):
        calls.append((root, targets, timeout_s))
        return True, "2 passed"

    gateway = DistributedSandboxGateway(provider="local", max_workers=2, budget_usd=0)
    result = asyncio.run(
        gateway.validate(_request(tmp_path, requested_workers=5), local_runner=runner)
    )
    assert result.passed is True
    assert result.worker_count == 2
    assert result.attempts == 2
    assert len(calls) == 2


def test_remote_provider_requires_explicit_budget_and_adapter(tmp_path: Path) -> None:
    gateway = DistributedSandboxGateway(provider="remote", max_workers=2, budget_usd=0)
    result = asyncio.run(gateway.validate(_request(tmp_path)))
    assert result.passed is False
    assert result.status == "provider_unavailable"

    async def remote_runner(_root, _targets, _timeout):
        return True, "remote pass"

    gateway = DistributedSandboxGateway(
        provider="remote", budget_usd=1, remote_runner=remote_runner
    )
    denied = asyncio.run(gateway.validate(_request(tmp_path, max_cost_usd=2)))
    assert denied.status == "budget_denied"
