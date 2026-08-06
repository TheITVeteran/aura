"""Bounded scale-out validation for self-modification candidates.

The gateway deliberately separates *requesting* more verification compute from
provisioning it.  Local validation is always available; container validation is
opt-in and network-isolated; remote providers require an explicit adapter,
worker cap, and monetary budget.  An unavailable requested provider fails
closed instead of silently falling back to weaker evidence.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.subprocess_gateway import get_subprocess_gateway


@dataclass(frozen=True)
class SandboxSweepRequest:
    candidate_root: Path
    test_targets: tuple[str, ...]
    risk_tier: int
    requested_workers: int = 1
    timeout_s: int = 120
    max_cost_usd: float = 0.0
    require_network_isolation: bool = True


@dataclass(frozen=True)
class SandboxSweepResult:
    passed: bool
    status: str
    provider: str
    worker_count: int
    attempts: int
    duration_s: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Runner = Callable[[Path, tuple[str, ...], int], Awaitable[tuple[bool, str]]]


class DistributedSandboxGateway:
    """Owns bounded local/container/remote candidate validation policy."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        max_workers: int | None = None,
        budget_usd: float | None = None,
        container_image: str | None = None,
        remote_runner: Runner | None = None,
    ) -> None:
        self.provider = (
            (provider or os.getenv("AURA_DISTRIBUTED_SANDBOX_PROVIDER", "disabled")).strip().lower()
        )
        self.max_workers = max(
            1, int(max_workers or os.getenv("AURA_DISTRIBUTED_SANDBOX_MAX_WORKERS", "2"))
        )
        self.budget_usd = max(
            0.0,
            float(
                budget_usd
                if budget_usd is not None
                else os.getenv("AURA_DISTRIBUTED_SANDBOX_BUDGET_USD", "0")
            ),
        )
        self.container_image = container_image or os.getenv(
            "AURA_DISTRIBUTED_SANDBOX_IMAGE", "python:3.12-slim"
        )
        self.remote_runner = remote_runner

    async def validate(
        self,
        request: SandboxSweepRequest,
        *,
        local_runner: Runner | None = None,
    ) -> SandboxSweepResult:
        started = time.monotonic()
        provider = self.provider
        workers = min(max(1, request.requested_workers), self.max_workers)

        if request.max_cost_usd > self.budget_usd:
            return self._result(
                started,
                provider,
                workers,
                "budget_denied",
                [
                    f"requested cost ${request.max_cost_usd:.2f} exceeds configured budget ${self.budget_usd:.2f}"
                ],
            )
        if provider in {"", "disabled", "off", "none"}:
            return self._result(
                started,
                "disabled",
                workers,
                "provider_unavailable",
                ["distributed sandbox provider is disabled"],
            )
        if provider == "local":
            if local_runner is None:
                return self._result(
                    started,
                    provider,
                    workers,
                    "runner_unavailable",
                    ["local sandbox runner was not supplied"],
                )
            return await self._run_attempts(request, provider, workers, local_runner, started)
        if provider == "container":
            if shutil.which("docker") is None:
                return self._result(
                    started,
                    provider,
                    workers,
                    "provider_unavailable",
                    ["docker executable is unavailable"],
                )
            return await self._run_attempts(
                request, provider, workers, self._container_runner, started
            )
        if provider == "remote":
            if self.remote_runner is None:
                return self._result(
                    started,
                    provider,
                    workers,
                    "provider_unavailable",
                    ["remote provider adapter is not configured"],
                )
            if self.budget_usd <= 0.0:
                return self._result(
                    started,
                    provider,
                    workers,
                    "budget_denied",
                    ["remote validation requires a positive explicit budget"],
                )
            return await self._run_attempts(request, provider, workers, self.remote_runner, started)
        return self._result(
            started,
            provider,
            workers,
            "unsupported_provider",
            [f"unsupported sandbox provider: {provider}"],
        )

    async def _run_attempts(
        self,
        request: SandboxSweepRequest,
        provider: str,
        workers: int,
        runner: Runner,
        started: float,
    ) -> SandboxSweepResult:
        evidence: list[dict[str, Any]] = []
        errors: list[str] = []
        # Multiple independent attempts catch order/flakiness problems without
        # creating unbounded process fan-out on a 64 GB workstation.
        for attempt in range(workers):
            ok, detail = await runner(
                request.candidate_root, request.test_targets, request.timeout_s
            )
            evidence.append({"attempt": attempt + 1, "passed": ok, "detail": detail[-2000:]})
            if not ok:
                errors.append(f"attempt {attempt + 1}: {detail[-500:]}")
        return SandboxSweepResult(
            passed=not errors,
            status="passed" if not errors else "failed",
            provider=provider,
            worker_count=workers,
            attempts=workers,
            duration_s=round(time.monotonic() - started, 4),
            evidence=evidence,
            errors=errors,
        )

    async def _container_runner(
        self, root: Path, targets: tuple[str, ...], timeout_s: int
    ) -> tuple[bool, str]:
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            "4g",
            "--cpus",
            "2",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=512m",
            "-v",
            f"{root}:/workspace:ro",
            "-w",
            "/workspace",
            self.container_image,
            "python",
            "-m",
            "pytest",
            "-x",
            *targets,
        ]
        result = await get_subprocess_gateway().run_async(
            command,
            capture_output=True,
            timeout=timeout_s,
            source="self_modification.distributed_sandbox.container",
            accelerator_capability="auto",
        )
        detail = (result.stdout or "") + "\n" + (result.stderr or "")
        return result.returncode == 0, detail

    @staticmethod
    def _result(
        started: float, provider: str, workers: int, status: str, errors: list[str]
    ) -> SandboxSweepResult:
        return SandboxSweepResult(
            passed=False,
            status=status,
            provider=provider,
            worker_count=workers,
            attempts=0,
            duration_s=round(time.monotonic() - started, 4),
            errors=errors,
        )


__all__ = ["DistributedSandboxGateway", "SandboxSweepRequest", "SandboxSweepResult"]
