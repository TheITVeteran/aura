#!/usr/bin/env python3
"""Kernel boot diagnostic CLI.

This tool lives outside runtime core because it is an operator diagnostic, not
part of Aura's boot path. It returns a structured result, records degradations
with recovery actions, and fails closed when the kernel cannot boot or when the
LLM organ resolves to a non-live fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger("Aura.FinalKernelDiagnostic")

_DIAGNOSTIC_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


@dataclass
class KernelDiagnosticResult:
    ok: bool
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_final_diagnostic_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "final_kernel_diagnostic",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=severity in {"degraded", "critical"},
        extra=extra,
    )


async def run_final_kernel_diagnostic(
    *,
    service_container_factory: Callable[[], Any] | None = None,
    cognitive_service_registrar: Callable[[Any], Any] | None = None,
    kernel_factory: Callable[[Any, Any], Any] | None = None,
    vault_factory: Callable[[], Any] | None = None,
    config_factory: Callable[[], Any] | None = None,
) -> KernelDiagnosticResult:
    """Boot a kernel in-memory and verify the LLM organ resolves to a live backend."""
    kernel = None
    try:
        if service_container_factory is None:
            from core.container import ServiceContainer

            service_container_factory = ServiceContainer
        if cognitive_service_registrar is None:
            from core.providers.cognitive_provider import register_cognitive_services

            cognitive_service_registrar = register_cognitive_services
        if config_factory is None:
            from core.kernel.aura_kernel import KernelConfig

            config_factory = KernelConfig
        if vault_factory is None:
            from core.state.state_repository import StateRepository

            def vault_factory() -> Any:
                return StateRepository(db_path=":memory:", is_vault_owner=True)

        if kernel_factory is None:
            from core.kernel.aura_kernel import AuraKernel

            def kernel_factory(config: Any, vault: Any) -> Any:
                return AuraKernel(config=config, vault=vault)

        container = service_container_factory()
        cognitive_service_registrar(container)

        config = config_factory()
        vault = vault_factory()
        kernel = kernel_factory(config, vault)

        boot_report = await kernel.boot()
        llm = getattr(kernel, "organs", {}).get("llm")
        if llm is None:
            return KernelDiagnosticResult(
                ok=False,
                status="missing_llm_organ",
                details={"boot_report": boot_report},
                error="LLM organ was not registered after kernel boot.",
            )

        instance = llm.get_instance()
        instance_class = instance.__class__.__name__
        if "MockLLM" in instance_class:
            return KernelDiagnosticResult(
                ok=False,
                status="fallback_llm_resolved",
                details={
                    "boot_report": boot_report,
                    "llm_instance_class": instance_class,
                },
                error="LLM organ resolved to MockLLM instead of a live backend.",
            )

        return KernelDiagnosticResult(
            ok=True,
            status="kernel_boot_live_llm_verified",
            details={
                "boot_report": boot_report,
                "llm_instance_class": instance_class,
            },
        )
    except _DIAGNOSTIC_RECOVERABLE_ERRORS as exc:
        _record_final_diagnostic_degradation(
            exc,
            action="diagnostic failed closed and returned a structured operator result",
            severity="degraded",
        )
        logger.exception("Final kernel diagnostic failed closed: %s", exc)
        return KernelDiagnosticResult(
            ok=False,
            status="diagnostic_failed_closed",
            error=str(exc),
            details={"error_type": type(exc).__qualname__},
        )
    finally:
        if kernel is not None:
            try:
                await kernel.stop()
            except _DIAGNOSTIC_RECOVERABLE_ERRORS as exc:
                _record_final_diagnostic_degradation(
                    exc,
                    action="diagnostic completed but kernel cleanup degraded",
                    severity="warning",
                )
                logger.warning("Final kernel diagnostic cleanup degraded: %s", exc)


async def _main() -> int:
    logging.basicConfig(level=logging.INFO)
    result = await run_final_kernel_diagnostic()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
