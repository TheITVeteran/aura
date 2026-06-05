"""Canonical dynamic execution gateway.

Dynamic code execution is only acceptable behind one auditable owner. This
gateway centralizes compile/exec calls so production code can route sandboxed
execution through a single governed surface instead of sprinkling raw exec,
eval, or compile calls through runtime modules.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from types import CodeType
from typing import Any

from core.governance_context import governance_runtime_active, require_governance

logger = logging.getLogger("Aura.DynamicExecutionGateway")

_DYNAMIC_EXECUTION_DOMAINS = (
    "tool_execution",
    "self_modification",
    "state_mutation",
)


class DynamicExecutionGateway:
    """Single owner for controlled dynamic compile/execute operations."""

    def _authorize(self, operation: str, source: str) -> None:
        if not isinstance(source, str) or not source.strip() or "\n" in source or "\r" in source:
            raise ValueError("dynamic execution source must be a non-empty single-line label")
        if governance_runtime_active():
            require_governance(
                f"dynamic_execution_gateway.{operation}:{source}",
                strict=True,
                allowed_domains=_DYNAMIC_EXECUTION_DOMAINS,
            )

    def compile_source(
        self,
        source_code: str,
        *,
        filename: str,
        mode: str = "exec",
        source: str,
        compiler: Callable[..., CodeType] | None = None,
    ) -> CodeType:
        """Compile source code through the dynamic-execution owner."""

        if not isinstance(source_code, str):
            raise TypeError("source_code must be a string")
        if mode not in {"exec", "eval", "single"}:
            raise ValueError(f"unsupported compile mode: {mode}")
        self._authorize("compile_source", source)
        compile_fn = compiler or compile
        return compile_fn(source_code, filename=filename, mode=mode)

    def execute_code_object(
        self,
        code_object: CodeType,
        *,
        globals_dict: dict[str, Any],
        locals_dict: dict[str, Any] | None = None,
        source: str,
    ) -> dict[str, Any]:
        """Execute a precompiled code object inside provided namespaces."""

        if not isinstance(code_object, CodeType):
            raise TypeError("code_object must be a compiled CodeType")
        if not isinstance(globals_dict, dict):
            raise TypeError("globals_dict must be a dict")
        if locals_dict is not None and not isinstance(locals_dict, dict):
            raise TypeError("locals_dict must be a dict when provided")
        self._authorize("execute_code_object", source)
        target_locals = locals_dict if locals_dict is not None else globals_dict
        exec(code_object, globals_dict, target_locals)
        return target_locals


_gateway: DynamicExecutionGateway | None = None


def get_dynamic_execution_gateway() -> DynamicExecutionGateway:
    global _gateway
    if _gateway is None:
        _gateway = DynamicExecutionGateway()
    return _gateway


__all__ = ["DynamicExecutionGateway", "get_dynamic_execution_gateway"]
