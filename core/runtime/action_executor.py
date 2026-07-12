"""Canonical governed transaction boundary for consequential actions."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import subprocess
import urllib.parse
import uuid
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from core.being.body_state_service import BodyStateService
from core.being.welfare_state import WelfareState
from core.being.welfare_transaction import WelfareTransaction
from core.governance.will import ActionDomain, get_will
from core.governance_context import (
    GovernanceViolation,
    governed_scope,
    require_governance,
)
from core.memory.memory_write_gateway import get_memory_write_gateway
from core.runtime.action_verification import (
    EffectVerifier,
    capture_pre_action_state,
    default_action_expectation,
    observe_action_effect,
)
from core.runtime.desktop_action_gateway import get_desktop_action_gateway
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.flags import FlagKind, declare
from core.runtime.network_gateway import get_network_gateway
from core.runtime.post_action_receipt import (
    PostActionReceipt,
    get_post_action_receipt_store,
)
from core.runtime.skill_contract import (
    ActionExpectation,
    SkillStatus,
    apply_action_expectation_payload,
)
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.state.state_gateway import get_state_gateway

logger = logging.getLogger("Aura.ActionExecutor")

_VERIFIER_TIMEOUT_FLAG = declare(
    "AURA_ACTION_VERIFIER_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=5.0,
    description="Maximum time allowed for post-action observed-effect verification",
    owner="core.runtime.action_executor",
)
_ACTION_EXECUTOR_RECOVERABLE_ERRORS = (
    AttributeError,
    GovernanceViolation,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    subprocess.SubprocessError,
    TimeoutError,
    TypeError,
    ValueError,
)
_SENSITIVE_PARAM_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "session_id",
    "token",
)
_ACTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
EffectHandler = Callable[
    [Mapping[str, Any]],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]
_HANDLER_DOMAINS = frozenset(
    {
        ActionDomain.ENVIRONMENT_ACTION,
        ActionDomain.EXTERNAL_ACTION,
        ActionDomain.NETWORK_CALL,
    }
)


class ActionExecutor:
    """Execute, observe, and receipt one consequential action."""

    @classmethod
    async def request_network_transport(
        cls,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        data: bytes | str | None = None,
        timeout_s: float = 30.0,
        source: str,
        read_only: bool = False,
    ) -> dict[str, Any]:
        """Run network IO inside an already-approved ActionExecutor transaction."""
        source_text = str(source or "").strip()
        if not source_text.startswith("world_bridge:"):
            raise ValueError("network transport source must be owned by world_bridge")
        require_governance(
            "action_executor.request_network_transport",
            strict=True,
            allowed_domains=(
                ActionDomain.ENVIRONMENT_ACTION.value,
                ActionDomain.EXTERNAL_ACTION.value,
                ActionDomain.NETWORK_CALL.value,
            ),
        )
        result = await get_network_gateway().request_async(
            method=method,
            url=url,
            headers=headers,
            data=data,
            timeout=timeout_s,
            source=source_text,
            read_only=read_only,
        )
        if not isinstance(result, Mapping):
            return {
                "ok": False,
                "status_code": 0,
                "error": "network_gateway_returned_non_mapping_result",
            }
        return dict(result)

    @classmethod
    async def execute(
        cls,
        *,
        domain: ActionDomain | str,
        action_name: str,
        params: dict[str, Any],
        source: str = "unknown",
        predicted_welfare_delta: dict[str, float] | None = None,
        rollback_target: str | None = None,
        expectation: ActionExpectation | Mapping[str, Any] | None = None,
        effect_handler: EffectHandler | None = None,
        effect_verifier: EffectVerifier | None = None,
        execution_timeout_s: float | None = None,
        verification_timeout_s: float | None = None,
        action_id: str | None = None,
    ) -> dict[str, Any]:
        domain = _coerce_domain(domain)
        action_name = _coerce_action_name(action_name)
        params = _coerce_params(params)
        handler_name = _validate_effect_handler(
            domain,
            effect_handler=effect_handler,
            effect_verifier=effect_verifier,
        )
        execution_timeout = _coerce_execution_timeout(execution_timeout_s)
        expectation_contract = _coerce_expectation(expectation) or default_action_expectation(
            domain,
            action_name,
        )
        action_id = _coerce_action_id(action_id)
        request_digest = _stable_digest(
            {
                "domain": domain.value,
                "action_name": action_name,
                "params": params,
                "source": source,
                "effect_handler": handler_name,
                "expectation": expectation_contract.to_dict(),
            }
        )

        will = get_will()
        decision = will.decide(
            content=_safe_action_summary(action_name, params),
            source=source,
            domain=domain,
            priority=0.5,
            context={
                "action_id": action_id,
                "request_digest": request_digest,
                "expectation_objective": expectation_contract.objective[:500],
                "rollback_target_declared": bool(rollback_target),
            },
        )
        if not decision.is_approved():
            logger.warning(
                "ActionExecutor refused %s in domain %s",
                action_name,
                domain.value,
            )
            return {
                "ok": False,
                "status": SkillStatus.BLOCKED_BY_POLICY.value,
                "error": f"Will refused action: {decision.reason}",
                "will_receipt_id": decision.receipt_id,
                "action_expectation": expectation_contract.to_dict(),
                "action_id": action_id,
                "request_digest": request_digest,
                "transport_succeeded": False,
                "effect_verified": False,
                "retry_safe": False,
                "manual_reconciliation_required": False,
            }

        will_receipt_id = str(decision.receipt_id)
        body_service = BodyStateService.get()
        welfare_service = WelfareState.get()
        tx = WelfareTransaction.begin(
            domain=domain.value,
            action=f"{action_name} ({source})",
            welfare_before=welfare_service.last_outputs,
            body_before=body_service.snapshot(),
            predicted_welfare_delta=predicted_welfare_delta,
            will_receipt_id=will_receipt_id,
        )

        result: dict[str, Any]
        pre_state: dict[str, Any] = {}
        try:
            async with governed_scope(decision):
                pre_state = await capture_pre_action_state(domain, params)
                result = await cls._dispatch(
                    domain=domain,
                    action_name=action_name,
                    params=params,
                    source=source,
                    will_receipt_id=will_receipt_id,
                    expectation=expectation_contract,
                    effect_handler=effect_handler,
                    execution_timeout_s=execution_timeout,
                )
                observation = await observe_action_effect(
                    domain,
                    params,
                    result,
                    pre_state=pre_state,
                    verifier=effect_verifier,
                    verifier_timeout_s=(
                        float(verification_timeout_s)
                        if verification_timeout_s is not None
                        else float(_VERIFIER_TIMEOUT_FLAG.value())
                    ),
                )
                result.update(observation)
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor",
                exc,
                action=f"recorded failed action transaction for {action_name}",
            )
            logger.error("Error executing action %s: %s", action_name, exc, exc_info=True)
            result = {
                "ok": False,
                "status": SkillStatus.FAILED_RECOVERABLE.value,
                "error": str(exc),
                "effect_verified": False,
                "verification_evidence": {
                    "observation": {
                        "effect_verified": False,
                        "reason": "execution_exception",
                        "error_type": type(exc).__qualname__,
                    }
                },
            }

        transport_succeeded = bool(result.get("ok", False))
        result["action_id"] = action_id
        result["request_digest"] = request_digest
        result["transport_succeeded"] = transport_succeeded
        if result.get("ok", False):
            result["status"] = (
                SkillStatus.SUCCESS_VERIFIED.value
                if result.get("effect_verified") is True
                else SkillStatus.SUCCESS_UNVERIFIED.value
            )
            result = apply_action_expectation_payload(
                action_name,
                result,
                expectation_contract,
            )
        else:
            result.setdefault("status", SkillStatus.FAILED_RECOVERABLE.value)
            result["action_expectation"] = expectation_contract.to_dict()

        status = str(result.get("status") or SkillStatus.FAILED_RECOVERABLE.value)
        effect_verified = result.get("effect_verified") is True
        result["retry_safe"] = bool(
            result.get("retry_safe") is True
            and not effect_verified
            and not transport_succeeded
        )
        result["manual_reconciliation_required"] = bool(
            transport_succeeded and not effect_verified
        )
        tx_outcome = _transaction_outcome(status, bool(result.get("ok", False)))
        error_msg = str(result.get("error") or "") if not result.get("ok", False) else ""
        tx_record = None
        welfare_transaction_completed = False
        try:
            tx_record = tx.complete(
                outcome=tx_outcome,
                welfare_after=welfare_service.last_outputs,
                body_after=body_service.snapshot(),
                error=error_msg,
            )
            welfare_transaction_completed = True
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor",
                exc,
                action=f"effect lane completed but welfare transaction closure failed for {action_name}",
                enforce_failure_policy=False,
            )
            result["ok"] = False
            result["status"] = (
                SkillStatus.PARTIAL_SUCCESS.value
                if transport_succeeded
                else SkillStatus.FAILED_RECOVERABLE.value
            )
            result["error"] = _append_error(
                result.get("error"),
                f"welfare_transaction_completion_failed:{exc}",
            )
            result["manual_reconciliation_required"] = transport_succeeded
            result["retry_safe"] = False
            status = str(result["status"])
            error_msg = str(result["error"])

        if tx_record is not None:
            try:
                will.record_outcome(will_receipt_id, tx_record)
            except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
                record_degradation(
                    "action_executor",
                    exc,
                    action=f"continued after Will outcome reinforcement failed for {action_name}",
                )

        result["will_receipt_id"] = will_receipt_id
        result["welfare_transaction_id"] = tx.tx_id
        result["welfare_transaction_completed"] = welfare_transaction_completed
        output_hash = _stable_digest(result)
        actual_outcome = (
            tx_record.outcome
            if tx_record is not None
            else _transaction_outcome(status, bool(result.get("ok", False)))
        )
        body_delta = tx_record.body_delta if tx_record is not None else {}
        post_receipt = PostActionReceipt(
            receipt_id=f"post-{uuid.uuid4()}",
            will_receipt_id=will_receipt_id,
            executor_name=action_name,
            actual_outcome=actual_outcome,
            output_hash=output_hash,
            error_status=error_msg,
            welfare_transaction_id=tx.tx_id,
            body_delta=body_delta,
            memory_delta=(
                {"record_id": result.get("record_id")}
                if result.get("record_id")
                else {}
            ),
            rollback_target=rollback_target,
            status=status,
            effect_verified=effect_verified,
            action_expectation=_bounded_receipt_mapping(expectation_contract.to_dict()),
            verification_evidence=_bounded_receipt_mapping(
                result.get("verification_evidence") or {}
            ),
            action_id=action_id,
            domain=domain.value,
            source=str(source or "unknown")[:240],
            request_digest=request_digest,
            transport_succeeded=transport_succeeded,
            retry_safe=bool(result.get("retry_safe", False)),
            manual_reconciliation_required=bool(
                result.get("manual_reconciliation_required", False)
            ),
            welfare_transaction_completed=welfare_transaction_completed,
        )
        try:
            await get_post_action_receipt_store().record_async(post_receipt)
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor",
                exc,
                severity="degraded",
                action=f"action effect occurred but post-action receipt failed for {action_name}",
                enforce_failure_policy=False,
            )
            result["ok"] = False
            result["status"] = (
                SkillStatus.PARTIAL_SUCCESS.value
                if transport_succeeded
                else SkillStatus.FAILED_RECOVERABLE.value
            )
            result["error"] = _append_error(
                result.get("error"),
                f"post_action_receipt_persistence_failed:{exc}",
            )
            result["receipt_persisted"] = False
            result["post_action_receipt_attempt_id"] = post_receipt.receipt_id
            result["manual_reconciliation_required"] = transport_succeeded
            result["retry_safe"] = False
            return result

        result["post_action_receipt_id"] = post_receipt.receipt_id
        result["receipt_persisted"] = True
        return result

    @staticmethod
    async def _dispatch(
        *,
        domain: ActionDomain,
        action_name: str,
        params: dict[str, Any],
        source: str,
        will_receipt_id: str,
        expectation: ActionExpectation,
        effect_handler: EffectHandler | None,
        execution_timeout_s: float,
    ) -> dict[str, Any]:
        if effect_handler is not None:
            return await _invoke_effect_handler(
                effect_handler,
                {
                    "domain": domain.value,
                    "action_name": action_name,
                    "params": dict(params),
                    "source": source,
                    "will_receipt_id": will_receipt_id,
                    "action_expectation": expectation.to_dict(),
                },
                timeout_s=execution_timeout_s,
            )
        if domain == ActionDomain.TOOL_EXECUTION:
            if "argv" in params:
                proc = await get_subprocess_gateway().run_async(
                    argv=params["argv"],
                    cwd=params.get("cwd"),
                    env=params.get("env"),
                    timeout=params.get("timeout", 30.0),
                    source=source,
                )
                return {
                    "ok": proc.returncode == 0,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "exit_code": proc.returncode,
                }
            from core.container import ServiceContainer

            engine = ServiceContainer.get("capability_engine", default=None)
            if engine is None or not hasattr(engine, "execute"):
                return {"ok": False, "error": "capability_engine_unavailable"}
            raw_result = await engine.execute(
                action_name,
                params,
                context={
                    "source": source,
                    "will_receipt_id": will_receipt_id,
                    "action_executor_managed_welfare_transaction": True,
                    "action_expectation": expectation.to_dict(),
                },
            )
            return _coerce_result(raw_result)

        if domain == ActionDomain.FILE_WRITE:
            gateway = get_file_write_gateway()
            path = _coerce_path_param(params.get("path"), "path")
            operation = str(params.get("op") or "").strip().lower()
            if operation == "ensure_directory":
                directory = await gateway.ensure_directory_async(path, source=source)
                return {"ok": True, "path": directory, "directory_created": True}
            if operation == "delete":
                deleted = await gateway.delete_path_async(
                    path,
                    recursive=bool(params.get("recursive", False)),
                    source=source,
                )
                return {"ok": True, "path": str(path), "deleted": deleted}
            if operation == "move":
                destination = _coerce_path_param(
                    params.get("destination"),
                    "destination",
                )
                final = await gateway.move_path_async(
                    path,
                    destination,
                    source=source,
                )
                return {"ok": True, "path": str(path), "destination": final}
            if operation == "copy":
                destination = _coerce_path_param(
                    params.get("destination"),
                    "destination",
                )
                final = await gateway.copy_path_async(
                    path,
                    destination,
                    source=source,
                )
                return {"ok": True, "path": str(path), "destination": final}
            if "text" in params:
                await gateway.write_text_async(
                    path,
                    params["text"],
                    encoding=str(params.get("encoding") or "utf-8"),
                    source=source,
                )
                return {"ok": True, "path": str(path)}
            if "payload" in params:
                await gateway.write_bytes_async(path, params["payload"], source=source)
                return {"ok": True, "path": str(path)}
            if "obj" in params:
                await gateway.write_json_async(
                    path,
                    params["obj"],
                    schema_version=int(params.get("schema_version", 1)),
                    schema_name=params.get("schema_name"),
                    source=source,
                )
                return {"ok": True, "path": str(path)}
            return {"ok": False, "error": "invalid_file_write_params"}

        if domain in {
            ActionDomain.NETWORK_CALL,
            ActionDomain.CLOUD_CALL,
            ActionDomain.CLOUD_FALLBACK,
        }:
            network_result = await get_network_gateway().request_async(
                method=params.get("method", "GET"),
                url=params.get("url", ""),
                headers=params.get("headers"),
                data=params.get("data"),
                timeout=params.get("timeout", 30.0),
                source=source,
            )
            if not isinstance(network_result, Mapping):
                return {
                    "ok": False,
                    "error": "network_gateway_returned_non_mapping_result",
                }
            return dict(network_result)

        if domain in {ActionDomain.ENVIRONMENT_ACTION, ActionDomain.EXTERNAL_ACTION}:
            desktop_result = await get_desktop_action_gateway().run_applescript_async(
                params.get("script", ""),
                source=source,
                timeout=params.get("timeout", 15.0),
            )
            if not isinstance(desktop_result, Mapping):
                return {
                    "ok": False,
                    "error": "desktop_gateway_returned_non_mapping_result",
                }
            return dict(desktop_result)

        if domain == ActionDomain.MEMORY_WRITE:
            from core.runtime.gateways import MemoryWriteRequest

            memory_receipt = await get_memory_write_gateway().write(
                MemoryWriteRequest(
                    content=str(params.get("content", "")),
                    metadata=dict(params.get("metadata", {}) or {}),
                    receipt_id=will_receipt_id,
                    cause=source,
                )
            )
            return {
                "ok": True,
                "record_id": memory_receipt.record_id,
                "receipt_id": memory_receipt.receipt_id,
                "bytes_written": memory_receipt.bytes_written,
            }

        if domain == ActionDomain.STATE_MUTATION:
            from core.runtime.gateways import StateMutationRequest

            key = str(params.get("key") or "")
            new_value = params.get("new_value", params.get("value"))
            state_gateway = get_state_gateway()
            state_receipt = await state_gateway.mutate(
                StateMutationRequest(
                    key=key,
                    new_value=new_value,
                    receipt_id=will_receipt_id,
                    cause=source,
                    domain=str(params.get("state_domain") or "world_state"),
                )
            )
            state_domain = str(params.get("state_domain") or "world_state")
            readback = await state_gateway.read(
                key,
                default=object(),
                domain=state_domain,
                fresh=True,
            )
            return {
                "ok": True,
                "key": state_receipt.key,
                "old_value": state_receipt.old_value,
                "new_value": state_receipt.new_value,
                "receipt_id": state_receipt.receipt_id,
                "readback_verified": readback == new_value,
            }

        if domain == ActionDomain.SELF_MODIFICATION:
            from core.self_modification.safe_modification_harness import run_self_mod_test

            tested = await run_self_mod_test(
                params.get("patch_path"),
                params.get("test_command"),
            )
            return {
                "ok": bool(tested.get("passed", False)),
                "test_output": tested.get("output", ""),
                "canary_passed": bool(tested.get("passed", False)),
                "applied": bool(tested.get("applied", False)),
            }
        return {"ok": False, "error": f"unsupported_action_domain:{domain.value}"}


def _coerce_domain(domain: ActionDomain | str) -> ActionDomain:
    if isinstance(domain, ActionDomain):
        return domain
    try:
        return ActionDomain(str(domain))
    except ValueError as exc:
        raise ValueError(f"unsupported action domain: {domain}") from exc


def _coerce_action_name(action_name: str) -> str:
    if not isinstance(action_name, str):
        raise TypeError("action_name must be a string")
    text = action_name.strip()
    if not text:
        raise ValueError("action_name must not be empty")
    return text[:160]


def _coerce_params(params: dict[str, Any]) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    return dict(params)


def _coerce_result(raw_result: Any) -> dict[str, Any]:
    if isinstance(raw_result, dict):
        result = dict(raw_result)
        result.setdefault("ok", bool(result.get("ok", False)))
        return result
    return {"ok": bool(raw_result), "result": raw_result}


def _validate_effect_handler(
    domain: ActionDomain,
    *,
    effect_handler: EffectHandler | None,
    effect_verifier: EffectVerifier | None,
) -> str:
    if effect_handler is None:
        return ""
    if domain not in _HANDLER_DOMAINS:
        raise ValueError(
            f"custom effect handlers are not permitted for action domain {domain.value}"
        )
    if effect_verifier is None:
        raise ValueError("custom effect handlers require an independent effect_verifier")
    return _callable_name(effect_handler)


def _coerce_execution_timeout(value: float | None) -> float:
    if value is None:
        return 60.0
    try:
        timeout_s = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution_timeout_s must be numeric") from exc
    if timeout_s <= 0:
        raise ValueError("execution_timeout_s must be positive")
    return min(timeout_s, 600.0)


async def _invoke_effect_handler(
    handler: EffectHandler,
    context: Mapping[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    async def invoke() -> Mapping[str, Any]:
        if inspect.iscoroutinefunction(handler):
            value = await handler(dict(context))
        else:
            value = await asyncio.to_thread(handler, dict(context))
            if inspect.isawaitable(value):
                value = await value
        if not isinstance(value, Mapping):
            raise TypeError("effect handler must return a mapping")
        return value

    completed = await asyncio.wait_for(
        asyncio.gather(invoke(), return_exceptions=True),
        timeout=timeout_s,
    )
    raw_result = completed[0]
    if isinstance(raw_result, asyncio.CancelledError):
        raise raw_result
    if isinstance(raw_result, BaseException):
        raise RuntimeError(
            f"effect handler {_callable_name(handler)} failed: {raw_result}"
        ) from raw_result
    return _coerce_result(raw_result)


def _callable_name(value: Callable[..., Any]) -> str:
    module = str(getattr(value, "__module__", "") or "")
    qualname = str(
        getattr(value, "__qualname__", "")
        or getattr(value, "__name__", "")
        or type(value).__qualname__
    )
    return f"{module}.{qualname}".strip(".")[:240]


def _coerce_expectation(
    value: ActionExpectation | Mapping[str, Any] | None,
) -> ActionExpectation | None:
    if value is None:
        return value
    if isinstance(value, ActionExpectation):
        raw = value.to_dict()
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError("expectation must be ActionExpectation, mapping, or None")
    return ActionExpectation(
        objective=str(raw.get("objective") or "")[:1000],
        acceptance_criteria=_string_list(raw.get("acceptance_criteria")),
        required_evidence=_string_list(raw.get("required_evidence")),
        required_evidence_present=_string_list(raw.get("required_evidence_present")),
        user_visible_effect=(
            str(raw.get("user_visible_effect"))[:1000]
            if raw.get("user_visible_effect") is not None
            else None
        ),
        repair_hint=str(raw.get("repair_hint") or "")[:1000],
        rollback_hint=str(raw.get("rollback_hint") or "")[:1000],
        allow_partial=bool(raw.get("allow_partial", True)),
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item)[:500] for item in list(value)[:64] if str(item).strip()]


def _coerce_action_id(value: str | None) -> str:
    if value is None:
        return f"action-{uuid.uuid4()}"
    text = str(value).strip()
    if not _ACTION_ID_PATTERN.fullmatch(text):
        raise ValueError(
            "action_id must be 1-160 letters, digits, dot, colon, dash, or underscore"
        )
    return text


def _coerce_path_param(value: Any, label: str) -> str | Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"file action {label} must be a string or Path")
    if not str(value).strip():
        raise ValueError(f"file action {label} must not be empty")
    return value


def _safe_action_summary(action_name: str, params: Mapping[str, Any]) -> str:
    summarized: dict[str, Any] = {}
    for key, value in params.items():
        key_text = str(key)
        if any(marker in key_text.casefold() for marker in _SENSITIVE_PARAM_MARKERS):
            summarized[key_text] = "[REDACTED]"
        elif key_text in {"content", "payload", "script", "text"}:
            length = len(value) if hasattr(value, "__len__") else 0
            summarized[key_text] = f"<{type(value).__name__}:{length}>"
        elif isinstance(value, Mapping):
            summarized[key_text] = f"<mapping:{len(value)}>"
        elif isinstance(value, (list, tuple, set)):
            summarized[key_text] = _safe_sequence_summary(value)
        elif key_text.casefold() in {"uri", "url"}:
            summarized[key_text] = _safe_url_summary(str(value))
        else:
            summarized[key_text] = str(value)[:160]
    encoded = json.dumps(summarized, sort_keys=True, default=str)
    return f"{action_name} params={encoded}"[:1000]


def _transaction_outcome(status: str, ok: bool) -> str:
    if ok and status == SkillStatus.SUCCESS_VERIFIED.value:
        return "success"
    if status in {
        SkillStatus.SUCCESS_UNVERIFIED.value,
        SkillStatus.PARTIAL_SUCCESS.value,
    }:
        return "partial"
    return "failure"


def _safe_sequence_summary(value: Any) -> list[Any]:
    summarized: list[Any] = []
    redact_next = False
    for raw_item in list(value)[:16]:
        if isinstance(raw_item, Mapping):
            summarized.append(_safe_nested_mapping_summary(raw_item))
            continue
        item = str(raw_item)
        lowered = item.casefold()
        if redact_next:
            summarized.append("[REDACTED]")
            redact_next = False
            continue
        if lowered.startswith(("http://", "https://")):
            summarized.append(_safe_url_summary(item))
            continue
        if any(marker in lowered for marker in _SENSITIVE_PARAM_MARKERS):
            if "=" in item:
                summarized.append(item.split("=", 1)[0][:80] + "=[REDACTED]")
            elif item.lstrip().startswith("-"):
                summarized.append(item[:80])
                redact_next = True
            else:
                summarized.append("[REDACTED]")
            continue
        summarized.append(item[:80])
    return summarized


def _safe_nested_mapping_summary(value: Mapping[Any, Any]) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    for raw_key, raw_value in list(value.items())[:32]:
        key = str(raw_key)[:80]
        lowered = key.casefold()
        if lowered == "value" or any(
            marker in lowered for marker in _SENSITIVE_PARAM_MARKERS
        ):
            summarized[key] = "[REDACTED]"
        elif lowered in {"content", "payload", "script", "text"}:
            length = len(raw_value) if hasattr(raw_value, "__len__") else 0
            summarized[key] = f"<{type(raw_value).__name__}:{length}>"
        elif lowered in {"uri", "url"}:
            summarized[key] = _safe_url_summary(str(raw_value))
        elif isinstance(raw_value, Mapping):
            summarized[key] = _safe_nested_mapping_summary(raw_value)
        elif isinstance(raw_value, (list, tuple, set)):
            summarized[key] = _safe_sequence_summary(raw_value)
        else:
            summarized[key] = str(raw_value)[:160]
    return summarized


def _safe_url_summary(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return urllib.parse.urlunsplit(
            (parsed.scheme, host + port, parsed.path, "", "")
        )[:240]
    except ValueError:
        return "<invalid-url>"


def _append_error(existing: Any, new_error: str) -> str:
    current = str(existing or "").strip()
    addition = str(new_error or "").strip()
    if not current:
        return addition[:1000]
    if not addition or addition in current:
        return current[:1000]
    return f"{current}; {addition}"[:1000]


def _stable_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value, seen=set(), depth=0)
    return "sha256:" + digest.hexdigest()


def _update_digest(
    digest: Any,
    value: Any,
    *,
    seen: set[int],
    depth: int,
) -> None:
    if depth > 48:
        digest.update(b"<max-depth>")
        return
    if value is None or isinstance(value, (bool, int, float)):
        digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        for offset in range(0, len(encoded), 1024 * 1024):
            digest.update(encoded[offset : offset + 1024 * 1024])
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        digest.update(f"bytes:{len(payload)}:".encode("ascii"))
        digest.update(payload)
        return
    container_id = id(value)
    if container_id in seen:
        digest.update(b"<cycle>")
        return
    if isinstance(value, Mapping):
        seen.add(container_id)
        digest.update(b"{")
        for key in sorted(value, key=lambda item: (type(item).__qualname__, str(item))):
            _update_digest(digest, key, seen=seen, depth=depth + 1)
            _update_digest(digest, value[key], seen=seen, depth=depth + 1)
        digest.update(b"}")
        seen.remove(container_id)
        return
    if isinstance(value, (list, tuple)):
        seen.add(container_id)
        digest.update(b"[")
        for item in value:
            _update_digest(digest, item, seen=seen, depth=depth + 1)
        digest.update(b"]")
        seen.remove(container_id)
        return
    if isinstance(value, (set, frozenset)):
        seen.add(container_id)
        digest.update(b"<set>")
        for item in sorted(value, key=lambda entry: (type(entry).__qualname__, str(entry))):
            _update_digest(digest, item, seen=seen, depth=depth + 1)
        seen.remove(container_id)
        return
    _update_digest(
        digest,
        f"{type(value).__module__}.{type(value).__qualname__}:{value}",
        seen=seen,
        depth=depth + 1,
    )


def _bounded_receipt_mapping(value: Any) -> dict[str, Any]:
    state = {"items": 0, "truncated": False}
    bounded = _bounded_receipt_value(value, state=state, depth=0)
    if isinstance(bounded, dict):
        result = bounded
    else:
        result = {"value": bounded}
    if state["truncated"]:
        result = dict(result)
        result["_truncated"] = True
        result["_original_digest"] = _stable_digest(value)
    return result


def _bounded_receipt_value(
    value: Any,
    *,
    state: dict[str, Any],
    depth: int,
) -> Any:
    state["items"] = int(state["items"]) + 1
    if depth > 8 or int(state["items"]) > 512:
        state["truncated"] = True
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= 2048:
            return value
        state["truncated"] = True
        return {
            "prefix": value[:2048],
            "characters": len(value),
            "digest": _stable_digest(value),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "type": "bytes",
            "bytes": len(payload),
            "digest": _stable_digest(payload),
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 128 or int(state["items"]) > 512:
                state["truncated"] = True
                break
            result[str(key)[:240]] = _bounded_receipt_value(
                item,
                state=state,
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if len(items) > 128:
            state["truncated"] = True
        return [
            _bounded_receipt_value(item, state=state, depth=depth + 1)
            for item in items[:128]
            if int(state["items"]) <= 512
        ]
    text = str(value)
    if len(text) > 2048:
        state["truncated"] = True
        text = text[:2048]
    return text


__all__ = ["ActionExecutor"]
