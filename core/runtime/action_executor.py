"""core/runtime/action_executor.py — Canonical Action Executor.

Coordinates the entire lifecycle of consequential actions: Will approval, WelfareTransaction, execution, publishing to the ConsequenceBus, and writing the PostActionReceipt.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from typing import Any

from core.being.body_state_service import BodyStateService
from core.being.welfare_state import WelfareState
from core.being.welfare_transaction import WelfareTransaction
from core.governance.will import ActionDomain, get_will
from core.governance_context import GovernanceViolation, governed_scope
from core.memory.memory_write_gateway import get_memory_write_gateway
from core.runtime.desktop_action_gateway import get_desktop_action_gateway
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.network_gateway import get_network_gateway
from core.runtime.post_action_receipt import PostActionReceipt, get_post_action_receipt_store
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.ActionExecutor")
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


class ActionExecutor:
    """Canonical facade for all governed and transacted Aura actions."""

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
    ) -> dict[str, Any]:
        domain = _coerce_domain(domain)
        action_name = _coerce_action_name(action_name)
        params = _coerce_params(params)

        # 1. Ask Will for pre-action receipt
        will = get_will()
        # Create a decodable text summary of what is happening
        content_summary = f"{action_name} params={json.dumps(params, default=str, sort_keys=True)}"[:400]
        decision = will.decide(
            content=content_summary,
            source=source,
            domain=domain,
            priority=0.5,
        )

        if not decision.is_approved():
            logger.warning("🚫 ActionExecutor: Will refused action %s in domain %s", action_name, domain.value)
            return {
                "ok": False,
                "status": "refused",
                "error": f"Will refused action: {decision.reason}",
            }

        will_receipt_id = decision.receipt_id

        # 2. Get current state to begin WelfareTransaction
        body_service = BodyStateService.get()
        welfare_service = WelfareState.get()

        body_before = body_service.snapshot()
        welfare_before = welfare_service.last_outputs

        tx = WelfareTransaction.begin(
            domain=domain.value,
            action=f"{action_name} ({source})",
            welfare_before=welfare_before,
            body_before=body_before,
            predicted_welfare_delta=predicted_welfare_delta,
            will_receipt_id=will_receipt_id,
        )

        # 3. Call the correct backend gateway
        result: dict[str, Any] = {"ok": False}
        try:
            async with governed_scope(decision):
                if domain == ActionDomain.TOOL_EXECUTION:
                    if "argv" in params:
                        proc = get_subprocess_gateway().run(
                            argv=params["argv"],
                            cwd=params.get("cwd"),
                            env=params.get("env"),
                            timeout=params.get("timeout", 30.0),
                            source=source,
                        )
                        result = {
                            "ok": proc.returncode == 0,
                            "stdout": proc.stdout,
                            "stderr": proc.stderr,
                            "exit_code": proc.returncode,
                        }
                    else:
                        from core.container import ServiceContainer

                        engine = ServiceContainer.get("capability_engine", default=None)
                        if engine and hasattr(engine, "execute"):
                            raw_result = await engine.execute(
                                action_name,
                                params,
                                context={
                                    "source": source,
                                    "will_receipt_id": will_receipt_id,
                                    "action_executor_managed_welfare_transaction": True,
                                },
                            )
                            result = _coerce_result(raw_result)
                        else:
                            result = {"ok": False, "error": "capability_engine_unavailable"}
                elif domain == ActionDomain.FILE_WRITE:
                    gateway = get_file_write_gateway()
                    path = params.get("path")
                    if "text" in params:
                        await gateway.write_text_async(path, params["text"], source=source)
                        result = {"ok": True, "path": str(path)}
                    elif "payload" in params:
                        await gateway.write_bytes_async(path, params["payload"], source=source)
                        result = {"ok": True, "path": str(path)}
                    elif "obj" in params:
                        gateway.write_json(
                            path,
                            params["obj"],
                            schema_version=int(params.get("schema_version", 1)),
                            schema_name=params.get("schema_name"),
                            source=source,
                        )
                        result = {"ok": True, "path": str(path)}
                    else:
                        result = {"ok": False, "error": "invalid_file_write_params"}
                elif domain in (ActionDomain.NETWORK_CALL, ActionDomain.CLOUD_CALL, ActionDomain.CLOUD_FALLBACK):
                    gateway = get_network_gateway()
                    result = gateway.request(
                        method=params.get("method", "GET"),
                        url=params.get("url", ""),
                        headers=params.get("headers"),
                        data=params.get("data"),
                        timeout=params.get("timeout", 30.0),
                        source=source,
                    )
                elif domain in (ActionDomain.ENVIRONMENT_ACTION, ActionDomain.EXTERNAL_ACTION):
                    gateway = get_desktop_action_gateway()
                    result = gateway.run_applescript(
                        params.get("script", ""),
                        source=source,
                        timeout=params.get("timeout", 15.0),
                    )
                elif domain == ActionDomain.MEMORY_WRITE:
                    from core.runtime.gateways import MemoryWriteRequest

                    gateway = get_memory_write_gateway()
                    req = MemoryWriteRequest(
                        content=str(params.get("content", "")),
                        metadata=dict(params.get("metadata", {}) or {}),
                        receipt_id=will_receipt_id,
                        cause=source,
                    )
                    receipt = await gateway.write(req)
                    result = {
                        "ok": True,
                        "record_id": receipt.record_id,
                        "receipt_id": receipt.receipt_id,
                        "bytes_written": receipt.bytes_written,
                    }
                elif domain == ActionDomain.SELF_MODIFICATION:
                    from core.self_modification.safe_modification_harness import run_self_mod_test

                    test_res = await run_self_mod_test(params.get("patch_path"), params.get("test_command"))
                    result = {
                        "ok": bool(test_res.get("passed", False)),
                        "test_output": test_res.get("output", ""),
                        "canary_passed": bool(test_res.get("passed", False)),
                    }
                else:
                    result = {"ok": False, "error": f"unsupported_action_domain:{domain.value}"}
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor",
                exc,
                action=f"recorded failed post-action receipt for {action_name}",
            )
            logger.error("Error executing action %s: %s", action_name, exc, exc_info=True)
            result = {"ok": False, "error": str(exc)}

        # 4. Complete WelfareTransaction
        body_after = body_service.snapshot()
        welfare_after = welfare_service.last_outputs

        error_msg = result.get("error", "") if not result.get("ok") else ""
        tx_record = tx.complete(
            outcome="success" if result.get("ok") else "failure",
            welfare_after=welfare_after,
            body_after=body_after,
            error=error_msg,
        )

        # Let the UnifiedWill learn from the actual outcome
        try:
            will.record_outcome(will_receipt_id, tx_record)
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor",
                exc,
                action=f"continued after Will outcome reinforcement failed for {action_name}",
            )
            logger.debug("Failed to record outcome in Will: %s", exc)

        # 5. Record PostActionReceipt
        output_data = json.dumps(result, default=str)
        output_hash = "sha256:" + hashlib.sha256(output_data.encode("utf-8")).hexdigest()

        post_receipt = PostActionReceipt(
            receipt_id="post_" + hashlib.sha256(f"{will_receipt_id}:{time.time()}".encode()).hexdigest()[:12],
            will_receipt_id=will_receipt_id,
            executor_name=action_name,
            actual_outcome=tx_record.outcome,
            output_hash=output_hash,
            error_status=error_msg,
            welfare_transaction_id=tx.tx_id,
            body_delta=tx_record.body_delta,
            memory_delta={"record_id": result.get("record_id")} if "record_id" in result else {},
            rollback_target=rollback_target,
        )
        get_post_action_receipt_store().record(post_receipt)

        return result


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


__all__ = ["ActionExecutor"]
