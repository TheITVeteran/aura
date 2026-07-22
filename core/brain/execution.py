# core/brain/execution.py
import asyncio
import inspect
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.brain.trace_logger import TraceLogger
from core.runtime.errors import FallbackClassification, record_degradation

_MAX_RETRIES = 20
_SENSITIVE_META_MARKERS = ("secret", "password", "passwd", "token", "key", "credential", "auth")


def _finite(value: Any, *, default: float, low: float, high: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return max(low, min(high, num))


def _copy_meta(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Copy caller metadata (so we never return their mutable dict by reference)
    and redact secret-bearing keys before it can enter traces/results."""
    out: dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        if isinstance(k, str) and any(m in k.lower() for m in _SENSITIVE_META_MARKERS):
            out[k] = "***redacted***"
        else:
            out[k] = v
    return out

_EXECUTION_RECOVERABLE_ERRORS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _record_execution_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        "execution",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


@dataclass
class ExecResult:
    ok: bool
    result: Any = None
    error: str | None = None
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionManager:
    """
    Responsible for executing actions safely.
    - action_fn: Callable[[str], Any] or async
    - safety_check(action_name, context) -> bool (allow/deny)
    - supports timeouts, retries, and safe-mode gating for dangerous actions
    """

    def __init__(
        self, trace: TraceLogger, safe_mode: bool = True, dangerous_whitelist: set | None = None
    ):
        self.trace = trace
        self.safe_mode = safe_mode
        self.dangerous_whitelist = dangerous_whitelist or set()

    async def execute(
        self,
        action_name: str,
        action_fn: Callable[..., Any],
        context: str = "",
        timeout_seconds: float = 30.0,
        retries: int = 1,
        retry_delay: float = 1.0,
        allow_danger: bool = False,
        metadata: dict[str, Any] | None = None,
        success_predicate: Callable[[Any], bool] | None = None,
        **legacy_kwargs: Any,
    ) -> ExecResult:
        if "timeout" in legacy_kwargs:
            timeout_seconds = float(legacy_kwargs.pop("timeout"))
        if legacy_kwargs:
            raise TypeError(f"Unsupported execution options: {sorted(legacy_kwargs)}")
        metadata = _copy_meta(metadata)
        # NaN would slip past a bare `<= 0` check; validate finiteness first.
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Execution timeout must be a positive, finite number.")
        # retries is an ATTEMPT count (min 1); cap it so a huge value cannot
        # monopolize the lane.
        try:
            retries = max(1, min(_MAX_RETRIES, int(retries)))
        except (TypeError, ValueError):
            retries = 1
        retry_delay = _finite(retry_delay, default=1.0, low=0.0, high=60.0)
        operation_id = uuid.uuid4().hex
        metadata.setdefault("operation_id", operation_id)
        # safety gating
        if self.safe_mode and not allow_danger and action_name in self.dangerous_whitelist:
            msg = f"Action '{action_name}' denied by safe_mode"
            self.trace.log(
                {
                    "type": "execution_denied",
                    "action": action_name,
                    "reason": msg,
                    "context": context[:200],
                }
            )
            return ExecResult(ok=False, error=msg, duration=0.0, metadata=metadata)

        # Monotonic clock: a wall-clock jump must not make a duration negative.
        start = time.monotonic()
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                res = await asyncio.wait_for(_invoke_action(action_fn), timeout=timeout_seconds)
                dur = time.monotonic() - start
                # A returned object is not automatically a success — honor an
                # explicit success predicate and treat obvious failure envelopes
                # (False/None, or a dict carrying ok=False / an error) as failures.
                if not _looks_successful(res, success_predicate):
                    last_err = _failure_reason(res)
                    self.trace.log({
                        "type": "execution_unsuccessful_result", "action": action_name,
                        "attempt": attempt, "operation_id": operation_id,
                    })
                    if attempt < retries:
                        await asyncio.sleep(retry_delay)
                        continue
                    return ExecResult(ok=False, result=res, error=last_err, duration=dur, metadata=metadata)
                self.trace.log({
                    "type": "execution", "action": action_name, "ok": True,
                    "duration": dur, "attempt": attempt, "operation_id": operation_id,
                })
                return ExecResult(ok=True, result=res, duration=dur, metadata=metadata)
            except TimeoutError:
                # The wait_for cancels our await, but a sync callable already
                # running in a worker thread cannot be forced to stop — mark the
                # outcome uncertain so a retry is not assumed side-effect-free.
                last_err = "timeout"
                metadata["outcome"] = "uncertain_timeout"
                self.trace.log({
                    "type": "execution_timeout", "action": action_name, "attempt": attempt,
                    "timeout": timeout_seconds, "operation_id": operation_id,
                })
            except asyncio.CancelledError:
                self.trace.log(
                    {"type": "execution_cancelled", "action": action_name, "attempt": attempt}
                )
                raise
            except _EXECUTION_RECOVERABLE_ERRORS as e:
                _record_execution_degradation(
                    e,
                    action="returned failed execution result after action callable failed",
                    extra={"action": action_name, "attempt": attempt},
                )
                last_err = str(e)
                self.trace.log({
                    "type": "execution_exception", "action": action_name,
                    "attempt": attempt, "error": last_err, "operation_id": operation_id,
                })
            except Exception as e:  # noqa: BLE001 — unexpected faults become a receipt, never escape mid-effect
                _record_execution_degradation(
                    e,
                    action="converted an unexpected execution fault into a failed result receipt",
                    severity="degraded",
                    extra={"action": action_name, "attempt": attempt, "operation_id": operation_id},
                )
                last_err = f"unexpected:{type(e).__name__}:{e}"
                self.trace.log({
                    "type": "execution_unexpected_error", "action": action_name,
                    "attempt": attempt, "error": last_err[:300], "operation_id": operation_id,
                })
            # retry backoff
            if attempt < retries:
                await asyncio.sleep(retry_delay)
        dur = time.monotonic() - start
        return ExecResult(ok=False, error=last_err, duration=dur, metadata=metadata)


def _looks_successful(res: Any, predicate: Callable[[Any], bool] | None) -> bool:
    """Judge whether a returned object represents success.

    With a caller ``predicate`` the caller decides. By default we reject only an
    explicit failure envelope (a dict with ok=False or an error) so a void
    (None) result from an ordinary side-effecting action stays successful.
    """
    if predicate is not None:
        try:
            return bool(predicate(res))
        except Exception:  # noqa: BLE001 — a predicate fault is a failed judgment
            return False
    if isinstance(res, dict):
        if res.get("ok") is False:
            return False
        if res.get("error") and res.get("ok") is not True:
            return False
    return True


def _failure_reason(res: Any) -> str:
    if isinstance(res, dict):
        return str(res.get("error") or res.get("reason") or "unsuccessful_result")[:200]
    return "unsuccessful_result"


async def _invoke_action(action_fn: Callable[..., Any]) -> Any:
    if inspect.iscoroutinefunction(action_fn):
        return await action_fn()
    result = await asyncio.to_thread(action_fn)
    if inspect.isawaitable(result):
        return await result
    return result
