"""core/learning/crsm_closure_scheduler.py — autonomous CRSM→LoRA loop closer.

The gap this fills: Aura captures high-salience moments into a synthetic
training corpus (the CRSM→LoRA bridge, the producer) and a monitor
(crsm_loop_monitor) reports the loop OPEN once untrained captures pass a
threshold — but nothing ever *closed* it on its own. The weight-compounding
scheduler harvests DPO pairs from the verifier harness, a different source;
it never consumed the CRSM capture corpus. So on any long-lived instance the
health poll honestly reported "proof integrity degraded: CRSM→LoRA loop OPEN
(N captures untrained)" forever, because closure required an operator to run
``training/train_and_fuse.py --crsm-delta`` by hand.

This scheduler is the missing executor. It mirrors the compounding
scheduler's discipline exactly — the same idle gate, the same Will contract,
the same governed scope, the same honest deferral — but its trigger is "the
loop is OPEN" and its action is the bounded CRSM delta train/fuse the monitor
already computes in ``next_action`` (train_and_fuse writes the consumed
marker on success, so a clean run closes the loop as a *verified fact*).

Default-OFF: ``AURA_CRSM_AUTOCLOSE`` must be set to enable it. Weight
mutation on the resident 32B is heavy (dequant→fp16→merge, ~2.5x transient
RAM), so it fires only in deep maintenance idle, with real free-RAM headroom,
under Will approval, single-flight, once per cooldown. Every decline is
recorded with its reason; it never fails silently and never fights the user
for memory.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CRSMClosureScheduler")

_RECOVERABLE = (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError)

SERVICE_NAME = "crsm_closure"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


class CRSMClosureScheduler:
    """Periodic, governed, idle-gated closer for the CRSM→LoRA loop."""

    def __init__(self, orchestrator: Any = None) -> None:
        self._orchestrator = orchestrator
        self._task: asyncio.Task[Any] | None = None
        self._active = False
        self._running_cycle = False
        self.check_interval_s = float(_env_int("AURA_CRSM_AUTOCLOSE_CHECK_INTERVAL_S", 900))
        self.cooldown_s = float(_env_int("AURA_CRSM_AUTOCLOSE_COOLDOWN_S", 6 * 3600))
        # A 32B CRSM delta fuse peaks well above the resident model; require
        # genuine headroom before we ever start one. Conservative by default.
        self.min_free_gb = float(_env_int("AURA_CRSM_AUTOCLOSE_MIN_FREE_GB", 40))
        # A 32B CRSM delta at the default 600 iters plus the fuse can run
        # 60-90 min; a timeout below that wastes the whole pass. Budget 3h and
        # let AURA_CRSM_DELTA_ITERS shorten the run rather than the clock.
        self.train_timeout_s = float(_env_int("AURA_CRSM_AUTOCLOSE_TIMEOUT_S", 3 * 3600))
        self._state_path = (
            Path(os.getenv("AURA_STATE_DIR", str(Path.home() / ".aura" / "run")))
            / "crsm_closure_state.json"
        )

    async def start(self) -> None:
        if not _env_flag("AURA_CRSM_AUTOCLOSE", False):
            logger.info(
                "CRSM→LoRA autonomous closure disabled (set AURA_CRSM_AUTOCLOSE=1 to "
                "let Aura close her own capture loop in deep idle)."
            )
            return
        if self._task is not None and not self._task.done():
            return
        from core.utils.task_tracker import get_task_tracker

        self._active = True
        self._task = get_task_tracker().create_task(
            self._run(), name="crsm_closure_scheduler", owner=SERVICE_NAME
        )
        logger.info(
            "🔁 CRSM→LoRA closure scheduler online (check every %.0fs, cooldown %.0fs, "
            "min free %.0fGB).",
            self.check_interval_s, self.cooldown_s, self.min_free_gb,
        )

    async def stop(self) -> None:
        self._active = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown best-effort
                pass

    async def _run(self) -> None:
        # Late first check: never compete with boot warmup.
        await asyncio.sleep(min(self.check_interval_s, 600.0))
        while self._active:
            try:
                await self._maybe_close()
            except asyncio.CancelledError:
                raise
            except _RECOVERABLE as exc:
                record_degradation(
                    SERVICE_NAME, exc, action="skipped this closure check; next check unaffected"
                )
            await asyncio.sleep(self.check_interval_s)

    # ── state / cooldown ──────────────────────────────────────────────────
    def _load_state(self) -> dict[str, Any]:
        try:
            if self._state_path.exists():
                return dict(json.loads(self._state_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        return {}

    def _record_attempt(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self._state_path,
                json.dumps({"last_attempt_at": time.time()}),
            )
        except OSError as exc:
            record_degradation(SERVICE_NAME, exc, action="continued without persisting cooldown")

    # ── admission ─────────────────────────────────────────────────────────
    def _loop_is_open(self) -> tuple[bool, dict[str, Any]]:
        from core.consciousness.crsm_loop_monitor import get_crsm_loop_monitor

        state = get_crsm_loop_monitor().loop_state()
        return state.get("state") == "open", state

    def _idle_allows(self) -> bool:
        from core.runtime.background_policy import (
            MAINTENANCE_BACKGROUND_POLICY,
            background_activity_allowed,
        )

        return bool(
            background_activity_allowed(
                self._orchestrator, profile=MAINTENANCE_BACKGROUND_POLICY
            )
        )

    def _ram_admits(self) -> tuple[bool, str]:
        """Deep-idle unloads the cortex, but never assume — check real free RAM."""
        try:
            import psutil

            free_gb = psutil.virtual_memory().available / (1024**3)
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            # Fail closed: no reading means no training.
            return False, f"ram_probe_unavailable:{type(exc).__name__}"
        if free_gb < self.min_free_gb:
            return False, f"insufficient_free_ram:{free_gb:.1f}GB<{self.min_free_gb:.0f}GB"
        return True, f"free_ram:{free_gb:.1f}GB"

    def _will_approval(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Weight mutation is governed: no Will, no training. Fail closed."""
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=f"crsm_loop_closure:{context.get('unconsumed', '?')}_captures",
                source=SERVICE_NAME,
                domain=ActionDomain.STATE_MUTATION,
                priority=0.7,
                context=context,
            )
            if decision.is_approved():
                return True, str(getattr(decision, "receipt_id", "approved"))
            return False, str(getattr(decision, "reason", "denied"))
        except _RECOVERABLE as exc:
            record_degradation(
                SERVICE_NAME,
                exc,
                action="blocked closure because Will approval was unavailable",
            )
            return False, f"will_unavailable:{type(exc).__name__}"

    # ── triggers ──────────────────────────────────────────────────────────
    async def _maybe_close(self) -> None:
        if self._running_cycle or not _env_flag("AURA_CRSM_AUTOCLOSE", False):
            return
        since_last = time.time() - float(self._load_state().get("last_attempt_at", 0.0))
        if since_last < self.cooldown_s:
            return
        is_open, _state = self._loop_is_open()
        if not is_open:
            return
        if not self._idle_allows():
            return
        await self._execute_closure(reason="scheduled_idle")

    async def run_closure_now(self, *, reason: str = "manual") -> dict[str, Any]:
        """On-demand governed closure — the seam operators and the RSI loop call.

        Bypasses the idle gate and cooldown (the caller chose the moment) but
        never the single-flight guard, RAM admission, or Will approval.
        """
        if self._running_cycle:
            return {"status": "blocked", "reasons": ["closure_already_running"]}
        if not _env_flag("AURA_CRSM_AUTOCLOSE", False):
            return {"status": "blocked", "reasons": ["disabled_by_env"]}
        return await self._execute_closure(reason=reason)

    async def _execute_closure(self, *, reason: str) -> dict[str, Any]:
        """The one governed execution core behind both triggers."""
        self._running_cycle = True
        self._record_attempt()
        try:
            is_open, state = self._loop_is_open()
            if not is_open:
                return {"status": "noop", "reasons": ["loop_not_open"], "loop": state}

            ram_ok, ram_reason = self._ram_admits()
            if not ram_ok:
                # Deferral is honest backpressure, not a failure — the captures
                # are preserved and the next idle window retries.
                logger.info("CRSM closure deferred (%s): %s", reason, ram_reason)
                return {"status": "deferred", "reasons": [ram_reason], "loop": state}

            approved, will_reason = self._will_approval(
                {"reason": reason, "unconsumed": state.get("unconsumed"), "readiness": state}
            )
            if not approved:
                logger.info("Will declined CRSM closure (%s): %s", reason, will_reason)
                return {"status": "will_declined", "reasons": [will_reason], "loop": state}

            from core.consciousness.crsm_loop_monitor import get_crsm_loop_monitor

            monitor = get_crsm_loop_monitor()
            action = monitor.next_action()
            command = [str(part) for part in (action.get("command") or [])]
            if not command:
                return {"status": "noop", "reasons": ["no_command"], "loop": state}

            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope(
                "crsm_closure_scheduler.train_fuse",
                domain="tool_execution",
                constraints={"artifact": "model_weights", "governed_by": "will+crsm_loop"},
            ):
                result = await self._run_training(command)

            if result.get("returncode") == 0:
                closed_state = monitor.loop_state()
                logger.info(
                    "🔁 CRSM→LoRA loop closed via %s: %s",
                    reason, closed_state.get("reason", ""),
                )
                return {"status": "closed", "loop": closed_state, "reason": reason}

            record_degradation(
                SERVICE_NAME,
                RuntimeError(f"crsm_train_fuse_rc={result.get('returncode')}"),
                action="left CRSM loop open after a non-zero train/fuse; captures preserved",
            )
            return {
                "status": "train_failed",
                "returncode": result.get("returncode"),
                "stderr_tail": str(result.get("stderr", ""))[-500:],
                "loop": monitor.loop_state(),
            }
        finally:
            self._running_cycle = False

    async def _run_training(self, command: list[str]) -> dict[str, Any]:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        res = await get_subprocess_gateway().run_async(
            command,
            capture_output=True,
            timeout=self.train_timeout_s,
            offline_tooling=True,
            source="training_tooling:crsm_closure_scheduler",
        )
        return {
            "returncode": res.returncode,
            "stdout": (res.stdout or "")[-1000:],
            "stderr": (res.stderr or "")[-1000:],
        }

    def get_status(self) -> dict[str, Any]:
        try:
            _open, state = self._loop_is_open()
        except _RECOVERABLE:
            state = {}
        return {
            "enabled": _env_flag("AURA_CRSM_AUTOCLOSE", False),
            "running_cycle": self._running_cycle,
            "check_interval_s": self.check_interval_s,
            "cooldown_s": self.cooldown_s,
            "min_free_gb": self.min_free_gb,
            "last_attempt_at": self._load_state().get("last_attempt_at", 0.0),
            "loop": state,
        }


_scheduler: CRSMClosureScheduler | None = None


def get_crsm_closure_scheduler(orchestrator: Any = None) -> CRSMClosureScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = CRSMClosureScheduler(orchestrator)
    return _scheduler


def reset_crsm_closure_scheduler_for_test() -> None:
    global _scheduler
    _scheduler = None


__all__ = [
    "SERVICE_NAME",
    "CRSMClosureScheduler",
    "get_crsm_closure_scheduler",
    "reset_crsm_closure_scheduler_for_test",
]
