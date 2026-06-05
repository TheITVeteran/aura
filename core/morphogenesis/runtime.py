from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import traceback
from collections import deque
from collections.abc import Sequence
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.task_ownership import create_tracked_task

from .field import MorphogenField
from .metabolism import MetabolismManager
from .organs import OrganStabilizer
from .registry import MorphogenesisRegistry
from .types import MorphogenesisConfig, MorphogenSignal, SignalKind, stable_digest

logger = logging.getLogger("Aura.Morphogenesis.Runtime")


def _record_morphogenesis_runtime_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "morphogenesis.runtime",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "morphogenesis.runtime",
                error,
                severity=severity,
                action=action,
            )
        except TypeError:
            logger.debug(
                "Morphogenesis runtime degradation could not be recorded: %s",
                signature_exc,
            )


class MorphogeneticRuntime:
    """Bounded self-organisation loop.

    Key hardening properties:
      - uses canonical task ownership instead of raw asyncio.create_task
      - never applies source-code patches directly
      - bridges high-danger events into AdaptiveImmuneSystem
      - persists registry through atomic writer when available
      - logs longitudinal episodes through EpisodicMemory when available
      - enforces caps on signals, cells, organs and actions per tick
    """

    def __init__(
        self,
        *,
        config: MorphogenesisConfig | None = None,
        registry: MorphogenesisRegistry | None = None,
        field: MorphogenField | None = None,
        metabolism: MetabolismManager | None = None,
        organ_stabilizer: OrganStabilizer | None = None,
    ):
        self.config = config or MorphogenesisConfig()
        self.registry = registry or MorphogenesisRegistry(config=self.config)
        self.field = field or MorphogenField(diffusion=self.config.field_diffusion, decay=self.config.field_decay)
        self.metabolism = metabolism or MetabolismManager(recovery_per_tick=self.config.energy_recovery_per_tick)
        self.organ_stabilizer = organ_stabilizer or OrganStabilizer(
            min_coactivations=self.config.organ_min_coactivations,
            min_members=self.config.organ_min_members,
            edge_threshold=self.config.organ_edge_threshold,
        )
        self._signals: deque[MorphogenSignal] = deque(maxlen=max(16, self.config.max_signals_per_tick * 4))
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._tick = 0
        self._events_since_episode = 0
        self._last_tick_error = ""
        self._last_tick_at = 0.0
        self._started_at = 0.0
        self._episode_buffer: deque[dict[str, Any]] = deque(maxlen=32)
        self._consecutive_tick_failures = 0
        self._last_degradation_at = 0.0
        self._hooks_wired = False
        self._last_hook_results: dict[str, Any] = {}

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        if not self.config.enabled:
            logger.info("MorphogeneticRuntime disabled by config.")
            return
        try:
            self.registry.load()
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            self._last_degradation_at = time.time()
            _record_morphogenesis_runtime_degradation(
                exc,
                action="started morphogenesis runtime with fresh in-memory registry after registry load failed",
                severity="degraded",
            )
        self._stopping.clear()
        self._started_at = time.time()
        self._task = create_tracked_task(
            self._run_loop(),
            name="morphogenesis.runtime",
        )
        logger.info("MorphogeneticRuntime started.")

    async def stop(self) -> None:
        self._stopping.set()
        task = self._task
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # no-op: intentional
        try:
            await asyncio.to_thread(self.registry.save)
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            self._last_degradation_at = time.time()
            _record_morphogenesis_runtime_degradation(
                exc,
                action="completed morphogenesis shutdown while registry save failed; in-memory state was preserved until process exit",
                severity="critical",
            )
        logger.info("MorphogeneticRuntime stopped.")

    async def on_stop_async(self) -> None:
        """ServiceContainer lifecycle hook — ensures clean shutdown."""
        await self.stop()

    def emit_signal(self, signal: MorphogenSignal) -> None:
        if signal.ttl_ticks <= 0:
            return
        self._signals.append(signal)
        try:
            self.field.ingest_signal(signal)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_degradation_at = time.time()
            _record_morphogenesis_runtime_degradation(
                exc,
                action="queued morphogen signal but skipped field ingestion after field update failed",
                severity="degraded",
                extra={"signal_kind": str(signal.kind), "subsystem": signal.subsystem},
            )

    def mark_hooks_wired(self, results: dict[str, Any] | None = None, *, ok: bool) -> None:
        self._hooks_wired = ok
        self._last_hook_results = dict(results or {})

    def observe_exception(
        self,
        *,
        subsystem: str,
        exc: BaseException,
        source: str = "runtime_exception",
        danger: float = 0.75,
        stack_trace: str | None = None,
    ) -> MorphogenSignal:
        stack = stack_trace or "".join(traceback.format_exception(type(exc), exc, getattr(exc, "__traceback__", None)))
        sig = MorphogenSignal(
            kind=SignalKind.EXCEPTION,
            source=source,
            subsystem=subsystem,
            intensity=danger,
            payload={
                "exception_type": type(exc).__name__,
                "message": str(exc)[:500],
                "stack_trace": stack[-4000:],
            },
            ttl_ticks=8,
        )
        self.emit_signal(sig)
        return sig

    async def tick(self) -> dict[str, Any]:
        started = time.monotonic()
        self._tick += 1
        self._last_tick_at = time.time()

        # Heartbeat SelfHealing watchdog — proves the runtime is alive.
        try:
            from core.morphogenesis.hooks import heartbeat_self_healing
            heartbeat_self_healing()
        except (ImportError, AttributeError, RuntimeError):
            pass  # no-op: intentional

        resource = self.metabolism.pulse()
        self._emit_system_signals(resource_pressure=resource.pressure)

        # Modulate MetabolicCoordinator energy refill based on field pressure.
        # This is how morphogenesis influences resource allocation: under high
        # danger/pressure, background autonomous tasks slow down; under high
        # growth/curiosity, they speed up.
        if self._tick % 5 == 0:
            try:
                from core.morphogenesis.hooks import modulate_metabolic_energy
                modulate_metabolic_energy()
            except (ImportError, AttributeError, RuntimeError):
                pass  # no-op: intentional

        active_signals = self._consume_signals()
        for sig in active_signals:
            self.field.ingest_signal(sig)
        self.field.diffuse_step()

        if self.config.adaptive_immunity_bridge:
            await self._bridge_signals_to_immunity(active_signals)

        active_results = []
        activated_ids: list[str] = []
        success = True

        cells = self.registry.active_cells()[: self.config.max_cells]
        for cell in cells:
            priority = max(0.05, float(cell.manifest.criticality))
            self.metabolism.ensure_budget(
                cell.cell_id,
                priority=priority,
                baseline=cell.manifest.baseline_energy,
                max_energy=cell.manifest.max_energy,
            )
            result = await cell.tick(
                signals=active_signals,
                field=self.field,
                global_energy=self.metabolism.global_energy,
            )
            if result.activated:
                activated_ids.append(cell.cell_id)
                active_results.append(result.to_dict())
                success = success and result.success
                for out in result.emitted_signals:
                    self.emit_signal(out)
                if len(active_results) >= self.config.max_cell_actions_per_tick:
                    break

        if len(activated_ids) >= 2:
            task_signature = self._task_signature(active_signals)
            subsystem = self._dominant_subsystem(active_signals) or "composite"
            self.organ_stabilizer.observe_activation(
                activated_ids,
                success=success,
                task_signature=task_signature,
                subsystem=subsystem,
            )
            for organ in self.organ_stabilizer.discover():
                cell = self.registry.register_organ(organ)
                if cell is not None:
                    self.emit_signal(
                        MorphogenSignal(
                            kind=SignalKind.GROWTH,
                            source="organ_stabilizer",
                            subsystem=organ.subsystem,
                            intensity=min(0.95, organ.confidence),
                            payload={"organ": organ.to_dict()},
                            ttl_ticks=10,
                        )
                    )
                    # Record organ formation in episodic memory — this is how
                    # morphogenesis drives long-term behavioural development.
                    try:
                        from core.morphogenesis.hooks import record_organ_formation_episode
                        from core.runtime.task_ownership import fire_and_forget
                        # Compatibility contract: get_task_tracker().create_task(record_organ_formation_episode...)
                        fire_and_forget(
                            record_organ_formation_episode(organ.to_dict()),
                            name="morphogenesis.organ_episode",
                            bounded=True,
                        )
                    except (ImportError, AttributeError, RuntimeError):
                        pass  # no-op: intentional

        if self._tick % max(1, self.config.snapshot_every_ticks) == 0:
            await asyncio.to_thread(self.registry.save)

        await self._maybe_record_episode(active_results)

        return {
            "tick": self._tick,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
            "signals": [s.to_dict() for s in active_signals],
            "activated": activated_ids,
            "results": active_results,
            "resources": resource.to_dict(),
            "registry": self.registry.status(),
        }

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            sleep_s = max(0.05, self.config.tick_interval_s)
            try:
                if self._foreground_quiet_window_active():
                    self._last_tick_at = time.time()
                    with contextlib.suppress(Exception):
                        from core.morphogenesis.hooks import heartbeat_self_healing
                        heartbeat_self_healing()
                    await asyncio.sleep(max(0.5, self.config.tick_interval_s))
                    continue
                await self.tick()
                self._last_tick_error = ""
                self._consecutive_tick_failures = 0
            except asyncio.CancelledError:
                raise
            except (ImportError, AttributeError, RuntimeError) as exc:
                self._consecutive_tick_failures += 1
                self._last_degradation_at = time.time()
                sleep_s = min(5.0, sleep_s * (1 + self._consecutive_tick_failures))
                _record_morphogenesis_runtime_degradation(
                    exc,
                    action="emitted morphogenesis error signal and backed off after tick failure",
                    severity="degraded",
                    extra={"consecutive_tick_failures": self._consecutive_tick_failures},
                )
                self._last_tick_error = f"{type(exc).__name__}: {exc}"
                logger.error("Morphogenesis tick failed: %s", self._last_tick_error, exc_info=True)
                self.emit_signal(
                    MorphogenSignal(
                        kind=SignalKind.ERROR,
                        source="morphogenesis.runtime",
                        subsystem="morphogenesis",
                        intensity=0.82,
                        payload={"error": self._last_tick_error},
                    )
                )
            await asyncio.sleep(sleep_s)

    @staticmethod
    def _foreground_quiet_window_active() -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False
            quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
            return quiet_until > time.time()
        except (ImportError, AttributeError, RuntimeError):
            return False

    def _consume_signals(self) -> list[MorphogenSignal]:
        out: list[MorphogenSignal] = []
        while self._signals and len(out) < self.config.max_signals_per_tick:
            sig = self._signals.popleft()
            if sig.ttl_ticks <= 0:
                continue
            out.append(sig)
            # requeue if still alive
            new_ttl = sig.ttl_ticks - self.config.signal_decay_per_tick
            if new_ttl > 0 and sig.intensity > 0.02:
                self._signals.append(
                    MorphogenSignal(
                        kind=sig.kind,
                        source=sig.source,
                        subsystem=sig.subsystem,
                        intensity=sig.intensity * 0.92,
                        payload=sig.payload,
                        target_cell_id=sig.target_cell_id,
                        ttl_ticks=new_ttl,
                        timestamp=sig.timestamp,
                        signal_id=sig.signal_id,
                    )
                )
        return out

    def _emit_system_signals(self, *, resource_pressure: float) -> None:
        self.emit_signal(
            MorphogenSignal(
                kind=SignalKind.HEARTBEAT,
                source="morphogenesis.runtime",
                subsystem="global",
                intensity=0.18,
                ttl_ticks=2,
            )
        )
        if resource_pressure > 0.65:
            self.emit_signal(
                MorphogenSignal(
                    kind=SignalKind.RESOURCE_PRESSURE,
                    source="metabolism",
                    subsystem="global",
                    intensity=resource_pressure,
                    ttl_ticks=4,
                )
            )

        # Read existing Aura state opportunistically. All failures are non-fatal.
        try:
            from core.container import ServiceContainer
            liquid = ServiceContainer.get("liquid_state", default=None)
            if liquid is not None and hasattr(liquid, "get_status"):
                status = liquid.get_status()
                if isinstance(status, dict):
                    curiosity = float(status.get("curiosity", 0.0)) / 100.0
                    energy = float(status.get("energy", 50.0)) / 100.0
                    if curiosity > 0.45:
                        self.emit_signal(MorphogenSignal(kind=SignalKind.CURIOSITY, source="liquid_state", subsystem="cognition", intensity=curiosity, ttl_ticks=3))
                    if energy < 0.25:
                        self.emit_signal(MorphogenSignal(kind=SignalKind.HOMEOSTASIS, source="liquid_state", subsystem="global", intensity=1.0 - energy, ttl_ticks=3))
        except (ImportError, AttributeError, RuntimeError):
            pass  # no-op: intentional

    async def _bridge_signals_to_immunity(self, signals: Sequence[MorphogenSignal]) -> None:
        for sig in signals[:8]:
            kind = sig.kind.value if hasattr(sig.kind, "value") else str(sig.kind)
            if kind not in {SignalKind.ERROR.value, SignalKind.EXCEPTION.value, SignalKind.DANGER.value, SignalKind.RESOURCE_PRESSURE.value}:
                continue
            if sig.intensity < 0.55:
                continue
            try:
                from core.adaptation.adaptive_immunity import get_adaptive_immune_system
                immune = get_adaptive_immune_system()
                event = {
                    "type": kind,
                    "text": str(sig.payload.get("message") or sig.payload.get("error") or kind),
                    "subsystem": sig.subsystem,
                    "source": f"morphogenesis:{sig.source}",
                    "danger": float(sig.intensity),
                    "resource_pressure": float(sig.intensity if kind == SignalKind.RESOURCE_PRESSURE.value else sig.payload.get("resource_pressure", 0.0)),
                    "stack_trace": str(sig.payload.get("stack_trace", ""))[-4000:],
                    "exception_type": str(sig.payload.get("exception_type", "")),
                    "timestamp": sig.timestamp,
                    "error_signature": str(sig.payload.get("exception_type") or sig.payload.get("error") or kind)[:120],
                }
                result = immune.observe_event(event)
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, timeout=3.0)
            except (ImportError, AttributeError, RuntimeError) as exc:
                self._last_degradation_at = time.time()
                _record_morphogenesis_runtime_degradation(
                    exc,
                    action="kept morphogenesis tick active while adaptive immunity bridge skipped one signal",
                    severity="warning",
                    extra={"signal_kind": kind, "subsystem": sig.subsystem},
                )
                logger.debug("Adaptive immunity bridge skipped: %s", exc)

    async def _maybe_record_episode(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return
        self._events_since_episode += len(results)
        self._episode_buffer.extend(results[-8:])
        if self._events_since_episode < self.config.episode_every_events:
            return
        self._events_since_episode = 0

        try:
            from core.container import ServiceContainer
            mem = ServiceContainer.get("episodic_memory", default=None)
            if mem is None:
                try:
                    from core.memory.episodic_memory import get_episodic_memory
                    mem = get_episodic_memory()
                except (ImportError, AttributeError, RuntimeError):
                    mem = None
            if mem is None or not hasattr(mem, "record_episode_async"):
                return
            failures = [r for r in self._episode_buffer if not r.get("success", True)]
            await mem.record_episode_async(
                context="MorphogeneticRuntime self-organization cycle",
                action="cellular_tick_and_organ_stabilization",
                outcome=f"{len(self._episode_buffer)} cell activations, failures={len(failures)}",
                success=not failures,
                emotional_valence=-0.25 if failures else 0.18,
                tools_used=["morphogenesis_runtime"],
                lessons=[
                    "Stable co-activated cells can be formalized into organs",
                    "High-danger signals must route through adaptive immunity",
                ][: 1 + bool(failures)],
                importance=0.55 if not failures else 0.75,
                source="morphogenesis",
                metadata={"tick": self._tick, "failure_count": len(failures)},
            )
            self._episode_buffer.clear()
        except (ImportError, AttributeError, RuntimeError) as exc:
            self._last_degradation_at = time.time()
            _record_morphogenesis_runtime_degradation(
                exc,
                action="kept morphogenesis tick active while episodic episode write was skipped",
                severity="warning",
                extra={"buffered_events": len(self._episode_buffer)},
            )
            logger.debug("morphogenesis episode record skipped: %s", exc)

    @staticmethod
    def _task_signature(signals: Sequence[MorphogenSignal]) -> str:
        parts = []
        for s in signals[:5]:
            parts.append(str(s.kind.value if hasattr(s.kind, "value") else s.kind))
            if s.payload.get("task"):
                parts.append(str(s.payload.get("task"))[:80])
        return stable_digest(*parts, length=12) if parts else ""

    @staticmethod
    def _dominant_subsystem(signals: Sequence[MorphogenSignal]) -> str:
        if not signals:
            return ""
        counts: dict[str, float] = {}
        for s in signals:
            counts[s.subsystem] = counts.get(s.subsystem, 0.0) + s.intensity
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "running": bool(self._task and not self._task.done()),
            "tick": self._tick,
            "started_at": self._started_at,
            "last_tick_at": self._last_tick_at,
            "last_tick_error": self._last_tick_error,
            "consecutive_tick_failures": self._consecutive_tick_failures,
            "last_degradation_at": self._last_degradation_at,
            "hooks_wired": self._hooks_wired,
            "last_hook_results": self._last_hook_results,
            "queued_signals": len(self._signals),
            "field": self.field.to_dict(),
            "metabolism": self.metabolism.status(),
            "registry": self.registry.status(),
            "organs": self.organ_stabilizer.to_dict().get("known_organs", {}),
        }


_runtime_singleton: MorphogeneticRuntime | None = None


def get_morphogenetic_runtime() -> MorphogeneticRuntime:
    global _runtime_singleton
    if _runtime_singleton is None:
        _runtime_singleton = MorphogeneticRuntime()
    return _runtime_singleton
