import asyncio
import contextlib
import logging
import os
from dataclasses import asdict, dataclass

import psutil

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.MemoryMonitor")
_MEMORY_MONITOR_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    psutil.Error,
)


def _clamp_pressure(value: float) -> int:
    return max(0, min(100, int(value)))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        return float(default)


@dataclass(frozen=True)
class MemoryPressureSnapshot:
    pressure_pct: float
    available_gb: float
    total_gb: float
    warning_pct: float
    high_pct: float
    critical_pct: float
    emergency_pct: float
    min_available_gb: float
    level: str
    reason: str

    @property
    def warning(self) -> bool:
        return self.level in {"warning", "high", "critical", "emergency"}

    @property
    def high(self) -> bool:
        return self.level in {"high", "critical", "emergency"}

    @property
    def critical(self) -> bool:
        return self.level in {"critical", "emergency"}

    @property
    def emergency(self) -> bool:
        return self.level == "emergency"

    @property
    def should_gc(self) -> bool:
        return self.high

    @property
    def max_token_cap(self) -> int | None:
        if self.emergency:
            return 32
        if self.critical:
            return 64
        if self.high:
            return 192
        if self.warning:
            return 384
        return None

    @property
    def refuse_heavy_local_generation(self) -> bool:
        return self.emergency or self.available_gb < self.min_available_gb

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        payload = asdict(self)
        payload.update(
            {
                "warning": self.warning,
                "high": self.high,
                "critical": self.critical,
                "emergency": self.emergency,
                "should_gc": self.should_gc,
                "max_token_cap": self.max_token_cap,
                "refuse_heavy_local_generation": self.refuse_heavy_local_generation,
            }
        )
        return payload


def get_memory_pressure_snapshot() -> MemoryPressureSnapshot:
    """Return one canonical unified-memory pressure decision for runtime gates."""

    vm = psutil.virtual_memory()
    total_gb = float(getattr(vm, "total", 0) or 0) / float(1024**3)
    available_gb = float(getattr(vm, "available", 0) or 0) / float(1024**3)
    pressure_pct = float(getattr(vm, "percent", 0.0) or 0.0)
    if pressure_pct <= 0.0 and total_gb > 0.0:
        pressure_pct = max(0.0, min(100.0, (1.0 - (available_gb / total_gb)) * 100.0))

    if total_gb >= 60.0:
        warning_default = 78.0
        high_default = 84.0
        critical_default = 90.0
        emergency_default = 94.0
        min_available_default = 6.0
    else:
        warning_default = 72.0
        high_default = 80.0
        critical_default = 88.0
        emergency_default = 92.0
        min_available_default = 4.0

    warning_pct = _env_float("AURA_MEMORY_WARNING_PCT", warning_default)
    high_pct = _env_float("AURA_MEMORY_HIGH_PCT", high_default)
    critical_pct = _env_float("AURA_MEMORY_CRITICAL_PCT", critical_default)
    emergency_pct = _env_float("AURA_MEMORY_EMERGENCY_PCT", emergency_default)
    min_available_gb = _env_float("AURA_MEMORY_MIN_AVAILABLE_GB", min_available_default)

    if pressure_pct >= emergency_pct or available_gb < max(1.0, min_available_gb / 2.0):
        level = "emergency"
    elif pressure_pct >= critical_pct or available_gb < min_available_gb:
        level = "critical"
    elif pressure_pct >= high_pct:
        level = "high"
    elif pressure_pct >= warning_pct:
        level = "warning"
    else:
        level = "normal"

    reason = ""
    if level != "normal":
        reason = (
            f"memory_pressure:{pressure_pct:.1f}%/{available_gb:.1f}GB "
            f"(level={level}, critical>={critical_pct:.1f}%, emergency>={emergency_pct:.1f}%, "
            f"min_available={min_available_gb:.1f}GB)"
        )

    return MemoryPressureSnapshot(
        pressure_pct=pressure_pct,
        available_gb=available_gb,
        total_gb=total_gb,
        warning_pct=warning_pct,
        high_pct=high_pct,
        critical_pct=critical_pct,
        emergency_pct=emergency_pct,
        min_available_gb=min_available_gb,
        level=level,
        reason=reason,
    )


class AppleSiliconMemoryMonitor:
    """Monitors Unified Memory pressure on Apple Silicon (M1/M2/M3/M4/M5).
    
    Aura uses this to throttle background reasoning (ReasoningQueue)
    when memory pressure is high to avoid system swap lag.
    """
    def __init__(self, interval: float = 2.0, threshold: int = 85):
        self.interval = interval
        self.threshold = threshold
        self.is_running = False
        self._pressure = 0
        self._loop_task = None

    async def start(self):
        self.is_running = True
        # Use our new task tracker helper (hoisted from Part 5)
        from .task_tracker import fire_and_track
        self._loop_task = fire_and_track(self._monitor_loop(), name="MemoryMonitor")
        logger.info("Apple Silicon Memory Monitor active.")

    async def stop(self):
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task

    @property
    def pressure(self) -> int:
        """Returns 0-100 indicating memory pressure."""
        return self._pressure

    async def _monitor_loop(self):
        import gc as _gc
        last_gc_at = 0.0
        last_purge_at = 0.0
        while self.is_running:
            try:
                # Sample memory pressure off the event loop so watchdogs never
                # see a shell command or psutil hiccup as a global stall.
                self._pressure = await asyncio.to_thread(self._get_pressure_sysctl)
                if self._pressure >= self.threshold:
                    logger.warning(
                        "⚠️ HIGH MEMORY PRESSURE: %s%% (Threshold: %s%%)",
                        self._pressure,
                        self.threshold,
                    )
                    import time as _time
                    now = _time.monotonic()
                    # Run a generational gc once per minute when pressure is up.
                    # Sustained-growth recovery had no eviction step between the
                    # 85% warning and the 90% VRAM purge, so RAM kept climbing
                    # through the gap.
                    if now - last_gc_at > 60.0:
                        await asyncio.to_thread(_gc.collect)
                        last_gc_at = now
                    # Trigger VRAM purge if critical (kept on its own cooldown
                    # so we never spin-purge the GPU heap).
                    if self._pressure > 90 and now - last_purge_at > 30.0:
                        from core.managers.vram_manager import get_vram_manager
                        await asyncio.to_thread(get_vram_manager().purge)
                        last_purge_at = now

                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except _MEMORY_MONITOR_RECOVERABLE_ERRORS as e:
                record_degradation('memory_monitor', e)
                logger.error("Memory monitor error: %s", e)
                await asyncio.sleep(5)

    def _get_pressure_sysctl(self) -> int:
        """Return a safe system memory pressure sample using psutil."""
        try:
            mem = psutil.virtual_memory()
            percent = getattr(mem, "percent", None)
            if percent is not None:
                return _clamp_pressure(float(percent))

            total = int(getattr(mem, "total", 0) or 0)
            available = int(getattr(mem, "available", 0) or 0)
            if total > 0:
                return _clamp_pressure((1.0 - (available / total)) * 100.0)
            return 0
        except _MEMORY_MONITOR_RECOVERABLE_ERRORS as exc:
            record_degradation("memory_monitor", exc)
            logger.debug("Memory pressure sample failed: %s", exc)
            return 0
