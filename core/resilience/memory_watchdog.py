"""MemoryWatchdog: out-of-band memory ceiling enforcement.

Every other memory-enforcement path in Aura (MemoryGovernor,
AppleSiliconMemoryMonitor, VRAM purges) runs as an asyncio task. When the
host starts swapping, the event loop stalls — which means the enforcement
paths go blind at exactly the moment they are needed. Observed failure
this module exists to prevent: a single live chat turn pushed the process
tree from ~17 GB RSS to 110 GB, macOS exhausted swap, and the machine
froze while the in-loop governor never got scheduled again.

Like StallWatchdog, this runs in its own daemon thread so it keeps acting
even when the loop is wedged. It enforces a three-stage ladder over the
managed RSS (core process + all child workers):

- soft ceiling: schedule the in-loop MemoryGovernor sweep (graceful
  prune/unload). If the loop is healthy this is the whole story.
- hard ceiling: act from the thread itself, no event loop required —
  terminate heavyweight child workers (mlx/llama/Metal) and force a full
  gc pass.
- lethal ceiling: after consecutive confirmations and a hard action that
  failed to reclaim, write a tombstone with the recent samples and exit
  with a categorized status code. A clean, explained crash that a
  supervisor (or the operator) can restart beats freezing the host.

Swap exhaustion escalates the ladder: high swap with elevated managed RSS
is treated as the hard tier even if RSS alone is under the ceiling.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from core.runtime.errors import record_degradation
from core.utils.memory_monitor import process_memory_bytes

logger = logging.getLogger("Aura.Resilience.MemoryWatchdog")

_WATCHDOG_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    psutil.Error,
)

# Exit status for the lethal path. Chosen to be greppable and distinct:
# EX_SOFTWARE (70) — "internal software error" — categorized OOM abort.
MEMORY_ABORT_EXIT_CODE = 70

# Child workers the hard tier is allowed to terminate out-of-band. These
# are inference/runtime workers that the lane clients know how to respawn;
# killing them loses no durable state.
_HEAVY_WORKER_MARKERS = ("mlx_worker.py", "MTLCompilerService")

_TOMBSTONE_DIR = Path("data/error_logs/memory")
_DARWIN_CHILD_LIBPROC: Any | None = None
_DARWIN_CHILD_LIBPROC_UNAVAILABLE = False


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = str(os.environ.get(name, default) or default).strip().lower()
    return value if value in allowed else default


@dataclass(frozen=True)
class MemorySample:
    """One out-of-band measurement of the managed process tree."""

    core_rss_mb: float
    child_rss_mb: float
    swap_used_gb: float
    system_percent: float
    total_ram_gb: float
    sampled_at: float

    @property
    def managed_rss_mb(self) -> float:
        return self.core_rss_mb + self.child_rss_mb


@dataclass
class WatchdogAction:
    at: float
    tier: str
    detail: str
    managed_rss_mb: float


@dataclass
class _Thresholds:
    soft_mb: float
    hard_mb: float
    lethal_mb: float
    swap_hard_gb: float
    soft_cooldown_s: float = 30.0
    hard_cooldown_s: float = 60.0
    lethal_confirmations: int = 2
    boot_grace_s: float = 300.0

    @classmethod
    def from_environment(cls, total_ram_gb: float) -> _Thresholds:
        # Daily-use defaults for the 64 GB desktop path. The previous 48/56 GB
        # hard/lethal tiers were too late once Chrome, Safari, the UI, and
        # compressed MLX pages were present; macOS could cross into global
        # application-memory failure before Aura reclaimed. Scale down on
        # smaller machines while preserving explicit operator overrides.
        total_mb = max(8192.0, total_ram_gb * 1024.0)
        return cls(
            soft_mb=_env_float("AURA_MEMWATCH_SOFT_MB", min(32768.0, total_mb * 0.50)),
            hard_mb=_env_float("AURA_MEMWATCH_HARD_MB", min(40960.0, total_mb * 0.62)),
            lethal_mb=_env_float("AURA_MEMWATCH_LETHAL_MB", min(46080.0, total_mb * 0.70)),
            swap_hard_gb=_env_float(
                "AURA_MEMWATCH_SWAP_HARD_GB",
                min(8.0, max(2.0, total_ram_gb * 0.12)),
            ),
            soft_cooldown_s=_env_float("AURA_MEMWATCH_SOFT_COOLDOWN_S", 30.0),
            hard_cooldown_s=_env_float("AURA_MEMWATCH_HARD_COOLDOWN_S", 60.0),
            lethal_confirmations=max(
                2, int(_env_float("AURA_MEMWATCH_LETHAL_CONFIRMS", 2.0))
            ),
            boot_grace_s=_env_float("AURA_MEMWATCH_BOOT_GRACE_S", 300.0),
        )


def _phys_footprint_mb(pid: int) -> float:
    """Return the canonical RSS/phys-footprint memory sample in MB."""
    try:
        return float(process_memory_bytes(pid)) / float(1024 * 1024)
    except _WATCHDOG_RECOVERABLE_ERRORS:
        return 0.0


def _darwin_child_pids(root_pid: int, *, recursive: bool, max_children: int = 64) -> list[int]:
    """Return child pids via libproc on macOS without psutil's full ppid map.

    A live stall trace showed ``psutil.Process.children(recursive=True)`` stuck
    in the watchdog thread while the event loop was already wedged. On Darwin,
    ``proc_listchildpids`` gives a bounded direct-child query without a global
    process-table ppid map or a production raw-subprocess surface.
    """

    global _DARWIN_CHILD_LIBPROC, _DARWIN_CHILD_LIBPROC_UNAVAILABLE
    if sys.platform != "darwin" or _DARWIN_CHILD_LIBPROC_UNAVAILABLE:
        return []
    seen: set[int] = set()
    frontier = [int(root_pid)]
    deadline = time.monotonic() + 0.75
    try:
        if _DARWIN_CHILD_LIBPROC is None:
            _DARWIN_CHILD_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
            _DARWIN_CHILD_LIBPROC.proc_listchildpids.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            _DARWIN_CHILD_LIBPROC.proc_listchildpids.restype = ctypes.c_int
    except (AttributeError, OSError, TypeError, ValueError):
        _DARWIN_CHILD_LIBPROC_UNAVAILABLE = True
        return []

    while frontier and len(seen) < max_children and time.monotonic() < deadline:
        parent = frontier.pop(0)
        try:
            buffer = (ctypes.c_int * max_children)()
            count = int(
                _DARWIN_CHILD_LIBPROC.proc_listchildpids(
                    int(parent),
                    ctypes.byref(buffer),
                    ctypes.sizeof(buffer),
                )
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, ctypes.ArgumentError):
            _DARWIN_CHILD_LIBPROC_UNAVAILABLE = True
            break
        if count <= 0:
            break
        for raw_pid in list(buffer)[: min(count, max_children)]:
            pid = int(raw_pid)
            if pid <= 0:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            if recursive and len(seen) < max_children:
                frontier.append(pid)
    return list(seen)


def _child_processes(root_pid: int, *, recursive: bool = True) -> list[psutil.Process]:
    if sys.platform == "darwin":
        return [psutil.Process(pid) for pid in _darwin_child_pids(root_pid, recursive=recursive)]
    return psutil.Process(root_pid).children(recursive=recursive)


def default_sampler() -> MemorySample:
    proc = psutil.Process(os.getpid())
    core_rss = 0.0
    child_rss = 0.0
    try:
        core_rss = proc.memory_info().rss / (1024 * 1024)
    except _WATCHDOG_RECOVERABLE_ERRORS:
        pass
    # Compression-aware: managed memory is the larger of RSS and the
    # kernel's phys_footprint view of this process.
    core_rss = max(core_rss, _phys_footprint_mb(os.getpid()))
    try:
        for child in _child_processes(proc.pid, recursive=True):
            try:
                child_rss += max(
                    child.memory_info().rss / (1024 * 1024),
                    _phys_footprint_mb(child.pid),
                )
            except _WATCHDOG_RECOVERABLE_ERRORS:
                continue
    except _WATCHDOG_RECOVERABLE_ERRORS:
        pass
    try:
        swap_used_gb = psutil.swap_memory().used / (1024**3)
    except _WATCHDOG_RECOVERABLE_ERRORS:
        swap_used_gb = 0.0
    try:
        vm = psutil.virtual_memory()
        system_percent = float(getattr(vm, "percent", 0.0) or 0.0)
        total_ram_gb = float(getattr(vm, "total", 0) or 0) / (1024**3)
    except _WATCHDOG_RECOVERABLE_ERRORS:
        system_percent = 0.0
        total_ram_gb = 0.0
    return MemorySample(
        core_rss_mb=core_rss,
        child_rss_mb=child_rss,
        swap_used_gb=swap_used_gb,
        system_percent=system_percent,
        total_ram_gb=total_ram_gb,
        sampled_at=time.time(),
    )


def terminate_heavy_child_workers(grace_s: float = 2.0) -> int:
    """Terminate inference child workers out-of-band. Returns count killed."""
    killed = 0
    try:
        children = _child_processes(os.getpid(), recursive=True)
    except _WATCHDOG_RECOVERABLE_ERRORS as exc:
        logger.debug("MemoryWatchdog: child scan failed: %s", exc)
        return 0
    doomed: list[psutil.Process] = []
    for child in children:
        try:
            cmd = " ".join(child.cmdline())
        except _WATCHDOG_RECOVERABLE_ERRORS:
            continue
        if any(marker in cmd for marker in _HEAVY_WORKER_MARKERS):
            try:
                child.terminate()
                doomed.append(child)
                killed += 1
                logger.warning(
                    "🛑 [MEMWATCH] Terminated heavy worker pid=%s cmd=%s",
                    child.pid,
                    cmd[:160],
                )
            except _WATCHDOG_RECOVERABLE_ERRORS:
                continue
    if doomed:
        _, alive = psutil.wait_procs(doomed, timeout=grace_s)
        for child in alive:
            try:
                child.kill()
            except _WATCHDOG_RECOVERABLE_ERRORS:
                continue
    return killed


class MemoryWatchdog(threading.Thread):
    """Daemon thread enforcing hard memory ceilings independent of the loop."""

    def __init__(
        self,
        *,
        loop: Any = None,
        governor: Any = None,
        sample_interval_s: float | None = None,
        thresholds: _Thresholds | None = None,
        lethal_action: str | None = None,
        sampler: Callable[[], MemorySample] | None = None,
        worker_terminator: Callable[[], int] | None = None,
        gc_collect: Callable[[], int] | None = None,
        process_exit: Callable[[int], None] | None = None,
    ):
        super().__init__(daemon=True, name="AuraMemoryWatchdog")
        self._loop = loop
        self._governor = governor
        try:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
        except _WATCHDOG_RECOVERABLE_ERRORS:
            total_ram_gb = 64.0
        self.thresholds = thresholds or _Thresholds.from_environment(total_ram_gb)
        self.sample_interval_s = sample_interval_s if sample_interval_s is not None else _env_float(
            "AURA_MEMWATCH_INTERVAL_S", 3.0
        )
        self.lethal_action = lethal_action or _env_choice(
            "AURA_MEMWATCH_LETHAL_ACTION", "exit", ("exit", "shed", "off")
        )
        self._sampler = sampler or default_sampler
        self._worker_terminator = worker_terminator or terminate_heavy_child_workers
        self._gc_collect = gc_collect or gc.collect
        self._process_exit = process_exit or self._default_process_exit
        self._stop_event = threading.Event()
        self._started_at = time.monotonic()
        self._last_soft_action_at = 0.0
        self._last_hard_action_at = 0.0
        self._spike_count = 0
        self._spike_dumps = 0
        self._last_spike_dump_at = 0.0
        self._lethal_streak = 0
        self._hard_attempted_in_streak = False
        self._last_sample: MemorySample | None = None
        self._actions: list[WatchdogAction] = []
        self._tick_failures = 0

    # ── public surface ────────────────────────────────────────────────

    def health_snapshot(self) -> dict[str, Any]:
        sample = self._last_sample
        return {
            "running": self.is_alive(),
            "lethal_action": self.lethal_action,
            "sample_interval_s": self.sample_interval_s,
            "thresholds": asdict(self.thresholds),
            "tick_failures": self._tick_failures,
            "lethal_streak": self._lethal_streak,
            "last_sample": asdict(sample) if sample else None,
            "recent_actions": [asdict(a) for a in self._actions[-10:]],
        }

    @property
    def last_sample(self) -> MemorySample | None:
        return self._last_sample

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self is threading.current_thread() or not self.is_alive():
            return
        self.join(timeout=max(0.0, float(timeout_s)))
        if self.is_alive():
            raise TimeoutError(
                f"memory watchdog did not stop within {float(timeout_s):.1f}s"
            )

    # ── thread loop ───────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "🛡️ MemoryWatchdog active (soft=%.0fMB hard=%.0fMB lethal=%.0fMB "
            "swap_hard=%.1fGB interval=%.1fs lethal_action=%s)",
            self.thresholds.soft_mb,
            self.thresholds.hard_mb,
            self.thresholds.lethal_mb,
            self.thresholds.swap_hard_gb,
            self.sample_interval_s,
            self.lethal_action,
        )
        while not self._stop_event.is_set():
            try:
                self._tick()
                self._tick_failures = 0
            except _WATCHDOG_RECOVERABLE_ERRORS as exc:
                self._tick_failures += 1
                record_degradation(
                    "memory_watchdog",
                    exc,
                    severity="warning",
                    action="kept out-of-band memory watchdog alive after tick failure",
                )
                logger.debug("MemoryWatchdog tick failed: %s", exc)
            # Adaptive cadence: past the hard ceiling every second counts —
            # a runaway in-process allocation can add gigabytes between
            # relaxed samples.
            wait_s = self.sample_interval_s
            sample = self._last_sample
            if sample is not None and sample.managed_rss_mb >= self.thresholds.hard_mb:
                wait_s = min(1.0, wait_s)
            self._stop_event.wait(wait_s)

    # ── policy ────────────────────────────────────────────────────────

    # A routine MLX generation wires ~20GB in one sample interval; without
    # a throttle the spike dumper wrote 1,568 identical stack dumps (55MB)
    # in one live afternoon. First occurrences keep full diagnostics; the
    # steady state costs one counter increment.
    SPIKE_DUMP_MIN_INTERVAL_S = 600.0
    SPIKE_DUMP_LIFETIME_CAP = 12

    def _tick(self) -> None:
        sample = self._sampler()
        previous = self._last_sample
        self._last_sample = sample
        if (
            previous is not None
            and (sample.managed_rss_mb - previous.managed_rss_mb) > 8192.0
        ):
            self._record_footprint_spike(previous, sample)
        self._evaluate(sample, time.monotonic())

    def _record_footprint_spike(self, previous: MemorySample, sample: MemorySample) -> None:
        self._spike_count += 1
        why = (
            f"footprint spike {previous.managed_rss_mb:.0f}→"
            f"{sample.managed_rss_mb:.0f}MB in one interval "
            f"(spike #{self._spike_count} this process)"
        )
        now = time.monotonic()
        if self._spike_dumps >= self.SPIKE_DUMP_LIFETIME_CAP:
            if self._spike_dumps == self.SPIKE_DUMP_LIFETIME_CAP:
                self._spike_dumps += 1
                logger.warning(
                    "[MEMWATCH] %s — lifetime stack-dump cap (%d) reached; "
                    "further spikes are counted but not dumped.",
                    why,
                    self.SPIKE_DUMP_LIFETIME_CAP,
                )
            return
        if (
            self._last_spike_dump_at
            and (now - self._last_spike_dump_at) < self.SPIKE_DUMP_MIN_INTERVAL_S
        ):
            logger.info("[MEMWATCH] %s — stack dump throttled.", why)
            return
        self._last_spike_dump_at = now
        self._spike_dumps += 1
        self._dump_thread_stacks(why)

    def _evaluate(self, sample: MemorySample, now: float) -> str:
        """Apply the escalation ladder to one sample. Returns the tier acted on."""
        managed = sample.managed_rss_mb
        t = self.thresholds

        swap_escalation = (
            sample.swap_used_gb >= t.swap_hard_gb and managed >= t.soft_mb
        )

        if managed >= t.lethal_mb:
            return self._handle_lethal(sample, now)
        self._lethal_streak = 0
        self._hard_attempted_in_streak = False

        if managed >= t.hard_mb or swap_escalation:
            return self._handle_hard(sample, now, swap_escalation=swap_escalation)

        if managed >= t.soft_mb or sample.system_percent >= 92.0:
            return self._handle_soft(sample, now)

        return "none"

    def _handle_soft(self, sample: MemorySample, now: float) -> str:
        if (now - self._last_soft_action_at) < self.thresholds.soft_cooldown_s:
            return "soft_cooldown"
        self._last_soft_action_at = now
        self._remember("soft", sample, "scheduled governor sweep")
        logger.warning(
            "⚠️ [MEMWATCH] Soft ceiling: managed RSS %.0fMB (sys %.1f%%). "
            "Scheduling governor sweep.",
            sample.managed_rss_mb,
            sample.system_percent,
        )
        self._schedule_governor_sweep()
        return "soft"

    def _handle_hard(
        self, sample: MemorySample, now: float, *, swap_escalation: bool
    ) -> str:
        if (now - self._last_hard_action_at) < self.thresholds.hard_cooldown_s:
            return "hard_cooldown"
        self._last_hard_action_at = now
        reason = "swap exhaustion" if swap_escalation else "hard RSS ceiling"
        self._dump_thread_stacks(f"hard tier at {sample.managed_rss_mb:.0f}MB")
        logger.critical(
            "🚨 [MEMWATCH] %s: managed RSS %.0fMB swap %.1fGB. "
            "Out-of-band reclaim (terminate heavy workers + gc).",
            reason,
            sample.managed_rss_mb,
            sample.swap_used_gb,
        )
        killed = self._worker_terminator()
        collected = self._gc_collect()
        self._remember(
            "hard", sample, f"{reason}: killed={killed} gc_collected={collected}"
        )
        record_degradation(
            "memory_watchdog",
            RuntimeError(
                f"{reason}: managed RSS {sample.managed_rss_mb:.0f}MB, "
                f"swap {sample.swap_used_gb:.1f}GB"
            ),
            severity="critical",
            action=f"terminated {killed} heavy workers and forced gc out-of-band",
        )
        # Also nudge the graceful path in case the loop is still breathing.
        self._schedule_governor_sweep()
        return "hard"

    def _handle_lethal(self, sample: MemorySample, now: float) -> str:
        self._lethal_streak += 1
        in_boot_grace = (now - self._started_at) < self.thresholds.boot_grace_s

        if not self._hard_attempted_in_streak:
            # Always try reclaiming before considering the terminal action.
            self._hard_attempted_in_streak = True
            self._last_hard_action_at = now
            killed = self._worker_terminator()
            collected = self._gc_collect()
            self._remember(
                "lethal_reclaim",
                sample,
                f"pre-abort reclaim: killed={killed} gc_collected={collected}",
            )
            logger.critical(
                "🚨 [MEMWATCH] LETHAL ceiling: managed RSS %.0fMB ≥ %.0fMB. "
                "Reclaimed (killed=%d). Next confirmation aborts.",
                sample.managed_rss_mb,
                self.thresholds.lethal_mb,
                killed,
            )
            return "lethal_reclaim"

        if self._lethal_streak < self.thresholds.lethal_confirmations + 1:
            return "lethal_pending"

        if self.lethal_action == "off" or in_boot_grace:
            self._remember(
                "lethal_suppressed",
                sample,
                "boot grace" if in_boot_grace else "lethal_action=off",
            )
            logger.critical(
                "🚨 [MEMWATCH] Lethal ceiling persists (%.0fMB) but abort is "
                "suppressed (%s).",
                sample.managed_rss_mb,
                "boot grace" if in_boot_grace else "lethal_action=off",
            )
            return "lethal_suppressed"

        if self.lethal_action == "shed":
            self._last_hard_action_at = now
            killed = self._worker_terminator()
            self._remember("lethal_shed", sample, f"repeat shed: killed={killed}")
            return "lethal_shed"

        # lethal_action == "exit": categorized abort.
        self._write_tombstone(sample)
        self._remember("lethal_exit", sample, f"exit({MEMORY_ABORT_EXIT_CODE})")
        logger.critical(
            "💀 [MEMWATCH] Managed RSS %.0fMB exceeded lethal ceiling %.0fMB "
            "after reclaim attempts. Aborting with exit code %d to protect "
            "the host (tombstone written).",
            sample.managed_rss_mb,
            self.thresholds.lethal_mb,
            MEMORY_ABORT_EXIT_CODE,
        )
        self._process_exit(MEMORY_ABORT_EXIT_CODE)
        return "lethal_exit"

    # ── helpers ───────────────────────────────────────────────────────

    def _dump_thread_stacks(self, why: str) -> None:
        """Snapshot every thread's stack at memory-spike time.

        The 78GB compressed runaway died with the allocator anonymous.
        At hard tier the allocator is, with high probability, ON one of
        these stacks — faulthandler writes them without allocating
        Python objects, safe under pressure.
        """
        try:
            import faulthandler

            crash_dir = Path("data/error_logs/crash")
            crash_dir.mkdir(parents=True, exist_ok=True)
            spike_log = crash_dir / "memory_spike_stacks.log"
            try:
                # Allocation-light rotation: a stat + rename keeps the log
                # bounded (observed 54MB unrotated growth in live use).
                if spike_log.exists() and spike_log.stat().st_size > 16 * 1024 * 1024:
                    spike_log.replace(spike_log.with_suffix(".log.1"))
            except OSError:
                pass
            with open(spike_log, "a") as fh:
                fh.write(f"\n===== {why} pid={os.getpid()} at={time.time()} =====\n")
                fh.flush()
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("MemoryWatchdog stack dump failed: %s", exc)

    def _remember(self, tier: str, sample: MemorySample, detail: str) -> None:
        self._actions.append(
            WatchdogAction(
                at=time.time(),
                tier=tier,
                detail=detail[:240],
                managed_rss_mb=sample.managed_rss_mb,
            )
        )
        if len(self._actions) > 50:
            self._actions = self._actions[-50:]

    def _schedule_governor_sweep(self) -> None:
        loop = self._loop
        governor = self._governor
        if loop is None or governor is None:
            return
        try:
            if loop.is_closed():
                return

            def _kick() -> None:
                try:
                    from core.runtime.task_ownership import create_tracked_task

                    create_tracked_task(
                        governor.check(), name="memory_watchdog.governor_sweep"
                    )
                except _WATCHDOG_RECOVERABLE_ERRORS as exc:
                    logger.debug("MemoryWatchdog governor kick failed: %s", exc)

            loop.call_soon_threadsafe(_kick)
        except RuntimeError:
            return
        except _WATCHDOG_RECOVERABLE_ERRORS as exc:
            logger.debug("MemoryWatchdog could not schedule governor sweep: %s", exc)

    def _write_tombstone(self, sample: MemorySample) -> None:
        payload = {
            "schema": "aura.memory_watchdog.tombstone.v1",
            "reason": "managed RSS exceeded lethal ceiling after reclaim attempts",
            "exit_code": MEMORY_ABORT_EXIT_CODE,
            "written_at": time.time(),
            "thresholds": asdict(self.thresholds),
            "final_sample": asdict(sample),
            "recent_actions": [asdict(a) for a in self._actions[-20:]],
        }
        try:
            from core.runtime.atomic_writer import atomic_write_json

            _TOMBSTONE_DIR.mkdir(parents=True, exist_ok=True)
            path = _TOMBSTONE_DIR / f"oom_tombstone_{int(time.time())}.json"
            # Approved emergency writer: atomic_writer is an audited file
            # sink with no governed-gateway machinery to starve under OOM,
            # and a torn tombstone would be worse than none.
            atomic_write_json(
                path,
                payload,
                schema_version=1,
                schema_name="aura.memory_watchdog.tombstone",
            )
            logger.critical("💀 [MEMWATCH] Tombstone written: %s", path)
        except (OSError, RuntimeError, ImportError, TypeError, ValueError) as exc:
            logger.critical("💀 [MEMWATCH] Tombstone write failed: %s", exc)

    @staticmethod
    def _default_process_exit(code: int) -> None:
        # No logging.shutdown() here: flushing handlers can block
        # indefinitely under swap thrash — observed in the 115GB crash
        # where the lethal path never reached exit. The tombstone is
        # already on disk; die immediately.
        os._exit(code)


_WATCHDOG_SINGLETON: MemoryWatchdog | None = None
_WATCHDOG_LOCK = threading.Lock()


def get_memory_watchdog() -> MemoryWatchdog | None:
    return _WATCHDOG_SINGLETON


def start_memory_watchdog(*, loop: Any = None, governor: Any = None) -> MemoryWatchdog:
    """Start (or return) the process-wide memory watchdog thread."""
    global _WATCHDOG_SINGLETON
    with _WATCHDOG_LOCK:
        existing = _WATCHDOG_SINGLETON
        if existing is not None and existing.is_alive():
            if governor is not None:
                existing._governor = governor
            if loop is not None:
                existing._loop = loop
            return existing
        watchdog = MemoryWatchdog(loop=loop, governor=governor)
        watchdog.start()
        _WATCHDOG_SINGLETON = watchdog
        return watchdog


def stop_memory_watchdog() -> None:
    global _WATCHDOG_SINGLETON
    with _WATCHDOG_LOCK:
        if _WATCHDOG_SINGLETON is not None:
            _WATCHDOG_SINGLETON.stop()
            _WATCHDOG_SINGLETON = None
