"""core/runtime/foundations.py — one boot entry for the engineering spine.

Aura has grown a set of disciplines borrowed, clean-room, from projects
that earned them the expensive way: the Linux kernel (taint, lockdep, PSI,
OOM policy), LLVM (verifier, pass manager, sanitizers), Kubernetes
(reconcilers, admission, quota, probes, eviction, leases), ROS 2 (managed
lifecycles, QoS, declared parameters, bags, diagnostics), Chromium
(histograms, traces, memory-infra, field trials, layering), and flight
software from F Prime / Apollo / OpenMCT (telemetry dictionaries, command
sequencing, rate groups, restart protection, assertions).

Every one of those is worth nothing if it is a module nobody calls. This
file is the single place the runtime turns them on, in dependency order,
with one report describing what came up and what did not. `aura_main`
calls :func:`activate_foundations` once during boot; nothing else needs to
know the list.

Design rules, all deliberate:

* **On by default.** No activator is behind an opt-in flag. A discipline
  that has to be enabled is a discipline that is off in the incident you
  needed it for. Individual activators may be disabled for the
  foreground-only boot profile, which genuinely has no background lanes.
* **Never fatal.** An activator that fails records a degradation, marks
  itself down in the report, and lets boot proceed. The runtime existed
  before these existed; a validator must not become a new way to fail to
  start.
* **Report, don't hide.** The returned report is written into the runtime
  manifest and surfaced by the health contract, so "is lockdep actually
  on?" has an answer that does not require reading this file.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Foundations")

#: Host memory available-fraction below which the OOM policy starts
#: shedding, and below which it treats the situation as terminal.
SOFT_PRESSURE_AVAILABLE_FRACTION = 0.12
HARD_PRESSURE_AVAILABLE_FRACTION = 0.06

#: The sentinel's duty cycle. Long enough to be free, short enough that a
#: fast allocator cannot cross both thresholds between samples.
SENTINEL_INTERVAL_S = 5.0

#: A monotonic-vs-wall-clock divergence beyond this in one sentinel period
#: means the wall clock jumped (NTP step, sleep/wake, VM migration).
CLOCK_JUMP_TOLERANCE_S = 5.0


@dataclass
class ActivationResult:
    name: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "data": self.data}


class MemorySentinel:
    """The reclaim path, kept independent of every organ it may shed.

    Runs the jobs that must keep working when the rest of the runtime is
    stalled: memory-pressure reclaim, PSI memory accounting, and clock-jump
    detection. It is one small loop rather than three because these all
    need the same sample and the same independence.
    """

    def __init__(self, *, interval_s: float = SENTINEL_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._last_wall = time.time()
        self._last_mono = time.monotonic()
        self._memory_stalled = False
        self.samples = 0
        self.sheds = 0
        self.evictions = 0
        #: Re-scan the container for new shed candidates every ~60s.
        self.rescan_every = max(1, int(60.0 / max(interval_s, 0.1)))
        self._rescan_countdown = self.rescan_every

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="foundations.memory_sentinel")

    async def stop(self) -> None:
        self._stopping.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — shutdown must not raise
            pass
        # Leaving a stall open would peg PSI at 100% forever.
        self._end_memory_stall()

    def _begin_memory_stall(self) -> None:
        if self._memory_stalled:
            return
        from core.runtime.pressure_stall import Resource, get_pressure_monitor

        get_pressure_monitor().begin_stall(Resource.MEMORY)
        self._memory_stalled = True

    def _end_memory_stall(self) -> None:
        if not self._memory_stalled:
            return
        from core.runtime.pressure_stall import Resource, get_pressure_monitor

        get_pressure_monitor().end_stall(Resource.MEMORY)
        self._memory_stalled = False

    def _check_clock(self) -> None:
        wall = time.time()
        mono = time.monotonic()
        wall_delta = wall - self._last_wall
        mono_delta = mono - self._last_mono
        self._last_wall = wall
        self._last_mono = mono
        skew = wall_delta - mono_delta
        if abs(skew) > CLOCK_JUMP_TOLERANCE_S:
            from core.runtime.taint import TaintFlag, taint

            taint(
                TaintFlag.CLOCK_JUMP,
                f"wall clock moved {skew:+.1f}s relative to the monotonic clock; "
                "durations measured across this point are not trustworthy",
                subsystem="memory_sentinel",
            )

    def _sample_memory(self) -> tuple[int, int] | None:
        try:
            from core.runtime.resource_observation import get_resource_observer

            observation = get_resource_observer().memory()
            if not observation.available or observation.total_bytes <= 0:
                return None
            return int(observation.available_bytes), int(observation.total_bytes)
        except Exception:
            logger.debug("memory sentinel sample failed", exc_info=True)
            return None

    def _record_observability(self, available_fraction: float) -> None:
        """Feed the histograms, the trace counters, and memory attribution.

        The memory dump on every tick is what turns the open ~242MB/h soak
        question into an answerable one: two dumps and a diff name the
        component that grew, which neither RSS nor allocation-site
        profiling can do.
        """
        try:
            from core.observability.histograms import record
            from core.observability.trace_events import trace_counter
            from core.runtime.pressure_stall import Resource, pressure

            memory_pressure = pressure(Resource.MEMORY)
            record("Aura.Memory.AvailableFraction", available_fraction)
            record("Aura.Pressure.MemoryFull", memory_pressure * 100.0)
            trace_counter(
                "memory",
                {
                    "available_fraction": available_fraction,
                    "psi_memory_full": memory_pressure,
                    "psi_inference_full": pressure(Resource.INFERENCE),
                },
                category="resource",
            )
        except Exception:
            logger.debug("observability sampling failed", exc_info=True)
        try:
            from core.runtime.memory_infra import DetailLevel, get_memory_infra

            get_memory_infra().dump(
                DetailLevel.LIGHT
                if available_fraction <= SOFT_PRESSURE_AVAILABLE_FRACTION
                else DetailLevel.BACKGROUND
            )
        except Exception:
            logger.debug("memory dump failed", exc_info=True)

    def _evaluate(self) -> None:
        from core.runtime.oom_policy import get_oom_policy

        # Organs arrive after boot — lazily constructed services, hot-swapped
        # adapters, a model that was not resident at activation. Re-scanning
        # keeps the shed order complete instead of frozen at boot; discovery
        # is idempotent and never instantiates anything.
        self._rescan_countdown -= 1
        if self._rescan_countdown <= 0:
            self._rescan_countdown = self.rescan_every
            try:
                _register_oom_organs()
            except Exception:
                logger.debug("OOM organ rescan failed", exc_info=True)

        # Graded eviction runs before the crude OOM ladder: reclaim caches
        # first, evict BestEffort organs next, and only then let the OOM
        # policy pick a victim. Gated fail-OPEN — skipping protective work
        # because another process might be doing it is the wrong failure
        # direction. See lease.should_act_as_singleton.
        from core.runtime.lease import RUNTIME_LEASE, should_act_as_singleton

        if should_act_as_singleton(RUNTIME_LEASE):
            try:
                from core.runtime.eviction import get_eviction_manager

                outcome = get_eviction_manager().enforce()
                self.evictions += len(
                    [a for a in outcome.get("actions", ()) if a.get("action") == "evict"]
                )
            except Exception:
                logger.debug("eviction enforcement failed", exc_info=True)

        sample = self._sample_memory()
        if sample is None:
            return
        available, total = sample
        fraction = available / float(total)
        self.samples += 1
        self._record_observability(fraction)

        if fraction > SOFT_PRESSURE_AVAILABLE_FRACTION:
            self._end_memory_stall()
            return

        # Under soft pressure the runtime is waiting on reclaim whether or
        # not any single caller says so; that is exactly what PSI memory
        # pressure means.
        self._begin_memory_stall()

        policy = get_oom_policy()
        target = int(total * SOFT_PRESSURE_AVAILABLE_FRACTION * 1.5)
        reason = (
            f"host memory available {fraction * 100:.1f}% "
            f"({available / 1e9:.2f}GB of {total / 1e9:.2f}GB)"
        )
        events = policy.shed_until(
            target_free_bytes=target,
            free_bytes_now=lambda: (self._sample_memory() or (0, 1))[0],
            reason=reason,
        )
        self.sheds += len(events)

        after = self._sample_memory()
        after_fraction = (after[0] / float(after[1])) if after else fraction
        if after_fraction <= HARD_PRESSURE_AVAILABLE_FRACTION and policy.no_victim_available():
            policy.request_controlled_restart(reason)

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                self._check_clock()
                await asyncio.to_thread(self._evaluate)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — the reclaim path never dies
                from core.runtime.errors import record_degradation

                record_degradation(
                    "memory_sentinel",
                    exc,
                    severity="warning",
                    action="sentinel iteration skipped; loop continues",
                    enforce_failure_policy=False,
                )
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.interval_s)
            except TimeoutError:
                continue

    def report(self) -> dict[str, Any]:
        return {
            "running": self._task is not None and not self._task.done(),
            "interval_s": self.interval_s,
            "samples": self.samples,
            "sheds": self.sheds,
            "evictions": self.evictions,
            "memory_stall_open": self._memory_stalled,
        }


_SENTINEL: MemorySentinel | None = None


def get_memory_sentinel() -> MemorySentinel:
    global _SENTINEL
    if _SENTINEL is None:
        _SENTINEL = MemorySentinel()
    return _SENTINEL


# ══════════════════════════════════════════════════════════════════════
# Wave activators
# ══════════════════════════════════════════════════════════════════════

def _declare_pressure_capacities() -> dict[str, int]:
    """Tell PSI how many workers can contend for each resource.

    ``full`` pressure means *every* worker stalled, so these numbers decide
    whether the throughput-collapse signal is meaningful or trivially true.
    """
    from core.runtime.pressure_stall import Resource, declare_capacity

    cpus = max(1, os.cpu_count() or 1)
    capacities: dict[str, int] = {
        # Cognition lanes contend for compute; the host's core count is the
        # honest ceiling.
        str(Resource.CPU): cpus,
        # Reclaim is global: one waiter is everyone waiting.
        str(Resource.MEMORY): 1,
        # The durability lane is a small thread pool, not a single fd.
        str(Resource.IO): max(2, min(8, cpus // 2)),
        # One resident model unless the lane controller says otherwise.
        str(Resource.INFERENCE): _model_lane_capacity(),
        str(Resource.BUS): 1,
        str(Resource.LOCK): cpus,
    }
    for name, workers in capacities.items():
        declare_capacity(name, workers)
    return capacities


def _model_lane_capacity() -> int:
    try:
        from core.runtime.model_lane_control import get_model_lane_controller

        controller = get_model_lane_controller()
        for attr in ("max_lanes", "lane_capacity", "concurrency", "max_concurrent_lanes"):
            value = getattr(controller, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    except Exception:
        logger.debug("model lane capacity probe unavailable", exc_info=True)
    return 1


#: Organs whose loss is worse than the memory they hold. The kernel gives
#: init OOM_SCORE_ADJ_MIN for the same reason: some things must not be the
#: answer to "what should we kill".
IMMUNE_SERVICES: tuple[str, ...] = (
    "unified_will",
    "will",
    "event_bus",
    "container",
    "memory_facade",
    "flight_recorder",
    "shutdown_coordinator",
    "health",
    "orchestrator",
    "identity",
    "self_object",
)


def _register_oom_organs() -> dict[str, Any]:
    """Build the shed order before the pressure arrives.

    Discovery is by capability, not by name: any *already-instantiated*
    service exposing ``shed_memory()`` volunteers. Lazily-registered
    services are deliberately not instantiated here — constructing an organ
    in order to learn it could be shed under memory pressure is exactly
    backwards.
    """
    from core.container import ServiceContainer
    from core.runtime.oom_policy import OOM_SCORE_ADJ_MIN, register_organ

    registered: list[str] = []
    for name in IMMUNE_SERVICES:
        register_organ(
            name,
            oom_score_adj=OOM_SCORE_ADJ_MIN,
            rationale="load-bearing: losing it costs more than the memory it holds",
            recoverable=False,
        )
        registered.append(name)

    discovered: list[str] = []
    for service_name, instance in _instantiated_services(ServiceContainer).items():
        if service_name in IMMUNE_SERVICES:
            continue
        shed = getattr(instance, "shed_memory", None)
        if not callable(shed):
            continue
        adj = int(getattr(instance, "oom_score_adj", 0) or 0)
        footprint = getattr(instance, "memory_footprint_bytes", None)
        register_organ(
            service_name,
            oom_score_adj=adj,
            footprint=footprint if callable(footprint) else None,
            shed=shed,
            rationale=getattr(instance, "oom_rationale", "")
            or f"{service_name} exposes shed_memory()",
            recoverable=bool(getattr(instance, "oom_recoverable", True)),
        )
        discovered.append(service_name)

    return {"immune": registered, "sheddable": discovered}


def _instantiated_services(container: Any) -> dict[str, Any]:
    """Live instances only — never triggers a lazy factory."""
    out: dict[str, Any] = {}
    services = getattr(container, "_services", None)
    if not isinstance(services, dict):
        return out
    for name, descriptor in list(services.items()):
        instance = getattr(descriptor, "instance", None)
        if instance is not None:
            out[str(name)] = instance
    return out


async def _activate_kernel_discipline(*, foreground_only: bool) -> ActivationResult:
    """Wave 1 — taint register, lockdep, PSI, OOM policy, memory sentinel."""
    from core.runtime.lockdep import lockdep_report, note_event_loop_thread

    note_event_loop_thread()
    capacities = _declare_pressure_capacities()
    organs = _register_oom_organs()

    sentinel_started = False
    if not foreground_only:
        await get_memory_sentinel().start()
        sentinel_started = True
        try:
            from core.runtime.shutdown_coordinator import get_shutdown_coordinator

            get_shutdown_coordinator().register(
                get_memory_sentinel().stop,
                phase="task_supervisor",
                name="foundations.memory_sentinel",
                timeout=5.0,
            )
        except Exception:
            logger.debug("memory sentinel shutdown registration skipped", exc_info=True)

    return ActivationResult(
        name="kernel_discipline",
        ok=True,
        detail=(
            f"lockdep armed ({lockdep_report()['acquires_checked']} acquires checked), "
            f"PSI over {len(capacities)} resources, "
            f"{len(organs['immune'])} immune + {len(organs['sheddable'])} sheddable organs"
        ),
        data={
            "pressure_capacities": capacities,
            "oom_organs": organs,
            "memory_sentinel_started": sentinel_started,
        },
    )


async def _activate_verification(*, foreground_only: bool) -> ActivationResult:
    """Wave 2 — structural verifier, pass instrumentation, sanitizers."""
    # Importing registers the standing invariants; the module is a
    # declaration site, not a service.
    from core.pipeline.pass_manager import get_instrumentation, install_default_instrumentation
    from core.verify import runtime_invariants  # noqa: F401 — import registers
    from core.verify.invariants import get_registry, verify

    instrumentation = install_default_instrumentation()

    # The first verification runs over the runtime as boot left it. This is
    # the moment a structural regression is cheapest to see: before any
    # traffic, with the boot path still on the stack.
    report = verify()
    declared = len(get_registry().specs())

    return ActivationResult(
        name="verification",
        ok=report.ok,
        detail=(
            f"{declared} invariants declared, {report.summary()}; "
            f"pass instrumentation {'armed' if instrumentation['installed'] else 'already armed'}"
            + (
                f", opt-bisect limit={get_instrumentation().bisect_limit()}"
                if get_instrumentation().bisect_limit() is not None
                else ""
            )
        ),
        data={
            "invariants_declared": declared,
            "scopes": get_registry().scopes(),
            "boot_verification": report.to_dict(),
            "pass_instrumentation": instrumentation,
        },
    )


async def _activate_orchestration(*, foreground_only: bool) -> ActivationResult:
    """Wave 3 — admission, quota, eviction, controllers, leader election."""
    from core.runtime.eviction import get_eviction_manager
    from core.runtime.lease import RUNTIME_LEASE, get_elector
    from core.runtime.quota import install_quota_admission
    from core.runtime.reconcile import get_controller_manager

    quota_hook = install_quota_admission()
    oom_scores = get_eviction_manager().sync_oom_scores()
    started_controllers = await get_controller_manager().start_all()

    # Contend for the runtime lease. Not holding it is not an error — it
    # is the *answer*, and it is the answer that used to require reading
    # memory graphs after the duplicate-runtime cascade had already
    # happened. Never blocks boot.
    leader = False
    if not foreground_only:
        elector = get_elector(RUNTIME_LEASE)
        await elector.start()
        # One synchronous attempt so the boot report says something true
        # rather than "pending".
        leader = await elector.try_acquire_or_renew()
        try:
            from core.runtime.shutdown_coordinator import get_shutdown_coordinator

            coordinator = get_shutdown_coordinator()
            coordinator.register(
                elector.stop,
                phase="task_supervisor",
                name=f"lease.{RUNTIME_LEASE}",
                timeout=5.0,
            )
            coordinator.register(
                get_controller_manager().stop_all,
                phase="task_supervisor",
                name="controller_manager",
                timeout=10.0,
            )
        except Exception:
            logger.debug("orchestration shutdown registration skipped", exc_info=True)

    return ActivationResult(
        name="orchestration",
        ok=True,
        detail=(
            f"admission chain live (quota hook {'installed' if quota_hook else 'present'}), "
            f"{len(oom_scores)} QoS→OOM scores synced, "
            f"{len(started_controllers)} controller(s) started, "
            f"runtime lease {'HELD' if leader else 'not held'}"
        ),
        data={
            "quota_admission_installed": quota_hook,
            "qos_oom_scores": oom_scores,
            "controllers": started_controllers,
            "runtime_lease_held": leader,
        },
    )


#: Topics whose volume would evict everything else from the bus ring.
#: Excluding them is what keeps the ring's minute of history useful.
BAG_EXCLUDED_TOPICS: tuple[str, ...] = (
    "metrics.sample",
    "telemetry.tick",
    "substrate.activation",
    "heartbeat",
)

#: Organs adopted into managed lifecycles at boot. Adoption gives an
#: existing start/stop object a visible state and makes its deactivation
#: distinguishable from its failure, without rewriting it.
LIFECYCLE_ADOPTIONS: tuple[tuple[str, bool], ...] = (
    ("orchestrator", True),
    ("event_bus", True),
    ("memory_facade", True),
    ("autonomy_conductor", False),
    ("research_cycle", False),
    ("curiosity_engine", False),
    ("performance_guard", False),
    ("self_healing", False),
    ("viability", False),
    ("flagship_doctor_daemon", False),
)


def _declare_core_parameters() -> list[str]:
    """Give the thresholds this module already hard-codes a real home.

    Each was a literal that could not be found, justified, or changed
    without a restart. Declared, they are inventoried, range-checked,
    observable, and retunable on a live runtime.
    """
    from core.runtime.parameters import ParameterType, declare

    specs: tuple[tuple[str, Any, dict[str, Any]], ...] = (
        (
            "memory.soft_pressure_available_fraction",
            SOFT_PRESSURE_AVAILABLE_FRACTION,
            {
                "type": ParameterType.FLOAT,
                "description": "available-memory fraction below which reclaim and shedding begin",
                "owner": "core/runtime/foundations.py",
                "minimum": 0.01,
                "maximum": 0.9,
            },
        ),
        (
            "memory.hard_pressure_available_fraction",
            HARD_PRESSURE_AVAILABLE_FRACTION,
            {
                "type": ParameterType.FLOAT,
                "description": (
                    "available-memory fraction at which a controlled restart beats "
                    "waiting to be killed"
                ),
                "owner": "core/runtime/foundations.py",
                "minimum": 0.005,
                "maximum": 0.5,
            },
        ),
        (
            "memory.sentinel_interval_s",
            SENTINEL_INTERVAL_S,
            {
                "type": ParameterType.FLOAT,
                "description": "duty cycle of the independent reclaim sentinel",
                "owner": "core/runtime/foundations.py",
                "minimum": 1.0,
                "maximum": 60.0,
            },
        ),
        (
            "lockdep.loop_blocking_hold_ms",
            50.0,
            {
                "type": ParameterType.FLOAT,
                "description": "sync-lock hold on the loop thread beyond which lockdep reports",
                "owner": "core/runtime/lockdep.py",
                "minimum": 1.0,
                "maximum": 5000.0,
            },
        ),
        (
            "pressure.saturation_threshold",
            0.20,
            {
                "type": ParameterType.FLOAT,
                "description": "PSI full-pressure fraction at which a resource counts as saturated",
                "owner": "core/runtime/pressure_stall.py",
                "minimum": 0.01,
                "maximum": 1.0,
            },
        ),
        (
            "bus.ring_capacity",
            8192,
            {
                "type": ParameterType.INT,
                "description": "messages retained in the always-on bus ring",
                "owner": "core/observability/bus_recorder.py",
                "minimum": 256,
                "maximum": 131072,
            },
        ),
        (
            "diagnostics.stale_after_s",
            30.0,
            {
                "type": ParameterType.FLOAT,
                "description": "silence after which a diagnostic task is reported STALE",
                "owner": "core/health/diagnostics_aggregator.py",
                "minimum": 5.0,
                "maximum": 600.0,
            },
        ),
    )
    declared: list[str] = []
    for name, default, kwargs in specs:
        try:
            declare(name, default, **kwargs)
            declared.append(name)
        except (ValueError, TypeError) as exc:
            logger.warning("parameter %s could not be declared: %s", name, exc)
    return declared


def _adopt_lifecycles() -> dict[str, str]:
    """Adopt already-instantiated organs into managed lifecycles."""
    from core.container import ServiceContainer
    from core.runtime.lifecycle import adopt

    adopted: dict[str, str] = {}
    instances = _instantiated_services(ServiceContainer)
    for name, critical in LIFECYCLE_ADOPTIONS:
        instance = instances.get(name)
        if instance is None:
            continue
        organ = adopt(name, instance, critical=critical)
        if organ is not None:
            adopted[name] = str(organ.state)
    return adopted


async def _activate_middleware(*, foreground_only: bool) -> ActivationResult:
    """Wave 4 — lifecycles, bus QoS, parameters, bus ring, diagnostics."""
    from core.health.diagnostics_aggregator import (
        install_default_analyzers,
        install_runtime_diagnostics,
    )
    from core.observability.bus_recorder import get_bus_recorder
    from core.runtime.lifecycle import lifecycle_report

    parameters = _declare_core_parameters()
    adopted = _adopt_lifecycles()

    recorder = get_bus_recorder()
    recorder.exclude(*BAG_EXCLUDED_TOPICS)

    analyzers = install_default_analyzers()
    tasks = install_runtime_diagnostics()

    qos_topics = _declare_standard_topics()

    return ActivationResult(
        name="middleware",
        ok=True,
        detail=(
            f"{len(parameters)} parameters declared, {len(adopted)} organ(s) adopted "
            f"into managed lifecycles, {len(qos_topics)} QoS topics, "
            f"bus ring armed, {len(analyzers)} diagnostic analyzers over "
            f"{len(tasks)} tasks"
        ),
        data={
            "parameters": parameters,
            "lifecycles": adopted,
            "lifecycle_report": lifecycle_report()["by_state"],
            "qos_topics": qos_topics,
            "diagnostic_analyzers": analyzers,
            "diagnostic_tasks": tasks,
        },
    )


def _declare_standard_topics() -> list[str]:
    """Give the topics whose meaning depends on QoS an explicit contract.

    State topics get transient-local durability, which is what makes an
    organ that boots *after* a state announcement still learn the state
    instead of behaving as though it never changed.
    """
    from core.bus.qos import COMMAND, HEARTBEAT, SENSOR_DATA, STATE, declare_topic

    topics = {
        "runtime.state": STATE,
        "runtime.boot_phase": STATE,
        "cortex.lane_state": STATE,
        "autonomy.state": STATE,
        "health.verdict": STATE,
        "memory.pressure": STATE,
        "sensory.frame": SENSOR_DATA,
        "sensory.audio": SENSOR_DATA,
        "will.decision": COMMAND,
        "action.request": COMMAND,
        "mind.tick": HEARTBEAT,
    }
    for topic, profile in topics.items():
        declare_topic(topic, profile)
    return sorted(topics)


async def _activate_observability(*, foreground_only: bool) -> ActivationResult:
    """Wave 5 — histograms, traces, memory attribution, trials, Rule of Two."""
    from core.observability.histograms import install_standard_histograms
    from core.observability.trace_events import get_tracer, install_pass_tracing
    from core.runtime.memory_infra import DetailLevel, get_memory_infra, install_runtime_providers
    from core.security.rule_of_two import install_known_handlers, rule_of_two_report

    histograms = install_standard_histograms()
    tracing = install_pass_tracing()
    get_tracer().name_thread("runtime.main")

    providers = install_runtime_providers()
    # Take the first dump immediately: a leak report needs two points, and
    # the earlier one has to exist before the growth starts.
    baseline_dump = get_memory_infra().dump(DetailLevel.BACKGROUND)

    handlers = install_known_handlers()
    posture = rule_of_two_report()

    return ActivationResult(
        name="observability",
        ok=not posture["violations"],
        detail=(
            f"{len(histograms)} histograms declared, pass tracing "
            f"{'armed' if tracing else 'already armed'}, "
            f"{len(providers)} memory providers "
            f"({baseline_dump.attributed_bytes / 1e6:.0f}MB attributed of "
            f"{baseline_dump.process_rss_bytes / 1e6:.0f}MB RSS), "
            f"{len(handlers)} security postures declared"
            + (
                f", {len(posture['violations'])} RULE-OF-TWO VIOLATION(S)"
                if posture["violations"]
                else ""
            )
        ),
        data={
            "histograms": histograms,
            "pass_tracing": tracing,
            "memory_providers": providers,
            "baseline_dump": baseline_dump.to_dict()["attributed_fraction"],
            "rule_of_two": {
                "declared": handlers,
                "violations": posture["violations"],
                "at_the_limit": posture["at_the_limit"],
            },
        },
    )


#: (name, activator) in dependency order. Later waves append here; the
#: order is the boot order and is meaningful.
_ACTIVATORS: list[tuple[str, Callable[..., Any]]] = [
    ("kernel_discipline", _activate_kernel_discipline),
    ("verification", _activate_verification),
    ("orchestration", _activate_orchestration),
    ("middleware", _activate_middleware),
    ("observability", _activate_observability),
]


def register_activator(name: str, activator: Callable[..., Any]) -> None:
    """Append a wave activator. Idempotent by name."""
    for existing, _ in _ACTIVATORS:
        if existing == name:
            return
    _ACTIVATORS.append((name, activator))


_LAST_REPORT: dict[str, Any] = {"activated": False}


async def activate_foundations(*, foreground_only: bool = False) -> dict[str, Any]:
    """Turn on every borrowed discipline. Called once, from boot."""
    started = time.time()
    results: list[ActivationResult] = []
    for name, activator in _ACTIVATORS:
        try:
            result = activator(foreground_only=foreground_only)
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, ActivationResult):
                result = ActivationResult(name=name, ok=True, detail=str(result))
        except Exception as exc:  # noqa: BLE001 — a validator must not break boot
            from core.runtime.errors import record_degradation

            record_degradation(
                "foundations",
                exc,
                severity="warning",
                action=f"{name} activation skipped; runtime continues without it",
                enforce_failure_policy=False,
            )
            result = ActivationResult(name=name, ok=False, detail=repr(exc))
        results.append(result)
        logger.info(
            "%s foundations/%s — %s",
            "✅" if result.ok else "⚠️",
            result.name,
            result.detail or ("active" if result.ok else "unavailable"),
        )

    report = {
        "activated": True,
        "at": started,
        "duration_s": round(time.time() - started, 3),
        "foreground_only": foreground_only,
        "waves": [r.to_dict() for r in results],
        "ok": all(r.ok for r in results),
        "failed": [r.name for r in results if not r.ok],
    }
    _LAST_REPORT.clear()
    _LAST_REPORT.update(report)

    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("foundations_report", report, required=False)
    except Exception:
        logger.debug("foundations report registration skipped", exc_info=True)
    return report


def foundations_report() -> dict[str, Any]:
    return dict(_LAST_REPORT)


def reset_foundations_for_test() -> None:
    global _SENTINEL
    _SENTINEL = None
    _LAST_REPORT.clear()
    _LAST_REPORT["activated"] = False


__all__ = [
    "ActivationResult",
    "HARD_PRESSURE_AVAILABLE_FRACTION",
    "IMMUNE_SERVICES",
    "MemorySentinel",
    "SOFT_PRESSURE_AVAILABLE_FRACTION",
    "activate_foundations",
    "foundations_report",
    "get_memory_sentinel",
    "register_activator",
    "reset_foundations_for_test",
]
