"""Durable, fenced admission transactions for model-memory owners.

The generic runtime admission controller schedules work, while this controller
owns the model-memory envelope itself.  A reservation is persisted before any
required eviction, remains counted until spawn commits or aborts, and is fenced
so an abandoned caller cannot later publish a stale worker as the owner.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import hmac
import inspect
import logging
import os
import secrets
import signal
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from core.brain.lane_admission import (
    ActiveLane,
    LaneAdmissionController,
    QoSClass,
    classify_lane,
    lane_budget_gb,
)
from core.runtime.atomic_writer import (
    interprocess_file_lock,
    read_json_envelope,
)
from core.runtime.flags import FlagKind, declare
from core.runtime.resource_observation import ResourceObserver, get_resource_observer
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ModelLaneControl")

_SCHEMA_NAME = "aura.model_lane_control.v1"
_SCHEMA_VERSION = 1
_EVENT_LIMIT = 256
_TERMINAL_RESERVATION_LIMIT = 256
_OWNER = "core.runtime.model_lane_control"

_RESERVATION_TTL_FLAG = declare(
    "AURA_MODEL_LANE_RESERVATION_TTL_S",
    kind=FlagKind.FLOAT,
    default=300.0,
    description="Maximum lifetime of an uncommitted model-lane reservation",
    owner=_OWNER,
)
_OWNER_LEASE_TTL_FLAG = declare(
    "AURA_MODEL_LANE_OWNER_LEASE_TTL_S",
    kind=FlagKind.FLOAT,
    default=600.0,
    description="Fallback owner lease when no verifiable process identity exists",
    owner=_OWNER,
)
_EVICTION_TIMEOUT_FLAG = declare(
    "AURA_MODEL_LANE_EVICTION_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=45.0,
    description="Bound for required model-lane eviction and process reclamation",
    owner=_OWNER,
)


class LaneTransactionState(StrEnum):
    RESERVED = "reserved"
    EVICTING = "evicting"
    READY = "ready"
    COMMITTED = "committed"
    REFUSED = "refused"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


_ACTIVE_RESERVATION_STATES = {
    LaneTransactionState.RESERVED.value,
    LaneTransactionState.EVICTING.value,
    LaneTransactionState.READY.value,
}
_TERMINAL_RESERVATION_STATES = {
    LaneTransactionState.COMMITTED.value,
    LaneTransactionState.REFUSED.value,
    LaneTransactionState.CANCELLED.value,
    LaneTransactionState.EXPIRED.value,
}


@dataclass(frozen=True)
class ProcessIdentity:
    """PID identity protected against PID reuse by process creation time."""

    pid: int
    started_at: float

    def to_dict(self) -> dict[str, Any]:
        return {"pid": int(self.pid), "started_at": float(self.started_at)}

    @classmethod
    def current(cls, *, observer: ResourceObserver | None = None) -> ProcessIdentity:
        return process_identity_for_pid(os.getpid(), observer=observer)


def process_identity_for_pid(
    pid: int,
    *,
    observer: ResourceObserver | None = None,
) -> ProcessIdentity:
    pid = int(pid)
    if pid <= 0:
        return ProcessIdentity(0, 0.0)
    try:
        process = (observer or get_resource_observer()).process(pid)
        if process is not None:
            return ProcessIdentity(pid, float(process.create_time))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    return ProcessIdentity(pid, 0.0)


def _default_process_alive(
    identity: ProcessIdentity,
    *,
    observer: ResourceObserver | None = None,
) -> bool:
    if identity.pid <= 0 or identity.started_at <= 0.0:
        return False
    try:
        process = (observer or get_resource_observer()).process(identity.pid)
        if process is None:
            return False
        observed = float(process.create_time)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return abs(observed - identity.started_at) <= 0.5


def managed_process_group_alive(
    process_group_id: int,
    *,
    root_started_at: float = 0.0,
    observer: ResourceObserver | None = None,
) -> bool:
    """Return whether an isolated managed group still has a live member."""

    pgid = int(process_group_id)
    if pgid <= 0 or pgid == os.getpgrp():
        return False
    earliest_member = max(0.0, float(root_started_at) - 1.0)
    for process in (observer or get_resource_observer()).processes():
        try:
            pid = int(process.pid)
            if pid <= 0 or pid == os.getpid() or os.getpgid(pid) != pgid:
                continue
            started_at = float(process.create_time)
            if earliest_member and started_at < earliest_member:
                continue
            if str(process.status).lower() != "zombie":
                return True
        except (OSError, ProcessLookupError, RuntimeError, TypeError, ValueError):
            continue
    return False


@dataclass(frozen=True)
class LaneOwnerObservation:
    """One live process or in-process job that owns model memory."""

    owner_id: str
    model_path: str
    declared_gb: float
    purpose: str = "serve"
    observed_gb: float = 0.0
    process: ProcessIdentity = field(default_factory=lambda: ProcessIdentity(0, 0.0))
    priority: int = 50
    preemptible: bool = True
    last_user_facing_age_s: float | None = None
    lease_ttl_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.owner_id).strip():
            raise ValueError("lane owner_id must be non-empty")
        if not str(self.model_path).strip():
            raise ValueError("lane model_path must be non-empty")
        if float(self.declared_gb) <= 0.0:
            raise ValueError("lane declared_gb must be positive")


@dataclass(frozen=True)
class LaneClaim:
    """Capacity requested by one candidate before any eviction or spawn."""

    owner_id: str
    model_path: str
    request_gb: float
    purpose: str = "serve"
    priority: int = 50
    preemptible: bool = True
    foreground: bool = False
    allow_disruptive_eviction: bool = False
    allow_last_warm_eviction: bool = False
    reservation_ttl_s: float = 0.0
    owner_lease_ttl_s: float = 0.0
    request_id: str = field(default_factory=lambda: f"lane-{uuid.uuid4()}")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.owner_id).strip():
            raise ValueError("lane claim owner_id must be non-empty")
        if not str(self.model_path).strip():
            raise ValueError("lane claim model_path must be non-empty")
        if float(self.request_gb) <= 0.0:
            raise ValueError("lane claim request_gb must be positive")
        if not str(self.request_id).strip():
            raise ValueError("lane claim request_id must be non-empty")


@dataclass(frozen=True)
class LaneTransactionDecision:
    request_id: str
    transaction_id: str
    fencing_token: int
    admitted: bool
    ready_to_spawn: bool
    state: LaneTransactionState
    reason: str
    owner_id: str
    model_path: str
    lane: str
    qos: QoSClass
    request_gb: float
    committed_gb: float
    reserved_gb: float
    budget_gb: float
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""
    resource_observation_available: bool = False
    evict_owner_ids: tuple[str, ...] = ()
    evicted_owner_ids: tuple[str, ...] = ()
    receipt_id: str = ""
    replayed: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["qos"] = self.qos.value
        payload["evict_owner_ids"] = list(self.evict_owner_ids)
        payload["evicted_owner_ids"] = list(self.evicted_owner_ids)
        return payload


EvictCallback = Callable[[LaneOwnerObservation, str], bool | Awaitable[bool]]
ObserveCallback = Callable[
    [], Iterable[LaneOwnerObservation] | Awaitable[Iterable[LaneOwnerObservation]]
]
ReclaimCallback = Callable[[LaneClaim], bool | Awaitable[bool]]
CompensateCallback = Callable[[LaneOwnerObservation, str], bool | Awaitable[bool]]
ProcessAliveProbe = Callable[[ProcessIdentity], bool]
ProcessDiscoveryProbe = Callable[
    [Iterable[LaneOwnerObservation]],
    Iterable[LaneOwnerObservation],
]

_LOCAL_OWNER_ADAPTERS: dict[str, tuple[EvictCallback, CompensateCallback | None]] = {}
_LOCAL_OWNER_COMPENSATORS: dict[str, tuple[CompensateCallback, float]] = {}
_LOCAL_OWNER_ADAPTERS_LOCK = threading.RLock()
_COMPENSATOR_TTL_S = 300.0


def register_model_lane_owner_adapter(
    owner_id: str,
    *,
    evict: EvictCallback,
    compensate: CompensateCallback | None = None,
) -> None:
    with _LOCAL_OWNER_ADAPTERS_LOCK:
        now = time.monotonic()
        for stale_owner in [
            key for key, (_callback, expires_at) in _LOCAL_OWNER_COMPENSATORS.items()
            if expires_at <= now
        ]:
            _LOCAL_OWNER_COMPENSATORS.pop(stale_owner, None)
        _LOCAL_OWNER_COMPENSATORS.pop(str(owner_id), None)
        _LOCAL_OWNER_ADAPTERS[str(owner_id)] = (evict, compensate)


def unregister_model_lane_owner_adapter(owner_id: str) -> None:
    with _LOCAL_OWNER_ADAPTERS_LOCK:
        adapter = _LOCAL_OWNER_ADAPTERS.pop(str(owner_id), None)
        if adapter is not None and adapter[1] is not None:
            _LOCAL_OWNER_COMPENSATORS[str(owner_id)] = (
                adapter[1],
                time.monotonic() + _COMPENSATOR_TTL_S,
            )


async def _invoke_owned_callback(callback: Callable[..., Any], *args: Any) -> Any:
    """Invoke callbacks without allowing synchronous work to block the loop."""

    if inspect.iscoroutinefunction(callback):
        result = callback(*args)
    else:
        callback_name = str(
            getattr(callback, "__qualname__", "")
            or getattr(callback, "__name__", "")
            or type(callback).__qualname__
        )
        result = await run_owned_model_thread_call(
            lambda: callback(*args),
            operation_name=f"model_lane_callback:{callback_name}",
        )
    if inspect.isawaitable(result):
        result = await result
    return result


async def evict_registered_model_owner(owner: LaneOwnerObservation, reason: str) -> bool:
    with _LOCAL_OWNER_ADAPTERS_LOCK:
        adapter = _LOCAL_OWNER_ADAPTERS.get(owner.owner_id)
    if adapter is not None:
        callback, _compensate = adapter
        return bool(await _invoke_owned_callback(callback, owner, reason))
    return await evict_managed_process_owner(owner, reason)


async def compensate_registered_model_owner(
    owner: LaneOwnerObservation,
    reason: str,
) -> bool:
    retained_entry: tuple[CompensateCallback, float] | None = None
    with _LOCAL_OWNER_ADAPTERS_LOCK:
        adapter = _LOCAL_OWNER_ADAPTERS.get(owner.owner_id)
        callback = adapter[1] if adapter is not None else None
        if callback is None:
            retained = _LOCAL_OWNER_COMPENSATORS.get(owner.owner_id)
            if retained is not None:
                if retained[1] > time.monotonic():
                    retained_entry = retained
                    callback = retained[0]
                else:
                    _LOCAL_OWNER_COMPENSATORS.pop(owner.owner_id, None)
    if callback is None:
        return False
    restored = bool(await _invoke_owned_callback(callback, owner, reason))
    if restored and retained_entry is not None:
        with _LOCAL_OWNER_ADAPTERS_LOCK:
            if _LOCAL_OWNER_COMPENSATORS.get(owner.owner_id) == retained_entry:
                _LOCAL_OWNER_COMPENSATORS.pop(owner.owner_id, None)
    return restored


class ModelLaneControlError(RuntimeError):
    """The durable lane state could not prove a safe transition."""


def _json_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep state payloads JSON-safe without allowing serializer side effects."""

    result: dict[str, Any] = {}
    for key, item in dict(value).items():
        name = str(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[name] = item
        elif isinstance(item, (list, tuple)):
            result[name] = [
                entry if isinstance(entry, (str, int, float, bool)) or entry is None else str(entry)
                for entry in item
            ]
        else:
            result[name] = str(item)
    return result


def _path_size_gb(model_path: str) -> float:
    path = Path(str(model_path or "")).expanduser()
    try:
        if path.is_file():
            return float(path.stat().st_size) / float(1024**3)
        if not path.is_dir():
            return 0.0
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
        return float(total) / float(1024**3)
    except OSError:
        return 0.0


def estimate_model_job_footprint_gb(model_path: str, *, purpose: str) -> float:
    """Conservative declared peak for serving, training, fusion, or evaluation."""

    lowered = str(model_path or "").lower()
    artifact_gb = _path_size_gb(model_path)
    if artifact_gb > 0.0:
        base_gb = artifact_gb + max(1.0, artifact_gb * 0.25)
    elif any(token in lowered for token in ("72b", "solver")):
        base_gb = 41.0
    elif any(token in lowered for token in ("32b", "cortex", "zenith")):
        base_gb = 20.0 if any(token in lowered for token in ("4bit", "q4", "fused")) else 35.0
    elif "14b" in lowered:
        base_gb = 10.0
    elif any(token in lowered for token in ("stable-diffusion-xl", "sdxl")):
        base_gb = 16.0
    elif any(token in lowered for token in ("stable-diffusion", "diffusion")):
        base_gb = 7.0
    elif "7b" in lowered:
        base_gb = 5.0
    elif any(token in lowered for token in ("1.5b", "1p5b", "0.5b", "reflex")):
        base_gb = 2.0
    else:
        base_gb = 4.0

    normalized = str(purpose or "serve").strip().lower()
    if normalized in {"train", "compound"}:
        return max(base_gb + 4.0, base_gb * 1.8)
    if normalized == "fuse":
        return max(base_gb + 6.0, base_gb * 2.25)
    if normalized in {"benchmark", "evaluate", "eval"}:
        return max(base_gb + 1.0, base_gb * 1.15)
    return base_gb


@dataclass(frozen=True)
class _ModelCommandDescriptor:
    recognized: bool
    model_paths: tuple[str, ...]
    purpose: str


_PROBE_BUILTIN_CALLS = {"print", "float", "abs"}
_PROBE_ATTRIBUTE_CALLS = {"ones", "zeros", "eval", "sum", "item"}
_PROBE_FORBIDDEN_NAMES = {"exec", "eval", "open", "__import__", "compile", "getattr", "setattr"}
_REGISTERED_NON_MODEL_PROBE_MODULES = frozenset(
    {"core.runtime.integration_liveness_probe"},
)


def _is_registered_non_model_python_probe(argv: tuple[str, ...]) -> bool:
    """Recognize audited child modules that cannot load model weights."""

    try:
        module_index = argv.index("-m") + 1
        module = argv[module_index]
    except (ValueError, IndexError):
        return False
    return module in _REGISTERED_NON_MODEL_PROBE_MODULES


def _is_import_only_python_probe(argv: tuple[str, ...]) -> bool:
    """Return true when ``python -c`` is a bounded runtime health probe.

    Import/compute health probes mention accelerator libraries but do not
    load model weights.  Treating the library name alone as a model process
    forced those probes through an async child lifecycle and made the
    synchronous readiness check impossible.  Parse the inline program
    instead of trusting a source label or matching an ad-hoc command string.

    Accepted shape (fail-closed — anything else is treated as a model
    command): imports, single-name assignments, asserts, and calls limited
    to constant ``print``/``float``/``abs`` plus tiny-tensor attribute calls
    (``mx.ones``/``.sum``/``mx.eval``).  No string may look like a path, so
    a checkpoint can never be smuggled through the probe exemption.
    """

    try:
        inline_index = argv.index("-c")
        source = argv[inline_index + 1]
    except (ValueError, IndexError):
        return False
    if len(source) > 600:
        return False
    try:
        module = ast.parse(source, mode="exec")
    except SyntaxError:
        return False
    if not module.body:
        return False
    for statement in module.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom, ast.Pass, ast.Assert)):
            continue
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                return False
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            continue
        return False
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _PROBE_BUILTIN_CALLS:
                    return False
            elif isinstance(func, ast.Attribute):
                if func.attr not in _PROBE_ATTRIBUTE_CALLS:
                    return False
            else:
                return False
        elif isinstance(node, ast.Name) and node.id in _PROBE_FORBIDDEN_NAMES:
            return False
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 64 or "/" in node.value or "\\" in node.value:
                return False
    return True


def _model_command_descriptor(command: Iterable[str]) -> _ModelCommandDescriptor:
    argv = tuple(str(part) for part in command)
    if _is_import_only_python_probe(argv) or _is_registered_non_model_python_probe(argv):
        return _ModelCommandDescriptor(False, (), "serve")
    lowered = tuple(part.lower() for part in argv)
    joined = " ".join(lowered)
    recognized = any(
        marker in joined
        for marker in (
            "mlx_lm",
            "mlx-lm",
            "mlx_lm_lora",
            "heldout_eval.py",
            "selfplay_harvest.py",
            "bench_speculative_decoding.py",
            "front_door_demo.py",
            "generate_lora_consolidation_proof.py",
            "probe_nonparametric_memory.py",
            "zenith_v2_state_benchmark.py",
        )
    )
    if "front_door_demo.py" in joined and "--with-model" not in lowered:
        recognized = any(
            marker in joined
            for marker in (
                "mlx_lm",
                "mlx-lm",
                "mlx_lm_lora",
                "heldout_eval.py",
                "selfplay_harvest.py",
                "bench_speculative_decoding.py",
                "generate_lora_consolidation_proof.py",
                "probe_nonparametric_memory.py",
                "zenith_v2_state_benchmark.py",
            )
        )
    if not recognized:
        return _ModelCommandDescriptor(False, (), "serve")

    model_paths: list[str] = []
    for index, part in enumerate(lowered[:-1]):
        if part in {
            "--model",
            "--model-path",
            "--base-model",
            "--target",
            "--draft",
        }:
            candidate = argv[index + 1]
            if candidate and candidate not in model_paths:
                model_paths.append(candidate)

    if "fuse" in lowered:
        purpose = "fuse"
    elif (
        "--train" in lowered
        or " lora " in f" {joined} "
        or "mlx_lm.lora" in joined
        or "mlx_lm_lora" in joined
    ):
        purpose = "train"
    elif any(
        marker in joined
        for marker in (
            "heldout_eval",
            "benchmark",
            "selfplay_harvest",
            "front_door_demo",
            "probe_nonparametric_memory",
        )
    ):
        purpose = "benchmark"
    else:
        purpose = "serve"
    return _ModelCommandDescriptor(True, tuple(model_paths), purpose)


def infer_model_process_claim(
    command: Iterable[str],
    *,
    source: str,
    timeout_s: float,
) -> LaneClaim | None:
    """Build a fail-closed claim for known accelerator-owning commands."""

    argv = tuple(str(part) for part in command)
    descriptor = _model_command_descriptor(argv)
    if not descriptor.recognized:
        return None
    if not descriptor.model_paths:
        raise ModelLaneControlError(
            f"model_process_claim_missing_model_path:{source}"
        )
    model_path = descriptor.model_paths[0]
    purpose = descriptor.purpose
    request_gb = sum(
        estimate_model_job_footprint_gb(path, purpose=purpose)
        for path in descriptor.model_paths
    )

    request_id = f"model-process-{uuid.uuid4()}"
    return LaneClaim(
        owner_id=f"subprocess:{os.getpid()}:{request_id}",
        model_path=model_path,
        request_gb=request_gb,
        purpose=purpose,
        priority=80 if purpose in {"train", "fuse", "compound"} else 50,
        preemptible=True,
        foreground=False,
        reservation_ttl_s=max(60.0, float(timeout_s) + 30.0),
        owner_lease_ttl_s=max(60.0, float(timeout_s) + 30.0),
        request_id=request_id,
        metadata={
            "source": source,
            "command": list(argv),
            "inferred_model_process": True,
            "purpose": purpose,
        },
    )


def discover_external_model_processes(
    known_owners: Iterable[LaneOwnerObservation],
    *,
    observer: ResourceObserver | None = None,
) -> list[LaneOwnerObservation]:
    """Conservatively account for model jobs not registered by this controller."""

    known = list(known_owners)
    known_pids = {owner.process.pid for owner in known if owner.process.pid > 0}
    managed_groups = {
        int(dict(owner.metadata).get("process_group_id") or 0)
        for owner in known
        if bool(dict(owner.metadata).get("managed_model_process", False))
    }
    managed_groups.discard(0)
    observations: list[LaneOwnerObservation] = []
    resource_observer = observer or get_resource_observer()
    for process in resource_observer.processes():
        try:
            pid = int(process.pid)
            if pid <= 0 or pid == os.getpid() or pid in known_pids:
                continue
            command = tuple(str(part) for part in process.cmdline)
            descriptor = _model_command_descriptor(command)
            if not descriptor.recognized:
                continue
            try:
                process_group_id = int(os.getpgid(pid))
            except (OSError, ProcessLookupError, ValueError):
                process_group_id = 0
            if process_group_id > 0 and process_group_id in managed_groups:
                continue
            started_at = float(process.create_time)
            observed_gb = float(process.rss_bytes) / float(1024**3)
            model_paths = descriptor.model_paths
            if model_paths:
                model_path = model_paths[0]
                declared_gb = sum(
                    estimate_model_job_footprint_gb(path, purpose=descriptor.purpose)
                    for path in model_paths
                )
                identity_status = "resolved"
            else:
                model_path = f"unresolved:model-process:{pid}"
                declared_gb = lane_budget_gb(observer=resource_observer)
                identity_status = "unresolved_fail_closed"
            ancestor_pids = set(process.ancestor_pids)
            parent_owner = next(
                (owner for owner in known if owner.process.pid in ancestor_pids),
                None,
            )
            command_digest = hashlib.sha256(
                "\0".join(command).encode("utf-8", errors="replace")
            ).hexdigest()
            observations.append(
                LaneOwnerObservation(
                    owner_id=f"external-model:{pid}:{int(started_at * 1_000_000)}",
                    model_path=model_path,
                    declared_gb=max(0.1, declared_gb),
                    purpose=descriptor.purpose,
                    observed_gb=max(0.0, observed_gb),
                    process=ProcessIdentity(pid, started_at),
                    priority=90 if descriptor.purpose == "serve" else 70,
                    preemptible=False,
                    lease_ttl_s=30.0,
                    metadata={
                        "externally_discovered": True,
                        "model_identity_status": identity_status,
                        "command_name": Path(command[0]).name if command else str(
                            process.name or "unknown"
                        ),
                        "command_sha256": command_digest,
                        "process_group_id": process_group_id,
                        "process_tree_escape": parent_owner is not None,
                        "registered_parent_owner_id": (
                            parent_owner.owner_id if parent_owner is not None else ""
                        ),
                    },
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
    return observations


async def evict_managed_process_owner(owner: LaneOwnerObservation, reason: str) -> bool:
    """Terminate only a process identity that opted into managed preemption."""

    metadata = dict(owner.metadata)
    if not owner.preemptible or not bool(metadata.get("managed_model_process", False)):
        return False
    identity = owner.process
    pgid = int(metadata.get("process_group_id") or 0)
    current_pgid = os.getpgrp()
    isolated_group = bool(metadata.get("start_new_session", False))

    def _tree_alive() -> bool:
        return _default_process_alive(identity) or (
            isolated_group
            and managed_process_group_alive(
                pgid,
                root_started_at=identity.started_at,
            )
        )

    if not _tree_alive():
        return True
    try:
        if isolated_group and pgid > 0 and pgid != current_pgid:
            os.killpg(pgid, signal.SIGTERM)
        elif _default_process_alive(identity):
            os.kill(identity.pid, signal.SIGTERM)
        else:
            return False
    except ProcessLookupError:
        return not _tree_alive()
    except (OSError, ValueError):
        return False

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _tree_alive():
            return True
        await asyncio.sleep(0.1)
    try:
        if isolated_group and pgid > 0 and pgid != current_pgid:
            os.killpg(pgid, signal.SIGKILL)
        elif _default_process_alive(identity):
            os.kill(identity.pid, signal.SIGKILL)
        else:
            return False
    except ProcessLookupError:
        return not _tree_alive()
    except (OSError, ValueError):
        return False
    kill_deadline = time.monotonic() + 5.0
    while time.monotonic() < kill_deadline:
        if not _tree_alive():
            return True
        await asyncio.sleep(0.1)
    logger.error(
        "Managed model process survived TERM/KILL owner=%s pid=%s reason=%s",
        owner.owner_id,
        identity.pid,
        reason,
    )
    return False


async def wait_for_model_job_headroom(
    claim: LaneClaim,
    *,
    timeout_s: float = 20.0,
    observer: ResourceObserver | None = None,
) -> bool:
    """Re-observe physical RAM after policy admission and any eviction."""

    bounded_timeout_s = max(0.0, float(timeout_s))
    deadline = time.monotonic() + bounded_timeout_s
    max_observations = max(1, int(bounded_timeout_s / 0.25) + 2)
    available_gb = 0.0
    resource_observer = observer or get_resource_observer()
    for _attempt in range(max_observations):
        try:
            memory = resource_observer.memory()
            if not memory.available:
                return False
            available_gb = float(memory.available_bytes) / float(1024**3)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        if available_gb >= float(claim.request_gb):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            break
        await asyncio.sleep(min(0.25, remaining_s))
    logger.warning(
        "Model job lacks physical headroom owner=%s available=%.2fGB requested=%.2fGB",
        claim.owner_id,
        available_gb,
        claim.request_gb,
    )
    return False


async def run_owned_model_thread_call[T](
    operation: Callable[[], T],
    *,
    operation_name: str,
    timeout_s: float | None = None,
) -> T:
    """Do not let coroutine cancellation outlive an in-process model call."""

    from core.utils.task_tracker import get_task_tracker

    task = get_task_tracker().create_task(
        asyncio.to_thread(operation),
        name=f"OwnedModelCall:{operation_name}",
    )
    try:
        if timeout_s is None:
            return await asyncio.shield(task)
        done, _pending = await asyncio.wait(
            {task},
            timeout=max(0.0, float(timeout_s)),
            return_when=asyncio.ALL_COMPLETED,
        )
        if task in done:
            return cast("T", task.result())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as worker_error:  # noqa: BLE001 - preserve timeout contract after native drain.
            raise TimeoutError(
                f"owned_model_call_timed_out:{operation_name}"
            ) from worker_error
        raise TimeoutError(f"owned_model_call_timed_out:{operation_name}")
    except asyncio.CancelledError:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as cleanup_error:  # noqa: BLE001 - cancellation wins after exact worker drain.
            logger.warning(
                "Owned model call failed while cancellation drained operation=%s: %s",
                operation_name,
                cleanup_error,
            )
        raise


async def prepare_model_lane_claim(
    claim: LaneClaim,
    *,
    controller: ModelLaneController | None = None,
    evict: EvictCallback = evict_registered_model_owner,
    compensate: CompensateCallback = compensate_registered_model_owner,
    reclaim: ReclaimCallback = wait_for_model_job_headroom,
) -> tuple[ModelLaneController, LaneTransactionDecision]:
    """Reserve and synchronously satisfy every pre-spawn model-lane obligation."""

    lane_controller = controller or get_model_lane_controller()
    reclaim_callback: ReclaimCallback = reclaim
    if reclaim is wait_for_model_job_headroom:
        async def _reclaim_with_controller_observer(candidate: LaneClaim) -> bool:
            return await wait_for_model_job_headroom(
                candidate,
                observer=lane_controller.resource_observer,
            )

        reclaim_callback = _reclaim_with_controller_observer
    from core.utils.task_tracker import get_task_tracker

    reserve_task = get_task_tracker().create_task(
        lane_controller.reserve(claim),
        name=f"ModelLaneReserve:{claim.owner_id}",
    )
    try:
        decision = await asyncio.shield(reserve_task)
    except asyncio.CancelledError:
        decision = await asyncio.shield(reserve_task)
        if decision.admitted:
            await asyncio.shield(
                lane_controller.cancel(
                    decision,
                    reason="model_lane_reservation_cancelled",
                    compensate=compensate,
                )
            )
        raise
    try:
        if not decision.admitted:
            raise ModelLaneControlError(
                f"model_lane_admission_refused:{decision.reason}:receipt={decision.receipt_id}"
            )
        if not decision.ready_to_spawn:
            decision = await lane_controller.prepare(
                decision,
                evict=evict,
                observe=lambda: asyncio.to_thread(lane_controller.owner_observations),
                reclaim=reclaim_callback,
                compensate=compensate,
            )
        if not decision.ready_to_spawn:
            raise ModelLaneControlError(
                f"model_lane_admission_cancelled:{decision.reason}:receipt={decision.receipt_id}"
            )
        if not await lane_controller._call_bool(reclaim_callback, claim):
            cancelled = await lane_controller.cancel(
                decision,
                reason="physical_model_headroom_unavailable",
                compensate=compensate,
            )
            raise ModelLaneControlError(
                f"model_lane_admission_cancelled:{cancelled.reason}:receipt={cancelled.receipt_id}"
            )
        return lane_controller, decision
    except asyncio.CancelledError:
        await asyncio.shield(
            lane_controller.cancel(
                decision,
                reason="model_lane_preparation_cancelled",
                compensate=compensate,
            )
        )
        raise
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        if decision.admitted and decision.state.value not in _TERMINAL_RESERVATION_STATES:
            await lane_controller.cancel(
                decision,
                reason="model_lane_preparation_failed",
                compensate=compensate,
            )
        raise


class ModelLaneController:
    """Cross-process capacity ledger and model-lane transaction owner."""

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        receipt_store: Any | None = None,
        process_alive: ProcessAliveProbe | None = None,
        process_discovery: ProcessDiscoveryProbe | None = discover_external_model_processes,
        policy: LaneAdmissionController | None = None,
        observer: ResourceObserver | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        configured_state_path = str(
            os.environ.get("AURA_MODEL_LANE_STATE_PATH", "") or ""
        ).strip()
        self.state_path = Path(
            state_path
            or configured_state_path
            or (state_root() / "run" / "model_lane_control.json")
        )
        self.lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        self._receipt_store = receipt_store
        self._observer = observer
        self._process_alive = process_alive or (
            lambda identity: _default_process_alive(
                identity,
                observer=self.resource_observer,
            )
        )
        self._process_discovery: ProcessDiscoveryProbe | None
        if process_discovery is discover_external_model_processes:
            self._process_discovery = lambda known: discover_external_model_processes(
                known,
                observer=self.resource_observer,
            )
        else:
            self._process_discovery = process_discovery
        self._policy = policy or LaneAdmissionController(observer=observer)
        self._clock = clock
        self._thread_lock = threading.RLock()
        self._last_discovery_monotonic = 0.0

    @property
    def resource_observer(self) -> ResourceObserver:
        return self._observer or get_resource_observer()

    def _refresh_external_owners(self, *, force: bool = False) -> None:
        if self._process_discovery is None:
            return
        observed_monotonic = time.monotonic()
        with self._thread_lock:
            if (
                not force
                and observed_monotonic - self._last_discovery_monotonic < 1.0
            ):
                return
            self._last_discovery_monotonic = observed_monotonic
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            self._prune_locked(state, now)
            known = [
                self._record_to_observation(record)
                for record in state["owners"].values()
                if not bool(dict(record.get("metadata") or {}).get("externally_discovered"))
            ]
        try:
            discovered = list(self._process_discovery(known))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            with self._thread_lock:
                self._last_discovery_monotonic = 0.0
            raise ModelLaneControlError(
                f"external_model_process_discovery_failed:{type(exc).__name__}"
            ) from exc
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            self._prune_locked(state, now)
            represented_pids = {
                int(record.get("process", {}).get("pid") or 0)
                for record in state["owners"].values()
                if not bool(dict(record.get("metadata") or {}).get("externally_discovered"))
                and isinstance(record.get("process"), Mapping)
            }
            discovered_ids = {item.owner_id for item in discovered}
            for owner_id, record in list(state["owners"].items()):
                metadata = dict(record.get("metadata") or {})
                if not bool(metadata.get("externally_discovered")):
                    continue
                identity = self._identity_from_record(record)
                if identity.pid in represented_pids:
                    state["owners"].pop(owner_id, None)
                    self._append_event(
                        state,
                        "external_owner_reconciled",
                        at=now,
                        owner_id=owner_id,
                        pid=identity.pid,
                    )
                elif owner_id not in discovered_ids and not self._process_alive(identity):
                    state["owners"].pop(owner_id, None)
            self._sync_observations_locked(state, discovered, now)
            self._save_locked(state)

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "generation": 0,
            "owners": {},
            "reservations": {},
            "events": [],
        }

    def _load_locked(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._empty_state()
        try:
            envelope = read_json_envelope(self.state_path)
        except (OSError, ValueError) as exc:
            raise ModelLaneControlError(f"model_lane_state_unreadable:{type(exc).__name__}") from exc
        if (
            envelope.get("schema_name") != _SCHEMA_NAME
            or int(envelope.get("schema_version") or 0) != _SCHEMA_VERSION
        ):
            raise ModelLaneControlError("model_lane_state_schema_mismatch")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ModelLaneControlError("model_lane_state_payload_invalid")
        owners = payload.get("owners")
        reservations = payload.get("reservations")
        events = payload.get("events")
        if not isinstance(owners, dict) or not isinstance(reservations, dict) or not isinstance(events, list):
            raise ModelLaneControlError("model_lane_state_shape_invalid")
        payload.setdefault("generation", 0)
        return payload

    def _save_locked(self, state: dict[str, Any]) -> None:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "model_lane_control.persist",
            domain="state_mutation",
        ):
            get_file_write_gateway().write_json(
                self.state_path,
                state,
                schema_version=_SCHEMA_VERSION,
                schema_name=_SCHEMA_NAME,
                indent=None,
                source="core.runtime.model_lane_control",
            )

    @staticmethod
    def _append_event(state: dict[str, Any], event: str, **fields: Any) -> None:
        events = state.setdefault("events", [])
        events.append({"event": event, **fields})
        del events[:-_EVENT_LIMIT]

    @staticmethod
    def _identity_from_record(record: Mapping[str, Any], key: str = "process") -> ProcessIdentity:
        raw = record.get(key)
        if not isinstance(raw, Mapping):
            return ProcessIdentity(0, 0.0)
        return ProcessIdentity(int(raw.get("pid") or 0), float(raw.get("started_at") or 0.0))

    def _record_alive(self, record: Mapping[str, Any], *, key: str = "process") -> bool:
        identity = self._identity_from_record(record, key)
        return bool(identity.pid > 0 and identity.started_at > 0.0 and self._process_alive(identity))

    def _owner_process_tree_alive(self, record: Mapping[str, Any]) -> bool:
        identity = self._identity_from_record(record)
        if identity.pid > 0 and identity.started_at > 0.0 and self._process_alive(identity):
            return True
        metadata = dict(record.get("metadata") or {})
        return bool(
            metadata.get("managed_model_process", False)
            and metadata.get("start_new_session", False)
            and managed_process_group_alive(
                int(metadata.get("process_group_id") or 0),
                root_started_at=identity.started_at,
                observer=self.resource_observer,
            )
        )

    def _prune_locked(self, state: dict[str, Any], now: float) -> bool:
        changed = False
        reservations = state["reservations"]
        for transaction_id, record in list(reservations.items()):
            status = str(record.get("state") or "")
            if status not in _ACTIVE_RESERVATION_STATES:
                continue
            expired = float(record.get("expires_at") or 0.0) <= now
            controller_dead = not self._record_alive(record, key="controller_process")
            if not expired and not controller_dead:
                continue
            record["state"] = LaneTransactionState.EXPIRED.value
            record["reason"] = "reservation_owner_dead" if controller_dead else "reservation_ttl_expired"
            record["terminal_at"] = now
            compensation = dict(record.get("compensation") or {})
            pending_compensation = [
                str(owner_id)
                for owner_id in record.get("evicted_owner_ids") or ()
                if str(owner_id) not in compensation
            ]
            record["compensation_pending_owner_ids"] = pending_compensation
            record.setdefault("compensation_claims", {})
            self._append_event(
                state,
                "reservation_expired",
                at=now,
                transaction_id=transaction_id,
                reason=record["reason"],
            )
            changed = True

        for record in reservations.values():
            pending_owner_ids = set(record.get("compensation_pending_owner_ids") or ())
            claims = record.get("compensation_claims")
            if not pending_owner_ids or not isinstance(claims, dict):
                continue
            for owner_id, claim in list(claims.items()):
                if owner_id not in pending_owner_ids or not isinstance(claim, Mapping):
                    claims.pop(owner_id, None)
                    changed = True
                    continue
                if float(claim.get("expires_at") or 0.0) <= now:
                    claims.pop(owner_id, None)
                    changed = True

        active_transactions = {
            transaction_id
            for transaction_id, record in reservations.items()
            if str(record.get("state") or "") in _ACTIVE_RESERVATION_STATES
        }
        owners = state["owners"]
        for owner_id, record in list(owners.items()):
            identity = self._identity_from_record(record)
            has_identity = identity.pid > 0 and identity.started_at > 0.0
            alive = self._owner_process_tree_alive(record) if has_identity else False
            lease_expired = float(record.get("lease_expires_at") or 0.0) <= now
            lease_mode = str(dict(record.get("metadata") or {}).get("lease_mode") or "process")
            heartbeat_expired = lease_mode == "heartbeat" and lease_expired
            if heartbeat_expired and has_identity and alive:
                metadata = dict(record.get("metadata") or {})
                if not bool(metadata.get("heartbeat_lease_stale", False)):
                    metadata["heartbeat_lease_stale"] = True
                    metadata["preemptible_before_heartbeat_stale"] = bool(
                        record.get("preemptible", True)
                    )
                    record["metadata"] = metadata
                    record["preemptible"] = False
                    self._append_event(
                        state,
                        "owner_heartbeat_stale_fail_closed",
                        at=now,
                        owner_id=owner_id,
                        fencing_token=int(record.get("fencing_token") or 0),
                    )
                    changed = True
                continue
            if (
                (has_identity and not alive)
                or (not has_identity and lease_expired)
                or heartbeat_expired
            ):
                owners.pop(owner_id, None)
                self._append_event(
                    state,
                    "owner_reaped",
                    at=now,
                    owner_id=owner_id,
                    reason=(
                        "heartbeat_lease_expired"
                        if heartbeat_expired
                        else "process_dead"
                        if has_identity
                        else "owner_lease_expired"
                    ),
                )
                changed = True
                continue
            pending = str(record.get("eviction_requested_by") or "")
            if pending and pending not in active_transactions:
                record["eviction_requested_by"] = ""
                record["eviction_fencing_token"] = 0
                changed = True
        terminal = sorted(
            (
                record
                for record in reservations.values()
                if str(record.get("state") or "") in _TERMINAL_RESERVATION_STATES
                and not record.get("compensation_pending_owner_ids")
            ),
            key=lambda record: float(record.get("terminal_at") or record.get("created_at") or 0.0),
            reverse=True,
        )
        for record in terminal[_TERMINAL_RESERVATION_LIMIT:]:
            request_id = str(record.get("request_id") or "")
            if request_id and reservations.pop(request_id, None) is not None:
                changed = True
        return changed

    @staticmethod
    def _owner_lease_ttl(observation: LaneOwnerObservation | LaneClaim) -> float:
        requested = float(getattr(observation, "lease_ttl_s", 0.0) or 0.0)
        if isinstance(observation, LaneClaim):
            requested = float(observation.owner_lease_ttl_s or 0.0)
        return max(5.0, requested or float(_OWNER_LEASE_TTL_FLAG.value()))

    @staticmethod
    def _reservation_ttl(claim: LaneClaim) -> float:
        return max(5.0, float(claim.reservation_ttl_s or _RESERVATION_TTL_FLAG.value()))

    def _sync_observations_locked(
        self,
        state: dict[str, Any],
        observations: Iterable[LaneOwnerObservation],
        now: float,
    ) -> None:
        for observation in observations:
            process = observation.process
            if process.pid > 0 and process.started_at > 0.0 and not self._process_alive(process):
                continue
            lane, qos = classify_lane(observation.model_path, purpose=observation.purpose)
            existing = state["owners"].get(observation.owner_id, {})
            state["owners"][observation.owner_id] = {
                "owner_id": observation.owner_id,
                "model_path": observation.model_path,
                "purpose": observation.purpose,
                "lane": lane,
                "qos": qos.value,
                "declared_gb": float(observation.declared_gb),
                "observed_gb": max(0.0, float(observation.observed_gb)),
                "priority": int(observation.priority),
                "preemptible": bool(observation.preemptible),
                "last_user_facing_age_s": observation.last_user_facing_age_s,
                "process": process.to_dict(),
                "registered_at": float(existing.get("registered_at") or now),
                "heartbeat_at": now,
                "lease_ttl_s": self._owner_lease_ttl(observation),
                "lease_expires_at": now + self._owner_lease_ttl(observation),
                "fencing_token": int(existing.get("fencing_token") or 0),
                "eviction_requested_by": str(existing.get("eviction_requested_by") or ""),
                "eviction_fencing_token": int(existing.get("eviction_fencing_token") or 0),
                "metadata": _json_metadata(observation.metadata),
            }

    def _record_to_observation(self, record: Mapping[str, Any]) -> LaneOwnerObservation:
        age = record.get("last_user_facing_age_s")
        return LaneOwnerObservation(
            owner_id=str(record.get("owner_id") or ""),
            model_path=str(record.get("model_path") or ""),
            purpose=str(record.get("purpose") or "serve"),
            declared_gb=float(record.get("declared_gb") or 0.0),
            observed_gb=float(record.get("observed_gb") or 0.0),
            process=ModelLaneController._identity_from_record(record),
            priority=int(record.get("priority") or 50),
            preemptible=bool(record.get("preemptible", True)),
            last_user_facing_age_s=float(age) if age is not None else None,
            lease_ttl_s=max(
                0.0,
                float(record.get("lease_ttl_s") or 0.0)
                or float(record.get("lease_expires_at") or 0.0) - self._clock(),
            ),
            metadata=dict(record.get("metadata") or {}),
        )

    @staticmethod
    def _claim_matches_record(claim: LaneClaim, record: Mapping[str, Any]) -> bool:
        """Require an idempotent replay to be the exact original claim."""

        return bool(
            str(record.get("owner_id") or "") == claim.owner_id
            and str(record.get("model_path") or "") == claim.model_path
            and str(record.get("purpose") or "serve") == claim.purpose
            and float(record.get("request_gb") or 0.0) == float(claim.request_gb)
            and int(record.get("priority") or 0) == int(claim.priority)
            and bool(record.get("preemptible", True)) is bool(claim.preemptible)
            and bool(record.get("foreground", False)) is bool(claim.foreground)
            and bool(record.get("allow_disruptive_eviction", False))
            is bool(claim.allow_disruptive_eviction)
            and bool(record.get("allow_last_warm_eviction", False))
            is bool(claim.allow_last_warm_eviction)
            and float(record.get("reservation_ttl_s") or 0.0)
            == float(claim.reservation_ttl_s or 0.0)
            and float(record.get("requested_owner_lease_ttl_s") or 0.0)
            == float(claim.owner_lease_ttl_s or 0.0)
            and dict(record.get("metadata") or {}) == _json_metadata(claim.metadata)
        )

    @staticmethod
    def _observation_payload(observation: LaneOwnerObservation) -> dict[str, Any]:
        return {
            "owner_id": observation.owner_id,
            "model_path": observation.model_path,
            "declared_gb": observation.declared_gb,
            "purpose": observation.purpose,
            "observed_gb": observation.observed_gb,
            "process": observation.process.to_dict(),
            "priority": observation.priority,
            "preemptible": observation.preemptible,
            "last_user_facing_age_s": observation.last_user_facing_age_s,
            "lease_ttl_s": observation.lease_ttl_s,
            "metadata": _json_metadata(observation.metadata),
        }

    @staticmethod
    def _capacity_totals(
        state: Mapping[str, Any], *, exclude_request_id: str = ""
    ) -> tuple[float, float]:
        committed = sum(
            float(record.get("declared_gb") or 0.0)
            for record in dict(state.get("owners") or {}).values()
        )
        reserved = sum(
            float(record.get("request_gb") or 0.0)
            for request_id, record in dict(state.get("reservations") or {}).items()
            if request_id != exclude_request_id
            and str(record.get("state") or "") in _ACTIVE_RESERVATION_STATES
        )
        return committed, reserved

    @staticmethod
    def _record_to_decision(record: Mapping[str, Any], *, replayed: bool = False) -> LaneTransactionDecision:
        state = LaneTransactionState(str(record.get("state") or LaneTransactionState.REFUSED.value))
        lane, qos = classify_lane(
            str(record.get("model_path") or ""),
            purpose=str(record.get("purpose") or "serve"),
        )
        return LaneTransactionDecision(
            request_id=str(record.get("request_id") or ""),
            transaction_id=str(record.get("transaction_id") or ""),
            fencing_token=int(record.get("fencing_token") or 0),
            admitted=state in {
                LaneTransactionState.RESERVED,
                LaneTransactionState.EVICTING,
                LaneTransactionState.READY,
                LaneTransactionState.COMMITTED,
            },
            ready_to_spawn=state in {LaneTransactionState.READY, LaneTransactionState.COMMITTED},
            state=state,
            reason=str(record.get("reason") or ""),
            owner_id=str(record.get("owner_id") or ""),
            model_path=str(record.get("model_path") or ""),
            lane=lane,
            qos=qos,
            request_gb=float(record.get("request_gb") or 0.0),
            committed_gb=float(record.get("committed_gb") or 0.0),
            reserved_gb=float(record.get("reserved_gb") or 0.0),
            budget_gb=float(record.get("budget_gb") or 0.0),
            observation_source=str(record.get("observation_source") or "unavailable"),
            observation_scenario_id=str(record.get("observation_scenario_id") or ""),
            resource_observation_available=bool(
                record.get("resource_observation_available", False)
            ),
            evict_owner_ids=tuple(str(item) for item in record.get("evict_owner_ids") or ()),
            evicted_owner_ids=tuple(str(item) for item in record.get("evicted_owner_ids") or ()),
            receipt_id=str(record.get("terminal_receipt_id") or ""),
            replayed=replayed,
        )

    @staticmethod
    def _terminal_receipt_id(request_id: str) -> str:
        digest = hashlib.sha256(str(request_id).encode("utf-8")).hexdigest()[:32]
        return f"resource_admission-lane-{digest}"

    @staticmethod
    def _eviction_receipt_id(transaction_id: str, owner_id: str) -> str:
        digest = hashlib.sha256(f"{transaction_id}:{owner_id}".encode()).hexdigest()[:32]
        return f"resource_admission-eviction-{digest}"

    def _receipt_store_instance(self) -> Any:
        if self._receipt_store is not None:
            return self._receipt_store
        from core.runtime.receipts import get_receipt_store

        return get_receipt_store()

    def _emit_terminal_receipt(self, record: Mapping[str, Any]) -> str:
        from core.runtime.receipts import ResourceAdmissionReceipt

        receipt_id = self._terminal_receipt_id(str(record.get("request_id") or ""))
        receipt = ResourceAdmissionReceipt(
            receipt_id=receipt_id,
            cause=f"model_lane:{record.get('owner_id', '')}",
            created_at=float(record.get("terminal_at") or record.get("created_at") or self._clock()),
            request_id=str(record.get("request_id") or ""),
            owner=str(record.get("owner_id") or ""),
            work_class="model_lane",
            lane=str(record.get("lane") or ""),
            priority=int(record.get("priority") or 0),
            decision=str(record.get("state") or ""),
            reason=str(record.get("reason") or ""),
            lease_id=str(record.get("transaction_id") or ""),
            pressure={
                "budget_gb": float(record.get("budget_gb") or 0.0),
                "committed_gb": float(record.get("committed_gb") or 0.0),
                "reserved_gb": float(record.get("reserved_gb") or 0.0),
                "request_gb": float(record.get("request_gb") or 0.0),
                "observation_source": str(
                    record.get("observation_source") or "unavailable"
                ),
                "observation_scenario_id": str(
                    record.get("observation_scenario_id") or ""
                ),
                "resource_observation_available": bool(
                    record.get("resource_observation_available", False)
                ),
            },
            metadata={
                **dict(record.get("metadata") or {}),
                "fencing_token": int(record.get("fencing_token") or 0),
                "evict_owner_ids": list(record.get("evict_owner_ids") or ()),
                "evicted_owner_ids": list(record.get("evicted_owner_ids") or ()),
                "compensation": dict(record.get("compensation") or {}),
            },
        )
        emitted = self._receipt_store_instance().emit(receipt)
        return str(emitted.receipt_id)

    def _durable_receipt_exists(self, receipt_id: str) -> bool:
        receipt = self._receipt_store_instance().get(str(receipt_id))
        if receipt is None:
            return False
        if str(getattr(receipt, "kind", "")) != "resource_admission":
            raise ModelLaneControlError("model_lane_receipt_id_kind_collision")
        return True

    def _emit_eviction_receipt(
        self,
        reservation: Mapping[str, Any],
        *,
        owner: LaneOwnerObservation,
        outcome: str,
        reason: str,
        completed_at: float,
    ) -> str:
        from core.runtime.receipts import ResourceAdmissionReceipt

        receipt_id = self._eviction_receipt_id(
            str(reservation.get("transaction_id") or ""), owner.owner_id
        )
        if self._durable_receipt_exists(receipt_id):
            return receipt_id
        receipt = ResourceAdmissionReceipt(
            receipt_id=receipt_id,
            cause=f"model_lane_eviction:{reservation.get('transaction_id', '')}",
            created_at=completed_at,
            request_id=str(reservation.get("request_id") or ""),
            owner=owner.owner_id,
            work_class="model_lane_eviction",
            lane=classify_lane(owner.model_path, purpose=owner.purpose)[0],
            priority=int(owner.priority),
            decision=outcome,
            reason=reason,
            lease_id=str(reservation.get("transaction_id") or ""),
            pressure={
                "declared_gb": owner.declared_gb,
                "observed_gb": owner.observed_gb,
                "observation_source": str(
                    reservation.get("observation_source") or "unavailable"
                ),
                "observation_scenario_id": str(
                    reservation.get("observation_scenario_id") or ""
                ),
            },
            metadata={
                "candidate_owner_id": str(reservation.get("owner_id") or ""),
                "candidate_model_path": str(reservation.get("model_path") or ""),
                "fencing_token": int(reservation.get("fencing_token") or 0),
                "target_process": owner.process.to_dict(),
            },
        )
        emitted = self._receipt_store_instance().emit(receipt)
        return str(emitted.receipt_id)

    def _persist_terminal_receipt(self, request_id: str) -> str:
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            record = state["reservations"].get(request_id)
            if not isinstance(record, dict):
                raise ModelLaneControlError("lane_reservation_missing_for_receipt")
            existing = str(record.get("terminal_receipt_id") or "")
            if existing:
                return existing
            if record.get("compensation_pending_owner_ids"):
                raise ModelLaneControlError(
                    "lane_terminal_receipt_blocked_by_pending_compensation"
                )
            receipt_id = self._terminal_receipt_id(request_id)
            if not self._durable_receipt_exists(receipt_id):
                receipt_id = self._emit_terminal_receipt(record)
            record["terminal_receipt_id"] = receipt_id
            self._save_locked(state)
            return receipt_id

    def _adopt_terminal_receipt_if_present(self, request_id: str) -> str:
        receipt_id = self._terminal_receipt_id(request_id)
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            record = state["reservations"].get(str(request_id))
            if not isinstance(record, dict):
                raise ModelLaneControlError("lane_reservation_missing_for_receipt_adoption")
            existing = str(record.get("terminal_receipt_id") or "")
            if existing and existing != receipt_id:
                raise ModelLaneControlError("lane_terminal_receipt_identity_mismatch")
            if not existing and not self._durable_receipt_exists(receipt_id):
                return ""
            if not existing:
                record["terminal_receipt_id"] = receipt_id
                self._append_event(
                    state,
                    "terminal_receipt_recovered",
                    at=self._clock(),
                    request_id=str(request_id),
                    receipt_id=receipt_id,
                )
                self._save_locked(state)
        return receipt_id

    def _persist_missing_terminal_receipts(self) -> None:
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            request_ids = [
                str(request_id)
                for request_id, record in state["reservations"].items()
                if str(record.get("state") or "") in _TERMINAL_RESERVATION_STATES
                and not str(record.get("terminal_receipt_id") or "")
                and not record.get("compensation_pending_owner_ids")
            ]
        for request_id in request_ids:
            self._persist_terminal_receipt(request_id)

    def _claim_expired_compensation_sync(
        self,
    ) -> tuple[str, str, str, LaneOwnerObservation] | None:
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            changed = self._prune_locked(state, now)
            for request_id, record in state["reservations"].items():
                if str(record.get("state") or "") != LaneTransactionState.EXPIRED.value:
                    continue
                pending = [
                    str(owner_id)
                    for owner_id in record.get("compensation_pending_owner_ids") or ()
                ]
                claims = record.setdefault("compensation_claims", {})
                evicted = dict(record.get("evicted_owners") or {})
                for owner_id in pending:
                    if owner_id in claims:
                        continue
                    payload = evicted.get(owner_id)
                    if not isinstance(payload, Mapping):
                        record.setdefault("compensation", {})[owner_id] = False
                        record["compensation_pending_owner_ids"] = [
                            candidate
                            for candidate in record.get(
                                "compensation_pending_owner_ids"
                            )
                            or ()
                            if str(candidate) != owner_id
                        ]
                        self._append_event(
                            state,
                            "expired_compensation_unrecoverable",
                            at=now,
                            request_id=str(request_id),
                            owner_id=owner_id,
                            reason="evicted_owner_payload_missing",
                        )
                        changed = True
                        continue
                    claim_token = str(uuid.uuid4())
                    claims[owner_id] = {
                        "token": claim_token,
                        "claimed_at": now,
                        "expires_at": now + 30.0,
                        "claimant_process": ProcessIdentity.current(
                            observer=self.resource_observer
                        ).to_dict(),
                    }
                    self._append_event(
                        state,
                        "expired_compensation_claimed",
                        at=now,
                        request_id=str(request_id),
                        owner_id=owner_id,
                    )
                    self._save_locked(state)
                    return (
                        str(request_id),
                        str(record.get("transaction_id") or ""),
                        claim_token,
                        self._record_to_observation(payload),
                    )
            if changed:
                self._save_locked(state)
        return None

    def _finish_expired_compensation_sync(
        self,
        *,
        request_id: str,
        owner_id: str,
        claim_token: str,
        restored: bool,
    ) -> None:
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            record = state["reservations"].get(str(request_id))
            if not isinstance(record, dict):
                raise ModelLaneControlError("expired_compensation_reservation_missing")
            claims = record.get("compensation_claims")
            if not isinstance(claims, dict):
                raise ModelLaneControlError("expired_compensation_claim_lost")
            claim = claims.get(owner_id)
            if not isinstance(claim, Mapping) or not hmac.compare_digest(
                str(claim.get("token") or ""), str(claim_token)
            ):
                raise ModelLaneControlError("expired_compensation_claim_lost")
            record.setdefault("compensation", {})[owner_id] = bool(restored)
            record["compensation_pending_owner_ids"] = [
                candidate
                for candidate in record.get("compensation_pending_owner_ids") or ()
                if str(candidate) != owner_id
            ]
            claims.pop(owner_id, None)
            self._append_event(
                state,
                "expired_compensation_finished",
                at=now,
                request_id=str(request_id),
                owner_id=owner_id,
                restored=bool(restored),
            )
            self._save_locked(state)

    def _release_expired_compensation_claim_sync(
        self,
        *,
        request_id: str,
        owner_id: str,
        claim_token: str,
    ) -> None:
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            record = state["reservations"].get(str(request_id))
            if not isinstance(record, dict):
                return
            claims = record.get("compensation_claims")
            if not isinstance(claims, dict):
                return
            claim = claims.get(owner_id)
            if not isinstance(claim, Mapping) or not hmac.compare_digest(
                str(claim.get("token") or ""), str(claim_token)
            ):
                return
            claims.pop(owner_id, None)
            self._save_locked(state)

    async def reconcile_expired_compensations(
        self,
        *,
        compensate: CompensateCallback = compensate_registered_model_owner,
        max_compensations: int = _TERMINAL_RESERVATION_LIMIT,
    ) -> int:
        """Recover owners displaced by a reservation that died before commit."""

        completed = 0
        for _attempt in range(max(0, int(max_compensations))):
            claimed = await asyncio.to_thread(self._claim_expired_compensation_sync)
            if claimed is None:
                break
            request_id, transaction_id, claim_token, owner = claimed
            try:
                restored = await self._call_bool(
                    compensate,
                    owner,
                    f"compensate_expired_candidate:{transaction_id}",
                )
            except asyncio.CancelledError:
                await asyncio.shield(
                    asyncio.to_thread(
                        self._release_expired_compensation_claim_sync,
                        request_id=request_id,
                        owner_id=owner.owner_id,
                        claim_token=claim_token,
                    )
                )
                raise
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                restored = False
            await asyncio.shield(
                asyncio.to_thread(
                    self._finish_expired_compensation_sync,
                    request_id=request_id,
                    owner_id=owner.owner_id,
                    claim_token=claim_token,
                    restored=restored,
                )
            )
            completed += 1
        else:
            logger.warning(
                "Expired model-lane compensation drain reached per-call bound=%d",
                max_compensations,
            )
        await asyncio.to_thread(self._persist_missing_terminal_receipts)
        return completed

    async def reserve(
        self,
        claim: LaneClaim,
        *,
        observations: Iterable[LaneOwnerObservation] = (),
    ) -> LaneTransactionDecision:
        await self.reconcile_expired_compensations()
        return await asyncio.to_thread(self.reserve_sync, claim, observations=observations)

    def reserve_sync(
        self,
        claim: LaneClaim,
        *,
        observations: Iterable[LaneOwnerObservation] = (),
    ) -> LaneTransactionDecision:
        self._refresh_external_owners(force=True)
        now = self._clock()
        terminal = False
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            self._prune_locked(state, now)
            pending_compensation = {
                str(request_id)
                for request_id, record in state["reservations"].items()
                if record.get("compensation_pending_owner_ids")
            }
            if pending_compensation and claim.request_id not in pending_compensation:
                self._save_locked(state)
                raise ModelLaneControlError(
                    "model_lane_expired_compensation_pending:"
                    + ",".join(sorted(pending_compensation))
                )
            self._sync_observations_locked(state, list(observations), now)
            existing = state["reservations"].get(claim.request_id)
            if isinstance(existing, dict):
                if not self._claim_matches_record(claim, existing):
                    raise ModelLaneControlError("lane_request_id_reused_with_different_claim")
                self._save_locked(state)
                decision = self._record_to_decision(existing, replayed=True)
                terminal = decision.state.value in _TERMINAL_RESERVATION_STATES
            else:
                lane, qos = classify_lane(claim.model_path, purpose=claim.purpose)
                committed, reserved = self._capacity_totals(
                    state,
                    exclude_request_id=claim.request_id,
                )
                committed_owner_conflict = claim.owner_id in state["owners"]
                reservation_owner_conflict = any(
                    request_id != claim.request_id
                    and str(record.get("owner_id") or "") == claim.owner_id
                    and str(record.get("state") or "") in _ACTIVE_RESERVATION_STATES
                    for request_id, record in state["reservations"].items()
                )
                active: list[ActiveLane] = []
                for owner_id, owner in state["owners"].items():
                    try:
                        owner_qos = QoSClass(str(owner.get("qos") or QoSClass.BEST_EFFORT.value))
                    except ValueError:
                        owner_qos = QoSClass.BEST_EFFORT
                    age = owner.get("last_user_facing_age_s")
                    active.append(
                        ActiveLane(
                            lane=str(owner.get("lane") or "auxiliary"),
                            qos=owner_qos,
                            footprint_gb=float(owner.get("declared_gb") or 0.0),
                            model_path=str(owner_id),
                            last_user_facing_age_s=float(age) if age is not None else None,
                        )
                    )
                if reserved > 0.0:
                    active.append(
                        ActiveLane(
                            lane="fenced_reservations",
                            qos=QoSClass.GUARANTEED,
                            footprint_gb=reserved,
                            model_path="__fenced_reservations__",
                        )
                    )
                arithmetic = self._policy.admit(
                    model_path=claim.model_path,
                    request_gb=claim.request_gb,
                    active=active,
                    purpose=claim.purpose,
                    allow_disruptive_eviction=claim.allow_disruptive_eviction,
                )
                evict_owner_ids = tuple(
                    owner_id
                    for owner_id in arithmetic.evict_first
                    if owner_id != "__fenced_reservations__"
                )
                reason = arithmetic.reason
                # The legacy arithmetic helper retains an advisory mode for
                # diagnostics. A durable transaction is a production safety
                # boundary and therefore never converts an envelope breach
                # into a reservation merely because advisory mode was set.
                admitted = bool(arithmetic.admitted)
                if committed_owner_conflict:
                    admitted = False
                    reason = f"owner_id_already_committed:{claim.owner_id}"
                elif reservation_owner_conflict:
                    admitted = False
                    reason = f"owner_id_reservation_in_flight:{claim.owner_id}"
                if admitted:
                    blocked_target = next(
                        (
                            owner_id
                            for owner_id in evict_owner_ids
                            if not bool(state["owners"].get(owner_id, {}).get("preemptible", True))
                        ),
                        "",
                    )
                    if blocked_target:
                        admitted = False
                        reason = f"required_eviction_not_preemptible:{blocked_target}"
                    pending_target = next(
                        (
                            owner_id
                            for owner_id in evict_owner_ids
                            if str(
                                state["owners"].get(owner_id, {}).get("eviction_requested_by") or ""
                            )
                        ),
                        "",
                    )
                    if pending_target:
                        admitted = False
                        reason = f"required_eviction_already_fenced:{pending_target}"
                    serving_owner_ids = {
                        owner_id
                        for owner_id, owner in state["owners"].items()
                        if str(owner.get("purpose") or "serve") == "serve"
                    }
                    targeted_serving = serving_owner_ids.intersection(evict_owner_ids)
                    remaining_warm = serving_owner_ids.difference(evict_owner_ids)
                    if (
                        targeted_serving
                        and not remaining_warm
                        and not claim.allow_last_warm_eviction
                    ):
                        admitted = False
                        reason = "disruption_budget:last_warm_lane"

                generation = int(state.get("generation") or 0) + 1
                state["generation"] = generation
                transaction_id = f"lane-txn-{uuid.uuid4()}"
                reservation_state = (
                    LaneTransactionState.EVICTING
                    if admitted and evict_owner_ids
                    else LaneTransactionState.READY
                    if admitted
                    else LaneTransactionState.REFUSED
                )
                record = {
                    "request_id": claim.request_id,
                    "transaction_id": transaction_id,
                    "fencing_token": generation,
                    "state": reservation_state.value,
                    "reason": reason,
                    "owner_id": claim.owner_id,
                    "model_path": claim.model_path,
                    "purpose": claim.purpose,
                    "lane": lane,
                    "qos": qos.value,
                    "priority": int(claim.priority),
                    "preemptible": bool(claim.preemptible),
                    "foreground": bool(claim.foreground),
                    "allow_disruptive_eviction": bool(claim.allow_disruptive_eviction),
                    "allow_last_warm_eviction": bool(claim.allow_last_warm_eviction),
                    "reservation_ttl_s": float(claim.reservation_ttl_s or 0.0),
                    "requested_owner_lease_ttl_s": float(
                        claim.owner_lease_ttl_s or 0.0
                    ),
                    "request_gb": float(claim.request_gb),
                    "committed_gb": committed,
                    "reserved_gb": reserved,
                    "budget_gb": arithmetic.budget_gb,
                    "observation_source": arithmetic.observation_source,
                    "observation_scenario_id": arithmetic.observation_scenario_id,
                    "resource_observation_available": (
                        arithmetic.resource_observation_available
                    ),
                    "evict_owner_ids": list(evict_owner_ids if admitted else ()),
                    "evicted_owner_ids": [],
                    "evicted_owners": {},
                    "eviction_receipt_ids": {},
                    "compensation": {},
                    "controller_process": ProcessIdentity.current(
                        observer=self.resource_observer
                    ).to_dict(),
                    "created_at": now,
                    "expires_at": now + self._reservation_ttl(claim),
                    "owner_lease_ttl_s": self._owner_lease_ttl(claim),
                    "metadata": _json_metadata(claim.metadata),
                    "terminal_at": now if not admitted else 0.0,
                    "terminal_receipt_id": "",
                }
                state["reservations"][claim.request_id] = record
                if admitted:
                    for owner_id in evict_owner_ids:
                        owner = state["owners"].get(owner_id)
                        if isinstance(owner, dict):
                            owner["eviction_requested_by"] = transaction_id
                            owner["eviction_fencing_token"] = generation
                self._append_event(
                    state,
                    "reservation_created" if admitted else "reservation_refused",
                    at=now,
                    request_id=claim.request_id,
                    transaction_id=transaction_id,
                    fencing_token=generation,
                    reason=reason,
                    evict_owner_ids=list(evict_owner_ids if admitted else ()),
                )
                self._save_locked(state)
                decision = self._record_to_decision(record)
                terminal = not admitted
        self._persist_missing_terminal_receipts()
        if terminal:
            receipt_id = self._persist_terminal_receipt(claim.request_id)
            return LaneTransactionDecision(**{**decision.__dict__, "receipt_id": receipt_id})
        return decision

    async def _call_observer(self, callback: ObserveCallback) -> list[LaneOwnerObservation]:
        result = await _invoke_owned_callback(callback)
        return list(result)

    @staticmethod
    async def _call_bool(callback: Callable[..., Any], *args: Any) -> bool:
        return bool(await _invoke_owned_callback(callback, *args))

    async def prepare(
        self,
        decision: LaneTransactionDecision,
        *,
        evict: EvictCallback,
        observe: ObserveCallback,
        reclaim: ReclaimCallback | None = None,
        compensate: CompensateCallback | None = None,
        timeout_s: float | None = None,
    ) -> LaneTransactionDecision:
        if not decision.admitted:
            return decision
        if decision.ready_to_spawn:
            return decision
        timeout = max(0.1, float(timeout_s or _EVICTION_TIMEOUT_FLAG.value()))
        deadline = time.monotonic() + timeout
        evicted: list[LaneOwnerObservation] = []

        for owner_id in decision.evict_owner_ids:
            with self._thread_lock, interprocess_file_lock(self.lock_path):
                state = self._load_locked()
                record = state["reservations"].get(decision.request_id)
                if not isinstance(record, dict):
                    raise ModelLaneControlError("lane_reservation_missing_during_eviction")
                self._assert_fence(record, decision)
                owner_record = state["owners"].get(owner_id)
                if not isinstance(owner_record, dict):
                    continue
                owner = self._record_to_observation(owner_record)
                reservation_copy = dict(record)

            reason = f"evicted_for:{decision.transaction_id}:{decision.owner_id}"
            outcome = "eviction_failed"
            detail = "eviction_callback_refused"
            completed_at = self._clock()
            try:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining <= 0.0:
                    raise TimeoutError("eviction deadline elapsed before callback")
                accepted = await asyncio.wait_for(
                    self._call_bool(evict, owner, reason),
                    timeout=remaining,
                )
                if accepted:
                    while time.monotonic() < deadline:
                        observations = await self._call_observer(observe)
                        matching = next(
                            (item for item in observations if item.owner_id == owner_id),
                            None,
                        )
                        process_alive = self._owner_process_tree_alive(owner_record)
                        in_process_owner = bool(
                            dict(owner.metadata).get("in_process_model_owner", False)
                        )
                        if matching is None and (in_process_owner or not process_alive):
                            outcome = "evicted"
                            detail = (
                                "in_process_model_unloaded_and_owner_absent"
                                if in_process_owner
                                else "process_dead_and_owner_absent"
                            )
                            break
                        await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                    else:
                        detail = "eviction_timeout"
            except asyncio.CancelledError:
                await asyncio.shield(
                    self.cancel(
                        decision,
                        reason=f"eviction_cancelled:{owner_id}",
                        compensate=compensate,
                        evicted=evicted,
                    )
                )
                raise
            except TimeoutError:
                detail = "eviction_callback_timeout"
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                detail = f"eviction_error:{type(exc).__name__}:{exc}"
            completed_at = self._clock()
            receipt_id = await asyncio.to_thread(
                self._emit_eviction_receipt,
                reservation_copy,
                owner=owner,
                outcome=outcome,
                reason=detail,
                completed_at=completed_at,
            )
            if outcome != "evicted":
                compensation_candidates = list(evicted)
                if detail == "eviction_callback_timeout":
                    compensation_candidates.append(owner)
                return await self.cancel(
                    decision,
                    reason=f"required_eviction_failed:{owner_id}:{detail}",
                    compensate=compensate,
                    evicted=compensation_candidates,
                    eviction_receipt=(owner_id, receipt_id),
                )
            evicted.append(owner)
            with self._thread_lock, interprocess_file_lock(self.lock_path):
                state = self._load_locked()
                record = state["reservations"].get(decision.request_id)
                if not isinstance(record, dict):
                    raise ModelLaneControlError("lane_reservation_missing_after_eviction")
                self._assert_fence(record, decision)
                state["owners"].pop(owner_id, None)
                evicted_ids = record.setdefault("evicted_owner_ids", [])
                if owner_id not in evicted_ids:
                    evicted_ids.append(owner_id)
                record.setdefault("evicted_owners", {})[owner_id] = self._observation_payload(owner)
                record.setdefault("eviction_receipt_ids", {})[owner_id] = receipt_id
                self._append_event(
                    state,
                    "owner_evicted",
                    at=completed_at,
                    transaction_id=decision.transaction_id,
                    owner_id=owner_id,
                    receipt_id=receipt_id,
                )
                self._save_locked(state)

        claim = await asyncio.to_thread(self._claim_for_reclamation_sync, decision)
        if reclaim is not None:
            try:
                reclaimed = await self._call_bool(reclaim, claim)
            except asyncio.CancelledError:
                await asyncio.shield(
                    self.cancel(
                        decision,
                        reason="reclamation_cancelled",
                        compensate=compensate,
                        evicted=evicted,
                    )
                )
                raise
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                reclaimed = False
                logger.warning("Model-lane reclamation probe failed: %s", exc)
            if not reclaimed:
                return await self.cancel(
                    decision,
                    reason="resource_reclamation_unobserved",
                    compensate=compensate,
                    evicted=evicted,
                )

        observations = await self._call_observer(observe)
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            now = self._clock()
            self._prune_locked(state, now)
            self._sync_observations_locked(state, observations, now)
            record = state["reservations"].get(decision.request_id)
            if not isinstance(record, dict):
                raise ModelLaneControlError("lane_reservation_missing_before_ready")
            self._assert_fence(record, decision)
            committed, reserved = self._capacity_totals(
                state,
                exclude_request_id=decision.request_id,
            )
            observer = self.resource_observer
            budget_gb = lane_budget_gb(observer=observer)
            provenance = observer.provenance
            if committed + reserved + decision.request_gb > budget_gb:
                over_budget = True
            else:
                over_budget = False
                record["state"] = LaneTransactionState.READY.value
                record["reason"] = "required_evictions_reclaimed" if evicted else "capacity_reserved"
                record["committed_gb"] = committed
                record["reserved_gb"] = reserved
                record["budget_gb"] = budget_gb
                record["observation_source"] = provenance.source.value
                record["observation_scenario_id"] = provenance.scenario_id
                record["resource_observation_available"] = bool(
                    observer.memory().available
                )
                record["ready_at"] = now
                self._append_event(
                    state,
                    "reservation_ready",
                    at=now,
                    transaction_id=decision.transaction_id,
                    fencing_token=decision.fencing_token,
                )
                self._save_locked(state)
                return self._record_to_decision(record)
        if over_budget:
            return await self.cancel(
                decision,
                reason="capacity_changed_after_eviction",
                compensate=compensate,
                evicted=evicted,
            )
        raise ModelLaneControlError("unreachable_lane_prepare_state")

    @staticmethod
    def _assert_fence(record: Mapping[str, Any], decision: LaneTransactionDecision) -> None:
        if str(record.get("transaction_id") or "") != decision.transaction_id:
            raise ModelLaneControlError("lane_transaction_identity_mismatch")
        if int(record.get("fencing_token") or 0) != decision.fencing_token:
            raise ModelLaneControlError("stale_lane_fencing_token")
        if str(record.get("state") or "") in _TERMINAL_RESERVATION_STATES:
            raise ModelLaneControlError(f"lane_transaction_already_terminal:{record.get('state')}")

    @classmethod
    def _assert_committed_replay(
        cls,
        record: Mapping[str, Any],
        decision: LaneTransactionDecision,
        process: ProcessIdentity,
    ) -> None:
        if str(record.get("transaction_id") or "") != decision.transaction_id:
            raise ModelLaneControlError("lane_transaction_identity_mismatch")
        if int(record.get("fencing_token") or 0) != decision.fencing_token:
            raise ModelLaneControlError("stale_lane_fencing_token")
        if str(record.get("owner_id") or "") != decision.owner_id:
            raise ModelLaneControlError("lane_committed_replay_owner_mismatch")
        if str(record.get("model_path") or "") != decision.model_path:
            raise ModelLaneControlError("lane_committed_replay_model_mismatch")
        if cls._identity_from_record(record, key="committed_process") != process:
            raise ModelLaneControlError("lane_committed_replay_process_mismatch")

    @staticmethod
    def _claim_from_record(record: Mapping[str, Any]) -> LaneClaim:
        return LaneClaim(
            owner_id=str(record.get("owner_id") or ""),
            model_path=str(record.get("model_path") or ""),
            request_gb=float(record.get("request_gb") or 0.0),
            purpose=str(record.get("purpose") or "serve"),
            priority=int(record.get("priority") or 0),
            preemptible=bool(record.get("preemptible", True)),
            foreground=bool(record.get("foreground", False)),
            allow_disruptive_eviction=bool(record.get("allow_disruptive_eviction", False)),
            allow_last_warm_eviction=bool(record.get("allow_last_warm_eviction", False)),
            reservation_ttl_s=float(record.get("reservation_ttl_s") or 0.0),
            owner_lease_ttl_s=float(record.get("requested_owner_lease_ttl_s") or 0.0),
            request_id=str(record.get("request_id") or ""),
            metadata=dict(record.get("metadata") or {}),
        )

    def _claim_for_reclamation_sync(self, decision: LaneTransactionDecision) -> LaneClaim:
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            record = state["reservations"].get(decision.request_id)
            if not isinstance(record, dict):
                raise ModelLaneControlError("lane_reservation_missing_before_reclamation")
            self._assert_fence(record, decision)
            return self._claim_from_record(record)

    async def commit(
        self,
        decision: LaneTransactionDecision,
        *,
        process: ProcessIdentity,
        observed_gb: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> LaneTransactionDecision:
        return await asyncio.to_thread(
            self.commit_sync,
            decision,
            process=process,
            observed_gb=observed_gb,
            metadata=metadata,
        )

    def commit_sync(
        self,
        decision: LaneTransactionDecision,
        *,
        process: ProcessIdentity,
        observed_gb: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> LaneTransactionDecision:
        if process.pid <= 0 or process.started_at <= 0.0 or not self._process_alive(process):
            raise ModelLaneControlError("candidate_process_identity_not_live")
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            if self._prune_locked(state, now):
                self._save_locked(state)
            record = state["reservations"].get(decision.request_id)
            if not isinstance(record, dict):
                raise ModelLaneControlError("lane_reservation_missing_at_commit")
            if str(record.get("state") or "") == LaneTransactionState.COMMITTED.value:
                self._assert_committed_replay(record, decision, process)
                return self._record_to_decision(record, replayed=True)
            self._assert_fence(record, decision)
            if str(record.get("state") or "") != LaneTransactionState.READY.value:
                raise ModelLaneControlError("lane_reservation_not_ready_to_commit")
            delegation = record.get("delegation")
            if isinstance(delegation, Mapping):
                delegated_process = self._identity_from_record(
                    delegation,
                    key="consumed_process",
                )
                if delegated_process.pid > 0 and delegated_process != process:
                    raise ModelLaneControlError("candidate_process_identity_not_delegated_child")
            existing = state["owners"].get(decision.owner_id)
            if isinstance(existing, dict):
                raise ModelLaneControlError("lane_owner_already_committed")
            lane, qos = classify_lane(
                decision.model_path,
                purpose=str(record.get("purpose") or "serve"),
            )
            state["owners"][decision.owner_id] = {
                "owner_id": decision.owner_id,
                "model_path": decision.model_path,
                "purpose": str(record.get("purpose") or "serve"),
                "lane": lane,
                "qos": qos.value,
                "declared_gb": decision.request_gb,
                "observed_gb": max(0.0, float(observed_gb)),
                "priority": int(record.get("priority") or 0),
                "preemptible": bool(record.get("preemptible", True)),
                "last_user_facing_age_s": None,
                "process": process.to_dict(),
                "registered_at": now,
                "heartbeat_at": now,
                "lease_ttl_s": float(record.get("owner_lease_ttl_s") or 600.0),
                "lease_expires_at": now + float(record.get("owner_lease_ttl_s") or 600.0),
                "fencing_token": decision.fencing_token,
                "eviction_requested_by": "",
                "eviction_fencing_token": 0,
                "metadata": {
                    **dict(record.get("metadata") or {}),
                    **_json_metadata(metadata or {}),
                },
            }
            record["state"] = LaneTransactionState.COMMITTED.value
            record["reason"] = "spawn_committed"
            record["terminal_at"] = now
            record["committed_process"] = process.to_dict()
            self._append_event(
                state,
                "reservation_committed",
                at=now,
                transaction_id=decision.transaction_id,
                owner_id=decision.owner_id,
                fencing_token=decision.fencing_token,
                pid=process.pid,
            )
            self._save_locked(state)
            committed = self._record_to_decision(record)
        receipt_id = self._persist_terminal_receipt(decision.request_id)
        return LaneTransactionDecision(**{**committed.__dict__, "receipt_id": receipt_id})

    async def cancel(
        self,
        decision: LaneTransactionDecision,
        *,
        reason: str,
        compensate: CompensateCallback | None = None,
        evicted: Iterable[LaneOwnerObservation] = (),
        eviction_receipt: tuple[str, str] | None = None,
    ) -> LaneTransactionDecision:
        terminal, stored_evicted = await asyncio.to_thread(
            self._mark_cancelled_sync,
            decision,
            reason=reason,
            eviction_receipt=eviction_receipt,
        )
        recovered_receipt = await asyncio.to_thread(
            self._adopt_terminal_receipt_if_present,
            decision.request_id,
        )
        if recovered_receipt:
            return LaneTransactionDecision(
                **{**terminal.__dict__, "receipt_id": recovered_receipt, "replayed": True}
            )
        evicted_list = list(evicted) or stored_evicted
        compensation: dict[str, Any] = {}
        if compensate is not None:
            for owner in reversed(evicted_list):
                try:
                    compensation[owner.owner_id] = await self._call_bool(
                        compensate,
                        owner,
                        f"compensate_failed_candidate:{decision.transaction_id}",
                    )
                except asyncio.CancelledError:
                    compensation[owner.owner_id] = False
                    raise
                except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                    compensation[owner.owner_id] = False
        if compensation:
            await asyncio.to_thread(
                self._record_compensation_sync,
                decision.request_id,
                compensation,
            )
        receipt_id = await asyncio.to_thread(
            self._persist_terminal_receipt,
            decision.request_id,
        )
        return LaneTransactionDecision(**{**terminal.__dict__, "receipt_id": receipt_id})

    def _mark_cancelled_sync(
        self,
        decision: LaneTransactionDecision,
        *,
        reason: str,
        eviction_receipt: tuple[str, str] | None = None,
    ) -> tuple[LaneTransactionDecision, list[LaneOwnerObservation]]:
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            self._prune_locked(state, now)
            record = state["reservations"].get(decision.request_id)
            if not isinstance(record, dict):
                raise ModelLaneControlError("lane_reservation_missing_at_cancel")
            status = str(record.get("state") or "")
            if status not in _TERMINAL_RESERVATION_STATES:
                self._assert_fence(record, decision)
                record["state"] = LaneTransactionState.CANCELLED.value
                record["reason"] = str(reason or "candidate_cancelled")
                record["terminal_at"] = now
                if eviction_receipt is not None:
                    owner_id, receipt_id = eviction_receipt
                    record.setdefault("eviction_receipt_ids", {})[owner_id] = receipt_id
                for owner in state["owners"].values():
                    if str(owner.get("eviction_requested_by") or "") == decision.transaction_id:
                        owner["eviction_requested_by"] = ""
                        owner["eviction_fencing_token"] = 0
                self._append_event(
                    state,
                    "reservation_cancelled",
                    at=now,
                    transaction_id=decision.transaction_id,
                    reason=record["reason"],
                )
                self._save_locked(state)
            evicted = [
                self._record_to_observation(payload)
                for payload in dict(record.get("evicted_owners") or {}).values()
                if isinstance(payload, Mapping)
            ]
            return self._record_to_decision(record, replayed=status in _TERMINAL_RESERVATION_STATES), evicted

    def _record_compensation_sync(
        self,
        request_id: str,
        compensation: Mapping[str, Any],
    ) -> None:
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            record = state["reservations"].get(str(request_id))
            if not isinstance(record, dict):
                raise ModelLaneControlError("lane_reservation_missing_for_compensation")
            receipt_id = self._terminal_receipt_id(str(request_id))
            if str(record.get("terminal_receipt_id") or "") or self._durable_receipt_exists(
                receipt_id
            ):
                raise ModelLaneControlError("lane_compensation_after_terminal_receipt")
            record["compensation"] = _json_metadata(compensation)
            self._save_locked(state)

    def cancel_sync(
        self,
        decision: LaneTransactionDecision,
        *,
        reason: str,
        compensation: Mapping[str, Any] | None = None,
        eviction_receipt: tuple[str, str] | None = None,
    ) -> LaneTransactionDecision:
        terminal, _evicted = self._mark_cancelled_sync(
            decision,
            reason=reason,
            eviction_receipt=eviction_receipt,
        )
        recovered_receipt = self._adopt_terminal_receipt_if_present(decision.request_id)
        if recovered_receipt:
            return LaneTransactionDecision(
                **{**terminal.__dict__, "receipt_id": recovered_receipt, "replayed": True}
            )
        if compensation:
            self._record_compensation_sync(decision.request_id, compensation)
        receipt_id = self._persist_terminal_receipt(decision.request_id)
        return LaneTransactionDecision(**{**terminal.__dict__, "receipt_id": receipt_id})

    async def release_owner(
        self,
        owner_id: str,
        *,
        fencing_token: int | None = None,
        reason: str = "owner_released",
    ) -> bool:
        return await asyncio.to_thread(
            self.release_owner_sync,
            owner_id,
            fencing_token=fencing_token,
            reason=reason,
        )

    @staticmethod
    def _owner_holder_is_alive(owner: Mapping[str, Any], *, owner_id: str) -> bool:
        """Whether the process that registered this owner still exists.

        Owner ids carry their holder pid (``mlx:<pid>:<model path>``). A dead
        holder cannot release its own claim, so without this a crashed worker
        keeps the lane fenced for the life of the runtime. Unknown or
        unparseable holders are treated as ALIVE — reaping on a guess would be
        the mirror of the bug it fixes.
        """
        pid = owner.get("holder_pid") or owner.get("pid")
        if pid is None:
            parts = str(owner_id).split(":")
            for part in parts[1:2]:
                if part.isdigit():
                    pid = part
                    break
        if isinstance(pid, bool) or not isinstance(pid, (int, str)):
            return True
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return True
        if pid_int <= 0:
            return True
        try:
            os.kill(pid_int, 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True
        return True

    def release_owner_sync(
        self,
        owner_id: str,
        *,
        fencing_token: int | None = None,
        reason: str = "owner_released",
    ) -> bool:
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            owner = state["owners"].get(str(owner_id))
            if not isinstance(owner, dict):
                # A MISSING owner is a settled release, not a failure. Returning
                # False for it made the caller unable to tell "already gone"
                # from "someone else owns it now", and mlx_client fences the
                # lane on False — so a clean release left admission blocked:
                #   durable_owner_release_not_confirmed:mlx:<pid>:<model>
                #   -> lane left FENCED ... admission stays blocked until it is
                # Live 2026-07-25, repeating, with no path back to service.
                self._append_event(
                    state,
                    "owner_release_noop",
                    at=now,
                    owner_id=str(owner_id),
                    reason=f"already_released:{reason}",
                )
                self._save_locked(state)
                return True
            current_token = int(owner.get("fencing_token") or 0)
            if fencing_token is not None and current_token != int(fencing_token):
                # A token mismatch is a REAL conflict — a newer owner holds the
                # lane — unless the registered holder is provably dead, in
                # which case it is a corpse holding the door shut. Reap it the
                # way admission leases are reaped for dead holders.
                if not self._owner_holder_is_alive(owner, owner_id=str(owner_id)):
                    state["owners"].pop(str(owner_id), None)
                    self._append_event(
                        state,
                        "owner_reaped",
                        at=now,
                        owner_id=str(owner_id),
                        fencing_token=current_token,
                        reason=f"holder_died:{reason}",
                    )
                    self._save_locked(state)
                    return True
                return False
            state["owners"].pop(str(owner_id), None)
            self._append_event(
                state,
                "owner_released",
                at=now,
                owner_id=str(owner_id),
                fencing_token=current_token,
                reason=str(reason),
            )
            self._save_locked(state)
            return True

    async def heartbeat_owner(
        self,
        owner_id: str,
        *,
        fencing_token: int,
        observed_gb: float | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self.heartbeat_owner_sync,
            owner_id,
            fencing_token=fencing_token,
            observed_gb=observed_gb,
        )

    def heartbeat_owner_sync(
        self,
        owner_id: str,
        *,
        fencing_token: int,
        observed_gb: float | None = None,
    ) -> bool:
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            owner = state["owners"].get(str(owner_id))
            if not isinstance(owner, dict):
                return False
            if int(owner.get("fencing_token") or 0) != int(fencing_token):
                return False
            owner["heartbeat_at"] = now
            owner["lease_expires_at"] = now + float(
                owner.get("lease_ttl_s") or _OWNER_LEASE_TTL_FLAG.value()
            )
            metadata = dict(owner.get("metadata") or {})
            if bool(metadata.pop("heartbeat_lease_stale", False)):
                owner["preemptible"] = bool(
                    metadata.pop("preemptible_before_heartbeat_stale", False)
                )
                self._append_event(
                    state,
                    "owner_heartbeat_recovered",
                    at=now,
                    owner_id=str(owner_id),
                    fencing_token=int(fencing_token),
                )
            owner["metadata"] = metadata
            if observed_gb is not None:
                owner["observed_gb"] = max(0.0, float(observed_gb))
            self._save_locked(state)
            return True

    async def update_owner_preemptibility(
        self,
        owner_id: str,
        *,
        fencing_token: int,
        preemptible: bool,
    ) -> bool:
        return await asyncio.to_thread(
            self.update_owner_preemptibility_sync,
            owner_id,
            fencing_token=fencing_token,
            preemptible=preemptible,
        )

    def update_owner_preemptibility_sync(
        self,
        owner_id: str,
        *,
        fencing_token: int,
        preemptible: bool,
    ) -> bool:
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            owner = state["owners"].get(str(owner_id))
            if not isinstance(owner, dict):
                return False
            if int(owner.get("fencing_token") or 0) != int(fencing_token):
                return False
            if str(owner.get("eviction_requested_by") or ""):
                return False
            if bool(dict(owner.get("metadata") or {}).get("heartbeat_lease_stale")) and preemptible:
                return False
            owner["preemptible"] = bool(preemptible)
            owner["heartbeat_at"] = now
            owner["lease_expires_at"] = now + float(
                owner.get("lease_ttl_s") or _OWNER_LEASE_TTL_FLAG.value()
            )
            self._append_event(
                state,
                "owner_preemptibility_updated",
                at=now,
                owner_id=str(owner_id),
                fencing_token=int(fencing_token),
                preemptible=bool(preemptible),
            )
            self._save_locked(state)
            return True

    def owner_observations(self) -> list[LaneOwnerObservation]:
        self._refresh_external_owners()
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            if self._prune_locked(state, now):
                self._save_locked(state)
        self._persist_missing_terminal_receipts()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            return [
                self._record_to_observation(record)
                for record in state["owners"].values()
            ]

    async def issue_inherited_claim(
        self,
        decision: LaneTransactionDecision,
        *,
        ttl_s: float = 30.0,
    ) -> str:
        return await asyncio.to_thread(
            self.issue_inherited_claim_sync,
            decision,
            ttl_s=ttl_s,
        )

    def issue_inherited_claim_sync(
        self,
        decision: LaneTransactionDecision,
        *,
        ttl_s: float = 30.0,
    ) -> str:
        token = secrets.token_urlsafe(32)
        token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            if self._prune_locked(state, now):
                self._save_locked(state)
            record = state["reservations"].get(decision.request_id)
            if not isinstance(record, dict):
                raise ModelLaneControlError("lane_reservation_missing_for_delegation")
            self._assert_fence(record, decision)
            if str(record.get("state") or "") != LaneTransactionState.READY.value:
                raise ModelLaneControlError("lane_reservation_not_ready_for_delegation")
            record["delegation"] = {
                "token_sha256": token_sha256,
                "issued_at": now,
                "expires_at": now + max(5.0, float(ttl_s)),
                "parent_process": ProcessIdentity.current(
                    observer=self.resource_observer
                ).to_dict(),
                "consumed_at": 0.0,
                "consumed_process": ProcessIdentity(0, 0.0).to_dict(),
            }
            self._append_event(
                state,
                "reservation_delegated",
                at=now,
                transaction_id=decision.transaction_id,
                owner_id=decision.owner_id,
                fencing_token=decision.fencing_token,
            )
            self._save_locked(state)
        return token

    def validate_inherited_claim(
        self,
        *,
        owner_id: str,
        request_id: str,
        model_path: str,
        purpose: str,
        delegation_token: str,
        child_pid: int,
        parent_pid: int,
    ) -> bool:
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            if self._prune_locked(state, now):
                self._save_locked(state)
            record = state["reservations"].get(str(request_id))
            if not isinstance(record, Mapping):
                return False
            if str(record.get("owner_id") or "") != str(owner_id):
                return False
            if str(record.get("model_path") or "") != str(model_path):
                return False
            if str(record.get("purpose") or "serve") != str(purpose):
                return False
            state_name = str(record.get("state") or "")
            if state_name not in _ACTIVE_RESERVATION_STATES | {
                LaneTransactionState.COMMITTED.value
            }:
                return False
            delegation = record.get("delegation")
            if not isinstance(delegation, dict):
                return False
            expected_digest = str(delegation.get("token_sha256") or "")
            supplied_digest = hashlib.sha256(
                str(delegation_token or "").encode("utf-8")
            ).hexdigest()
            if not expected_digest or not hmac.compare_digest(expected_digest, supplied_digest):
                return False
            child_identity = process_identity_for_pid(
                child_pid,
                observer=self.resource_observer,
            )
            consumed_identity = self._identity_from_record(
                delegation,
                key="consumed_process",
            )
            if consumed_identity.pid > 0:
                return consumed_identity == child_identity and self._process_alive(child_identity)
            if float(delegation.get("expires_at") or 0.0) <= now:
                return False
            parent_identity = self._identity_from_record(
                delegation,
                key="parent_process",
            )
            if parent_identity.pid != int(parent_pid) or not self._process_alive(parent_identity):
                return False
            if not self._process_alive(child_identity):
                return False
            try:
                child_process = self.resource_observer.process(int(child_pid))
                if child_process is None:
                    return False
                observed_parent_pid = int(child_process.ppid)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
            if observed_parent_pid != int(parent_pid):
                return False
            if state_name == LaneTransactionState.COMMITTED.value:
                committed_identity = self._identity_from_record(record, key="committed_process")
                owner = state["owners"].get(str(owner_id))
                if committed_identity != child_identity or not isinstance(owner, Mapping):
                    return False
            delegation["consumed_at"] = now
            delegation["consumed_process"] = child_identity.to_dict()
            self._append_event(
                state,
                "reservation_delegation_consumed",
                at=now,
                transaction_id=str(record.get("transaction_id") or ""),
                owner_id=str(owner_id),
                child_pid=int(child_pid),
            )
            self._save_locked(state)
            return True

    def validate_inherited_child_claim(
        self,
        *,
        owner_id: str,
        request_id: str,
        model_path: str,
        purpose: str,
        delegation_token: str,
        child_pid: int,
        parent_pid: int,
        requested_gb: float,
        child_model_path: str,
        child_purpose: str,
    ) -> bool:
        """Authorize a nested model child inside one delegated pipeline lane.

        The outer worker must first prove the one-time inherited delegation and
        must itself be the committed process owner. Only reservations that
        explicitly opted into nested children may reuse their isolated process
        group, and no child may exceed the parent's admitted capacity.
        """
        if not self.validate_inherited_claim(
            owner_id=owner_id,
            request_id=request_id,
            model_path=model_path,
            purpose=purpose,
            delegation_token=delegation_token,
            child_pid=child_pid,
            parent_pid=parent_pid,
        ):
            return False
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            if self._prune_locked(state, now):
                self._save_locked(state)
            record = state["reservations"].get(str(request_id))
            owner = state["owners"].get(str(owner_id))
            if not isinstance(record, Mapping) or not isinstance(owner, Mapping):
                return False
            if str(record.get("state") or "") != LaneTransactionState.COMMITTED.value:
                return False
            metadata = dict(record.get("metadata") or {})
            if metadata.get("allow_inherited_model_children") is not True:
                return False
            allowed_purposes = {
                str(value)
                for value in (metadata.get("allowed_inherited_model_purposes") or ())
                if str(value)
            }
            if str(child_purpose) not in allowed_purposes:
                return False
            try:
                child_path = Path(str(child_model_path)).expanduser().resolve()
                parent_path = Path(str(record.get("model_path") or "")).expanduser().resolve()
                allowed_roots = tuple(
                    Path(str(value)).expanduser().resolve()
                    for value in (metadata.get("allowed_inherited_model_roots") or ())
                    if str(value)
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                return False
            if child_path != parent_path and not any(
                child_path == root or child_path.is_relative_to(root)
                for root in allowed_roots
            ):
                return False
            if float(requested_gb) <= 0.0:
                return False
            if float(requested_gb) > float(record.get("request_gb") or 0.0):
                return False
            owner_metadata = dict(owner.get("metadata") or {})
            if owner_metadata.get("managed_model_process") is not True:
                return False
            if owner_metadata.get("start_new_session") is not True:
                return False
            try:
                return int(owner_metadata.get("process_group_id") or 0) == os.getpgrp()
            except (OSError, TypeError, ValueError):
                return False

    def snapshot(self) -> dict[str, Any]:
        self._refresh_external_owners()
        now = self._clock()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            if self._prune_locked(state, now):
                self._save_locked(state)
        self._persist_missing_terminal_receipts()
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            state = self._load_locked()
            committed, reserved = self._capacity_totals(state)
            return {
                "schema": _SCHEMA_NAME,
                "generation": int(state.get("generation") or 0),
                "budget_gb": lane_budget_gb(observer=self.resource_observer),
                "observation_source": self.resource_observer.provenance.source.value,
                "observation_scenario_id": self.resource_observer.provenance.scenario_id,
                "committed_gb": committed,
                "reserved_gb": reserved,
                "owners": list(state["owners"].values()),
                "reservations": list(state["reservations"].values()),
                "events": list(state["events"]),
            }

    def is_alive(self) -> bool:
        try:
            self.snapshot()
            return True
        except (OSError, RuntimeError, ValueError):
            return False

    def is_ready(self) -> bool:
        return self.is_alive()

    def get_status(self) -> dict[str, Any]:
        return self.snapshot()


class InProcessModelLaneLease:
    """Heartbeat-backed owner for a model loaded inside the Aura process."""

    def __init__(
        self,
        *,
        controller: ModelLaneController,
        decision: LaneTransactionDecision,
        heartbeat_interval_s: float,
    ) -> None:
        self.controller = controller
        self.decision = decision
        self._heartbeat_interval_s = max(1.0, float(heartbeat_interval_s))
        from core.utils.task_tracker import get_task_tracker

        self._heartbeat_task: asyncio.Task[Any] | None = get_task_tracker().create_task(
            self._heartbeat_loop(), name=f"ModelLaneHeartbeat:{decision.owner_id}"
        )
        self._released = False

    async def _heartbeat_loop(self) -> None:
        while not self._released:
            await asyncio.sleep(self._heartbeat_interval_s)
            if self._released:
                return
            try:
                alive = await self.controller.heartbeat_owner(
                    self.decision.owner_id,
                    fencing_token=self.decision.fencing_token,
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.error(
                    "Asynchronous model lane heartbeat failed owner=%s: %s",
                    self.decision.owner_id,
                    exc,
                )
                alive = False
            if not alive:
                logger.error(
                    "In-process model lane lost its fencing lease owner=%s token=%s",
                    self.decision.owner_id,
                    self.decision.fencing_token,
                )
                from core.runtime.shutdown_coordinator import request_shutdown

                request_shutdown(f"in_process_model_lane_fence_lost:{self.decision.owner_id}")
                return

    async def release(self, *, reason: str = "in_process_model_released") -> bool:
        if self._released:
            return False
        self._released = True
        task, self._heartbeat_task = self._heartbeat_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        unregister_model_lane_owner_adapter(self.decision.owner_id)
        return await self.controller.release_owner(
            self.decision.owner_id,
            fencing_token=self.decision.fencing_token,
            reason=reason,
        )

    async def set_preemptible(self, preemptible: bool) -> bool:
        if self._released:
            return False
        return await self.controller.update_owner_preemptibility(
            self.decision.owner_id,
            fencing_token=self.decision.fencing_token,
            preemptible=preemptible,
        )


class SynchronousInProcessModelLaneLease:
    """Heartbeat-backed owner for models loaded by synchronous worker APIs."""

    def __init__(
        self,
        *,
        controller: ModelLaneController,
        decision: LaneTransactionDecision,
        heartbeat_interval_s: float,
    ) -> None:
        self.controller = controller
        self.decision = decision
        self._heartbeat_interval_s = max(1.0, float(heartbeat_interval_s))
        self._released = False
        self._stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"ModelLaneHeartbeatSync:{decision.owner_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval_s):
            if self._released:
                return
            try:
                alive = self.controller.heartbeat_owner_sync(
                    self.decision.owner_id,
                    fencing_token=self.decision.fencing_token,
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.error(
                    "Synchronous model lane heartbeat failed owner=%s: %s",
                    self.decision.owner_id,
                    exc,
                )
                alive = False
            if alive:
                continue
            logger.critical(
                "Synchronous in-process model lost fencing lease owner=%s token=%s",
                self.decision.owner_id,
                self.decision.fencing_token,
            )
            from core.runtime.shutdown_coordinator import request_shutdown

            request_shutdown(f"sync_in_process_model_lane_fence_lost:{self.decision.owner_id}")
            return

    def release(self, *, reason: str = "sync_in_process_model_released") -> bool:
        if self._released:
            return False
        self._released = True
        self._stop_event.set()
        if self._heartbeat_thread is not threading.current_thread():
            self._heartbeat_thread.join(timeout=max(1.0, self._heartbeat_interval_s + 0.5))
        unregister_model_lane_owner_adapter(self.decision.owner_id)
        return self.controller.release_owner_sync(
            self.decision.owner_id,
            fencing_token=self.decision.fencing_token,
            reason=reason,
        )

    def set_preemptible(self, preemptible: bool) -> bool:
        if self._released:
            return False
        return self.controller.update_owner_preemptibility_sync(
            self.decision.owner_id,
            fencing_token=self.decision.fencing_token,
            preemptible=preemptible,
        )


def acquire_synchronous_in_process_model_lane(
    *,
    owner_id: str,
    model_path: str,
    purpose: str,
    request_gb: float | None = None,
    priority: int = 50,
    preemptible: bool = True,
    owner_lease_ttl_s: float = 180.0,
    evict: EvictCallback | None = None,
    compensate: CompensateCallback | None = None,
    metadata: Mapping[str, Any] | None = None,
    controller: ModelLaneController | None = None,
) -> SynchronousInProcessModelLaneLease:
    if is_shutdown_requested():
        raise ModelLaneControlError("sync_in_process_model_admission_refused:runtime_shutdown")
    lane_controller = controller or get_model_lane_controller()
    stable_owner_id = f"in-process-sync:{os.getpid()}:{owner_id}"
    claim = LaneClaim(
        owner_id=stable_owner_id,
        model_path=model_path,
        request_gb=(
            float(request_gb)
            if request_gb is not None
            else estimate_model_job_footprint_gb(model_path, purpose=purpose)
        ),
        purpose=purpose,
        priority=int(priority),
        preemptible=bool(preemptible),
        owner_lease_ttl_s=max(15.0, float(owner_lease_ttl_s)),
        request_id=f"sync-in-process-model-{uuid.uuid4()}",
        metadata={
            **_json_metadata(metadata or {}),
            "lease_mode": "heartbeat",
            "in_process_model_owner": True,
            "synchronous_loader": True,
        },
    )
    decision = lane_controller.reserve_sync(claim)
    if not decision.admitted:
        raise ModelLaneControlError(
            f"sync_in_process_model_admission_refused:{decision.reason}:receipt={decision.receipt_id}"
        )
    if not decision.ready_to_spawn:
        cancelled = lane_controller.cancel_sync(
            decision,
            reason="sync_in_process_model_requires_async_eviction",
        )
        raise ModelLaneControlError(
            f"sync_in_process_model_admission_cancelled:{cancelled.reason}:receipt={cancelled.receipt_id}"
        )
    try:
        memory = lane_controller.resource_observer.memory()
        if not memory.available:
            raise RuntimeError(memory.error or "memory observation unavailable")
        available_gb = float(memory.available_bytes) / float(1024**3)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        lane_controller.cancel_sync(decision, reason="sync_model_headroom_unobservable")
        raise ModelLaneControlError("sync_in_process_model_headroom_unobservable") from exc
    if available_gb < claim.request_gb:
        cancelled = lane_controller.cancel_sync(
            decision,
            reason="sync_in_process_model_headroom_unavailable",
        )
        raise ModelLaneControlError(
            f"sync_in_process_model_admission_cancelled:{cancelled.reason}:receipt={cancelled.receipt_id}"
        )
    try:
        committed = lane_controller.commit_sync(
            decision,
            process=ProcessIdentity.current(observer=lane_controller.resource_observer),
            metadata={
                "lease_mode": "heartbeat",
                "in_process_model_owner": True,
                "synchronous_loader": True,
            },
        )
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        lane_controller.cancel_sync(decision, reason="sync_in_process_model_commit_failed")
        raise
    if evict is not None:
        register_model_lane_owner_adapter(
            stable_owner_id,
            evict=evict,
            compensate=compensate,
        )
    return SynchronousInProcessModelLaneLease(
        controller=lane_controller,
        decision=committed,
        heartbeat_interval_s=max(5.0, float(owner_lease_ttl_s) / 3.0),
    )


async def acquire_in_process_model_lane(
    *,
    owner_id: str,
    model_path: str,
    purpose: str,
    request_gb: float | None = None,
    priority: int = 80,
    preemptible: bool = True,
    owner_lease_ttl_s: float = 180.0,
    evict: EvictCallback | None = None,
    compensate: CompensateCallback | None = None,
    metadata: Mapping[str, Any] | None = None,
    controller: ModelLaneController | None = None,
) -> InProcessModelLaneLease:
    if is_shutdown_requested():
        raise ModelLaneControlError("in_process_model_admission_refused:runtime_shutdown")
    lane_controller = controller or get_model_lane_controller()
    stable_owner_id = f"in-process:{os.getpid()}:{owner_id}"
    claim = LaneClaim(
        owner_id=stable_owner_id,
        model_path=model_path,
        request_gb=(
            float(request_gb)
            if request_gb is not None
            else estimate_model_job_footprint_gb(model_path, purpose=purpose)
        ),
        purpose=purpose,
        priority=int(priority),
        preemptible=bool(preemptible),
        owner_lease_ttl_s=max(15.0, float(owner_lease_ttl_s)),
        request_id=f"in-process-model-{uuid.uuid4()}",
        metadata={
            **_json_metadata(metadata or {}),
            "lease_mode": "heartbeat",
            "in_process_model_owner": True,
        },
    )
    decision = await lane_controller.reserve(
        claim,
        observations=await asyncio.to_thread(lane_controller.owner_observations),
    )
    if not decision.admitted:
        raise ModelLaneControlError(
            f"in_process_model_admission_refused:{decision.reason}:receipt={decision.receipt_id}"
        )
    if not decision.ready_to_spawn:
        decision = await lane_controller.prepare(
            decision,
            evict=evict_registered_model_owner,
            observe=lambda: asyncio.to_thread(lane_controller.owner_observations),
            reclaim=lambda candidate: wait_for_model_job_headroom(
                candidate,
                observer=lane_controller.resource_observer,
            ),
            compensate=compensate_registered_model_owner,
        )
    if not decision.ready_to_spawn:
        raise ModelLaneControlError(
            f"in_process_model_admission_cancelled:{decision.reason}:receipt={decision.receipt_id}"
        )
    if not await wait_for_model_job_headroom(
        claim,
        observer=lane_controller.resource_observer,
    ):
        cancelled = await lane_controller.cancel(
            decision,
            reason="in_process_model_headroom_unavailable",
            compensate=compensate_registered_model_owner,
        )
        raise ModelLaneControlError(
            f"in_process_model_admission_cancelled:{cancelled.reason}:receipt={cancelled.receipt_id}"
        )
    try:
        committed = await lane_controller.commit(
            decision,
            process=ProcessIdentity.current(observer=lane_controller.resource_observer),
            metadata={"lease_mode": "heartbeat", "in_process_model_owner": True},
        )
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        await lane_controller.cancel(decision, reason="in_process_model_commit_failed")
        raise
    if evict is not None:
        register_model_lane_owner_adapter(
            stable_owner_id,
            evict=evict,
            compensate=compensate,
        )
    return InProcessModelLaneLease(
        controller=lane_controller,
        decision=committed,
        heartbeat_interval_s=max(5.0, float(owner_lease_ttl_s) / 3.0),
    )


@contextlib.asynccontextmanager
async def in_process_model_lane(
    **kwargs: Any,
) -> AsyncIterator[InProcessModelLaneLease]:
    lease = await acquire_in_process_model_lane(**kwargs)
    try:
        yield lease
    finally:
        await lease.release()


class StandaloneModelLaneLease:
    """Process-scoped lease for directly invoked model tools."""

    def __init__(
        self,
        *,
        controller: ModelLaneController | None,
        decision: LaneTransactionDecision | None,
        inherited: bool,
    ) -> None:
        self.controller = controller
        self.decision = decision
        self.inherited = inherited
        self._released = False

    @property
    def active(self) -> bool:
        """Whether this process still owns or validly inherits the load lease."""

        return not self._released

    def release(self, *, reason: str = "standalone_model_tool_finished") -> bool:
        if self._released:
            return False
        self._released = True
        if self.inherited or self.controller is None or self.decision is None:
            return False
        return self.controller.release_owner_sync(
            self.decision.owner_id,
            fencing_token=self.decision.fencing_token,
            reason=reason,
        )


def _run_standalone_async(awaitable: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        async def _await_value() -> Any:
            return await awaitable

        return asyncio.run(_await_value())
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()
    raise ModelLaneControlError(
        "standalone_model_lane_called_inside_event_loop; use in_process_model_lane"
    )


def acquire_standalone_model_lane(
    *,
    owner_id: str,
    model_path: str,
    purpose: str,
    request_gb: float | None = None,
    priority: int = 80,
    preemptible: bool = True,
    metadata: Mapping[str, Any] | None = None,
    controller: ModelLaneController | None = None,
) -> StandaloneModelLaneLease:
    if is_shutdown_requested():
        raise ModelLaneControlError("standalone_model_admission_refused:runtime_shutdown")
    lane_controller = controller or get_model_lane_controller()
    inherited_owner = str(os.environ.get("AURA_MODEL_LANE_INHERITED_OWNER_ID", "") or "")
    inherited_request = str(os.environ.get("AURA_MODEL_LANE_INHERITED_REQUEST_ID", "") or "")
    inherited_model = str(os.environ.get("AURA_MODEL_LANE_INHERITED_MODEL_PATH", "") or "")
    inherited_purpose = str(os.environ.get("AURA_MODEL_LANE_INHERITED_PURPOSE", "") or "")
    inherited_token = str(os.environ.get("AURA_MODEL_LANE_DELEGATION_TOKEN", "") or "")
    if (
        inherited_owner
        and inherited_request
        and inherited_model == str(model_path)
        and inherited_purpose == str(purpose)
        and inherited_token
        and lane_controller.validate_inherited_claim(
            owner_id=inherited_owner,
            request_id=inherited_request,
            model_path=model_path,
            purpose=purpose,
            delegation_token=inherited_token,
            child_pid=os.getpid(),
            parent_pid=os.getppid(),
        )
    ):
        return StandaloneModelLaneLease(
            controller=None,
            decision=None,
            inherited=True,
        )

    claim = LaneClaim(
        owner_id=f"standalone:{os.getpid()}:{owner_id}",
        model_path=model_path,
        request_gb=(
            float(request_gb)
            if request_gb is not None
            else estimate_model_job_footprint_gb(model_path, purpose=purpose)
        ),
        purpose=purpose,
        priority=int(priority),
        preemptible=bool(preemptible),
        reservation_ttl_s=300.0,
        owner_lease_ttl_s=300.0,
        request_id=f"standalone-model-{uuid.uuid4()}",
        metadata={
            **_json_metadata(metadata or {}),
            "standalone_model_tool": True,
            "lease_mode": "process",
        },
    )
    decision = lane_controller.reserve_sync(
        claim,
        observations=lane_controller.owner_observations(),
    )
    if not decision.admitted:
        raise ModelLaneControlError(
            f"standalone_model_admission_refused:{decision.reason}:receipt={decision.receipt_id}"
        )
    if not decision.ready_to_spawn:
        decision = _run_standalone_async(
            lane_controller.prepare(
                decision,
                evict=evict_registered_model_owner,
                observe=lambda: asyncio.to_thread(lane_controller.owner_observations),
                reclaim=lambda candidate: wait_for_model_job_headroom(
                    candidate,
                    observer=lane_controller.resource_observer,
                ),
                compensate=compensate_registered_model_owner,
            )
        )
    if not decision.ready_to_spawn:
        raise ModelLaneControlError(
            f"standalone_model_admission_cancelled:{decision.reason}:receipt={decision.receipt_id}"
        )
    if not _run_standalone_async(
        wait_for_model_job_headroom(
            claim,
            observer=lane_controller.resource_observer,
        )
    ):
        cancelled = lane_controller.cancel_sync(
            decision,
            reason="standalone_model_headroom_unavailable",
        )
        raise ModelLaneControlError(
            f"standalone_model_admission_cancelled:{cancelled.reason}:receipt={cancelled.receipt_id}"
        )
    committed = lane_controller.commit_sync(
        decision,
        process=ProcessIdentity.current(observer=lane_controller.resource_observer),
        metadata={"standalone_model_tool": True, "lease_mode": "process"},
    )
    return StandaloneModelLaneLease(
        controller=lane_controller,
        decision=committed,
        inherited=False,
    )


@contextlib.contextmanager
def standalone_model_lane(**kwargs: Any) -> Iterator[StandaloneModelLaneLease]:
    lease = acquire_standalone_model_lane(**kwargs)
    try:
        yield lease
    finally:
        lease.release()


_CONTROLLER: ModelLaneController | None = None
_CONTROLLER_LOCK = threading.Lock()


def get_model_lane_controller() -> ModelLaneController:
    global _CONTROLLER
    if _CONTROLLER is None:
        with _CONTROLLER_LOCK:
            if _CONTROLLER is None:
                _CONTROLLER = ModelLaneController()
    return _CONTROLLER


def reset_model_lane_controller_for_test() -> None:
    """Clear process-local lane ownership between hermetic tests."""

    global _CONTROLLER
    with _CONTROLLER_LOCK:
        _CONTROLLER = None
    with _LOCAL_OWNER_ADAPTERS_LOCK:
        _LOCAL_OWNER_ADAPTERS.clear()
        _LOCAL_OWNER_COMPENSATORS.clear()


def reset_model_lane_controller() -> None:
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        _CONTROLLER = None


__all__ = [
    "InProcessModelLaneLease",
    "SynchronousInProcessModelLaneLease",
    "StandaloneModelLaneLease",
    "LaneClaim",
    "LaneOwnerObservation",
    "LaneTransactionDecision",
    "LaneTransactionState",
    "ModelLaneControlError",
    "ModelLaneController",
    "ProcessIdentity",
    "acquire_in_process_model_lane",
    "acquire_synchronous_in_process_model_lane",
    "acquire_standalone_model_lane",
    "compensate_registered_model_owner",
    "estimate_model_job_footprint_gb",
    "evict_managed_process_owner",
    "evict_registered_model_owner",
    "get_model_lane_controller",
    "infer_model_process_claim",
    "in_process_model_lane",
    "managed_process_group_alive",
    "prepare_model_lane_claim",
    "process_identity_for_pid",
    "reset_model_lane_controller_for_test",
    "reset_model_lane_controller",
    "register_model_lane_owner_adapter",
    "run_owned_model_thread_call",
    "unregister_model_lane_owner_adapter",
    "standalone_model_lane",
    "wait_for_model_job_headroom",
]
