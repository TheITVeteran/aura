"""core/runtime/lockdep.py — runtime lock-order validation.

Clean-room adoption of the Linux kernel's `CONFIG_PROVE_LOCKING`
("lockdep") discipline.

The idea that makes lockdep worth more than every deadlock post-mortem
combined: **you do not have to observe the deadlock.** Lockdep watches the
*order* in which locks are taken and builds a dependency graph across the
whole process lifetime. The moment any execution takes lock B while
holding lock A, and any other execution has ever taken A while holding B,
it reports — even though the two orderings never actually raced. A latent
ABBA deadlock that needs a one-in-a-million interleaving is found on the
first run that exercises both paths in any order.

Aura needs this specifically. The runtime mixes `threading.Lock` (service
container, registries, caches), `asyncio.Lock` (lane admission, model
lifecycle), and cross-thread handoffs (`asyncio.to_thread`, the MLX
worker). The recorded incidents — a welfare deadlock, a 20-minute event
loop freeze on an on-loop fsync — are exactly the class lockdep catches
before it bites.

Four hazard classes are checked, all live, all cheap:

1. **Order inversion** (the classic): the acquisition-order graph would
   gain a cycle. Reported at the acquire that would close the cycle,
   naming the other call site that established the opposing edge.
2. **Sync lock held across an await**: a `threading.Lock` is held while
   the same task awaits an async primitive. On a single-threaded event
   loop this is a self-deadlock waiting for a second task to want the
   same lock — and, even when it does not deadlock, it blocks the loop.
3. **Self-deadlock**: a non-reentrant lock re-acquired by the same
   execution context.
4. **Loop-blocking hold**: a sync lock held past a deadline while the
   event loop thread owns it, which is the freeze signature.

Overhead is a handful of dict/set operations per acquire. It is on by
default and stays on in production — a validator that only runs in test
builds only finds the bugs tests already reach.

Usage
-----
Prefer the wrappers, which are drop-in for the stdlib primitives::

    from core.runtime.lockdep import checked_lock, checked_async_lock

    _CACHE_LOCK = checked_lock("probe_cache", rank=LockRank.LEAF)

    async def refresh():
        async with checked_async_lock("model_lane", rank=LockRank.LANE):
            ...

An already-constructed lock is adopted with :func:`instrument`. Code that
must run with no locks held (fsync, subprocess spawn, a blocking join)
asserts it with :func:`assert_no_locks_held`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from core.runtime.taint import TaintFlag, taint

logger = logging.getLogger("Aura.Lockdep")

#: A sync lock held longer than this while the event loop thread owns it is
#: reported as a loop-blocking hold. 50ms is well past any legitimate
#: in-memory critical section and well under a user-perceptible stall.
LOOP_BLOCKING_HOLD_S = 0.05

#: Max distinct splats retained. Deadlock reports are deduplicated by
#: signature, so this bounds pathological churn, not real findings.
MAX_SPLATS = 256

#: How long an async lock acquisition may take before it is reported as a
#: splat and the wait gives up. Generous — a legitimate critical section
#: is milliseconds — but finite, because an unbounded acquire is exactly
#: the wedge this module exists to find.
ACQUIRE_BUDGET_S = 30.0


class LockRank(IntEnum):
    """Optional declared nesting levels, mirroring lockdep's subclasses.

    When two locks both declare a rank, acquiring a lower-or-equal rank
    while holding a higher one is a violation *even if no cycle has been
    observed yet* — declared order is stronger evidence than observed
    order. Locks with rank UNRANKED fall back to pure cycle detection.
    """

    UNRANKED = 0
    #: Process-wide singletons and registries — taken first, held briefly.
    REGISTRY = 10
    #: Subsystem-level coordination (lanes, schedulers, supervisors).
    LANE = 20
    #: Per-resource state (a model handle, a connection, a session).
    RESOURCE = 30
    #: Innermost data structures. Nothing may be taken while holding one.
    LEAF = 40


@dataclass(frozen=True)
class Splat:
    """One lockdep report. Named after the kernel's term for the dump."""

    kind: str
    signature: str
    message: str
    held: tuple[str, ...]
    acquiring: str
    at: float
    stack: tuple[str, ...]
    context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "signature": self.signature,
            "message": self.message,
            "held": list(self.held),
            "acquiring": self.acquiring,
            "at": self.at,
            "stack": list(self.stack),
            "context": self.context,
        }


@dataclass
class _HeldLock:
    name: str
    rank: LockRank
    is_async: bool
    acquired_at: float
    thread_ident: int
    site: str


@dataclass
class _ContextState:
    """Per-execution-context held-lock stack."""

    label: str
    held: list[_HeldLock] = field(default_factory=list)


class LockdepValidator:
    """The dependency graph and the four checks over it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # class -> set of classes that have been acquired while holding it
        self._edges: dict[str, set[str]] = {}
        # (before, after) -> the site that first established the edge
        self._edge_site: dict[tuple[str, str], str] = {}
        self._ranks: dict[str, LockRank] = {}
        self._contexts: dict[tuple[str, int], _ContextState] = {}
        self._splats: dict[str, Splat] = {}
        self._splat_counts: dict[str, int] = {}
        self._acquires = 0
        self._loop_thread_ident: int | None = None

    # ── execution context identity ────────────────────────────────────
    @staticmethod
    def _context_key() -> tuple[tuple[str, int], str]:
        """Identify the execution context.

        Two coroutines interleave on one thread, so a task is its own
        context; plain threads key on their ident.
        """
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is not None:
            return ("task", id(task)), f"task:{task.get_name()}"
        ident = threading.get_ident()
        return ("thread", ident), f"thread:{threading.current_thread().name}"

    def note_loop_thread(self) -> None:
        """Record which thread runs the event loop, for hazard 4."""
        self._loop_thread_ident = threading.get_ident()

    @staticmethod
    def _site(depth: int = 4) -> str:
        stack = traceback.extract_stack(limit=depth + 2)
        for frame in reversed(stack[:-1]):
            if "lockdep.py" in frame.filename:
                continue
            return f"{frame.filename.rsplit('/', 2)[-1]}:{frame.lineno}"
        return "<unknown>"

    @staticmethod
    def _stack(limit: int = 12) -> tuple[str, ...]:
        frames = traceback.extract_stack(limit=limit + 4)
        return tuple(
            f"{f.filename.rsplit('/', 2)[-1]}:{f.lineno} in {f.name}"
            for f in frames
            if "lockdep.py" not in f.filename
        )[-limit:]

    # ── the checks ────────────────────────────────────────────────────
    def register(self, name: str, rank: LockRank) -> None:
        if rank is LockRank.UNRANKED:
            return
        with self._lock:
            prior = self._ranks.get(name)
            if prior is not None and prior != rank:
                raise ValueError(
                    f"lock {name!r} already declared with rank {prior.name}; "
                    f"refusing conflicting rank {rank.name} — a lock has one "
                    "position in the order"
                )
            self._ranks[name] = rank

    def _creates_cycle(self, new_edge_from: str, new_edge_to: str) -> list[str] | None:
        """Does adding from→to close a cycle? Returns the offending path."""
        if new_edge_from == new_edge_to:
            return [new_edge_from, new_edge_to]
        # A cycle exists iff `to` already reaches `from`.
        seen: set[str] = set()
        stack: list[tuple[str, list[str]]] = [(new_edge_to, [new_edge_to])]
        while stack:
            node, path = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for nxt in self._edges.get(node, ()):  # noqa: SIM118 — set default
                if nxt == new_edge_from:
                    return path + [nxt]
                stack.append((nxt, path + [nxt]))
        return None

    def on_acquire(
        self,
        name: str,
        *,
        rank: LockRank,
        is_async: bool,
        reentrant: bool,
    ) -> None:
        key, label = self._context_key()
        site = self._site()
        now = time.monotonic()

        with self._lock:
            self._acquires += 1
            state = self._contexts.get(key)
            if state is None:
                state = _ContextState(label=label)
                self._contexts[key] = state
            held = state.held

            # 3. self-deadlock on a non-reentrant lock
            if not reentrant and any(h.name == name for h in held):
                self._splat_locked(
                    kind="self_deadlock",
                    signature=f"self:{name}",
                    message=(
                        f"non-reentrant lock {name!r} re-acquired by {label} "
                        f"at {site} while already held"
                    ),
                    held=held,
                    acquiring=name,
                    context=label,
                )

            # 2. sync lock held across an await
            if is_async:
                sync_held = [h for h in held if not h.is_async]
                if sync_held:
                    names = ", ".join(h.name for h in sync_held)
                    self._splat_locked(
                        kind="sync_held_across_await",
                        signature=f"await:{names}->{name}",
                        message=(
                            f"{label} awaits async lock {name!r} at {site} while "
                            f"holding blocking lock(s) [{names}] taken at "
                            f"[{', '.join(h.site for h in sync_held)}] — the "
                            "blocking lock is held across a suspension point"
                        ),
                        held=held,
                        acquiring=name,
                        context=label,
                    )

            for h in held:
                # 1a. declared rank inversion
                if (
                    h.rank is not LockRank.UNRANKED
                    and rank is not LockRank.UNRANKED
                    and rank <= h.rank
                    and h.name != name
                ):
                    self._splat_locked(
                        kind="rank_inversion",
                        signature=f"rank:{h.name}->{name}",
                        message=(
                            f"lock {name!r} (rank {rank.name}) acquired at {site} "
                            f"while holding {h.name!r} (rank {h.rank.name}) taken "
                            f"at {h.site} — declared order requires strictly "
                            "increasing rank"
                        ),
                        held=held,
                        acquiring=name,
                        context=label,
                    )

                # 1b. observed order inversion
                if h.name == name:
                    continue
                edge = (h.name, name)
                if name in self._edges.get(h.name, ()):  # already known-good order
                    continue
                cycle = self._creates_cycle(h.name, name)
                if cycle is not None:
                    opposing = self._edge_site.get((name, h.name), "<unknown site>")
                    self._splat_locked(
                        kind="order_inversion",
                        signature="cycle:" + "->".join(sorted({*cycle, h.name, name})),
                        message=(
                            f"acquiring {name!r} while holding {h.name!r} at {site} "
                            f"closes the cycle {' -> '.join(cycle)}; the opposing "
                            f"order was established at {opposing}. This is an ABBA "
                            "deadlock that has not happened yet"
                        ),
                        held=held,
                        acquiring=name,
                        context=label,
                    )
                self._edges.setdefault(h.name, set()).add(name)
                self._edge_site.setdefault(edge, site)

            held.append(
                _HeldLock(
                    name=name,
                    rank=rank,
                    is_async=is_async,
                    acquired_at=now,
                    thread_ident=threading.get_ident(),
                    site=site,
                )
            )

    def on_release(self, name: str, *, is_async: bool) -> None:
        key, label = self._context_key()
        now = time.monotonic()
        with self._lock:
            state = self._contexts.get(key)
            if state is None:
                return
            for idx in range(len(state.held) - 1, -1, -1):
                if state.held[idx].name == name:
                    entry = state.held.pop(idx)
                    break
            else:
                return
            if not state.held:
                self._contexts.pop(key, None)

            # 4. loop-blocking hold
            held_for = now - entry.acquired_at
            if (
                not is_async
                and held_for > LOOP_BLOCKING_HOLD_S
                and self._loop_thread_ident is not None
                and entry.thread_ident == self._loop_thread_ident
            ):
                self._splat_locked(
                    kind="loop_blocking_hold",
                    signature=f"hold:{name}@{entry.site}",
                    message=(
                        f"blocking lock {name!r} taken at {entry.site} was held "
                        f"{held_for * 1000:.0f}ms on the event loop thread "
                        f"(limit {LOOP_BLOCKING_HOLD_S * 1000:.0f}ms) — the loop "
                        "could not make progress for that window"
                    ),
                    held=state.held,
                    acquiring=name,
                    context=label,
                )

    # ── reporting ─────────────────────────────────────────────────────
    def _splat_locked(
        self,
        *,
        kind: str,
        signature: str,
        message: str,
        held: list[_HeldLock],
        acquiring: str,
        context: str,
    ) -> None:
        """Caller holds self._lock."""
        self._splat_counts[signature] = self._splat_counts.get(signature, 0) + 1
        if signature in self._splats:
            return  # already reported; count is enough
        if len(self._splats) >= MAX_SPLATS:
            return
        splat = Splat(
            kind=kind,
            signature=signature,
            message=message,
            held=tuple(h.name for h in held),
            acquiring=acquiring,
            at=time.time(),
            stack=self._stack(),
            context=context,
        )
        self._splats[signature] = splat
        # Report outside the caller's critical path as much as possible:
        # logging and taint are cheap and must not be deferred, because a
        # process that deadlocks after this point still needs the record.
        logger.error("🔒 LOCKDEP %s: %s", kind, message)
        taint(
            TaintFlag.LOCK_ORDER,
            f"{kind}: {message[:200]}",
            subsystem="lockdep",
        )
        try:
            from core.runtime.errors import record_degradation

            record_degradation(
                "lockdep",
                RuntimeError(message),
                severity="warning",
                action=f"recorded {kind} splat; execution continued",
                extra={"kind": kind, "signature": signature, "held": list(splat.held)},
                enforce_failure_policy=False,
            )
        except Exception:  # noqa: BLE001 — never let reporting break the lock
            logger.debug("lockdep degradation record failed", exc_info=True)

    def splats(self) -> list[Splat]:
        with self._lock:
            return sorted(self._splats.values(), key=lambda s: s.at)

    def report(self) -> dict[str, Any]:
        with self._lock:
            edges = {k: sorted(v) for k, v in self._edges.items()}
            splats = [s.to_dict() for s in sorted(self._splats.values(), key=lambda s: s.at)]
            counts = dict(self._splat_counts)
            acquires = self._acquires
            ranks = {k: v.name for k, v in self._ranks.items()}
            live = {
                state.label: [h.name for h in state.held]
                for state in self._contexts.values()
                if state.held
            }
        for entry in splats:
            entry["occurrences"] = counts.get(entry["signature"], 1)
        return {
            "clean": not splats,
            "acquires_checked": acquires,
            "known_locks": sorted(set(edges) | set(ranks)),
            "declared_ranks": ranks,
            "order_edges": edges,
            "splats": splats,
            "currently_held": live,
        }

    def report_external(self, *, kind: str, signature: str, message: str, held: list[str]) -> None:
        """Record a hazard observed by a caller rather than by an acquire.

        Deduplicated by signature exactly like an internally detected splat,
        so a hot path (every fsync, every spawn) reports the first
        occurrence and then only counts.
        """
        key, label = self._context_key()
        with self._lock:
            self._splat_counts[signature] = self._splat_counts.get(signature, 0) + 1
            if signature in self._splats or len(self._splats) >= MAX_SPLATS:
                return
            self._splats[signature] = Splat(
                kind=kind,
                signature=signature,
                message=message,
                held=tuple(held),
                acquiring="(external)",
                at=time.time(),
                stack=self._stack(),
                context=label,
            )
        logger.error("🔒 LOCKDEP %s: %s", kind, message)
        taint(TaintFlag.LOCK_ORDER, f"{kind}: {message[:200]}", subsystem="lockdep")

    def already_reported(self, signature: str) -> bool:
        with self._lock:
            return signature in self._splats

    def held_names(self) -> list[str]:
        key, _ = self._context_key()
        with self._lock:
            state = self._contexts.get(key)
            return [h.name for h in state.held] if state else []

    def reset_for_test(self) -> None:
        with self._lock:
            self._edges.clear()
            self._edge_site.clear()
            self._ranks.clear()
            self._contexts.clear()
            self._splats.clear()
            self._splat_counts.clear()
            self._acquires = 0


_VALIDATOR = LockdepValidator()


def get_validator() -> LockdepValidator:
    return _VALIDATOR


class CheckedLock:
    """Drop-in for ``threading.Lock`` / ``threading.RLock`` with validation."""

    __slots__ = ("_lock", "_name", "_rank", "_reentrant")

    def __init__(self, name: str, *, rank: LockRank = LockRank.UNRANKED, reentrant: bool = False):
        self._name = name
        self._rank = rank
        self._reentrant = reentrant
        self._lock = threading.RLock() if reentrant else threading.Lock()
        _VALIDATOR.register(name, rank)

    @property
    def name(self) -> str:
        return self._name

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        _VALIDATOR.on_acquire(
            self._name, rank=self._rank, is_async=False, reentrant=self._reentrant
        )
        acquired = self._lock.acquire(blocking, timeout)
        if not acquired:
            _VALIDATOR.on_release(self._name, is_async=False)
        return acquired

    def release(self) -> None:
        self._lock.release()
        _VALIDATOR.on_release(self._name, is_async=False)

    def __enter__(self) -> CheckedLock:
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def locked(self) -> bool:
        held = self._lock.acquire(blocking=False)
        if held:
            self._lock.release()
            return False
        return True


class CheckedAsyncLock:
    """Drop-in for ``asyncio.Lock`` with validation."""

    __slots__ = ("_budget_s", "_lock", "_name", "_rank")

    def __init__(
        self,
        name: str,
        *,
        rank: LockRank = LockRank.UNRANKED,
        budget_s: float = ACQUIRE_BUDGET_S,
    ):
        self._name = name
        self._rank = rank
        self._budget_s = budget_s
        self._lock = asyncio.Lock()
        _VALIDATOR.register(name, rank)

    @property
    def name(self) -> str:
        return self._name

    async def _acquire_within_budget(self) -> bool:
        """Acquire, bounded. Exceeding the budget IS the deadlock report.

        An unbounded ``await lock.acquire()`` is the wedge this module
        exists to find, so it would be incoherent for lockdep's own lock
        to contain one. A lock that cannot be taken within the budget is
        reported as a splat — with the holders named — and the timeout
        propagates rather than becoming an invisible stall.
        """
        _VALIDATOR.on_acquire(self._name, rank=self._rank, is_async=True, reentrant=False)
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self._budget_s)
        except TimeoutError:
            _VALIDATOR.on_release(self._name, is_async=True)
            held = _VALIDATOR.held_names()
            _VALIDATOR.report_external(
                kind="acquire_timeout",
                signature=f"acquire_timeout:{self._name}",
                message=(
                    f"async lock {self._name!r} could not be acquired within "
                    f"{self._budget_s:.0f}s. Either a holder is wedged, or this is "
                    f"the deadlock the order graph predicted. Held here: {held or 'nothing'}"
                ),
                held=held,
            )
            raise
        except BaseException:  # noqa: BLE001 — release bookkeeping precedes propagation
            _VALIDATOR.on_release(self._name, is_async=True)
            raise
        return True

    async def acquire(self) -> bool:
        return await self._acquire_within_budget()

    def release(self) -> None:
        self._lock.release()
        _VALIDATOR.on_release(self._name, is_async=True)

    async def __aenter__(self) -> CheckedAsyncLock:
        await self._acquire_within_budget()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.release()

    def locked(self) -> bool:
        return self._lock.locked()


def checked_lock(
    name: str, *, rank: LockRank = LockRank.UNRANKED, reentrant: bool = False
) -> CheckedLock:
    return CheckedLock(name, rank=rank, reentrant=reentrant)


def checked_async_lock(name: str, *, rank: LockRank = LockRank.UNRANKED) -> CheckedAsyncLock:
    return CheckedAsyncLock(name, rank=rank)


@contextmanager
def instrument(name: str, *, rank: LockRank = LockRank.UNRANKED) -> Iterator[None]:
    """Validate a critical section guarded by a lock this module does not own.

    Wrap the ``with existing_lock:`` body so the ordering is still recorded::

        with existing_lock, instrument("legacy_cache", rank=LockRank.LEAF):
            ...
    """
    _VALIDATOR.on_acquire(name, rank=rank, is_async=False, reentrant=False)
    try:
        yield
    finally:
        _VALIDATOR.on_release(name, is_async=False)


class LockHeldError(RuntimeError):
    """Raised by :func:`assert_no_locks_held` in strict callers."""


def assert_no_locks_held(operation: str, *, strict: bool = False) -> list[str]:
    """Guard a blocking operation that must never run under a lock.

    fsync, subprocess spawn, and thread joins are the recorded offenders.
    Returns the offending lock names (empty when clean); raises in strict
    mode, otherwise reports and lets the caller proceed.
    """
    held = _VALIDATOR.held_names()
    if not held:
        return []
    message = (
        f"{operation} attempted while holding {held} — a blocking operation "
        "under a lock is how the runtime freezes"
    )
    _VALIDATOR.report_external(
        kind="blocking_op_under_lock",
        signature=f"blocking:{operation}:{','.join(sorted(held))}",
        message=message,
        held=held,
    )
    if strict:
        raise LockHeldError(message)
    return held


def lockdep_report() -> dict[str, Any]:
    return _VALIDATOR.report()


def lockdep_clean() -> bool:
    return not _VALIDATOR.splats()


def note_event_loop_thread() -> None:
    """Called once from the runtime boot path on the loop thread."""
    _VALIDATOR.note_loop_thread()


def reset_lockdep_for_test() -> None:
    _VALIDATOR.reset_for_test()


__all__ = [
    "LOOP_BLOCKING_HOLD_S",
    "CheckedAsyncLock",
    "CheckedLock",
    "LockHeldError",
    "LockRank",
    "LockdepValidator",
    "Splat",
    "assert_no_locks_held",
    "checked_async_lock",
    "checked_lock",
    "get_validator",
    "instrument",
    "lockdep_clean",
    "lockdep_report",
    "note_event_loop_thread",
    "reset_lockdep_for_test",
]
