"""core/runtime/quota.py — resource quotas, limit ranges, and QoS classes.

Clean-room adoption of Kubernetes ResourceQuota, LimitRange, and the
requests-vs-limits model that produces QoS classes.

The distinction Kubernetes makes and Aura did not is between a **request**
and a **limit**:

* A *request* is what the organ is guaranteed. The scheduler will not
  over-commit it, and the organ can rely on it existing.
* A *limit* is the ceiling it may burst to when the resource happens to be
  free. Exceeding it is throttled or denied.

That one distinction produces everything else. An organ whose requests
equal its limits is **Guaranteed** — it never gets more, and it never gets
evicted for using what it asked for. An organ with a request below its
limit is **Burstable** — it may use the slack, and it gives the slack back
first. An organ that declares nothing is **BestEffort** — it runs on
leftovers and is the first thing shed. Aura's organs have exactly this
structure (a resident model that must not be squeezed, background research
that should use whatever is spare) and expressed it nowhere, so under
pressure the runtime had no principled way to decide who loses.

Quota enforcement is registered as an **admission hook**, which is where
Kubernetes puts it too, and for the same reason: a budget checked at the
point of use is a budget checked wherever somebody remembered. Checked at
admission, it is checked everywhere by construction.

Resources are Aura's real contended ones — tokens, tool calls, durable
writes, inference seconds, subprocess spawns, network requests — not a
copy of the container ones, because those are not what runs out here.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Quota")


class ResourceKind(StrEnum):
    """What actually runs out in this runtime."""

    TOKENS = "tokens"
    TOOL_CALLS = "tool_calls"
    DURABLE_WRITES = "durable_writes"
    INFERENCE_SECONDS = "inference_seconds"
    CPU_SECONDS = "cpu_seconds"
    MEMORY_BYTES = "memory_bytes"
    NETWORK_REQUESTS = "network_requests"
    SUBPROCESSES = "subprocesses"


class QosClass(StrEnum):
    #: requests == limits for everything declared. Never evicted for using
    #: what it asked for.
    GUARANTEED = "guaranteed"
    #: requests < limits somewhere. May burst; gives the slack back first.
    BURSTABLE = "burstable"
    #: declares nothing. Runs on leftovers, shed first.
    BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class ResourceSpec:
    """One organ's declared requests and limits."""

    name: str
    requests: dict[str, float] = field(default_factory=dict)
    limits: dict[str, float] = field(default_factory=dict)
    priority: int = 0

    @property
    def qos_class(self) -> QosClass:
        if not self.requests and not self.limits:
            return QosClass.BEST_EFFORT
        if not self.limits:
            return QosClass.BURSTABLE
        # Guaranteed requires every limited resource to have an equal request.
        for kind, limit in self.limits.items():
            if abs(self.requests.get(kind, 0.0) - limit) > 1e-9:
                return QosClass.BURSTABLE
        return QosClass.GUARANTEED if self.requests else QosClass.BURSTABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requests": dict(self.requests),
            "limits": dict(self.limits),
            "priority": self.priority,
            "qos_class": str(self.qos_class),
        }


@dataclass(frozen=True)
class LimitRange:
    """Defaults and bounds applied to any spec that omits them.

    A LimitRange is how a system stops depending on every author
    remembering to declare a budget: unset fields get the default, and
    out-of-band declarations are rejected at admission rather than
    discovered at exhaustion.
    """

    default_requests: dict[str, float] = field(default_factory=dict)
    default_limits: dict[str, float] = field(default_factory=dict)
    minimum: dict[str, float] = field(default_factory=dict)
    maximum: dict[str, float] = field(default_factory=dict)

    def apply(self, spec: ResourceSpec) -> ResourceSpec:
        resource_requests = {**self.default_requests, **spec.requests}
        limits = {**self.default_limits, **spec.limits}
        # A request above its own limit is incoherent; clamp it rather
        # than admit a spec that can never be satisfied.
        for kind, limit in limits.items():
            if resource_requests.get(kind, 0.0) > limit:
                resource_requests[kind] = limit
        return ResourceSpec(
            name=spec.name,
            requests=resource_requests,
            limits=limits,
            priority=spec.priority,
        )

    def violations(self, spec: ResourceSpec) -> list[str]:
        problems: list[str] = []
        for kind, floor in self.minimum.items():
            value = spec.limits.get(kind, spec.requests.get(kind))
            if value is not None and value < floor:
                problems.append(f"{kind}={value} is below the minimum {floor}")
        for kind, ceiling in self.maximum.items():
            value = spec.limits.get(kind, spec.requests.get(kind))
            if value is not None and value > ceiling:
                problems.append(f"{kind}={value} exceeds the maximum {ceiling}")
        return problems


@dataclass
class QuotaVerdict:
    allowed: bool
    scope: str
    reason: str = ""
    exceeded: dict[str, tuple[float, float]] = field(default_factory=dict)
    reservation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "scope": self.scope,
            "reason": self.reason,
            "exceeded": {k: list(v) for k, v in self.exceeded.items()},
            "reservation_id": self.reservation_id,
        }


@dataclass
class _Window:
    """A rolling budget window. ``period_s <= 0`` means "for all time"."""

    hard: dict[str, float]
    period_s: float
    used: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)

    def maybe_roll(self) -> bool:
        if self.period_s <= 0:
            return False
        now = time.monotonic()
        if now - self.started_at < self.period_s:
            return False
        self.used.clear()
        self.started_at = now
        return True


class QuotaRegistry:
    """Scoped budgets with atomic reserve/commit/release."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, _Window] = {}
        self._specs: dict[str, ResourceSpec] = {}
        self._limit_range: LimitRange = LimitRange()
        self._reservations: dict[str, tuple[str, dict[str, float]]] = {}
        self._counter = 0
        self.denials = 0
        self._denial_reasons: dict[str, int] = {}

    # ── declaration ───────────────────────────────────────────────────
    def set_limit_range(self, limit_range: LimitRange) -> None:
        with self._lock:
            self._limit_range = limit_range

    def limit_range(self) -> LimitRange:
        with self._lock:
            return self._limit_range

    def declare(self, spec: ResourceSpec) -> ResourceSpec:
        """Register an organ's requests/limits, defaulted by the LimitRange."""
        with self._lock:
            resolved = self._limit_range.apply(spec)
            self._specs[resolved.name] = resolved
            return resolved

    def spec(self, name: str) -> ResourceSpec | None:
        with self._lock:
            return self._specs.get(name)

    def specs(self) -> dict[str, ResourceSpec]:
        with self._lock:
            return dict(self._specs)

    def qos_class(self, name: str) -> QosClass:
        spec = self.spec(name)
        return spec.qos_class if spec else QosClass.BEST_EFFORT

    def set_quota(self, scope: str, hard: dict[str, float], *, period_s: float = 0.0) -> None:
        """Set a scope's hard budget. ``period_s`` makes it a rolling window."""
        with self._lock:
            existing = self._windows.get(scope)
            if existing is not None and existing.period_s == period_s:
                existing.hard = dict(hard)
                return
            self._windows[scope] = _Window(hard=dict(hard), period_s=period_s)

    # ── enforcement ───────────────────────────────────────────────────
    def check(self, scope: str, amounts: dict[str, float]) -> QuotaVerdict:
        """Would this fit? Does not reserve."""
        with self._lock:
            return self._check_locked(scope, amounts)

    def _check_locked(self, scope: str, amounts: dict[str, float]) -> QuotaVerdict:
        window = self._windows.get(scope)
        if window is None:
            return QuotaVerdict(allowed=True, scope=scope)
        window.maybe_roll()
        exceeded: dict[str, tuple[float, float]] = {}
        for kind, amount in amounts.items():
            hard = window.hard.get(kind)
            if hard is None:
                continue
            projected = window.used.get(kind, 0.0) + float(amount)
            if projected > hard:
                exceeded[kind] = (projected, hard)
        if exceeded:
            detail = ", ".join(
                f"{kind} would reach {used:.4g} of {hard:.4g}"
                for kind, (used, hard) in sorted(exceeded.items())
            )
            return QuotaVerdict(
                allowed=False,
                scope=scope,
                reason=f"quota {scope!r} exceeded: {detail}",
                exceeded=exceeded,
            )
        return QuotaVerdict(allowed=True, scope=scope)

    def reserve(self, scope: str, amounts: dict[str, float]) -> QuotaVerdict:
        """Atomically check and consume. Release with :meth:`release`."""
        with self._lock:
            verdict = self._check_locked(scope, amounts)
            if not verdict.allowed:
                self.denials += 1
                for kind in verdict.exceeded:
                    key = f"{scope}:{kind}"
                    self._denial_reasons[key] = self._denial_reasons.get(key, 0) + 1
                return verdict
            window = self._windows.get(scope)
            if window is not None:
                for kind, amount in amounts.items():
                    window.used[kind] = window.used.get(kind, 0.0) + float(amount)
            self._counter += 1
            reservation_id = f"{scope}#{self._counter}"
            self._reservations[reservation_id] = (scope, dict(amounts))
            verdict.reservation_id = reservation_id
            return verdict

    def release(self, reservation_id: str) -> bool:
        """Give back an unused reservation (the work did not happen)."""
        with self._lock:
            entry = self._reservations.pop(reservation_id, None)
            if entry is None:
                return False
            scope, amounts = entry
            window = self._windows.get(scope)
            if window is not None:
                for kind, amount in amounts.items():
                    window.used[kind] = max(0.0, window.used.get(kind, 0.0) - float(amount))
            return True

    def commit(self, reservation_id: str) -> bool:
        """Keep the consumption; forget the ability to release it."""
        with self._lock:
            return self._reservations.pop(reservation_id, None) is not None

    def usage(self, scope: str) -> dict[str, Any]:
        with self._lock:
            window = self._windows.get(scope)
            if window is None:
                return {"scope": scope, "unbounded": True}
            window.maybe_roll()
            return {
                "scope": scope,
                "period_s": window.period_s,
                "hard": dict(window.hard),
                "used": dict(window.used),
                "headroom": {
                    kind: hard - window.used.get(kind, 0.0)
                    for kind, hard in window.hard.items()
                },
                "age_s": round(time.monotonic() - window.started_at, 2),
            }

    def report(self) -> dict[str, Any]:
        with self._lock:
            scopes = list(self._windows)
            specs = {name: spec.to_dict() for name, spec in self._specs.items()}
            denials = dict(self._denial_reasons)
            outstanding = len(self._reservations)
        by_class: dict[str, list[str]] = {}
        for name, spec in specs.items():
            by_class.setdefault(spec["qos_class"], []).append(name)
        return {
            "scopes": {scope: self.usage(scope) for scope in scopes},
            "specs": specs,
            "by_qos_class": by_class,
            "denials": self.denials,
            "denials_by_resource": denials,
            "outstanding_reservations": outstanding,
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._windows.clear()
            self._specs.clear()
            self._reservations.clear()
            self._limit_range = LimitRange()
            self._counter = 0
            self.denials = 0
            self._denial_reasons.clear()


_REGISTRY = QuotaRegistry()


def get_quota_registry() -> QuotaRegistry:
    return _REGISTRY


def declare_resources(
    name: str,
    *,
    requests: dict[str, float] | None = None,
    limits: dict[str, float] | None = None,
    priority: int = 0,
) -> ResourceSpec:
    """Declare an organ's budget. This is what assigns its QoS class."""
    return _REGISTRY.declare(
        ResourceSpec(
            name=name,
            requests=dict(requests or {}),
            limits=dict(limits or {}),
            priority=priority,
        )
    )


def install_quota_admission() -> bool:
    """Register quota enforcement as an admission hook.

    Kubernetes enforces ResourceQuota at admission rather than at the
    point of use, because a budget checked at the point of use is checked
    only where somebody remembered. Idempotent.
    """
    from core.runtime.admission import (
        AdmissionRequest,
        AdmissionResponse,
        FailurePolicy,
        get_admission_chain,
        validating,
    )

    chain = get_admission_chain()
    if any(h.name == "quota.enforce" for h in chain.hooks()):
        return False

    @validating(
        "quota.enforce",
        order=10,
        failure_policy=FailurePolicy.FAIL,
        owner="core/runtime/quota.py",
    )
    def _enforce(request: AdmissionRequest) -> AdmissionResponse:
        amounts = request.context.get("resources")
        if not isinstance(amounts, dict) or not amounts:
            return AdmissionResponse.allow()
        scope = str(request.context.get("quota_scope") or request.principal)
        verdict = _REGISTRY.reserve(scope, {str(k): float(v) for k, v in amounts.items()})
        if not verdict.allowed:
            return AdmissionResponse.deny(verdict.reason)
        # Hand the reservation id back so a caller that abandons the work
        # can return the budget instead of leaking it.
        request.context["quota_reservation"] = verdict.reservation_id
        return AdmissionResponse.allow()

    return True


def quota_report() -> dict[str, Any]:
    return _REGISTRY.report()


def reset_quota_for_test() -> None:
    _REGISTRY.reset_for_test()


__all__ = [
    "LimitRange",
    "QosClass",
    "QuotaRegistry",
    "QuotaVerdict",
    "ResourceKind",
    "ResourceSpec",
    "declare_resources",
    "get_quota_registry",
    "install_quota_admission",
    "quota_report",
    "reset_quota_for_test",
]
