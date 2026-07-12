"""core/brain/lane_admission.py — declarative memory admission for model lanes.

The dominant historical reliability ceiling on this host is model-serving
memory over-commitment: lanes spawn imperatively, each spot-checking
*instantaneous* free RAM, so concurrent warmups + trainers + a resident 32B
can jointly commit more than the host holds. The result is the recorded
stall → force-kill → cold-reload doom loop.

This module is the Kubernetes-style fix (roadmap K3): every model lane
*declares* a memory request up front, a single controller admits only within
an explicit host budget, and QoS classes decide who yields when the budget
is contended:

  GUARANTEED  — the primary cortex. Never named as an eviction candidate.
  BURSTABLE   — brainstem / reflex / solver. Evictable for a GUARANTEED
                candidate, in largest-footprint-first order.
  BEST_EFFORT — trainers, compounding, auxiliary lanes. Evicted first.

The controller is deliberately pure arithmetic + bookkeeping: callers pass
the observed active lanes and the candidate's projected footprint (the
existing, battle-tested projection in mlx_client). Decisions come in three
shapes:

  * fits            → admitted, no conditions.
  * fits-if-yield   → admitted, with an ``evict_first`` advisory naming the
                      lower-QoS lanes that must yield. The existing swap
                      machinery (and, next, the K1 reconciler) executes it.
  * envelope breach → REFUSED with a named reason. This is the case that
                      today ends in an OOM SIGKILL with empty stderr —
                      e.g. the 72B solver over a committed host. Refusing
                      here is aerospace envelope protection (A4), not a
                      regression: the spawn was never going to survive.

Every decision is recorded in a bounded in-memory ring for the health
surface and the incident narrator. ``AURA_LANE_ADMISSION`` selects
``enforce`` (default) or ``advise`` (log-only; the kill switch).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.flags import FlagKind, declare
from core.runtime.resource_observation import ResourceObserver, get_resource_observer

logger = logging.getLogger("Aura.LaneAdmission")

_DECISION_RING_SIZE = 64

_BUDGET_GB_FLAG = declare(
    "AURA_LANE_BUDGET_GB",
    kind=FlagKind.FLOAT,
    default=0.0,
    description="Absolute lane memory budget in GB; 0 = derive from fraction",
    owner="core.brain.lane_admission",
)
_BUDGET_FRACTION_FLAG = declare(
    "AURA_LANE_BUDGET_FRACTION",
    kind=FlagKind.FLOAT,
    default=0.72,
    description="Fraction of host RAM all model lanes may jointly commit",
    owner="core.brain.lane_admission",
)
_EVICTION_SHIELD_FLAG = declare(
    "AURA_LANE_EVICTION_SHIELD_S",
    kind=FlagKind.FLOAT,
    default=180.0,
    description="Seconds after a user-facing turn during which a lane is shielded from eviction",
    owner="core.brain.lane_admission",
)
_ADMISSION_MODE_FLAG = declare(
    "AURA_LANE_ADMISSION",
    kind=FlagKind.STRING,
    default="enforce",
    description="Lane admission mode: enforce (refusals bind) or advise (log-only kill switch)",
    owner="core.brain.lane_admission",
)


class QoSClass(StrEnum):
    GUARANTEED = "guaranteed"
    BURSTABLE = "burstable"
    BEST_EFFORT = "best_effort"


_QOS_RANK = {QoSClass.BEST_EFFORT: 0, QoSClass.BURSTABLE: 1, QoSClass.GUARANTEED: 2}


def classify_lane(model_path: str, *, purpose: str = "serve") -> tuple[str, QoSClass]:
    """Map a model path (and purpose) to a (lane, QoS) declaration.

    Token matching mirrors the projection heuristics in mlx_client so the
    two layers never disagree about which lane a path belongs to.
    """
    if purpose in {"train", "compound", "fuse", "benchmark", "evaluate", "eval"}:
        return "trainer", QoSClass.BEST_EFFORT
    lowered = str(model_path or "").lower()
    if any(token in lowered for token in ("72b", "solver")):
        return "solver", QoSClass.BURSTABLE
    if any(token in lowered for token in ("32b", "cortex", "zenith")):
        return "cortex", QoSClass.GUARANTEED
    if any(token in lowered for token in ("14b", "7b", "brainstem")):
        return "brainstem", QoSClass.BURSTABLE
    if any(token in lowered for token in ("1.5b", "1p5b", "0.5b", "reflex")):
        return "reflex", QoSClass.BURSTABLE
    return "auxiliary", QoSClass.BEST_EFFORT


@dataclass(frozen=True)
class ActiveLane:
    """One observed live model lane (a running worker holding memory)."""

    lane: str
    qos: QoSClass
    footprint_gb: float
    model_path: str = ""
    # Age since this lane last completed a user-facing generation; recently
    # user-facing lanes are shielded from eviction advisories (the 180s
    # cortex-protection precedent from the live thrash findings).
    last_user_facing_age_s: float | None = None


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    lane: str
    qos: QoSClass
    request_gb: float
    committed_gb: float
    budget_gb: float
    evict_first: tuple[str, ...] = ()
    enforced: bool = True
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""
    resource_observation_available: bool = False
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "lane": self.lane,
            "qos": str(self.qos),
            "request_gb": round(self.request_gb, 2),
            "committed_gb": round(self.committed_gb, 2),
            "budget_gb": round(self.budget_gb, 2),
            "evict_first": list(self.evict_first),
            "enforced": self.enforced,
            "observation_source": self.observation_source,
            "observation_scenario_id": self.observation_scenario_id,
            "resource_observation_available": self.resource_observation_available,
            "at": self.at,
        }


def _host_total_gb(observer: ResourceObserver | None = None) -> tuple[float, bool]:
    memory = (observer or get_resource_observer()).memory()
    if memory.available and memory.total_bytes > 0:
        return float(memory.total_bytes) / float(1024**3), True
    return 64.0, False


def _budget_for_total_gb(host_total_gb: float) -> float:
    absolute = float(_BUDGET_GB_FLAG.value())
    if absolute > 0.0:
        return absolute
    fraction = max(0.30, min(0.95, float(_BUDGET_FRACTION_FLAG.value())))
    return host_total_gb * fraction


def lane_budget_gb(*, observer: ResourceObserver | None = None) -> float:
    """The total memory envelope all model lanes may jointly commit.

    Default: 72% of host RAM. On the 64 GB production host that is ~46 GB —
    room for the resident 4-bit 32B (~20), brainstem (~5), reflex (~2) and a
    trainer burst, while the OS, the orchestrator process, and page cache
    keep the remainder. Override with AURA_LANE_BUDGET_GB (absolute) or
    AURA_LANE_BUDGET_FRACTION.
    """
    host_total_gb, _available = _host_total_gb(observer)
    return _budget_for_total_gb(host_total_gb)


def _eviction_shield_s() -> float:
    return float(_EVICTION_SHIELD_FLAG.value())


def enforcement_mode() -> str:
    mode = str(_ADMISSION_MODE_FLAG.value()).strip().lower()
    if mode in {"off", "0", "false", "advise", "advisory"}:
        return "advise"
    return "enforce"


class LaneAdmissionController:
    """Pure budget/QoS arithmetic with a bounded decision ring.

    Thread-safe; safe to call from worker-spawn executor threads.
    """

    def __init__(self, *, observer: ResourceObserver | None = None) -> None:
        self._observer = observer
        self._decisions: deque[AdmissionDecision] = deque(maxlen=_DECISION_RING_SIZE)
        self._lock = threading.Lock()

    # ── decision engine ────────────────────────────────────────────

    def admit(
        self,
        *,
        model_path: str,
        request_gb: float,
        active: Iterable[ActiveLane],
        purpose: str = "serve",
        allow_disruptive_eviction: bool = False,
    ) -> AdmissionDecision:
        lane, qos = classify_lane(model_path, purpose=purpose)
        request_gb = max(0.0, float(request_gb))
        observer = self._observer or get_resource_observer()
        provenance = observer.provenance
        host_total, observation_available = _host_total_gb(observer)
        budget = _budget_for_total_gb(host_total)
        # A lane replacing itself (worker recycle) must not double-count:
        # callers exclude the candidate's own lane from `active`.
        lanes = [active_lane for active_lane in active if active_lane.footprint_gb > 0.0]
        committed = sum(active_lane.footprint_gb for active_lane in lanes)

        if committed + request_gb <= budget:
            decision = self._record(
                AdmissionDecision(
                    admitted=True,
                    reason="fits",
                    lane=lane,
                    qos=qos,
                    request_gb=request_gb,
                    committed_gb=committed,
                    budget_gb=budget,
                    observation_source=provenance.source.value,
                    observation_scenario_id=provenance.scenario_id,
                    resource_observation_available=observation_available,
                )
            )
            return decision

        # Over budget: can lower-QoS lanes yield enough room? The recently-
        # user-facing shield protects warm lanes from background churn, but
        # a GUARANTEED candidate (the primary cortex) outranks it — the
        # cortex must always be able to come up.
        shield_s = _eviction_shield_s()
        evictable = [
            active_lane
            for active_lane in lanes
            if (
                allow_disruptive_eviction
                or _QOS_RANK[active_lane.qos] < _QOS_RANK[qos]
            )
            and (
                allow_disruptive_eviction
                or qos is QoSClass.GUARANTEED
                or active_lane.last_user_facing_age_s is None
                or active_lane.last_user_facing_age_s >= shield_s
            )
        ]
        # Best-effort first, then largest footprint — free the most with the
        # least collateral.
        evictable.sort(
            key=lambda active_lane: (
                _QOS_RANK[active_lane.qos],
                -active_lane.footprint_gb,
            )
        )

        freed = 0.0
        chosen: list[ActiveLane] = []
        for candidate in evictable:
            if committed - freed + request_gb <= budget:
                break
            chosen.append(candidate)
            freed += candidate.footprint_gb

        if committed - freed + request_gb <= budget:
            decision = self._record(
                AdmissionDecision(
                    admitted=True,
                    reason="fits_after_yield",
                    lane=lane,
                    qos=qos,
                    request_gb=request_gb,
                    committed_gb=committed,
                    budget_gb=budget,
                    evict_first=tuple(
                        c.model_path or c.lane for c in chosen
                    ),
                    observation_source=provenance.source.value,
                    observation_scenario_id=provenance.scenario_id,
                    resource_observation_available=observation_available,
                )
            )
            return decision

        # Envelope breach: even with every eviction we own, this spawn
        # exceeds the host budget. Refuse with the arithmetic in the reason.
        enforced = enforcement_mode() == "enforce"
        decision = self._record(
            AdmissionDecision(
                admitted=False,
                reason=(
                    f"lane_budget_exceeded:{lane} request {request_gb:.1f}GB "
                    f"+ committed {committed - freed:.1f}GB (after max yield) "
                    f"> budget {budget:.1f}GB"
                ),
                lane=lane,
                qos=qos,
                request_gb=request_gb,
                committed_gb=committed,
                budget_gb=budget,
                evict_first=tuple(c.model_path or c.lane for c in chosen),
                enforced=enforced,
                observation_source=provenance.source.value,
                observation_scenario_id=provenance.scenario_id,
                resource_observation_available=observation_available,
            )
        )
        return decision

    # ── observability ──────────────────────────────────────────────

    def _record(self, decision: AdmissionDecision) -> AdmissionDecision:
        with self._lock:
            self._decisions.append(decision)
        if decision.admitted and decision.reason == "fits":
            logger.debug("Lane admission: %s", decision.to_dict())
        elif decision.admitted:
            logger.info("Lane admission (yield advised): %s", decision.to_dict())
        else:
            logger.warning("Lane admission REFUSED: %s", decision.to_dict())
        return decision

    def snapshot(self) -> dict[str, Any]:
        observer = self._observer or get_resource_observer()
        provenance = observer.provenance
        with self._lock:
            recent = [d.to_dict() for d in list(self._decisions)[-10:]]
        return {
            "alive": True,
            "ready": True,
            "budget_gb": round(lane_budget_gb(observer=observer), 2),
            "mode": enforcement_mode(),
            "observation_source": provenance.source.value,
            "observation_scenario_id": provenance.scenario_id,
            "recent_decisions": recent,
        }

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return True

    def get_status(self) -> dict[str, Any]:
        return self.snapshot()


_CONTROLLER: LaneAdmissionController | None = None
_CONTROLLER_LOCK = threading.Lock()


def get_lane_admission_controller() -> LaneAdmissionController:
    global _CONTROLLER
    if _CONTROLLER is None:
        with _CONTROLLER_LOCK:
            if _CONTROLLER is None:
                _CONTROLLER = LaneAdmissionController()
    return _CONTROLLER


def reset_lane_admission_controller_for_test() -> None:
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        _CONTROLLER = None
