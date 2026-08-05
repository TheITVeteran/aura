"""Operational owner for governed Reality Reach acceptance campaigns."""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from core.reality_reach.acceptance import (
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceCertificateStore,
    AcceptanceError,
    AcceptanceEvidenceClass,
    AcceptanceExecutor,
    ConnectorAcceptanceCertificate,
    ScalarAcceptancePlan,
    ScalarAcceptanceRunner,
)
from core.reality_reach.live import RealityReachService
from core.reality_reach.metrology import (
    AcquisitionChannel,
    AcquisitionMode,
    AcquisitionTask,
    EvidenceSource,
    RealityMetrologyService,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock

_GIT_OID = re.compile(r"^[0-9a-f]{40}$")
_MAX_HIL_COMPANION_CHANNELS = 63
SourceIdentityProvider = Callable[[], Mapping[str, Any]]


def capture_runtime_source_identity() -> Mapping[str, Any]:
    from core.config import config
    from core.runtime.launch_provenance import (
        collect_runtime_launch_provenance,
        collect_source_identity,
    )

    root = config.paths.project_root
    provenance = collect_runtime_launch_provenance(root)
    required = provenance.get("required") is True
    source = provenance.get("actual") if required else collect_source_identity(root)
    if not isinstance(source, Mapping):
        source = {}
    return {
        "identity_bound": bool(
            (provenance.get("verified") is True if required else True)
            and provenance.get("source_verified", True) is True
        ),
        "source_commit": str(source.get("commit_sha") or "").strip().lower(),
        "source_dirty": source.get("source_dirty") is True,
        "workspace_state_sha256": str(
            source.get("workspace_state_sha256") or ""
        ).strip().lower(),
    }


def _normalize_source_identity(raw: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(raw, Mapping) or raw.get("identity_bound") is not True:
        raise AcceptanceError("acceptance_runtime_source_identity_unbound")
    commit = str(raw.get("source_commit") or "").strip().lower()
    if not _GIT_OID.fullmatch(commit):
        raise AcceptanceError("acceptance_runtime_source_commit_invalid")
    if raw.get("source_dirty") is True:
        raise AcceptanceError("acceptance_runtime_source_dirty")
    workspace = str(raw.get("workspace_state_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", workspace):
        raise AcceptanceError("acceptance_runtime_workspace_identity_invalid")
    return {
        "source_commit_sha256": str(sha256_hex(canonical_json(commit))),
        "workspace_state_sha256": f"sha256:{workspace}",
    }


@dataclass(frozen=True, slots=True)
class ScalarAcceptanceRequest:
    campaign_id: str
    connector_id: str
    adapter_id: str
    target: float
    expected_source_commit_sha256: str
    evidence_class: AcceptanceEvidenceClass
    scenario_id: str = ""
    simulated_channel_ids: tuple[str, ...] = ()
    deadline_s: float = 5.0
    sample_interval_s: float = 0.1
    effect_hold_s: float = 0.25

    def __post_init__(self) -> None:
        channels = tuple(self.simulated_channel_ids)
        if len(channels) > _MAX_HIL_COMPANION_CHANNELS:
            raise ValueError(
                "simulated_channel_ids exceeds the HIL companion-channel bound"
            )
        if len(set(channels)) != len(channels):
            raise ValueError("simulated_channel_ids must be unique")
        object.__setattr__(self, "simulated_channel_ids", channels)
        if self.evidence_class is AcceptanceEvidenceClass.SIMULATION:
            if self.simulated_channel_ids:
                raise ValueError("simulation acceptance does not use HIL companion channels")
            return
        if self.evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP:
            if not self.scenario_id or not self.simulated_channel_ids:
                raise ValueError("HIL acceptance requires a scenario and simulated channels")
        elif self.simulated_channel_ids:
            raise ValueError("live acceptance cannot include simulated channels")
        interval = float(self.sample_interval_s)
        deadline = float(self.deadline_s)
        hold = float(self.effect_hold_s)
        if not math.isfinite(interval) or not 0.01 <= interval <= 0.5:
            raise ValueError("sample_interval_s must lie inside [0.01, 0.5]")
        if not math.isfinite(deadline) or not 0.5 <= deadline <= 60.0:
            raise ValueError("deadline_s must lie inside [0.5, 60]")
        if not math.isfinite(hold) or not max(0.05, interval * 2.0) <= hold <= 5.0:
            raise ValueError("effect_hold_s must span at least two sample intervals")


class RealityAcceptanceService:
    """Serialize, execute, persist, and summarize physical acceptance runs."""

    def __init__(
        self,
        reality: RealityReachService,
        metrology: RealityMetrologyService,
        *,
        store: AcceptanceCertificateStore | None = None,
        governed_executor: AcceptanceExecutor | None = None,
        pinned_source_identity: Mapping[str, Any],
        source_identity_provider: SourceIdentityProvider | None = None,
    ) -> None:
        if not isinstance(reality, RealityReachService):
            raise TypeError("reality must be a RealityReachService")
        if not isinstance(metrology, RealityMetrologyService):
            raise TypeError("metrology must be a RealityMetrologyService")
        self._reality = reality
        self._metrology = metrology
        self._store = store or AcceptanceCertificateStore()
        self._governed_executor = governed_executor
        if not isinstance(pinned_source_identity, Mapping):
            raise TypeError("pinned_source_identity must be a mapping")
        self._pinned_source_identity = dict(pinned_source_identity)
        self._source_identity_provider = (
            source_identity_provider or capture_runtime_source_identity
        )
        self._lock = checked_async_lock("reality_reach.acceptance_service")
        self._generation = 0
        self._active_campaign_id = ""
        self._last_result: dict[str, Any] | None = None
        self._last_failure: dict[str, Any] | None = None

    async def _source_binding(self, expected_sha256: str | None) -> dict[str, str]:
        raw = await asyncio.to_thread(self._source_identity_provider)
        pinned = _normalize_source_identity(self._pinned_source_identity)
        current = _normalize_source_identity(raw)
        if current != pinned:
            raise AcceptanceError("acceptance_runtime_source_drifted_since_boot")
        if (
            expected_sha256 is not None
            and current["source_commit_sha256"] != expected_sha256
        ):
            raise AcceptanceError("acceptance_runtime_source_commit_mismatch")
        return current

    async def preflight(self, adapter_id: str) -> dict[str, Any]:
        adapter = self._reality.scalar_acceptance_adapter(adapter_id)
        if adapter is None:
            raise AcceptanceError("acceptance_scalar_adapter_not_registered")
        source = await self._source_binding(None)
        capabilities = tuple(adapter.actuator_capabilities())
        observation_channels = tuple(
            dict.fromkeys(
                channel_id
                for capability in capabilities
                for channel_id in capability.observation_channels
            )
        )
        evidence_classes = (
            (AcceptanceEvidenceClass.SIMULATION.value,)
            if adapter.transport_class.value == "simulated"
            else (
                AcceptanceEvidenceClass.HARDWARE_IN_LOOP.value,
                AcceptanceEvidenceClass.LIVE.value,
            )
        )
        return {
            "ready": bool(capabilities and observation_channels and self._metrology.is_ready()),
            "adapter_id": adapter.adapter_id,
            "physical_identity_sha256": adapter.physical_identity_sha256,
            "transport_class": adapter.transport_class.value,
            "target_tolerance": adapter.effect_tolerance,
            "required_cases": list(REQUIRED_SCALAR_ACCEPTANCE_CASES),
            "observation_channels": list(observation_channels),
            "supported_evidence_classes": list(evidence_classes),
            **source,
            "trust_boundary": "producer_observation_not_independent_acceptance",
        }

    def _record_failure(
        self,
        request: ScalarAcceptanceRequest,
        error: BaseException,
        *,
        started_at_ns: int,
    ) -> None:
        error_type = type(error).__name__
        self._last_failure = {
            "campaign_id_sha256": str(
                sha256_hex(canonical_json(str(request.campaign_id)))
            ),
            "error_type": error_type,
            "error_sha256": str(
                sha256_hex(
                    canonical_json(
                        {
                            "error_type": error_type,
                            "message": str(error),
                        }
                    )
                )
            ),
            "started_at_ns": started_at_ns,
            "failed_at_ns": time.time_ns(),
        }

    def _task(self, request: ScalarAcceptanceRequest, adapter: Any) -> AcquisitionTask:
        capabilities = tuple(adapter.actuator_capabilities())
        live_channels = tuple(
            dict.fromkeys(
                channel_id
                for capability in capabilities
                for channel_id in capability.observation_channels
            )
        )
        if not live_channels:
            raise AcceptanceError("acceptance_adapter_has_no_observation_channels")
        mode = (
            AcquisitionMode.HARDWARE_IN_LOOP
            if request.evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP
            else AcquisitionMode.LIVE
        )
        channels = tuple(AcquisitionChannel(item) for item in live_channels)
        channels += tuple(
            AcquisitionChannel(item, EvidenceSource.SIMULATED)
            for item in request.simulated_channel_ids
        )
        duration_s = request.deadline_s + request.effect_hold_s + 1.0
        sample_count = int(math.ceil(duration_s / request.sample_interval_s)) + 1
        if sample_count > 1024:
            raise AcceptanceError("acceptance_metrology_sample_budget_exceeded")
        return AcquisitionTask(
            task_id=f"acceptance.{request.campaign_id}"[:128],
            channels=channels,
            mode=mode,
            sample_count=sample_count,
            sample_interval_s=request.sample_interval_s,
            timeout_s=duration_s + 5.0,
            scenario_id=request.scenario_id,
        )

    async def run(self, request: ScalarAcceptanceRequest) -> dict[str, Any]:
        if not isinstance(request, ScalarAcceptanceRequest):
            raise TypeError("request must be a ScalarAcceptanceRequest")
        adapter = self._reality.scalar_acceptance_adapter(request.adapter_id)
        if adapter is None:
            raise AcceptanceError("acceptance_scalar_adapter_not_registered")
        async with self._lock:
            self._generation += 1
            self._active_campaign_id = request.campaign_id
            started_at_ns = time.time_ns()
            try:
                source_before = await self._source_binding(
                    request.expected_source_commit_sha256
                )
                plan = ScalarAcceptancePlan(
                    campaign_id=request.campaign_id,
                    connector_id=request.connector_id,
                    target=request.target,
                    source_commit_sha256=source_before["source_commit_sha256"],
                    authority_receipt_id="authority.runtime.will",
                    evidence_class=request.evidence_class,
                    scenario_id=request.scenario_id,
                    deadline_s=request.deadline_s,
                    metrology_effect_hold_s=request.effect_hold_s,
                )
                if request.evidence_class is AcceptanceEvidenceClass.SIMULATION:
                    runner = ScalarAcceptanceRunner(adapter, self._reality, plan)
                else:
                    task = self._task(request, adapter)
                    runner = ScalarAcceptanceRunner(
                        adapter,
                        self._reality,
                        plan,
                        governed_executor=self._governed_executor,
                        metrology_acquirer=lambda operation: self._metrology.acquire_around(
                            task,
                            operation,
                        ),
                    )
                certificate = await runner.run()
                source_after = await self._source_binding(
                    request.expected_source_commit_sha256
                )
                if source_after != source_before:
                    raise AcceptanceError("acceptance_runtime_source_changed_during_run")
                self._store.persist(certificate)
                self._store.persist_evidence(
                    certificate,
                    runner.case_evidence,
                    metrology_receipt=runner.metrology_receipt,
                    governance_evidence=(
                        runner.governance_evidence
                        if certificate.governance_evidence_sha256
                        else None
                    ),
                )
                result = self._result(
                    certificate,
                    started_at_ns=started_at_ns,
                    source_identity=source_after,
                )
                self._last_result = result
                self._last_failure = None
                return dict(result)
            except asyncio.CancelledError as exc:
                self._record_failure(request, exc, started_at_ns=started_at_ns)
                raise
            except Exception as exc:
                self._record_failure(request, exc, started_at_ns=started_at_ns)
                raise
            finally:
                self._active_campaign_id = ""

    def _result(
        self,
        certificate: ConnectorAcceptanceCertificate,
        *,
        started_at_ns: int,
        source_identity: Mapping[str, str],
    ) -> dict[str, Any]:
        return {
            "campaign_id": certificate.campaign_id,
            "certificate_sha256": certificate.sha256,
            "evidence_class": next(iter(certificate.cases)).evidence_class.value,
            "deterministic_passed": certificate.deterministic_passed,
            "live_acceptance_passed": certificate.live_acceptance_passed,
            "metrology_evidence_sha256": certificate.metrology_evidence_sha256,
            "governance_evidence_sha256": certificate.governance_evidence_sha256,
            "source_commit_sha256": source_identity["source_commit_sha256"],
            "workspace_state_sha256": source_identity["workspace_state_sha256"],
            "started_at_ns": started_at_ns,
            "completed_at_ns": time.time_ns(),
        }

    def status(self) -> dict[str, Any]:
        source_blocker = ""
        try:
            _normalize_source_identity(self._pinned_source_identity)
        except AcceptanceError as exc:
            source_blocker = str(exc)
        return {
            "alive": True,
            "ready": bool(
                not self._active_campaign_id
                and not source_blocker
                and self._metrology.is_ready()
            ),
            "source_identity_blocker": source_blocker,
            "generation": self._generation,
            "active_campaign_id": self._active_campaign_id,
            "last_result": dict(self._last_result) if self._last_result else None,
            "last_failure": dict(self._last_failure) if self._last_failure else None,
            "store_root": str(self._store.root),
        }

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return bool(self.status()["ready"])


__all__ = [
    "RealityAcceptanceService",
    "ScalarAcceptanceRequest",
    "SourceIdentityProvider",
    "capture_runtime_source_identity",
]
