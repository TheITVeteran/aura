from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.health.system_health import get_runtime_health_contract
from core.runtime.health_contract import (
    HEALTH_CONTRACT_VERSION,
    RUNTIME_CONTRACT,
    HealthLevel,
    ServiceRequirement,
    ServiceTier,
    evaluate_health,
    required_probe_status,
    runtime_health_report,
)


@pytest.fixture(autouse=True)
def isolated_service_container():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


def _service_for(requirement: ServiceRequirement, *, failing_key: str | None = None) -> object:
    if requirement.liveness_check is None:
        return SimpleNamespace()
    live = requirement.container_key != failing_key
    return SimpleNamespace(**{requirement.liveness_check: lambda live=live: live})


def _register_contract_services(
    *,
    tiers: set[ServiceTier],
    failing_key: str | None = None,
) -> None:
    for requirement in RUNTIME_CONTRACT:
        if requirement.tier in tiers:
            ServiceContainer.register_instance(
                requirement.container_key,
                _service_for(requirement, failing_key=failing_key),
            )


def test_runtime_contract_report_marks_all_required_tiers_healthy():
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})

    report = runtime_health_report()

    assert report["contract_version"] == HEALTH_CONTRACT_VERSION
    assert report["status"] == HealthLevel.HEALTHY.value
    assert report["healthy"] is True
    assert report["operational"] is True
    assert report["status_code"] == 200
    assert report["tier_summary"]["critical"]["failed"] == 0
    assert report["tier_summary"]["important"]["failed"] == 0
    assert report["tier_summary"]["optional"]["missing"] > 0
    assert report["failures"]["critical"] == []


def test_runtime_contract_distinguishes_degraded_from_failed_runtime():
    _register_contract_services(tiers={ServiceTier.CRITICAL})

    verdict = evaluate_health()
    report = verdict.to_report()

    assert verdict.level == HealthLevel.DEGRADED
    assert report["healthy"] is False
    assert report["operational"] is True
    assert report["tier_summary"]["important"]["failed"] > 0
    assert report["failures"]["critical"] == []


def test_runtime_contract_fails_closed_on_critical_liveness_failure():
    _register_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="inference_gate",
    )

    report = runtime_health_report()

    assert report["status"] == HealthLevel.CRITICAL.value
    assert report["healthy"] is False
    assert report["operational"] is False
    assert report["status_code"] == 503
    assert report["tier_summary"]["critical"]["liveness_failed"] == 1
    assert report["failures"]["critical"][0]["container_key"] == "inference_gate"
    assert report["failures"]["critical"][0]["liveness"] == "failed"


def test_runtime_contract_reports_dead_when_no_critical_service_exists():
    report = runtime_health_report()

    assert report["status"] == HealthLevel.DEAD.value
    assert report["operational"] is False
    assert report["status_code"] == 503
    assert (
        report["tier_summary"]["critical"]["missing"] == report["tier_summary"]["critical"]["total"]
    )


def test_runtime_health_endpoint_uses_contract_status_code():
    _register_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="inference_gate",
    )

    response = asyncio.run(get_runtime_health_contract())
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["contract_version"] == HEALTH_CONTRACT_VERSION
    assert payload["failures"]["critical"][0]["container_key"] == "inference_gate"


def test_compute_orchestrator_is_liveness_checked_by_runtime_health_contract():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.container_key == "compute_orchestrator"
    }

    assert set(required) == {"compute_orchestrator"}
    assert required["compute_orchestrator"].tier == ServiceTier.IMPORTANT
    assert required["compute_orchestrator"].liveness_check == "is_alive"


def test_runtime_contract_requires_kernel_inference_memory_scheduler_and_tool_governance():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.tier == ServiceTier.CRITICAL
    }

    assert required["kernel_interface"].liveness_check == "is_ready"
    assert required["inference_gate"].liveness_check == "is_alive"
    assert required["llm_router"].tier == ServiceTier.CRITICAL
    assert required["state_repository"].liveness_check == "is_initialized"
    assert required["memory_facade"].liveness_check == "is_ready"
    assert required["scheduler"].liveness_check == "is_alive"
    assert required["unified_will"].liveness_check == "is_alive"
    assert required["authority_gateway"].liveness_check == "is_ready"
    assert required["capability_engine"].liveness_check == "is_ready"


def test_required_probe_summary_fails_if_scheduler_or_tool_governance_is_unhealthy():
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    ServiceContainer.register_instance("scheduler", SimpleNamespace(is_alive=lambda: False))

    report = runtime_health_report()
    probes = required_probe_status(report)

    assert report["status"] == HealthLevel.CRITICAL.value
    assert probes["scheduler"]["ok"] is False
    assert probes["tool_governance"]["ok"] is True
    assert probes["all_passed"] is False


def test_observability_readiness_uses_runtime_health_contract():
    from core.observability.metrics import check_readiness

    ServiceContainer.clear()

    result = check_readiness()

    assert result["ready"] is False
    assert result["status"] == "not_ready"
    assert any(issue.startswith("runtime_contract:") for issue in result["issues"])
    assert any(issue.startswith("required_probes:") for issue in result["issues"])
