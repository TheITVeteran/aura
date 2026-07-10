"""Contracts for the K2 probe split: startup / liveness / readiness.

Their conflation caused a real incident class: a loop-lag spike flipped the
health verdict, boot readiness went false, and the GUI sat on "Connecting
to runtime…" for 55 minutes over a fully conversational mind. These tests
pin the three independent semantics and the boot-status presentation rule
they exist to enforce: once startup latches, the shell NEVER presents
"booting" again — only "degraded".
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.health.boot_status import build_boot_health_snapshot
from core.runtime.errors import get_degradation_tracker
from core.runtime.health_contract import (
    RUNTIME_CONTRACT,
    HealthLevel,
    ProbeKind,
    ServiceRequirement,
    ServiceTier,
    evaluate_health,
    evaluate_probes,
    latch_startup_if_ready,
    probe_split_report,
    probes_from_report,
    reset_startup_latch_for_test,
    startup_complete_at,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_state():
    ServiceContainer.clear()
    get_degradation_tracker().reset()
    reset_startup_latch_for_test()
    yield
    ServiceContainer.clear()
    get_degradation_tracker().reset()
    reset_startup_latch_for_test()


def _service_for(requirement: ServiceRequirement, *, failing_key: str | None = None) -> object:
    if requirement.liveness_check is None:
        return SimpleNamespace()
    live = requirement.container_key != failing_key
    return SimpleNamespace(**{requirement.liveness_check: lambda live=live: live})


def _register(*, tiers: set[ServiceTier], failing_key: str | None = None) -> None:
    for requirement in RUNTIME_CONTRACT:
        if requirement.tier in tiers:
            ServiceContainer.register_instance(
                requirement.container_key,
                _service_for(requirement, failing_key=failing_key),
            )


def _healthy_orchestrator() -> SimpleNamespace:
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=3,
        start_time=None,
    )
    return SimpleNamespace(status=status, health_check=lambda: True)


_RUNTIME_STATE = {"sha256": "abc", "signature": "sig", "state": {}}


class TestStartupLatch:
    def test_readiness_pass_latches_startup(self):
        _register(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
        assert startup_complete_at() is None
        probes = evaluate_probes()
        assert probes["readiness"].ok
        assert probes["startup"].ok
        assert startup_complete_at() is not None

    def test_startup_survives_later_readiness_flap(self):
        """THE incident semantics: readiness may flap; startup never regresses."""
        _register(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
        evaluate_probes()
        latched_at = startup_complete_at()
        assert latched_at is not None

        # Now the spine degrades: re-register with a failing critical key.
        ServiceContainer.clear()
        failing = next(
            r.container_key for r in RUNTIME_CONTRACT if r.tier is ServiceTier.CRITICAL
        )
        _register(
            tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT}, failing_key=failing
        )
        probes = evaluate_probes()
        assert not probes["readiness"].ok
        assert probes["startup"].ok, "startup must stay latched through readiness flaps"
        assert startup_complete_at() == latched_at

    def test_before_latch_within_deadline_is_starting(self):
        probes = probes_from_report({"status": "degraded", "required_probes": {}})
        assert probes["startup"].ok
        assert "starting" in probes["startup"].reason

    def test_before_latch_past_deadline_is_wedged(self, monkeypatch):
        monkeypatch.setenv("AURA_STARTUP_DEADLINE_S", "0.0")
        probes = probes_from_report({"status": "degraded", "required_probes": {}})
        assert not probes["startup"].ok
        assert "wedged" in probes["startup"].reason

    def test_latch_helper_is_monotonic(self):
        latch_startup_if_ready(False)
        assert startup_complete_at() is None
        latch_startup_if_ready(True)
        first = startup_complete_at()
        latch_startup_if_ready(True)
        assert startup_complete_at() == first


class TestLivenessSemantics:
    def test_liveness_fails_only_on_dead(self):
        """No critical spine at all = DEAD = restart-worthy."""
        verdict = evaluate_health()
        assert verdict.level == HealthLevel.DEAD
        probes = probes_from_report(verdict.to_report())
        assert not probes["liveness"].ok

    def test_important_tier_failure_fails_neither_liveness_nor_readiness(self):
        """Important-tier degradation hurts UX; it neither stops traffic nor
        advises a restart (the conversation_operational precedent)."""
        failing = next(
            r.container_key for r in RUNTIME_CONTRACT if r.tier is ServiceTier.IMPORTANT
        )
        _register(
            tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT}, failing_key=failing
        )
        probes = evaluate_probes()
        assert probes["liveness"].ok, "flapping important services are not restart-worthy"
        assert probes["readiness"].ok, "important-tier degradation does not stop traffic"

    def test_critical_probe_member_failure_fails_readiness_not_liveness(self):
        """A failing critical probe-group member stops traffic — but the
        spine is present, so it is NOT restart-worthy. This distinction is
        the entire point of the split."""
        _register(
            tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
            failing_key="llm_router",
        )
        probes = evaluate_probes()
        assert not probes["readiness"].ok
        assert probes["liveness"].ok


class TestReportShape:
    def test_probe_split_report_is_serializable(self):
        _register(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
        report = probe_split_report()
        assert set(report) == {"startup", "liveness", "readiness"}
        for name, payload in report.items():
            assert payload["kind"] == name
            assert isinstance(payload["ok"], bool)
            assert isinstance(payload["reason"], str)

    def test_probe_kinds_are_stable_strings(self):
        assert [str(kind) for kind in ProbeKind] == ["startup", "liveness", "readiness"]


class TestBootStatusPresentation:
    """The rule the split exists for: no 'booting' after first readiness."""

    def test_degraded_after_latch_not_booting(self):
        _register(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
        evaluate_probes()  # latch startup
        assert startup_complete_at() is not None

        # Runtime contract now fails (empty container = nothing registered).
        ServiceContainer.clear()
        payload, http_status = build_boot_health_snapshot(
            _healthy_orchestrator(), _RUNTIME_STATE, is_gui_proxy=False
        )
        assert payload["status"] == "degraded", "post-latch must present degraded"
        assert payload["boot_phase"] == "runtime_degraded"
        assert payload["progress"] == 100, "boot is OVER; the runtime is degraded"
        assert payload["checks"]["startup_latched"] is True
        assert http_status == 503, "presentation changes; traffic gating does not"

    def test_booting_before_latch_is_unchanged(self):
        payload, http_status = build_boot_health_snapshot(
            _healthy_orchestrator(), _RUNTIME_STATE, is_gui_proxy=False
        )
        assert payload["status"] == "booting"
        assert payload["boot_phase"] == "kernel_warming"
        assert payload["checks"]["startup_latched"] is False
        assert http_status == 503

    def test_boot_payload_carries_probe_split(self):
        _register(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
        payload, _ = build_boot_health_snapshot(
            _healthy_orchestrator(), _RUNTIME_STATE, is_gui_proxy=False
        )
        assert set(payload["probes"]) == {"startup", "liveness", "readiness"}
        assert payload["probes"]["liveness"]["ok"] is True

    def test_status_message_for_degraded_runtime(self):
        _register(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
        evaluate_probes()
        ServiceContainer.clear()
        payload, _ = build_boot_health_snapshot(
            _healthy_orchestrator(), _RUNTIME_STATE, is_gui_proxy=False
        )
        assert "degraded" in payload["status_message"].lower()
