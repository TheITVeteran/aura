"""Tests for the consolidated system-integrity audit."""
from __future__ import annotations

import core.runtime.integrity_audit as ia


def test_audit_aggregates_signals_and_reports_structure():
    report = ia.run_integrity_audit(log=False)
    assert set(report) >= {"healthy", "concerns", "advisory", "degradations", "crsm_loop", "caa_readiness"}
    assert isinstance(report["concerns"], list)
    # CRSM loop + CAA readiness are real on this repo → expect them surfaced
    assert report["crsm_loop"] and report["caa_readiness"]


def test_caa_and_crsm_are_advisory_not_health_blocking():
    report = ia.run_integrity_audit(log=False)
    # Operational proof facts surface as ADVISORY when they remain open, never
    # as runtime-health concerns, so they cannot make launch report "degraded".
    if report["crsm_loop"].get("state") == "open":
        assert any("CRSM" in c for c in report["advisory"])
    else:
        assert not any("CRSM" in c for c in report["advisory"])
    assert report["caa_readiness"]["level"] in {"production", "validated", "mixed", "bootstrap"}
    if report["caa_readiness"].get("below_design_capacity"):
        assert any("CAA steering" in c for c in report["advisory"])
    assert not any("CAA steering" in c for c in report["concerns"])
    assert not any("CRSM" in c for c in report["concerns"])


def test_maybe_run_is_throttled(monkeypatch):
    ia._last_run = 0.0
    first = ia.maybe_run(interval_s=10_000)
    assert first is not None
    # immediate second call within the interval returns the cached report, no re-run
    ran = {"n": 0}
    orig = ia.run_integrity_audit
    monkeypatch.setattr(ia, "run_integrity_audit", lambda **k: ran.__setitem__("n", ran["n"] + 1) or orig(**k))
    ia.maybe_run(interval_s=10_000)
    assert ran["n"] == 0                      # throttled — did not re-run


def test_strict_mode_reflected(monkeypatch):
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")
    assert ia.strict_mode() is True
    assert ia.run_integrity_audit(log=False)["strict_mode"] is True


def test_report_names_failure_pressure_feeders(monkeypatch):
    """During a lockdown the integrity report must name the pressure's top
    contributing subsystems — no log archaeology to find the feeder."""
    import core.runtime.integrity_audit as ia

    monkeypatch.setattr(
        "core.health.degraded_events.get_unified_failure_state",
        lambda limit=25: {
            "pressure": 0.85,
            "count": 12,
            "critical": 2,
            "top_subsystems": ["mlx_warmup", "chat.cognitive_engine_reply"],
        },
    )

    report = ia.run_integrity_audit(log=False)

    assert report["failure_state"]["pressure"] == 0.85
    assert report["failure_state"]["top_subsystems"][0] == "mlx_warmup"
    assert any("failure pressure 0.85" in a for a in report["advisory"])
