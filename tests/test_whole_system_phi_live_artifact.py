"""The live-evidence contract for whole-system Φ.

The unit suite validates the instrument on known-answer systems; this test
validates THE MEASUREMENT — the checked-in artifact produced by
tools/measure_whole_system_phi.py against Aura's real runtime. It exists so
the July critique's demand ("has it measured Aura herself?") has a pinned,
regenerable answer: a real window, real channels, real governed
perturbation-versus-sham campaign, and an explicit scope claim. If the
artifact is missing or the evidence regresses, this fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "phi" / (
    "whole_system_live_report.json"
)


@pytest.fixture(scope="module")
def report() -> dict:
    assert ARTIFACT.is_file(), (
        "the live Φ measurement artifact is missing — regenerate with "
        ".venv/bin/python tools/measure_whole_system_phi.py"
    )
    envelope = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return envelope.get("payload", envelope)


def test_report_identity_and_provenance(report):
    assert report["schema"] == "aura.whole_system_phi_live_report.v1"
    assert report["mode"] in {"organ_host", "live_api"}
    assert report["git_commit"] and report["git_commit"] != "unknown"
    assert report["instrument"].endswith("integrated_information.py")


def test_scope_claim_is_honest(report):
    claim = report["scope_claim"]
    if report["mode"] == "organ_host":
        # the artifact must say out loud what it is NOT
        assert "NOT the full live mind" in claim
        assert "32B" in claim
    else:
        assert "LIVE desktop instance" in claim


def test_window_is_a_meaningful_period(report):
    assert report["window_seconds"] >= 600, "at least a 10-minute natural window"
    est = report["estimate_with_interventions"]
    assert est["n_samples"] >= 600
    assert est["n_channels"] >= 6, "a real multi-organ channel set"


def test_estimate_carries_the_full_evidence(report):
    est = report["estimate_with_interventions"]
    for key in ("estimator", "phi_raw", "z", "null_mean", "null_std",
                "ci_5", "ci_95", "mip", "grains", "emergent_grain_k",
                "diagnostics", "integration_established", "claim"):
        assert key in est, f"missing evidentiary field: {key}"
    assert est["phi_raw"] >= 0.0
    assert est["null_std"] >= 0.0
    assert est["ci_5"] <= est["ci_95"]
    assert isinstance(est["integration_established"], bool)
    assert "not a consciousness meter" in est["claim"]
    diag = est["diagnostics"]
    assert "stationarity_drift_sigma" in diag
    assert "dropped_dead_channels" in diag


def test_channels_are_real_organs_not_synthetic(report):
    est = report["estimate_with_interventions"]
    names = set(est["channel_names"])
    organs = {n.split(".")[0] for n in names}
    # a genuine multi-subsystem harvest, not a constructed ring
    assert len(organs) >= 3, f"too few organ families: {organs}"
    assert not any(n.startswith("ch") and n[2:].isdigit() for n in names), (
        "synthetic channel names in a live artifact"
    )


def test_perturbation_campaign_ran_and_beats_sham(report):
    if report["mode"] == "live_api":
        pytest.skip("the live instance is never perturbed by tooling")
    campaign = report["campaign"]
    assert campaign["trials_ran"] >= 4, "a real campaign, not a single anecdote"
    # every executed trial carries a Will receipt — the probe is governed
    for trial in campaign["trials"]:
        if trial.get("ran"):
            assert trial.get("will_receipt_id"), "ungoverned probe trial"
    # the causal signature: perturbed responses propagate more complexity
    # than sham windows. This is the artifact's evidentiary core.
    assert campaign["mean_evoked_complexity"] is not None
    assert campaign["mean_sham_evoked_complexity"] is not None
    assert (campaign["mean_evoked_complexity"]
            > campaign["mean_sham_evoked_complexity"]), (
        "perturbation did not beat sham — the causal evidence regressed"
    )


def test_workload_is_transparent(report):
    if report["mode"] == "live_api":
        pytest.skip("live mode has no injected workload")
    workload = report["workload"]
    assert workload["ticks"] >= 1000
    assert workload["coupling"], "the injected couplings must be declared"
    outcomes = workload["decision_outcomes"]
    assert sum(outcomes.values()) >= 1000, "real decision traffic"
