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


def _final_estimate(report: dict) -> dict:
    """The post-campaign estimate.

    ``estimate_with_interventions`` was renamed to ``estimate_after_campaign``:
    the old name asserted a causal role the interventions did not have (they
    reach only the discrete estimator, never the Gaussian rail that produces the
    headline Φ/z/p), and in the run that named itself that way every
    interventional row had been rejected. Both keys are read so historical
    artifacts stay checkable rather than being quietly regenerated away.
    """
    for key in ("estimate_after_campaign", "estimate_with_interventions"):
        if key in report:
            return report[key]
    raise AssertionError("artifact has no post-campaign estimate")


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
    est = _final_estimate(report)
    assert est["n_samples"] >= 600
    assert est["n_channels"] >= 6, "a real multi-organ channel set"


def test_estimate_carries_the_full_evidence(report):
    est = _final_estimate(report)
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
    est = _final_estimate(report)
    names = set(est["channel_names"])
    organs = {n.split(".")[0] for n in names}
    # a genuine multi-subsystem harvest, not a constructed ring
    assert len(organs) >= 3, f"too few organ families: {organs}"
    assert not any(n.startswith("ch") and n[2:].isdigit() for n in names), (
        "synthetic channel names in a live artifact"
    )


def _assert_campaign_contract(report: dict) -> None:
    campaign = report["campaign"]
    if report["mode"] == "live_api":
        assert campaign["trials_requested"] == 0
        assert campaign["trials_ran"] == 0
        assert "never perturbed" in campaign["note"]
        return
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


def test_perturbation_campaign_ran_and_beats_sham(report):
    _assert_campaign_contract(report)


def _assert_workload_contract(report: dict) -> None:
    if report["mode"] == "live_api":
        assert "workload" not in report
        assert "running naturally" in report["scope_claim"]
        return
    workload = report["workload"]
    assert workload["ticks"] >= 1000
    assert workload["coupling"], "the injected couplings must be declared"
    outcomes = workload["decision_outcomes"]
    assert sum(outcomes.values()) >= 1000, "real decision traffic"


def test_workload_is_transparent(report):
    _assert_workload_contract(report)


def test_live_api_mode_has_explicit_nonintervention_contract():
    live_report = {
        "mode": "live_api",
        "scope_claim": "The live mind is sampled read-only while running naturally.",
        "campaign": {
            "trials_requested": 0,
            "trials_ran": 0,
            "note": "live instance is never perturbed by tooling",
        },
    }

    _assert_campaign_contract(live_report)
    _assert_workload_contract(live_report)


# ---------------------------------------------------------------------------
# The honesty contract the original artifact did not have to satisfy
# ---------------------------------------------------------------------------


def test_a_positive_verdict_requires_a_resolvable_null(report):
    """"Integration established" must not rest on the test's own resolution.

    The checked-in run reported family-wise p = 0.047619 from 20 surrogates —
    exactly 1/(20+1), the smallest value that test can produce. It cleared the
    threshold only because it could not have been any smaller.

    Note what is NOT required here: that p sits above the floor. A genuinely
    integrated system beats *every* null draw, so its p is at the floor however
    many surrogates you run — that is a strong result, not a weak one. What
    matters is where the floor is. At 20 surrogates the floor (0.048) is
    indistinguishable from "barely significant"; at 500 it is 0.002, so beating
    all nulls means something.
    """
    est = _final_estimate(report)
    if not est.get("integration_established"):
        return  # a negative verdict needs no resolution guarantee

    from core.consciousness.integrated_information import PHI_MIN_CLAIM_SURROGATES

    surrogates = int(est.get("grain_selection_surrogates") or 0)
    p = float(est.get("grain_selection_p", 1.0))
    floor = 1.0 / (surrogates + 1) if surrogates else 1.0

    assert surrogates >= PHI_MIN_CLAIM_SURROGATES, (
        f"integration_established=True on only {surrogates} surrogates "
        f"(p={p:.6f}, floor={floor:.6f}); at least {PHI_MIN_CLAIM_SURROGATES} "
        "are needed before the threshold is resolvable at all"
    )


def test_interventional_rows_reach_the_estimator_or_are_reported_as_absent(report):
    """A campaign's interventions must be accounted for, not silently dropped.

    Five probe trials produced 13-channel rows that were handed to an estimator
    expecting 8 named channels; all five were projection-rejected. The artifact
    still called the result "estimate_with_interventions", and no test noticed
    because the only assertion was `n_interventional_transitions >= 0`.
    """
    if report["mode"] == "live_api":
        return  # never perturbed

    est = _final_estimate(report)
    macro = est.get("exact_macro") or {}
    if not macro:
        return

    accepted = int(macro.get("n_interventional_transitions", 0))
    rejected = int(macro.get("n_projection_rejected_transitions", 0))

    if rejected and not accepted:
        pytest.fail(
            f"all {rejected} interventional rows were rejected at projection — "
            "the campaign contributed nothing to the estimate. Pass "
            "channel_names to add_interventional_transitions."
        )


def test_the_artifact_does_not_imply_interventions_drove_the_gaussian_rail(report):
    """The Gaussian rail never consumes interventional rows.

    phi_raw / z / family-wise p — every number behind integration_established —
    come from the time-series estimator, which reads no interventions at all. An
    artifact must not let a reader infer the perturbations moved them.
    """
    if "interventions" not in report:
        return  # historical artifact predating the disclosure

    interventions = report["interventions"]
    assert "not_consumed_by" in interventions
    assert "Gaussian" in interventions["not_consumed_by"]


def test_regime_change_is_disclosed(report):
    """The post-campaign window is not the same regime plus interventions.

    The sample count rose 3600 → 3960: exactly the 90 s stabilization rest plus
    6 × 45 s inter-trial rests. A quiet decay-and-recovery regime was appended to
    a decision-workload regime, and coordinated affect drift during recovery can
    raise integration on its own. So a pre→post change cannot be attributed to
    the perturbations.
    """
    if report["mode"] == "live_api" or "regime_note" not in report:
        return  # historical artifact predating the disclosure

    note = report["regime_note"]
    assert "rest" in note.lower()
    assert "NOT evidence" in note
