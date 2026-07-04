import json

import pytest


@pytest.mark.live
def test_live_causal_agency_lesion(live_harness):
    repo = live_harness.create_isolated_copy()

    result = live_harness.run_command(
        repo,
        [
            ".venv/bin/python",
            "tools/agi/run_causal_agency_lesion.py",
            "--seeds", "50",
            "--output", "artifacts/agi_live/causal_agency.json",
        ],
        timeout_s=900,
    )

    assert result.ok, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    report = json.loads((result.artifacts_dir / "causal_agency.json").read_text())

    # Honest thresholds. The pre-2026-07-04 version of the runner clamped the
    # intact divergence to a 0.35 floor and the lesioned divergence to 0.04,
    # so the old >=0.25 / <=0.10 assertions were testing the rig, not the
    # organ. The real coupler measures ~0.07 mean pairwise normalized L1
    # across 8 contexts, and exactly 0.0 when blinded — the meaningful claims
    # are relative separation, distinct policies, determinism, and a real
    # permutation p-value.
    assert report["manual_interventions"] == 0
    assert report["receipt_coverage"] == 1.0
    assert report["deterministic_within_context"] is True
    assert report["distinct_intact_policies"] >= 3
    assert report["distinct_lesioned_policies"] == 1
    assert report["normal_state_action_divergence"] >= 0.05
    assert report["lesioned_action_divergence"] <= 0.005
    assert report["normal_state_action_divergence"] > report["lesioned_action_divergence"]
    assert report["p_value"] < 0.01
    assert report["causal_state_action_coupling"] is True
