import json

import pytest


@pytest.mark.live
def test_live_agi_capability_battery(live_harness):
    """Live subsystem-liveness battery.

    Asserts the battery actually probed the cognitive subsystems and reported
    REAL probe results. The prior version asserted a fabricated 17-category
    capability scorecard (Gaussian noise around a hardcoded mean vs hardcoded
    baselines); that has been removed, so this test now verifies honesty:
    the report carries no synthesized capability score and the real probes pass.
    """
    repo = live_harness.create_isolated_copy()

    result = live_harness.run_command(
        repo,
        [
            ".venv/bin/python",
            "tools/agi/run_agi_capability_battery.py",
            "--output", "artifacts/agi_live/capability_battery.json",
            "--markdown", "artifacts/agi_live/CAPABILITY_BATTERY_RESULTS.md",
        ],
        timeout_s=600,
    )

    assert result.ok, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    report = json.loads((result.artifacts_dir / "capability_battery.json").read_text())

    # Honest shape: subsystem liveness, not a fabricated capability score.
    assert report["measurement"] == "subsystem_liveness"
    assert "aura_scores" not in report  # the fabricated scorecard is gone
    assert "baselines_and_ablations" not in report

    probes = report["probes"]
    assert set(probes) == {
        "will_concurrency",
        "volition_deduplication",
        "agency_goal_completion",
        "steering_vector_library",
        "skill_surface_constraint",
    }
    # The real probes must actually pass on a live boot.
    assert report["all_probes_pass"] is True, f"probe failures: {probes}"
    assert report["probes_passed"] == report["probes_total"]
    assert report["capability_areas_probed"] == 17

    telemetry = report["live_telemetry"]
    assert telemetry["registered_skills"] > 0
    assert telemetry["will_probe_p50_ms"] >= 0.0
