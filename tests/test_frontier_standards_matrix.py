from __future__ import annotations

from tools.closeout.frontier_standards_matrix import STANDARDS, report


def test_frontier_standards_cover_requested_closeout_targets():
    keys = {standard.key for standard in STANDARDS}
    assert {
        "daily_runtime_reliability",
        "humanlike_conversation",
        "sci_fi_ai_capability",
        "phenomenal_building_blocks",
        "frontier_reasoning_outside_model",
        "superintelligence_trajectory",
        "os_control_frontier",
        "nethack_general_environment",
        "generally_capable_ai",
    } <= keys


def test_frontier_standards_have_no_source_or_validator_gaps():
    payload = report()
    assert payload["summary"]["gaps"] == 0
    assert payload["summary"]["mapped"] == payload["summary"]["total"]


def test_frontier_standards_keep_live_artifacts_separate_by_default():
    payload = report(require_live=True)
    by_key = {item["key"]: item for item in payload["standards"]}
    assert "artifacts/current/live_desktop_runtime" in by_key[
        "daily_runtime_reliability"
    ]["missing_live_artifacts"] or by_key["daily_runtime_reliability"]["status"] != "gap"
