from __future__ import annotations

from tools.closeout.frontier_standards_matrix import STANDARDS, report
from tools.closeout.operational_label_baselines import classify_evidence_path, excluded_evidence_paths


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
    assert payload["passed"] is True
    assert payload["summary"]["gaps"] == 0
    assert payload["summary"]["mapped"] == payload["summary"]["total"]


def test_frontier_standards_keep_live_artifacts_separate_by_default():
    payload = report(require_live=True)
    by_key = {item["key"]: item for item in payload["standards"]}
    assert "artifacts/current/live_desktop_runtime" in by_key[
        "daily_runtime_reliability"
    ]["missing_live_artifacts"] or by_key["daily_runtime_reliability"]["status"] != "gap"


def test_sci_fi_capabilities_route_to_real_organs_not_themed_silo():
    sci_fi = next(standard for standard in STANDARDS if standard.key == "sci_fi_ai_capability")

    assert "docs/FICTIONAL_AI_CAPABILITY_MAP.md" in sci_fi.source_paths
    assert all(
        not path.startswith("core/fictional_ai") for path in sci_fi.source_paths
    ), sci_fi.source_paths
    assert {
        "core/capability_engine.py",
        "core/runtime/desktop_action_gateway.py",
        "core/actuation/desktop_actuator.py",
        "core/perception/screen_perception.py",
        "core/social/social_imagination.py",
    } <= set(sci_fi.source_paths)
    assert "tests/test_derived_character_engines.py" in sci_fi.validator_paths


def test_sci_fi_reference_set_keeps_requested_systems_visible():
    sci_fi = next(standard for standard in STANDARDS if standard.key == "sci_fi_ai_capability")
    references = set(sci_fi.reference_models)

    assert {
        "JARVIS",
        "Cortana",
        "EDI",
        "MIST",
        "Pantheon UIs",
        "Safe Surf",
        "Data",
        "Kokoro/Koroko",
        "Skynet (defensive resilience only)",
        "Caine",
        "Samantha/SAM",
        "Jane",
        "HAL 9000 (anti-HAL directive conflict handling)",
        "GLaDOS (adaptive testing only)",
        "TARS/CASE",
        "The Machine",
    } <= references


def test_frontier_standards_do_not_use_mock_or_proxy_paths_as_evidence():
    excluded = set(excluded_evidence_paths())

    for standard in STANDARDS:
        cited = set(standard.source_paths) | set(standard.validator_paths)
        assert not (excluded & cited), standard.key
        assert all(
            classify_evidence_path(path) not in {"excluded_proxy_or_harness", "benchmark_proxy"}
            for path in standard.source_paths
        ), standard.key


def test_frontier_standards_use_runtime_sources_for_capability_claims():
    runtime_heavy = {
        "daily_runtime_reliability",
        "humanlike_conversation",
        "sci_fi_ai_capability",
        "phenomenal_building_blocks",
        "os_control_frontier",
        "generally_capable_ai",
    }

    for standard in STANDARDS:
        if standard.key not in runtime_heavy:
            continue
        assert any(
            classify_evidence_path(path) == "runtime_source" for path in standard.source_paths
        ), standard.key
