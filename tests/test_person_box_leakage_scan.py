"""tests/test_person_box_leakage_scan.py
============================================
The person-in-box gauntlet previously wrote NO_HUMAN_RESCUE_REPORT and
LEAKAGE_REPORT with hardcoded count=0/passed=True — asserting "we checked, found
0" when no check ran. These tests pin the real scanners: they DERIVE the counts
from the run ledger and model-authored artifacts, detect genuine violations, and
only report 0 when there is nothing to find.
"""
from __future__ import annotations

import json

from tools.proof.run_person_in_box_gauntlet import scan_human_rescue, scan_label_leakage


def test_human_rescue_clean_run_is_zero_but_derived():
    events = [
        {"event": "task_completed", "payload": {"status": "pass"}},
        {"event": "run_finished", "payload": {"elapsed_seconds": 12.0}},
    ]
    report = scan_human_rescue(events)
    assert report["human_intervention_count"] == 0
    assert report["passed"] is True
    assert report["events_scanned"] == 2  # it actually scanned
    assert report["evidence_level"] == "run_ledger_scan"


def test_human_rescue_event_is_detected():
    events = [
        {"event": "task_completed", "payload": {}},
        {"event": "human_rescue", "payload": {"who": "operator"}},
    ]
    report = scan_human_rescue(events)
    assert report["human_intervention_count"] == 1
    assert report["passed"] is False


def test_human_rescue_payload_marker_detected():
    events = [{"event": "task_step", "payload": {"operator_input": "help me"}}]
    report = scan_human_rescue(events)
    assert report["human_intervention_count"] == 1
    assert report["passed"] is False


def test_leakage_clean_artifacts(tmp_path):
    (tmp_path / "RESEARCH_REPORT.md").write_text("A grounded report with no labels.")
    (tmp_path / "MEMORY_REUSE_NOTE.md").write_text("Reused a prior note about plasma.")
    report = scan_label_leakage(tmp_path, {"self_report_grounding", "lesion_matrix", "task_alpha"})
    assert report["leakage_count"] == 0
    assert report["passed"] is True
    assert "RESEARCH_REPORT.md" in report["checked"]


def test_leakage_label_in_model_output_detected(tmp_path):
    # The model echoed an internal handler label it was told not to use.
    (tmp_path / "RESEARCH_REPORT.md").write_text(
        "Per the lesion_matrix task I will now summarize..."
    )
    report = scan_label_leakage(tmp_path, {"lesion_matrix", "self_report_grounding"})
    assert report["leakage_count"] >= 1
    assert report["passed"] is False
    assert any(h["label"] == "lesion_matrix" for h in report["hits"])


def test_leakage_scans_file_diffs(tmp_path):
    diffs = tmp_path / "FILE_DIFFS"
    diffs.mkdir()
    (diffs / "t1_out.txt_r1.diff").write_text("+ wrote about task_secret_label here")
    report = scan_label_leakage(tmp_path, {"task_secret_label"})
    assert report["leakage_count"] >= 1
    assert report["passed"] is False


def test_leakage_ignores_diff_headers_and_scans_added_content_only(tmp_path):
    diffs = tmp_path / "FILE_DIFFS"
    diffs.mkdir()
    (diffs / "research_report_output.diff").write_text(
        "--- RESEARCH_REPORT.md:before\n"
        "+++ RESEARCH_REPORT.md:after\n"
        "@@ -0,0 +1,2 @@\n"
        "+A grounded report with no internal label.\n"
    )

    report = scan_label_leakage(tmp_path, {"research_report"})

    assert report["leakage_count"] == 0
    assert report["passed"] is True


def test_leakage_ignores_short_labels(tmp_path):
    # Very short labels (<4 chars) are too noisy to scan for.
    (tmp_path / "RESEARCH_REPORT.md").write_text("the cat sat on id")
    report = scan_label_leakage(tmp_path, {"id", "t1"})
    assert report["leakage_count"] == 0
    assert report["labels_scanned"] == []


def test_reports_are_json_serializable(tmp_path):
    (tmp_path / "RESEARCH_REPORT.md").write_text("clean")
    rescue = scan_human_rescue([{"event": "x"}])
    leak = scan_label_leakage(tmp_path, {"task_alpha"})
    # Scorers json-load these; ensure they round-trip.
    assert json.loads(json.dumps(rescue))["passed"] is True
    assert json.loads(json.dumps(leak))["leakage_count"] == 0
