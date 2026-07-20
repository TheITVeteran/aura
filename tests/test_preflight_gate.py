"""Contracts for the release-preflight checklist (tools/release_preflight.py).

The checklist IS the procedure: these pins make sure no gate can be quietly
dropped or reordered, failures fail the flight (and skip later checks unless
--keep-going), and every run leaves a machine-readable receipt naming both
what ran and what was deliberately deferred.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import release_preflight as preflight  # noqa: E402

pytestmark = pytest.mark.unit


def _scripted_check(name: str, exit_code: int) -> preflight.PreflightCheck:
    return preflight.PreflightCheck(
        name=name,
        purpose=f"scripted exit for {name}",
        command=(sys.executable, "-c", f"raise SystemExit({exit_code})"),
        timeout_s=30.0,
    )


class TestChecklistPin:
    def test_the_pinned_checklist_names_and_order(self):
        """Removing or reordering a gate is a deliberate act — change this
        pin in the same commit that changes the checklist, with the why."""
        names = [check.name for check in preflight.default_checks()]
        assert names == [
            "compile",
            "lint",
            "smoke",
            "governance_lint",
            "security_scan",
            "enterprise_gate",
            "production_gate",
            "fresh_hard_deaths",
            # CP-scope control plane (SCOPE-001): registry/tracker coherence,
            # zero-unmapped source coverage, and the defect-fingerprint
            # ratchet are release-checklist items from 2026-07-18 onward.
            "reqproof_structural",
            # CP208 adds acceptance-granular completion and total-checkpoint
            # accounting as an independently recomputed release input.
            "reqproof_progress",
        ]

    def test_deferred_gates_are_named_with_reasons(self):
        assert set(preflight.DEFERRED_GATES) == {
            "full_suite_6_chunks",
            "startup_budget_probe",
            "endurance_soak",
        }
        for reason in preflight.DEFERRED_GATES.values():
            assert len(reason) > 10

    def test_triage_check_honors_exit_code(self):
        """make triage swallows failures with '|| true'; the preflight
        variant must NOT — fresh hard deaths ground the flight."""
        triage = next(
            check for check in preflight.default_checks()
            if check.name == "fresh_hard_deaths"
        )
        assert "|| true" not in " ".join(triage.command)
        assert "--window-days" in triage.command


class TestRunner:
    def test_all_pass_yields_pass_verdict(self):
        report = preflight.run_preflight(
            checks=(_scripted_check("a", 0), _scripted_check("b", 0))
        )
        assert report.verdict == "PASS"
        assert [result.status for result in report.results] == ["pass", "pass"]
        assert all(result.duration_s >= 0.0 for result in report.results)

    def test_failure_fails_fast_and_skips_the_rest(self):
        report = preflight.run_preflight(
            checks=(_scripted_check("a", 0), _scripted_check("b", 3), _scripted_check("c", 0))
        )
        assert report.verdict == "FAIL"
        assert [result.status for result in report.results] == ["pass", "fail", "skipped"]
        assert report.results[1].exit_code == 3

    def test_keep_going_runs_everything(self):
        report = preflight.run_preflight(
            checks=(_scripted_check("a", 1), _scripted_check("b", 0)),
            keep_going=True,
        )
        assert report.verdict == "FAIL"
        assert [result.status for result in report.results] == ["fail", "pass"]

    def test_failed_check_captures_output_tail(self):
        check = preflight.PreflightCheck(
            name="noisy",
            purpose="scripted exit",
            command=(
                sys.executable, "-c",
                "print('line1'); print('the-smoking-gun'); raise SystemExit(1)",
            ),
            timeout_s=30.0,
        )
        report = preflight.run_preflight(checks=(check,))
        assert "the-smoking-gun" in report.results[0].tail

    def test_receipt_schema_and_render(self, tmp_path):
        report = preflight.run_preflight(checks=(_scripted_check("a", 0),))
        receipt_path = tmp_path / "preflight.json"
        preflight._write_receipt(report, receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["schema"] == "aura.preflight.v1"
        assert receipt["verdict"] == "PASS"
        assert receipt["checks"][0]["name"] == "a"
        assert receipt["deferred_gates"]
        rendered = preflight._render(report)
        assert "PASS" in rendered and "deferred (deliberate)" in rendered

    def test_timeout_is_a_failure_not_a_hang(self):
        check = preflight.PreflightCheck(
            name="hang",
            purpose="scripted exit",
            command=(sys.executable, "-c", "import time; time.sleep(30)"),
            timeout_s=1.0,
        )
        report = preflight.run_preflight(checks=(check,))
        assert report.verdict == "FAIL"
        assert "timed out" in report.results[0].tail
