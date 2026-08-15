"""A closure the report cannot read must fail, not warn.

The CP126 ledger counts a finding closed when its status is one the reader
recognises. A row carrying an unrecognised status is excluded from that count
and reported as still open — with its commit, its note and its passing test
sitting right there in the file. `status` printed a warning and exited zero, so
`remediated_scope_named` sat uncounted, and eleven similar rows sat uncounted
for three days before it.

Warning about a defect and then reporting success is the shape this campaign
exists to close, so the ledger's own verifier now fails on it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "closeout" / "semantic_remediation_ledger.py"
LEDGER = (
    ROOT / "artifacts" / "closeout" / "semantic_review" / "cp126" / "REMEDIATION_LEDGER.jsonl"
)


def _run_verify(ledger_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), "verify", "--ledger", str(ledger_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=300,
    )


def test_the_live_ledger_has_no_unreadable_rows():
    result = _run_verify(LEDGER)
    assert "unreadable status    : 0" in result.stdout, result.stdout
    assert result.returncode == 0, result.stdout + result.stderr


def test_verify_fails_on_a_status_no_reader_recognises(tmp_path):
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    assert rows, "the ledger is empty; this test proves nothing"

    doctored = dict(rows[0])
    doctored["status"] = "remediated_scope_named"
    target = tmp_path / "ledger.jsonl"
    target.write_text(json.dumps(doctored) + "\n")

    result = _run_verify(target)
    assert result.returncode != 0, "an uncountable closure exited zero"
    assert "unreadable status    : 1" in result.stdout, result.stdout
    assert doctored["finding_id"] in result.stdout


def test_record_refuses_the_status_in_the_first_place():
    result = subprocess.run(
        [
            sys.executable, str(TOOL), "record",
            "--finding", "semantic-afa4ee984634c351",
            "--status", "remediated_scope_named",
            "--note", "n",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=300,
    )
    assert result.returncode == 2
    assert "--status must be one of" in result.stderr
