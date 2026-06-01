from __future__ import annotations

import json

from tools.closeout.run_codebase_closeout_audit import audit_file, build_closeout_audit


def test_audit_file_hashes_every_text_line(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("# comment\nx = 1\n# TODO: review\n", encoding="utf-8")

    ledger = tmp_path / "lines.jsonl"
    result = audit_file(sample, line_ledger_path=ledger)

    assert result.text is True
    assert result.code is True
    assert result.line_count == 3
    assert result.comment_lines == 2
    assert result.todo_markers == 1
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [record["line"] for record in records] == [1, 2, 3]
    assert all(record["sha256"] for record in records)


def test_closeout_audit_writes_checkpoint_bundle(tmp_path):
    out = tmp_path / "closeout"
    summary = build_closeout_audit(
        out_dir=out,
        allow_dirty=True,
        run_gates=False,
        max_files=12,
    )

    assert summary["verdict"] == "PASS"
    assert summary["tracked_file_count"] == 12
    assert summary["line_ledger_entries"] > 0
    assert summary["full_closeout_complete"] is False
    assert "all_issues_fixed" in summary["claim_not_supported"]

    for name in (
        "SOURCE_FILE_LEDGER.jsonl",
        "SOURCE_LINE_LEDGER.jsonl",
        "FINDINGS.json",
        "GATE_RESULTS.json",
        "CLOSEOUT_CHECKPOINT.json",
        "FINAL_VERDICT.txt",
        "MANIFEST.json",
    ):
        assert (out / name).exists(), name

    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "SOURCE_LINE_LEDGER.jsonl" in manifest["files"]
