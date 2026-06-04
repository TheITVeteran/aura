from __future__ import annotations

import json

import pytest

from tools.closeout.run_codebase_closeout_audit import audit_file, build_closeout_audit
from tools.closeout.semantic_review_ledger import (
    append_entries,
    build_arg_parser,
    build_review_entry,
    main as semantic_review_main,
    record_reviews_from_args,
    summarize_semantic_reviews,
)


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
    assert "SEMANTIC_REVIEW_STATUS.json" in manifest["files"]

    checkpoint = json.loads((out / "CLOSEOUT_CHECKPOINT.json").read_text(encoding="utf-8"))
    assert checkpoint["semantic_review"]["schema"] == "aura.closeout.semantic_review_status.v1"
    assert checkpoint["semantic_review"]["full_semantic_review_current"] is False


def test_semantic_review_ledger_marks_current_and_stale_reviews(tmp_path):
    sample = tmp_path / "reviewed.py"
    sample.write_text("x = 1\ny = 2\n", encoding="utf-8")
    ledger = tmp_path / "SEMANTIC_REVIEW_LEDGER.jsonl"

    entry = build_review_entry(
        sample,
        reviewer="codex",
        checkpoint_id="unit-checkpoint",
        note="reviewed control-flow and write behavior",
        tests=["python -m pytest tests/test_closeout_audit.py -q"],
        root=tmp_path,
    )
    append_entries(ledger, [entry])

    current = summarize_semantic_reviews(ledger_path=ledger, tracked_paths=[sample], root=tmp_path)
    assert current["full_semantic_review_current"] is True
    assert current["semantic_reviewed_line_count"] == 2
    assert current["stale_review_count"] == 0

    sample.write_text("x = 1\ny = 3\n", encoding="utf-8")
    stale = summarize_semantic_reviews(ledger_path=ledger, tracked_paths=[sample], root=tmp_path)
    assert stale["full_semantic_review_current"] is False
    assert stale["semantic_reviewed_line_count"] == 0
    assert stale["stale_review_count"] == 1
    assert stale["stale_reviews"][0]["reason"] == "file_hash_changed"


def test_semantic_review_ledger_marks_re_reviewed_stale_receipts_superseded(tmp_path):
    sample = tmp_path / "reviewed.py"
    sample.write_text("x = 1\ny = 2\n", encoding="utf-8")
    ledger = tmp_path / "SEMANTIC_REVIEW_LEDGER.jsonl"
    first = build_review_entry(sample, reviewer="codex", checkpoint_id="first", root=tmp_path)
    append_entries(ledger, [first])

    sample.write_text("x = 1\ny = 3\n", encoding="utf-8")
    second = build_review_entry(sample, reviewer="codex", checkpoint_id="second", root=tmp_path)
    append_entries(ledger, [second])

    summary = summarize_semantic_reviews(ledger_path=ledger, tracked_paths=[sample], root=tmp_path)
    assert summary["full_semantic_review_current"] is True
    assert summary["stale_review_count"] == 0
    assert summary["superseded_stale_review_count"] == 1


def test_semantic_review_ledger_marks_moved_reviews_superseded(tmp_path):
    old = tmp_path / "scripts" / "legacy_probe.py"
    new = tmp_path / "archive" / "legacy_probe.py"
    old.parent.mkdir()
    old.write_text("print('legacy')\n", encoding="utf-8")
    ledger = tmp_path / "SEMANTIC_REVIEW_LEDGER.jsonl"
    append_entries(ledger, [build_review_entry(old, reviewer="codex", checkpoint_id="old", root=tmp_path)])

    new.parent.mkdir()
    old.rename(new)
    append_entries(ledger, [build_review_entry(new, reviewer="codex", checkpoint_id="new", root=tmp_path)])

    summary = summarize_semantic_reviews(ledger_path=ledger, tracked_paths=[new], root=tmp_path)

    assert summary["full_semantic_review_current"] is True
    assert summary["orphan_review_count"] == 0
    assert summary["superseded_orphan_review_count"] == 1


def test_semantic_review_ledger_merges_reviewed_spans(tmp_path):
    sample = tmp_path / "spans.py"
    sample.write_text("a\nb\nc\nd\n", encoding="utf-8")
    ledger = tmp_path / "SEMANTIC_REVIEW_LEDGER.jsonl"
    append_entries(
        ledger,
        [
            build_review_entry(
                sample,
                reviewer="codex",
                checkpoint_id="span-1",
                first_line=1,
                last_line=2,
                root=tmp_path,
            ),
            build_review_entry(
                sample,
                reviewer="codex",
                checkpoint_id="span-2",
                first_line=3,
                last_line=4,
                root=tmp_path,
            ),
        ],
    )

    summary = summarize_semantic_reviews(ledger_path=ledger, tracked_paths=[sample], root=tmp_path)
    assert summary["full_semantic_review_current"] is True
    assert summary["reviewed_files"]["spans.py"]["spans"] == [{"first_line": 1, "last_line": 4}]


def test_semantic_review_record_requires_explicit_scope(tmp_path):
    args = build_arg_parser().parse_args(["record", "--ledger", str(tmp_path / "ledger.jsonl")])

    with pytest.raises(ValueError, match="requires explicit paths"):
        record_reviews_from_args(args)


def test_semantic_review_status_exits_cleanly_when_downstream_pipe_closes(tmp_path, monkeypatch):
    class BrokenStdout:
        def __init__(self):
            self.closed = False
            self.write_attempts = 0

        def write(self, _text):
            self.write_attempts += 1
            raise BrokenPipeError("downstream closed")

        def flush(self):
            return None

        def close(self):
            self.closed = True

    stdout = BrokenStdout()
    monkeypatch.setattr("tools.closeout.semantic_review_ledger.sys.stdout", stdout)

    assert semantic_review_main(["status", "--ledger", str(tmp_path / "ledger.jsonl")]) == 0
    assert stdout.write_attempts == 1
    assert stdout.closed is True


def test_semantic_review_uses_closeout_text_classification(tmp_path):
    sample = tmp_path / "invalid_utf8.txt"
    sample.write_bytes(b"valid line\n\xff\n")
    ledger = tmp_path / "SEMANTIC_REVIEW_LEDGER.jsonl"

    summary = summarize_semantic_reviews(ledger_path=ledger, tracked_paths=[sample], root=tmp_path)

    assert summary["tracked_text_file_count"] == 1
    assert summary["tracked_text_line_count"] == 2
