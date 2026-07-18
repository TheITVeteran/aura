from __future__ import annotations

import gzip
import json

import pytest

from tools.closeout.run_codebase_closeout_audit import audit_file, build_closeout_audit
from tools.closeout.semantic_review_ledger import (
    SEMANTIC_CAMPAIGN_SCHEMA,
    append_entries,
    build_arg_parser,
    build_review_entry,
    build_semantic_inventory_batch,
    build_semantic_inventory_entries,
    build_semantic_review_campaign,
    build_semantic_review_queue,
    carry_semantic_inventory,
    main as semantic_review_main,
    record_semantic_inventory_batch,
    record_reviews_from_args,
    semantic_campaign_receipt,
    summarize_semantic_inventory,
    summarize_semantic_reviews,
    validate_semantic_review_campaign,
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


def test_semantic_review_status_reports_code_coverage_and_unreviewed_queue(tmp_path):
    reviewed = tmp_path / "reviewed.py"
    reviewed.write_text("x = 1\n", encoding="utf-8")
    code = tmp_path / "large.py"
    code.write_text("a\nb\nc\n", encoding="utf-8")
    data = tmp_path / "data.json"
    data.write_text('{"ok": true}\n' * 5, encoding="utf-8")
    ledger = tmp_path / "SEMANTIC_REVIEW_LEDGER.jsonl"
    append_entries(
        ledger,
        [build_review_entry(reviewed, reviewer="codex", checkpoint_id="reviewed", root=tmp_path)],
    )

    summary = summarize_semantic_reviews(
        ledger_path=ledger,
        tracked_paths=[reviewed, code, data],
        root=tmp_path,
    )
    queue = build_semantic_review_queue(
        ledger_path=ledger,
        tracked_paths=[reviewed, code, data],
        root=tmp_path,
        code_only=True,
        limit=5,
    )

    assert summary["tracked_text_file_count"] == 3
    assert summary["tracked_code_file_count"] == 2
    assert summary["fully_reviewed_code_file_count"] == 1
    assert summary["unreviewed_file_count"] == 2
    assert summary["unreviewed_code_file_count"] == 1
    assert summary["unreviewed_files"][0]["file"] == "large.py"
    assert summary["unreviewed_files"][0]["code"] is True
    assert queue["files"] == [summary["unreviewed_files"][0]]


def test_semantic_review_excludes_its_mutating_ledger_from_source_coverage(tmp_path):
    source = tmp_path / "core.py"
    source.write_text("value = 1\n", encoding="utf-8")
    ledger = tmp_path / "SEMANTIC_REVIEW_LEDGER.jsonl"
    ledger.write_text("", encoding="utf-8")

    summary = summarize_semantic_reviews(
        ledger_path=ledger,
        tracked_paths=[source, ledger],
        root=tmp_path,
    )

    assert summary["tracked_text_file_count"] == 1
    assert summary["excluded_mutable_evidence_file_count"] == 1
    assert summary["excluded_mutable_evidence_files"] == [
        "SEMANTIC_REVIEW_LEDGER.jsonl"
    ]


def test_semantic_campaign_freezes_every_missing_span_before_remediation(tmp_path):
    reviewed = tmp_path / "core" / "reviewed.py"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_text("a\nb\nc\nd\n", encoding="utf-8")
    sibling = tmp_path / "core" / "sibling.py"
    sibling.write_text("e\nf\ng\n", encoding="utf-8")
    note = tmp_path / "docs" / "note.md"
    note.parent.mkdir(parents=True)
    note.write_text("h\ni\n", encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    append_entries(
        ledger,
        [
            build_review_entry(
                reviewed,
                reviewer="codex",
                checkpoint_id="partial",
                first_line=1,
                last_line=2,
                root=tmp_path,
            )
        ],
    )

    campaign = build_semantic_review_campaign(
        ledger_path=ledger,
        tracked_paths=[reviewed, sibling, note],
        root=tmp_path,
        batch_line_budget=3,
        max_span_lines=2,
        source_commit="frozen-commit",
        source_clean=True,
    )

    assert campaign["schema"] == SEMANTIC_CAMPAIGN_SCHEMA
    assert campaign["source_commit"] == "frozen-commit"
    assert campaign["edits_permitted_before_inventory_complete"] is False
    assert campaign["planned_file_count"] == 3
    assert campaign["planned_line_count"] == 7
    assert campaign["planned_span_count"] == 4
    assert campaign["batch_count"] == 3
    assert [batch["subsystem"] for batch in campaign["batches"]] == [
        "core",
        "core",
        "docs",
    ]
    spans = [span for batch in campaign["batches"] for span in batch["spans"]]
    reviewed_spans = [
        span for span in spans if span["file"] == "core/reviewed.py"
    ]
    assert [(span["first_line"], span["last_line"]) for span in reviewed_spans] == [
        (3, 4)
    ]
    assert all(batch["line_count"] <= 3 for batch in campaign["batches"])
    assert validate_semantic_review_campaign(
        campaign,
        root=tmp_path,
        source_commit="frozen-commit",
    )["passed"] is True


def test_semantic_campaign_validation_detects_source_drift(tmp_path):
    source = tmp_path / "core" / "service.py"
    source.parent.mkdir(parents=True)
    source.write_text("stable\n", encoding="utf-8")
    campaign = build_semantic_review_campaign(
        ledger_path=tmp_path / "ledger.jsonl",
        tracked_paths=[source],
        root=tmp_path,
        source_commit="before",
        source_clean=True,
    )

    source.write_text("changed\n", encoding="utf-8")
    validation = validate_semantic_review_campaign(
        campaign,
        root=tmp_path,
        source_commit="after",
    )

    assert validation["passed"] is False
    assert validation["issues"] == [
        "file_hash_changed:core/service.py",
        "planned_line_count_mismatch",
        "source_commit_changed",
    ]


def test_semantic_campaign_validation_rejects_structural_tampering(tmp_path):
    source = tmp_path / "core" / "service.py"
    source.parent.mkdir(parents=True)
    source.write_text("one\ntwo\n", encoding="utf-8")
    campaign = build_semantic_review_campaign(
        ledger_path=tmp_path / "ledger.jsonl",
        tracked_paths=[source],
        root=tmp_path,
        batch_line_budget=2,
        max_span_lines=2,
        source_commit="frozen",
        source_clean=True,
    )
    campaign["batches"][0]["line_count"] = 1
    campaign["planned_file_count"] = 2

    validation = validate_semantic_review_campaign(
        campaign,
        root=tmp_path,
        source_commit="frozen",
    )

    assert validation["passed"] is False
    assert "campaign_hash_mismatch" in validation["issues"]
    assert "batch_line_count_mismatch:semantic-batch-0001" in validation["issues"]
    assert "planned_file_count_mismatch" in validation["issues"]


def test_semantic_campaign_receipt_is_bounded(tmp_path):
    campaign = {
        "campaign_sha256": "abc123",
        "source_commit": "frozen",
        "source_clean": True,
        "planned_file_count": 10,
        "planned_span_count": 12,
        "planned_line_count": 9000,
        "batch_count": 3,
        "batches": [{"large": "payload"}],
    }

    receipt = semantic_campaign_receipt(
        campaign,
        output_path=tmp_path / "campaign.json",
    )

    assert receipt["schema"] == "aura.closeout.semantic_review_campaign_receipt.v1"
    assert receipt["campaign_sha256"] == "abc123"
    assert receipt["batch_count"] == 3
    assert "batches" not in receipt


def test_semantic_inventory_records_complete_hash_bound_batch(tmp_path):
    source = tmp_path / "core" / "service.py"
    source.parent.mkdir(parents=True)
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")
    campaign = build_semantic_review_campaign(
        ledger_path=tmp_path / "review-ledger.jsonl",
        tracked_paths=[source],
        root=tmp_path,
        batch_line_budget=3,
        max_span_lines=2,
        source_commit="frozen",
        source_clean=True,
    )
    batch_id = campaign["batches"][0]["batch_id"]
    materialized = build_semantic_inventory_batch(
        campaign,
        batch_id=batch_id,
        root=tmp_path,
        source_commit="frozen",
    )
    assert [span["content"] for span in materialized["spans"]] == [
        "one\ntwo",
        "three",
    ]
    submission = {
        "campaign_sha256": campaign["campaign_sha256"],
        "batch_id": batch_id,
        "reviews": [
            {
                "file": "core/service.py",
                "first_line": 1,
                "last_line": 2,
                "verdict": "finding",
                "summary": "The first span has one actionable correctness defect.",
                "findings": [
                    {
                        "severity": "high",
                        "category": "correctness",
                        "title": "Unvalidated state transition",
                        "description": "The transition is represented without validation.",
                        "repair_group": "core-state-validation",
                        "evidence_lines": [2],
                    }
                ],
                "dependencies": [
                    {"file": "core/model.py", "reason": "Defines the state contract."}
                ],
                "recommended_tests": ["pytest -q tests/test_state.py"],
            },
            {
                "file": "core/service.py",
                "first_line": 3,
                "last_line": 3,
                "verdict": "clean",
                "summary": "The final line is internally consistent and needs no repair.",
                "findings": [],
            },
        ],
    }
    inventory_path = tmp_path / "inventory.jsonl"

    status = record_semantic_inventory_batch(
        campaign,
        batch_id=batch_id,
        submission=submission,
        inventory_path=inventory_path,
        reviewer="codex",
        root=tmp_path,
        source_commit="frozen",
    )

    assert status["inventory_complete"] is True
    assert status["edits_permitted"] is True
    assert status["reviewed_span_count"] == 2
    assert status["reviewed_line_count"] == 3
    assert status["finding_count"] == 1
    assert status["finding_severity_counts"]["high"] == 1


def test_semantic_inventory_rejects_missing_span_note(tmp_path):
    source = tmp_path / "service.py"
    source.write_text("one\ntwo\n", encoding="utf-8")
    campaign = build_semantic_review_campaign(
        ledger_path=tmp_path / "review-ledger.jsonl",
        tracked_paths=[source],
        root=tmp_path,
        batch_line_budget=2,
        max_span_lines=1,
        source_commit="frozen",
        source_clean=True,
    )
    batch_id = campaign["batches"][0]["batch_id"]
    submission = {
        "campaign_sha256": campaign["campaign_sha256"],
        "batch_id": batch_id,
        "reviews": [
            {
                "file": "service.py",
                "first_line": 1,
                "last_line": 1,
                "verdict": "clean",
                "summary": "The first line has no actionable defect.",
            }
        ],
    }

    with pytest.raises(ValueError, match="missing 1 span"):
        build_semantic_inventory_entries(
            campaign,
            batch_id=batch_id,
            submission=submission,
            reviewer="codex",
        )


def test_semantic_inventory_detects_note_tampering(tmp_path):
    source = tmp_path / "service.py"
    source.write_text("stable\n", encoding="utf-8")
    campaign = build_semantic_review_campaign(
        ledger_path=tmp_path / "review-ledger.jsonl",
        tracked_paths=[source],
        root=tmp_path,
        source_commit="frozen",
        source_clean=True,
    )
    batch_id = campaign["batches"][0]["batch_id"]
    submission = {
        "campaign_sha256": campaign["campaign_sha256"],
        "batch_id": batch_id,
        "reviews": [
            {
                "file": "service.py",
                "first_line": 1,
                "last_line": 1,
                "verdict": "clean",
                "summary": "The line is stable and contains no actionable defect.",
            }
        ],
    }
    inventory_path = tmp_path / "inventory.jsonl"
    record_semantic_inventory_batch(
        campaign,
        batch_id=batch_id,
        submission=submission,
        inventory_path=inventory_path,
        reviewer="codex",
        root=tmp_path,
        source_commit="frozen",
    )
    entry = json.loads(inventory_path.read_text(encoding="utf-8"))
    entry["summary"] = "tampered after review"
    inventory_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    status = summarize_semantic_inventory(
        campaign,
        inventory_path=inventory_path,
        root=tmp_path,
        source_commit="frozen",
    )

    assert status["inventory_complete"] is False
    assert status["reviewed_span_count"] == 0
    assert "inventory_line_1:entry_hash_mismatch" in status["issues"]


def test_semantic_inventory_carries_only_hash_identical_spans_and_records_pending(
    tmp_path,
):
    first = tmp_path / "core" / "first.py"
    first.parent.mkdir(parents=True)
    first.write_text("stable = 1\n", encoding="utf-8")
    second = tmp_path / "core" / "second.py"
    second.write_text("version = 1\n", encoding="utf-8")
    old_campaign = build_semantic_review_campaign(
        ledger_path=tmp_path / "review-ledger.jsonl",
        tracked_paths=[first, second],
        root=tmp_path,
        batch_line_budget=10,
        max_span_lines=10,
        source_commit="old-commit",
        source_clean=True,
    )
    batch_id = old_campaign["batches"][0]["batch_id"]
    old_submission = {
        "campaign_sha256": old_campaign["campaign_sha256"],
        "batch_id": batch_id,
        "reviews": [
            {
                "file": span["file"],
                "first_line": span["first_line"],
                "last_line": span["last_line"],
                "verdict": "clean",
                "summary": f"The exact {span['file']} span has no actionable defect.",
                "findings": [],
            }
            for span in old_campaign["batches"][0]["spans"]
        ],
    }
    old_inventory = tmp_path / "old-inventory.jsonl"
    record_semantic_inventory_batch(
        old_campaign,
        batch_id=batch_id,
        submission=old_submission,
        inventory_path=old_inventory,
        reviewer="codex",
        root=tmp_path,
        source_commit="old-commit",
    )
    old_inventory_gz = tmp_path / "old-inventory.jsonl.gz"
    old_inventory_gz.write_bytes(gzip.compress(old_inventory.read_bytes(), mtime=0))

    second.write_text("version = 2\n", encoding="utf-8")
    new_campaign = build_semantic_review_campaign(
        ledger_path=tmp_path / "review-ledger.jsonl",
        tracked_paths=[first, second],
        root=tmp_path,
        batch_line_budget=10,
        max_span_lines=10,
        source_commit="new-commit",
        source_clean=True,
    )
    new_inventory = tmp_path / "new-inventory.jsonl"
    receipt = carry_semantic_inventory(
        old_campaign,
        old_inventory_path=old_inventory_gz,
        new_campaign=new_campaign,
        output_path=new_inventory,
        root=tmp_path,
        source_commit="new-commit",
    )

    assert receipt["old_reviewed_span_count"] == 2
    assert receipt["carried_span_count"] == 1
    assert receipt["changed_span_count"] == 1
    assert receipt["pending_span_count"] == 1
    carried = json.loads(new_inventory.read_text(encoding="utf-8"))
    assert carried["file"] == "core/first.py"
    assert carried["carried_from"]["source_commit"] == "old-commit"

    new_batch_id = new_campaign["batches"][0]["batch_id"]
    pending = build_semantic_inventory_batch(
        new_campaign,
        batch_id=new_batch_id,
        inventory_path=new_inventory,
        pending_only=True,
        root=tmp_path,
        source_commit="new-commit",
    )
    assert pending["pending_only"] is True
    assert pending["span_count"] == 1
    assert pending["spans"][0]["file"] == "core/second.py"
    pending_submission = {
        "campaign_sha256": new_campaign["campaign_sha256"],
        "batch_id": new_batch_id,
        "reviews": [
            {
                "file": "core/second.py",
                "first_line": 1,
                "last_line": 1,
                "verdict": "clean",
                "summary": "The changed second-file span has been reviewed again.",
                "findings": [],
            }
        ],
    }
    status = record_semantic_inventory_batch(
        new_campaign,
        batch_id=new_batch_id,
        submission=pending_submission,
        inventory_path=new_inventory,
        reviewer="codex",
        pending_only=True,
        root=tmp_path,
        source_commit="new-commit",
    )
    assert status["inventory_complete"] is True
    assert status["edits_permitted"] is True


def test_semantic_inventory_carry_rejects_tampered_archived_entry(tmp_path):
    source = tmp_path / "service.py"
    source.write_text("stable\n", encoding="utf-8")
    campaign = build_semantic_review_campaign(
        ledger_path=tmp_path / "review-ledger.jsonl",
        tracked_paths=[source],
        root=tmp_path,
        source_commit="frozen",
        source_clean=True,
    )
    batch_id = campaign["batches"][0]["batch_id"]
    inventory = tmp_path / "inventory.jsonl"
    record_semantic_inventory_batch(
        campaign,
        batch_id=batch_id,
        submission={
            "campaign_sha256": campaign["campaign_sha256"],
            "batch_id": batch_id,
            "reviews": [
                {
                    "file": "service.py",
                    "first_line": 1,
                    "last_line": 1,
                    "verdict": "clean",
                    "summary": "The stable line contains no actionable defect.",
                    "findings": [],
                }
            ],
        },
        inventory_path=inventory,
        reviewer="codex",
        root=tmp_path,
        source_commit="frozen",
    )
    entry = json.loads(inventory.read_text(encoding="utf-8"))
    entry["summary"] = "tampered after archival"
    inventory.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="entry_hash_mismatch"):
        carry_semantic_inventory(
            campaign,
            old_inventory_path=inventory,
            new_campaign=campaign,
            output_path=tmp_path / "new-inventory.jsonl",
            root=tmp_path,
            source_commit="frozen",
        )
