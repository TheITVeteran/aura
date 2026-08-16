"""A closure status the reader does not recognise silently reopens the work.

Twice now the CP126 ledger has carried a status no reader knew. Thirteen
rows the first time (``verified_already_remediated``), four the second
(``remediated_scope_named``). Both times the work was real, the note was
detailed, and the finding counted as OPEN — visible only to whoever
happened to run ``verify`` and read past the drift summary.

Counting an unknown status as open is the right default: a reader that
guessed would close findings on a typo. The defect is that nothing failed
at the moment the row was written, so the loss sat in the ledger for
however long it took someone to look.

This is that check. It runs with the suite, not on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "artifacts/closeout/semantic_review/cp126/REMEDIATION_LEDGER.jsonl"


def _valid_statuses() -> set[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "semantic_remediation_ledger",
        REPO_ROOT / "tools/closeout/semantic_remediation_ledger.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.VALID_STATUSES)


def _rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def _effective_rows() -> dict[str, dict]:
    """Last row per finding wins, which is how every reader treats it.

    The ledger is append-only, so a corrected closure is a NEW row rather
    than an edit. Checking every historical row would fail forever on the
    rows a correction exists to supersede.
    """
    latest: dict[str, dict] = {}
    for row in _rows():
        finding = str(row.get("finding_id", ""))
        if finding:
            latest[finding] = row
    return latest


def test_every_recorded_status_is_one_a_reader_knows():
    valid = _valid_statuses()
    unknown: dict[str, list[str]] = {}
    for finding, row in _effective_rows().items():
        status = str(row.get("status", ""))
        if status not in valid:
            unknown.setdefault(status, []).append(finding)
    assert not unknown, (
        "closure rows carry a status no reader recognises, so they count as "
        f"OPEN and the work is lost: { {k: v[:3] for k, v in unknown.items()} }. "
        "Either add the status to VALID_STATUSES with a comment saying what it "
        "asserts, or re-record the rows under an existing one."
    )


def test_a_status_that_names_a_residue_carries_one():
    """The residue is the point of `remediated_scope_named`."""
    for row in _effective_rows().values():
        if row.get("status") != "remediated_scope_named":
            continue
        note = str(row.get("note", "")).strip()
        assert note, (
            f"{row.get('finding_id')} claims a named scope and names none"
        )
        assert len(note) > 80, (
            f"{row.get('finding_id')}: a residue named in under eighty characters "
            "is a label, not a scope"
        )


#: Findings whose EFFECTIVE closure row carries no evidence, measured
#: 2026-08-16. Most were written before the tool required the field, by an
#: agent that did not record its own name either. The number may only go
#: down: a closure with nothing to re-run is a claim, and this campaign
#: exists because claims outlived the code that made them true.
EVIDENCE_FREE_ROWS_BASELINE = 345


def test_the_count_of_unevidenced_closures_only_shrinks():
    missing = [
        finding
        for finding, row in _effective_rows().items()
        if not row.get("evidence")
    ]
    assert len(missing) <= EVIDENCE_FREE_ROWS_BASELINE, (
        f"{len(missing)} closure rows carry no evidence, up from "
        f"{EVIDENCE_FREE_ROWS_BASELINE}. A closure with nothing to re-run is a "
        f"claim: {missing[-5:]}"
    )
    if len(missing) < EVIDENCE_FREE_ROWS_BASELINE:
        pytest.fail(
            f"only {len(missing)} rows lack evidence now, down from "
            f"{EVIDENCE_FREE_ROWS_BASELINE} — lower the baseline so the "
            "ratchet holds the gain"
        )
