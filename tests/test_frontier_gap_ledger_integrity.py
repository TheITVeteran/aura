"""CP126: ledger integrity and measurement hygiene for the frontier gap battery.

The gap ledger is the durable trend record a capability claim rests on, and
run_battery is the measurement that feeds it. Both fail by overstating, not
by crashing.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from core.brain.frontier_gap import (
    BATTERY_VERSION,
    CAPABILITY_EVIDENCE_CLASS,
    CONTROL_EVIDENCE_CLASS,
    MAX_PER_CLASS,
    SCHEMA_VERSION,
    ClassResult,
    GapLedger,
    build_battery,
    run_battery,
)


# ── ledger anti-rollback ───────────────────────────────────────────────────


def _control_ledger_dict(runs: list, pruned_count: int, pruned_through: str | None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": CONTROL_EVIDENCE_CLASS,
        "capability_claim_eligible": False,
        "retention": {
            "max_entries": 8,
            "pruned_count": pruned_count,
            "pruned_through_sha256": pruned_through,
            "retains_outputs_in_content_addressed_blobs": True,
        },
        "head_entry_sha256": pruned_through,
        "runs": runs,
        "trend": {},
    }


def test_ledger_refuses_deleted_history_behind_a_prune_anchor():
    """An empty run list beside a positive prune count is erased history.

    from_dict previously accepted it: an attacker could drop every entry and
    present any syntactically valid anchor as if it summarized them.
    """
    payload = _control_ledger_dict([], pruned_count=5, pruned_through="a" * 64)

    with pytest.raises(ValueError, match="retains no entries"):
        GapLedger.from_dict(payload, evidence_class=CONTROL_EVIDENCE_CLASS)


def test_ledger_still_accepts_a_fresh_empty_ledger():
    """A never-pruned ledger legitimately has no entries."""
    fresh = GapLedger(
        evidence_class=CONTROL_EVIDENCE_CLASS, capability_claim_eligible=False
    )
    payload = fresh.to_dict()
    assert payload["runs"] == []

    ledger = GapLedger.from_dict(payload, evidence_class=CONTROL_EVIDENCE_CLASS)

    assert ledger.runs == []
    assert ledger.pruned_count == 0


# ── ledger concurrency ─────────────────────────────────────────────────────


def _control_report(index: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "battery_version": BATTERY_VERSION,
        "evidence_class": CONTROL_EVIDENCE_CLASS,
        "capability_claim_eligible": False,
        "generated_at_unix": 1_700_000_000.0 + index,
        "overall_candidate_score": 0.5,
        "effective_n": 4,
    }


def test_concurrent_adds_keep_the_hash_chain_intact():
    """add() reads the previous head, writes a blob, appends, then prunes.

    Without a lock two threads can read the SAME head and append against it,
    which silently breaks the chain the ledger's integrity rests on.
    """
    ledger = GapLedger(evidence_class=CONTROL_EVIDENCE_CLASS, capability_claim_eligible=False)
    blobs: dict[str, dict] = {}
    blob_lock = threading.Lock()

    def writer(digest: str, snapshot: dict) -> None:
        with blob_lock:
            blobs[digest] = snapshot

    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            ledger.add(_control_report(index), evidence_blob_writer=writer)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not errors, errors
    assert len(ledger.runs) == 8
    # Every entry must chain to its predecessor, in order.
    previous = None
    for entry in ledger.runs:
        assert entry["previous_entry_sha256"] == previous
        previous = entry["entry_sha256"]
    # Every indexed entry must have a durable blob behind it.
    for entry in ledger.runs:
        assert entry["evidence_sha256"] in blobs


def test_blob_is_written_before_the_entry_is_indexed():
    """A failed blob write must not leave an index entry pointing at nothing."""
    ledger = GapLedger(evidence_class=CONTROL_EVIDENCE_CLASS, capability_claim_eligible=False)

    def failing_writer(_digest: str, _snapshot: dict) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError):
        ledger.add(_control_report(0), evidence_blob_writer=failing_writer)

    assert ledger.runs == [], "no entry may be indexed when its evidence was not stored"


# ── gap metric honesty ─────────────────────────────────────────────────────


def test_parity_and_superiority_are_distinguishable():
    """The clamped gap reads 0.0 for both; relative_position must not."""
    parity = ClassResult("math", n=10, candidate_correct=5, reference_score=0.5)
    superior = ClassResult("math", n=10, candidate_correct=9, reference_score=0.5)

    assert parity.gap == 0.0
    assert superior.gap == 0.0
    assert parity.relative_position == pytest.approx(0.0)
    assert superior.relative_position > 0.0


def test_zero_reference_is_reported_as_uninformative():
    """A reference that solved nothing gives no baseline to trail."""
    cell = ClassResult("math", n=10, candidate_correct=0, reference_score=0.0)

    assert cell.gap == 0.0
    assert cell.reference_uninformative is True
    payload = cell.to_dict()
    assert payload["reference_uninformative"] is True


def test_absent_reference_stays_unmeasured():
    cell = ClassResult("math", n=10, candidate_correct=7, reference_score=None)

    assert cell.gap is None
    assert cell.relative_position is None
    assert cell.reference_uninformative is False


# ── battery input contract ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("seed", "per_class"),
    [
        (1, 0),
        (1, -3),
        (1, True),               # bool is an int subclass
        (1, 2.5),
        (1, MAX_PER_CLASS + 1),
        (True, 4),
        ("seed", 4),
    ],
)
def test_build_battery_rejects_invalid_dimensions(seed, per_class):
    with pytest.raises(ValueError):
        build_battery(seed=seed, per_class=per_class)


def test_build_battery_accepts_a_valid_request():
    items = build_battery(seed=7, per_class=2)
    assert items
    assert len({item.item_id for item in items}) == len(items)


# ── measurement must not mutate what it measures ───────────────────────────


def test_run_battery_does_not_grade_to_the_foundry_by_default():
    """Writing a verdict per item during a benchmark contaminates future
    verifier selection with benchmark cases — an unreceipted side effect of
    merely taking a measurement."""
    import inspect

    signature = inspect.signature(run_battery)
    assert signature.parameters["grade_to_foundry"].default is False


def test_solver_exceptions_do_not_leak_detail_into_retained_evidence():
    """execution_error is retained in content-addressed blobs that outlive the
    run, so it must not carry backend paths, prompts, or credentials."""

    async def exploding_solver(_prompt: str, _task_type: str) -> str:
        raise RuntimeError("/Users/secret/path leaked token=abc123 prompt=<private>")

    report = asyncio.run(run_battery(exploding_solver, seed=3, per_class=1))

    blob = repr(report)
    assert "secret" not in blob
    assert "abc123" not in blob
    for item in report["items"]:
        if item.get("execution_error"):
            assert item["execution_error"] == "RuntimeError"


def test_capability_evidence_class_constant_is_stable():
    assert CAPABILITY_EVIDENCE_CLASS
