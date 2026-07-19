from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import (
    ARM_RESULT,
    COMMITTED,
    FAILED,
    STARTED,
    VERIFIED,
    CampaignJournal,
    CampaignJournalError,
    CampaignPlan,
    canonical_json_bytes,
)


def _plan(count: int = 2) -> CampaignPlan:
    return CampaignPlan.build(
        "resident-32b-frontier",
        [
            {
                "domain": "mathematics" if index % 2 == 0 else "coding",
                "seed": 100 + index,
                "task_sha256": f"{index + 1:064x}",
            }
            for index in range(count)
        ],
        metadata={"comparison": "rlc-vs-vanilla", "version": 3},
    )


def _complete(journal: CampaignJournal, cell_id: str) -> str:
    attempt_id = journal.start_cell(cell_id)
    journal.record_arm_result(
        cell_id,
        attempt_id,
        {"control": {"score": 0}, "treatment": {"score": 1}},
    )
    journal.record_verified(cell_id, attempt_id, {"accepted": True})
    journal.commit_cell(cell_id, attempt_id, {"raw_receipt_sha256": "a" * 64})
    return attempt_id


def _assert_code(expected: str, operation) -> None:
    with pytest.raises(CampaignJournalError) as exc_info:
        operation()
    assert exc_info.value.code == expected


def _rewrite_record(path: Path, index: int, mutate) -> None:
    lines = path.read_bytes().splitlines()
    record = json.loads(lines[index])
    mutate(record)
    base = {key: value for key, value in record.items() if key != "event_sha256"}
    import hashlib

    record["event_sha256"] = hashlib.sha256(canonical_json_bytes(base)).hexdigest()
    lines[index] = canonical_json_bytes(record)
    path.write_bytes(b"\n".join(lines) + b"\n")


def _append_validly_hashed_record(path: Path, record: dict) -> None:
    import hashlib

    base = {key: value for key, value in record.items() if key != "event_sha256"}
    record = {**base, "event_sha256": hashlib.sha256(canonical_json_bytes(base)).hexdigest()}
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def test_plan_and_cell_ids_are_deterministic_and_immutable() -> None:
    cells = [{"task": "alpha", "seed": 1}, {"task": "alpha", "seed": 1}]
    metadata = {"nested": {"issuer": "independent"}}
    first = CampaignPlan.build("campaign", cells, metadata=metadata)
    second = CampaignPlan.build("campaign", cells, metadata=metadata)

    assert first.plan_sha256 == second.plan_sha256
    assert first.cell_ids == second.cell_ids
    assert first.cell_ids[0] != first.cell_ids[1]
    assert first.to_dict() == second.to_dict()

    cells[0]["task"] = "mutated"
    metadata["nested"]["issuer"] = "mutated"
    assert first.cell_definition(first.cell_ids[0]) == {"seed": 1, "task": "alpha"}
    assert first.to_dict()["metadata"] == {"nested": {"issuer": "independent"}}

    rebuilt = CampaignPlan.from_dict(first.to_dict())
    assert rebuilt == first


def test_journal_replay_returns_only_committed_cells(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "campaign.jsonl"
    with CampaignJournal(path, plan) as journal:
        _complete(journal, plan.cell_ids[0])
        partial_attempt = journal.start_cell(plan.cell_ids[1])
        journal.record_arm_result(plan.cell_ids[1], partial_attempt, {"score": 0})

    with CampaignJournal(path, plan) as resumed:
        snapshot = resumed.resume()
        assert snapshot.committed_cell_ids == (plan.cell_ids[0],)
        assert snapshot.runnable_cell_ids == (plan.cell_ids[1],)
        assert snapshot.incomplete_cell_ids == (plan.cell_ids[1],)
        first_record = resumed.committed_records()[0]
        assert first_record["definition"] == plan.cell_definition(plan.cell_ids[0])
        assert first_record["result"]["treatment"]["score"] == 1
        assert first_record["verification"] == {"accepted": True}
        assert first_record["commit"] == {"raw_receipt_sha256": "a" * 64}

        retried_attempt = resumed.start_cell(plan.cell_ids[1])
        assert retried_attempt != partial_attempt
        resumed.record_arm_result(plan.cell_ids[1], retried_attempt, {"score": 1})
        resumed.record_verified(plan.cell_ids[1], retried_attempt, {"accepted": True})
        resumed.commit_cell(plan.cell_ids[1], retried_attempt)
        assert resumed.resume().committed_cell_ids == plan.cell_ids

    states = [json.loads(line)["event"] for line in path.read_bytes().splitlines()]
    assert states == [
        "PLAN",
        STARTED,
        ARM_RESULT,
        VERIFIED,
        COMMITTED,
        STARTED,
        ARM_RESULT,
        FAILED,
        STARTED,
        ARM_RESULT,
        VERIFIED,
        COMMITTED,
    ]


def test_fsync_sealed_result_can_be_verified_after_worker_exit(tmp_path: Path) -> None:
    plan = _plan(1)
    path = tmp_path / "campaign.jsonl"
    with CampaignJournal(path, plan) as worker:
        attempt_id = worker.start_cell(plan.cell_ids[0])
        event_sha256 = worker.record_arm_result(
            plan.cell_ids[0], attempt_id, {"text": "candidate output"}
        )
        assert worker.resume().sealed_cell_ids == plan.cell_ids
        assert worker.resume().committed_cell_ids == ()

    with CampaignJournal(path, plan) as verifier:
        records = verifier.result_records()
        assert len(records) == 1
        assert records[0]["state"] == ARM_RESULT
        assert records[0]["arm_result_event_sha256"] == event_sha256
        assert records[0]["result"] == {"text": "candidate output"}
        verifier.record_verified(
            plan.cell_ids[0], attempt_id, {"accepted": True}
        )
        verifier.commit_cell(plan.cell_ids[0], attempt_id)
        assert verifier.resume().committed_cell_ids == plan.cell_ids


@pytest.mark.parametrize(
    "boundary",
    [None, STARTED, ARM_RESULT, VERIFIED, FAILED, COMMITTED],
)
def test_interruption_at_every_state_boundary_is_deterministically_resumable(
    tmp_path: Path,
    boundary: str | None,
) -> None:
    plan = _plan(1)
    path = tmp_path / f"{boundary or 'genesis'}.jsonl"
    original_attempt: str | None = None
    with CampaignJournal(path, plan) as journal:
        if boundary is not None:
            original_attempt = journal.start_cell(plan.cell_ids[0])
        if boundary in {ARM_RESULT, VERIFIED, FAILED, COMMITTED}:
            journal.record_arm_result(plan.cell_ids[0], original_attempt, {"score": 1})
        if boundary in {VERIFIED, COMMITTED}:
            journal.record_verified(plan.cell_ids[0], original_attempt, {"accepted": True})
        if boundary == FAILED:
            journal.fail_cell(plan.cell_ids[0], original_attempt, reason="injected_crash")
        if boundary == COMMITTED:
            journal.commit_cell(plan.cell_ids[0], original_attempt)

    with CampaignJournal(path, plan) as resumed:
        snapshot = resumed.resume()
        if boundary == COMMITTED:
            assert snapshot.committed_cell_ids == plan.cell_ids
            assert snapshot.runnable_cell_ids == ()
            _assert_code("cell_already_committed", lambda: resumed.start_cell(plan.cell_ids[0]))
            return

        assert snapshot.committed_cell_ids == ()
        assert snapshot.runnable_cell_ids == plan.cell_ids
        expected_incomplete = (
            (plan.cell_ids[0],)
            if boundary
            in {
                STARTED,
                ARM_RESULT,
                VERIFIED,
            }
            else ()
        )
        assert snapshot.incomplete_cell_ids == expected_incomplete
        retry = _complete(resumed, plan.cell_ids[0])
        if original_attempt is not None:
            assert retry != original_attempt
        assert resumed.resume().committed_cell_ids == plan.cell_ids


def test_duplicate_results_commits_and_invalid_transitions_are_refused(tmp_path: Path) -> None:
    plan = _plan(1)
    with CampaignJournal(tmp_path / "campaign.jsonl", plan) as journal:
        cell_id = plan.cell_ids[0]
        attempt_id = journal.start_cell(cell_id)
        _assert_code(
            "cell_attempt_already_active",
            lambda: journal.start_cell(cell_id),
        )
        _assert_code(
            "invalid_transition",
            lambda: journal.record_verified(cell_id, attempt_id, {"accepted": True}),
        )
        _assert_code(
            "invalid_transition",
            lambda: journal.commit_cell(cell_id, attempt_id),
        )
        journal.record_arm_result(cell_id, attempt_id, {"score": 1})
        _assert_code(
            "duplicate_arm_result",
            lambda: journal.record_arm_result(cell_id, attempt_id, {"score": 1}),
        )
        journal.record_verified(cell_id, attempt_id, {"accepted": True})
        _assert_code(
            "invalid_transition",
            lambda: journal.record_verified(cell_id, attempt_id, {"accepted": True}),
        )
        journal.commit_cell(cell_id, attempt_id)
        _assert_code("duplicate_commit", lambda: journal.commit_cell(cell_id, attempt_id))


def test_final_manifest_requires_exact_complete_cell_set(tmp_path: Path) -> None:
    plan = _plan(2)
    path = tmp_path / "campaign.jsonl"
    manifest_path = tmp_path / "manifest.json"
    with CampaignJournal(path, plan) as journal:
        _complete(journal, plan.cell_ids[0])
        _assert_code("campaign_incomplete", lambda: journal.finalize(manifest_path))
        assert not manifest_path.exists()
        _complete(journal, plan.cell_ids[1])
        manifest = journal.finalize(manifest_path)

        assert manifest["plan_sha256"] == plan.plan_sha256
        assert manifest["journal_head_sha256"] == journal.resume().journal_head_sha256
        assert manifest["cell_count"] == 2
        assert [cell["cell_id"] for cell in manifest["cells"]] == list(plan.cell_ids)
        assert journal.finalize(manifest_path) == manifest

    disk_manifest = json.loads(manifest_path.read_bytes())
    material = {key: value for key, value in disk_manifest.items() if key != "manifest_sha256"}
    import hashlib

    assert (
        disk_manifest["manifest_sha256"]
        == hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    )


def test_finalization_is_byte_deterministic(tmp_path: Path) -> None:
    plan = _plan(2)
    manifests: list[bytes] = []
    journals: list[bytes] = []
    for suffix in ("a", "b"):
        directory = tmp_path / suffix
        directory.mkdir()
        journal_path = directory / "campaign.jsonl"
        manifest_path = directory / "manifest.json"
        with CampaignJournal(journal_path, plan) as journal:
            for cell_id in plan.cell_ids:
                _complete(journal, cell_id)
            journal.finalize(manifest_path)
        journals.append(journal_path.read_bytes())
        manifests.append(manifest_path.read_bytes())

    assert journals[0] == journals[1]
    assert manifests[0] == manifests[1]


def test_conflicting_existing_manifest_is_refused(tmp_path: Path) -> None:
    plan = _plan(1)
    manifest_path = tmp_path / "manifest.json"
    with CampaignJournal(tmp_path / "campaign.jsonl", plan) as journal:
        _complete(journal, plan.cell_ids[0])
        journal.finalize(manifest_path)
        manifest_path.write_text("{}\n", encoding="utf-8")
        _assert_code(
            "manifest_already_exists_with_different_content",
            lambda: journal.finalize(manifest_path),
        )


def test_torn_and_malformed_records_fail_closed(tmp_path: Path) -> None:
    plan = _plan(1)
    torn = tmp_path / "torn.jsonl"
    with CampaignJournal(torn, plan):
        pass
    with torn.open("ab") as stream:
        stream.write(b'{"event":"STARTED"')
        stream.flush()
        os.fsync(stream.fileno())
    _assert_code("journal_torn_record", lambda: CampaignJournal(torn, plan))

    malformed = tmp_path / "malformed.jsonl"
    with CampaignJournal(malformed, plan):
        pass
    with malformed.open("ab") as stream:
        stream.write(b"{}\n")
        stream.flush()
        os.fsync(stream.fileno())
    _assert_code("journal_event_shape_invalid", lambda: CampaignJournal(malformed, plan))


def test_hash_chain_tampering_is_detected_even_when_event_hash_is_recomputed(
    tmp_path: Path,
) -> None:
    plan = _plan(1)
    path = tmp_path / "campaign.jsonl"
    with CampaignJournal(path, plan) as journal:
        journal.start_cell(plan.cell_ids[0])

    _rewrite_record(
        path,
        1,
        lambda record: record.__setitem__("previous_event_sha256", "f" * 64),
    )
    _assert_code("journal_hash_chain_invalid", lambda: CampaignJournal(path, plan))


def test_invalid_transition_in_a_valid_hash_chain_fails_replay(tmp_path: Path) -> None:
    plan = _plan(1)
    path = tmp_path / "campaign.jsonl"
    with CampaignJournal(path, plan) as journal:
        attempt_id = journal.start_cell(plan.cell_ids[0])

    previous = json.loads(path.read_bytes().splitlines()[-1])
    _append_validly_hashed_record(
        path,
        {
            "schema": previous["schema"],
            "sequence": previous["sequence"] + 1,
            "plan_sha256": plan.plan_sha256,
            "previous_event_sha256": previous["event_sha256"],
            "event": VERIFIED,
            "cell_id": plan.cell_ids[0],
            "attempt_id": attempt_id,
            "payload": {"verification": {"accepted": True}},
        },
    )
    _assert_code("journal_invalid_transition", lambda: CampaignJournal(path, plan))


def test_unknown_cell_in_a_valid_hash_chain_fails_replay(tmp_path: Path) -> None:
    plan = _plan(1)
    path = tmp_path / "campaign.jsonl"
    with CampaignJournal(path, plan):
        pass

    previous = json.loads(path.read_bytes().splitlines()[-1])
    _append_validly_hashed_record(
        path,
        {
            "schema": previous["schema"],
            "sequence": previous["sequence"] + 1,
            "plan_sha256": plan.plan_sha256,
            "previous_event_sha256": previous["event_sha256"],
            "event": STARTED,
            "cell_id": "cell-" + "f" * 64,
            "attempt_id": "attempt-" + "e" * 64,
            "payload": {"attempt_number": 1},
        },
    )
    _assert_code("journal_cell_drift", lambda: CampaignJournal(path, plan))


def test_stale_plan_and_cell_identity_drift_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "campaign.jsonl"
    original = _plan(1)
    with CampaignJournal(path, original):
        pass

    stale = CampaignPlan.build(
        "resident-32b-frontier",
        [{"domain": "different", "seed": 999, "task_sha256": "f" * 64}],
        metadata={"comparison": "rlc-vs-vanilla", "version": 3},
    )
    _assert_code("journal_plan_drift", lambda: CampaignJournal(path, stale))

    document = original.to_dict()
    document["cells"][0]["cell_id"] = "cell-" + "f" * 64
    _assert_code("plan_hash_or_cell_identity_invalid", lambda: CampaignPlan.from_dict(document))


def test_same_process_and_os_level_concurrent_writers_are_refused(tmp_path: Path) -> None:
    plan = _plan(1)
    path = tmp_path / "campaign.jsonl"
    with CampaignJournal(path, plan):
        _assert_code("journal_writer_locked", lambda: CampaignJournal(path, plan))

        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from core.brain.llm.latent_cortex.campaign_journal import "
                    "CampaignJournal, CampaignJournalError, CampaignPlan; "
                    "p=CampaignPlan.build('resident-32b-frontier', "
                    "[{'domain':'mathematics','seed':100,'task_sha256':'%064x' % 1}], "
                    "metadata={'comparison':'rlc-vs-vanilla','version':3}); "
                    "\ntry:\n CampaignJournal(sys.argv[1], p)\nexcept CampaignJournalError as e:\n "
                    "print(e.code)\nelse:\n print('LOCK_FAILURE')\n"
                ),
                str(path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert child.stdout.strip() == "journal_writer_locked"


def test_attempt_ids_are_per_cell_per_attempt_and_deterministic(tmp_path: Path) -> None:
    plan = _plan(1)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    with CampaignJournal(first_path, plan) as first:
        attempt_one = first.start_cell(plan.cell_ids[0])
        first.fail_cell(plan.cell_ids[0], attempt_one, reason="retryable")
        attempt_two = first.start_cell(plan.cell_ids[0])

    with CampaignJournal(second_path, plan) as second:
        assert second.start_cell(plan.cell_ids[0]) == attempt_one
        second.fail_cell(plan.cell_ids[0], attempt_one, reason="retryable")
        assert second.start_cell(plan.cell_ids[0]) == attempt_two

    assert attempt_one.startswith("attempt-")
    assert attempt_two.startswith("attempt-")
    assert attempt_one != attempt_two


def test_duplicate_json_keys_and_noncanonical_records_fail_closed(tmp_path: Path) -> None:
    plan = _plan(1)
    duplicate = tmp_path / "duplicate.jsonl"
    with CampaignJournal(duplicate, plan):
        pass
    with duplicate.open("ab") as stream:
        stream.write(b'{"schema":"x","schema":"y"}\n')
        stream.flush()
        os.fsync(stream.fileno())
    _assert_code(
        "journal_event_duplicate_json_key",
        lambda: CampaignJournal(duplicate, plan),
    )

    noncanonical = tmp_path / "noncanonical.jsonl"
    with CampaignJournal(noncanonical, plan) as journal:
        journal.start_cell(plan.cell_ids[0])
    lines = noncanonical.read_bytes().splitlines()
    parsed = json.loads(lines[1])
    lines[1] = json.dumps(parsed, sort_keys=False).encode("utf-8")
    noncanonical.write_bytes(b"\n".join(lines) + b"\n")
    _assert_code("journal_event_noncanonical", lambda: CampaignJournal(noncanonical, plan))
