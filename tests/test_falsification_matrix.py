"""SPARK-070: the falsification matrix stays complete, bound, and replayable."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256
from core.brain.llm.latent_cortex.falsification_matrix import (
    MATRIX_ROWS,
    REQUIRED_ROW_IDS,
    ROW_BLOCKED,
    ROW_ENFORCED,
    ROW_RUNNABLE,
    FalsificationMatrixError,
    MatrixRow,
    assemble_falsification_matrix_receipt,
    replay_falsification_matrix_receipt,
    run_transition_matrix,
    validate_falsification_matrix,
)


@dataclass(frozen=True)
class _Task:
    prompt: str
    family: str
    depth: int
    seed: int


def _runnable_ids() -> set[str]:
    return {row.row_id for row in MATRIX_ROWS if row.status == ROW_RUNNABLE}


def _fake_row_results() -> dict[str, dict]:
    return {
        row_id: {"claim": {"experiment": f"mx_{row_id}", "tier": "SUPPORTED"}}
        for row_id in _runnable_ids()
    }


class TestRegistry:
    def test_registry_covers_every_required_row(self) -> None:
        receipt = validate_falsification_matrix()
        assert receipt["row_count"] == len(REQUIRED_ROW_IDS) == 12
        assert {row["row_id"] for row in receipt["rows"]} == set(REQUIRED_ROW_IDS)
        body = {
            key: value
            for key, value in receipt.items()
            if key != "registry_sha256"
        }
        assert receipt["registry_sha256"] == canonical_sha256(body)

    def test_blocked_rows_name_their_blockers(self) -> None:
        blocked = [row for row in MATRIX_ROWS if row.status == ROW_BLOCKED]
        assert blocked
        for row in blocked:
            assert row.blockers, row.row_id
        verifier_row = next(
            row for row in MATRIX_ROWS if row.row_id == "verifier_arms"
        )
        assert set(verifier_row.blockers) == set(range(39, 47))
        fast_weight_row = next(
            row for row in MATRIX_ROWS if row.row_id == "fast_weight_controls"
        )
        assert set(fast_weight_row.blockers) == {55, 56}

    def test_enforced_rows_bind_to_threat_model(self) -> None:
        enforced = [row for row in MATRIX_ROWS if row.status == ROW_ENFORCED]
        assert enforced
        for row in enforced:
            assert row.producer.startswith("threat_model:")

    def test_row_contracts_fail_closed(self) -> None:
        with pytest.raises(FalsificationMatrixError):
            MatrixRow(
                row_id="not_a_row",
                ledger_clause="a clause long enough",
                status=ROW_RUNNABLE,
                producer="x",
                blockers=(),
                notes="notes long enough",
            )
        with pytest.raises(FalsificationMatrixError):
            MatrixRow(
                row_id="verifier_arms",
                ledger_clause="a clause long enough",
                status=ROW_BLOCKED,
                producer="",
                blockers=(),
                notes="blocked without blockers",
            )
        with pytest.raises(FalsificationMatrixError):
            MatrixRow(
                row_id="verifier_arms",
                ledger_clause="a clause long enough",
                status=ROW_RUNNABLE,
                producer="",
                blockers=(),
                notes="runnable without producer",
            )


class TestTransitionMatrix:
    def test_table_separates_repair_from_damage(self) -> None:
        tasks = [_Task(f"p{i}", "modular", 2, i) for i in range(8)]
        # Shallow solves even tasks; deep solves tasks 0-5: task 6,7 wrong→
        # deep repairs 1,3,5 (odd<6), damages none... construct explicitly:
        shallow_ok = {0, 2, 4, 6}
        deep_ok = {0, 1, 2, 3, 4, 5}

        def solve(task, steps):
            ok = task.seed in (shallow_ok if steps == 1 else deep_ok)
            return (ok, 100 * steps)

        report = run_transition_matrix(
            solve, tasks, shallow_steps=1, deep_steps=4
        )
        assert report["table"] == {
            "wrong_to_right": 3,  # 1, 3, 5
            "right_to_wrong": 1,  # 6
            "unchanged_right": 3,  # 0, 2, 4
            "unchanged_wrong": 1,  # 7
        }
        assert report["n_tasks"] == 8
        assert report["claim"]["experiment"] == "mx_transition_matrix"

    def test_invalid_steps_fail_closed(self) -> None:
        with pytest.raises(FalsificationMatrixError):
            run_transition_matrix(
                lambda task, steps: (True, 1),
                [_Task("p", "modular", 2, 0)],
                shallow_steps=4,
                deep_steps=4,
            )


class TestReceipt:
    def test_assemble_and_replay_roundtrip(self) -> None:
        results = _fake_row_results()
        receipt = assemble_falsification_matrix_receipt(
            row_results=results,
            runner_identity={"mode": "unit"},
            threat_model_registry_sha256="a" * 64,
        )
        assert set(receipt["runnable_rows"]) == _runnable_ids()
        replay = replay_falsification_matrix_receipt(
            receipt, row_payloads=results
        )
        assert replay["replayed"] is True
        for row in receipt["rows"]:
            if row["status"] == ROW_RUNNABLE:
                assert row["claim_tiers"], row["row_id"]
            if row["status"] == ROW_ENFORCED:
                assert row["result_sha256"] == "a" * 64

    def test_missing_runnable_result_fails(self) -> None:
        results = _fake_row_results()
        results.pop("recurrence_depth_curves")
        with pytest.raises(FalsificationMatrixError) as error:
            assemble_falsification_matrix_receipt(
                row_results=results,
                runner_identity={"mode": "unit"},
                threat_model_registry_sha256="a" * 64,
            )
        assert error.value.code == "falsification_matrix_missing_result_row"

    def test_result_for_blocked_row_fails(self) -> None:
        results = _fake_row_results()
        results["verifier_arms"] = {"claim": {"experiment": "x", "tier": "PROVEN"}}
        with pytest.raises(FalsificationMatrixError) as error:
            assemble_falsification_matrix_receipt(
                row_results=results,
                runner_identity={"mode": "unit"},
                threat_model_registry_sha256="a" * 64,
            )
        assert error.value.code == "falsification_matrix_unknown_result_row"

    def test_replay_detects_payload_tampering(self) -> None:
        results = _fake_row_results()
        receipt = assemble_falsification_matrix_receipt(
            row_results=results,
            runner_identity={"mode": "unit"},
            threat_model_registry_sha256="a" * 64,
        )
        tampered = {
            **results,
            "recurrence_depth_curves": {
                "claim": {"experiment": "mx_recurrence_depth_curves", "tier": "PROVEN"}
            },
        }
        with pytest.raises(FalsificationMatrixError) as error:
            replay_falsification_matrix_receipt(receipt, row_payloads=tampered)
        assert error.value.code == "falsification_matrix_replay_payload_mismatch"

    def test_replay_detects_registry_drift(self) -> None:
        results = _fake_row_results()
        receipt = assemble_falsification_matrix_receipt(
            row_results=results,
            runner_identity={"mode": "unit"},
            threat_model_registry_sha256="a" * 64,
        )
        drifted_rows = [dict(row) for row in receipt["rows"]]
        for row in drifted_rows:
            if row["row_id"] == "verifier_arms":
                row["status"] = ROW_RUNNABLE
                row["producer"] = "invented"
                row["blockers"] = []
        body = {
            **{k: v for k, v in receipt.items() if k != "receipt_sha256"},
            "rows": drifted_rows,
        }
        forged = {**body, "receipt_sha256": canonical_sha256(body)}
        with pytest.raises(FalsificationMatrixError) as error:
            replay_falsification_matrix_receipt(forged)
        assert error.value.code == "falsification_matrix_registry_drift"

    def test_replay_detects_digest_tampering(self) -> None:
        receipt = assemble_falsification_matrix_receipt(
            row_results=_fake_row_results(),
            runner_identity={"mode": "unit"},
            threat_model_registry_sha256="a" * 64,
        )
        tampered = {**receipt, "runner_identity": {"mode": "forged"}}
        with pytest.raises(FalsificationMatrixError) as error:
            replay_falsification_matrix_receipt(tampered)
        assert error.value.code == "falsification_matrix_receipt_digest_mismatch"
