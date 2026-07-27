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

    def test_blocked_rows_name_why_they_are_blocked(self) -> None:
        """Every blocked row must be unblockable by someone.

        The previous version of this test asserted the exact blocker ids
        `set(range(39, 47))` and `{55, 56}` — which is precisely why those
        lists went stale and stayed stale: a test was enforcing them long
        after all ten of those items closed. Pinning the *reason* rather than
        a frozen id list is what makes the row honest as the ledger moves.
        """
        from core.brain.llm.latent_cortex.falsification_matrix import (
            BLOCKED_OPEN_SPARK_ITEMS,
            BLOCKED_PRODUCER_ABSENT,
            BLOCKED_REASONS,
        )

        blocked = [row for row in MATRIX_ROWS if row.status == ROW_BLOCKED]
        assert blocked
        for row in blocked:
            assert row.blocked_reason in BLOCKED_REASONS, row.row_id
            if row.blocked_reason == BLOCKED_OPEN_SPARK_ITEMS:
                # Waiting on an unlanded item must say which one.
                assert row.blockers, row.row_id
            elif row.blocked_reason == BLOCKED_PRODUCER_ABSENT:
                # Nothing in the ledger is being waited on; the gap is code.
                assert not row.blockers, row.row_id
            # acceptance_run_only may name the acceptance item (SPARK-070)
            # without that being a dependency on unlanded machinery.

    def test_no_row_is_blocked_on_an_item_that_already_closed(self) -> None:
        """The check that makes the stale-blocker class detectable.

        Parsed from the ledger rather than mirrored, because a mirrored copy
        is the thing that drifts.
        """
        from core.brain.llm.latent_cortex.falsification_matrix import (
            FalsificationMatrixError,
            open_ledger_items,
            validate_blockers_against_ledger,
        )

        open_items = open_ledger_items()
        assert open_items, "the ledger must still have open items to check against"
        report = validate_blockers_against_ledger(open_items=open_items)
        assert report["stale_blockers"] == []

        # And it must actually fire: with nothing open, every named blocker
        # is by definition closed.
        with pytest.raises(FalsificationMatrixError, match="blocker_closed"):
            validate_blockers_against_ledger(open_items=frozenset())

    def test_verifier_arms_waits_on_a_producer_not_a_ledger_item(self) -> None:
        """SPARK-039..046 all landed; the gap is code, not a dependency.

        fast_weight_controls was the other row in this state and has since
        gained `experiments.run_fast_weight_controls`, so it is runnable.
        verifier_arms stays blocked deliberately: its generative and
        counterfactual arms cannot run model-free, and a row marked runnable
        while only its exact-checker arms execute would restate the defect
        the verifier mesh was built to remove.
        """
        from core.brain.llm.latent_cortex.falsification_matrix import (
            BLOCKED_PRODUCER_ABSENT,
        )

        row = next(item for item in MATRIX_ROWS if item.row_id == "verifier_arms")
        assert row.blocked_reason == BLOCKED_PRODUCER_ABSENT
        assert row.blockers == ()
        assert not row.producer

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


class TestFastWeightControls:
    """SPARK-070's fast-weight row: the producer, and what it refuses to claim."""

    @staticmethod
    def _tasks():
        from core.brain.llm.latent_cortex.experiments import Task

        return {
            "khop": [
                Task(
                    prompt=f"q{index}",
                    answer=f"a{index}",
                    family="khop",
                    depth=2,
                    seed=index,
                )
                for index in range(6)
            ]
        }

    def test_row_is_runnable_and_names_its_producer(self) -> None:
        row = next(
            item for item in MATRIX_ROWS if item.row_id == "fast_weight_controls"
        )
        assert row.status == "runnable"
        assert row.producer == "experiments.run_fast_weight_controls"
        assert row.blockers == ()
        assert row.blocked_reason == ""

    def test_arms_are_counterbalanced_and_the_claim_is_on_versus_sham(self) -> None:
        """Beating 'off' only shows a perturbation helped; direction needs sham."""
        from core.brain.llm.latent_cortex.experiments import (
            run_fast_weight_controls,
        )

        def solve(task, arm):
            # 'on' genuinely helps; 'sham' matches 'off'. The claim must come
            # from the on-vs-sham contrast.
            success = arm == "on"
            return success, 100, None if arm == "off" else True

        report = run_fast_weight_controls(solve, self._tasks())
        assert set(report["arms"]) == {"off", "on", "sham"}
        assert report["erasure_integrity"]["integrity_proven"] is True
        assert report["erasure_integrity"]["erase_proven"] == 12  # 6 tasks x 2 arms
        # Every task ran all three arms, and not always in the same order.
        orders = {tuple(row["arms"]) for row in report["execution_order"]}
        assert len(orders) > 1, "arm order must be counterbalanced across tasks"
        assert report["claim"]["tier"] != "REFUTED_INTEGRITY"

    def test_an_unproven_erase_quarantines_the_task_rather_than_counting_it(
        self,
    ) -> None:
        """An unproven erase is not a passed check, and not a failed one either."""
        from core.brain.llm.latent_cortex.experiments import (
            run_fast_weight_controls,
        )

        def solve(task, arm):
            erased = None if arm == "off" else (None if task.seed == 0 else True)
            return arm == "on", 100, erased

        report = run_fast_weight_controls(solve, self._tasks())
        integrity = report["erasure_integrity"]
        assert integrity["quarantined_observations"] == 2  # both adapted arms
        assert integrity["integrity_proven"] is False
        assert report["claim"]["tier"] == "REFUTED_INTEGRITY"
        # The quarantined task is dropped whole: counting part of it would
        # silently unbalance the pairing.
        assert report["arms"]["on"]["khop"]["n"] == 5

    def test_a_refuted_erase_fails_the_whole_run(self) -> None:
        from core.brain.llm.latent_cortex.experiments import (
            run_fast_weight_controls,
        )

        def solve(task, arm):
            erased = None if arm == "off" else (task.seed != 0)
            return arm == "on", 100, erased

        report = run_fast_weight_controls(solve, self._tasks())
        assert report["erasure_integrity"]["erase_refuted"] == 2
        assert report["erasure_integrity"]["integrity_proven"] is False
        assert report["claim"]["tier"] == "REFUTED_INTEGRITY"

    def test_the_off_arm_cannot_claim_an_erase(self) -> None:
        from core.brain.llm.latent_cortex.experiments import (
            run_fast_weight_controls,
        )

        def solve(task, arm):
            return True, 100, True  # 'off' wrongly reports an erase

        with pytest.raises(ValueError, match="no delta to erase"):
            run_fast_weight_controls(solve, self._tasks())

    def test_malformed_solver_outcomes_are_refused(self) -> None:
        from core.brain.llm.latent_cortex.experiments import (
            run_fast_weight_controls,
        )

        with pytest.raises(ValueError, match="must return"):
            run_fast_weight_controls(lambda task, arm: True, self._tasks())
