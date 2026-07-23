"""CP126 execution_controller — evidence integrity, attribution, governance.

Each test pins one finding from artifacts/closeout/semantic_review/cp126/.
"""
from __future__ import annotations

import json

import pytest

from core.brain.llm.latent_cortex import execution_controller as ec
from core.brain.llm.latent_cortex.execution_controller import (
    ARMS,
    CONTROLLER_ROW_SCHEMA,
    MIN_TRIALS,
    ExecutionController,
    _bonferroni_z,
    _validated_region,
    context_bucket,
)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setenv("AURA_EXECUTION_CONTROLLER", "1")


def _controller(tmp_path, name="controller"):
    return ExecutionController(root=tmp_path / name)


def _record(controller, *, bucket, arm, score, success, latency=0.0):
    decision = {"bucket": bucket, "arm": arm, "mode": "explore"}
    controller._issue_decision(decision)
    return controller.record_outcome(
        bucket=bucket,
        arm=arm,
        verified_score=score,
        success=success,
        checked=True,
        wall_clock_s=latency,
        decision_id=decision["decision_id"],
    )


class TestDecisionOutcomeBinding:
    """3b3d44e8: outcomes are bound to the decision that produced them."""

    def test_outcome_without_a_token_is_refused(self, tmp_path):
        c = _controller(tmp_path)
        assert c.record_outcome(
            bucket="b", arm="base", verified_score=1.0, success=True, checked=True
        ) is False
        assert c.status()["cells"] == []

    def test_unknown_token_is_refused(self, tmp_path):
        c = _controller(tmp_path)
        assert c.record_outcome(
            bucket="b", arm="base", verified_score=1.0, success=True,
            checked=True, decision_id="deadbeef",
        ) is False

    def test_token_cannot_be_reused(self, tmp_path):
        c = _controller(tmp_path)
        decision = {"bucket": "b", "arm": "base", "mode": "explore"}
        c._issue_decision(decision)
        token = decision["decision_id"]
        kwargs = dict(
            bucket="b", arm="base", verified_score=1.0, success=True,
            checked=True, decision_id=token,
        )
        assert c.record_outcome(**kwargs) is True
        assert c.record_outcome(**kwargs) is False, "a token was replayable"

    def test_caller_cannot_credit_a_different_arm(self, tmp_path):
        c = _controller(tmp_path)
        decision = {"bucket": "b", "arm": "base", "mode": "observe"}
        c._issue_decision(decision)
        # The episode RAN base; crediting a treatment must be refused.
        assert c.record_outcome(
            bucket="b", arm="deeper_recurrence", verified_score=1.0,
            success=True, checked=True, decision_id=decision["decision_id"],
        ) is False

    def test_choose_issues_a_token(self, tmp_path):
        c = _controller(tmp_path)
        decision = c.choose(objective="q", domain="general", stakes=0.5, uncertainty=0.5)
        assert decision.get("decision_id")
        assert decision.get("decision_sha256")


class TestLedgerProvenanceAndReplay:
    """cbcb73c1 + 84c5f06c: rows carry identity and cannot be replayed."""

    def test_duplicate_rows_do_not_inflate_trials(self, tmp_path):
        root = tmp_path / "controller"
        c = _controller(tmp_path)
        _record(c, bucket="b", arm="base", score=0.9, success=True)
        raw = (root / "outcomes.jsonl").read_text()
        # An attacker/backup copies the row.
        with (root / "outcomes.jsonl").open("a") as handle:
            handle.write(raw)
        restored = ExecutionController(root=root)
        assert restored.status()["episodes_seen"] == 1, "a replayed row was folded"

    def test_rows_from_another_provenance_are_skipped_not_folded(self, tmp_path, monkeypatch):
        root = tmp_path / "controller"
        root.mkdir(parents=True)
        (root / "outcomes.jsonl").write_text(
            json.dumps(
                {
                    "schema": CONTROLLER_ROW_SCHEMA,
                    "episode_id": "a" * 32,
                    "provenance": "old-checkpoint",
                    "bucket": "b",
                    "arm": "base",
                    "verified_score": 1.0,
                    "success": True,
                    "checked": True,
                    "wall_clock_s": 1.0,
                }
            )
            + "\n"
        )
        monkeypatch.setenv(ec._PROVENANCE_ENV, "new-checkpoint")
        c = ExecutionController(root=root)
        status = c.status()
        assert status["episodes_seen"] == 0
        # Incompatible ≠ corrupt: integrity stays intact.
        assert status["restore_errors"] == 0
        assert c.integrity_ok() is True

    def test_schemaless_row_is_an_integrity_error(self, tmp_path):
        root = tmp_path / "controller"
        root.mkdir(parents=True)
        (root / "outcomes.jsonl").write_text(
            '{"bucket":"b","arm":"base","verified_score":1.0,"success":true,"checked":true}\n'
        )
        c = ExecutionController(root=root)
        assert c.status()["restore_errors"] == 1
        assert c.integrity_ok() is False


class TestCorruptionForcesObserveOnly:
    """af9ac58e: a partial ledger must block non-base selection."""

    def test_choose_refuses_to_explore_or_exploit(self, tmp_path):
        root = tmp_path / "controller"
        root.mkdir(parents=True)
        (root / "outcomes.jsonl").write_text("{corrupt\n")
        c = ExecutionController(root=root)
        for _ in range(12):
            decision = c.choose(
                objective="q", domain="general", stakes=0.5, uncertainty=0.5
            )
            assert decision["arm"] == "base"
            assert decision["mode"] == "observe_only_integrity"
            assert decision["evidence"]["blocked"] == "ledger_integrity_failed"


class TestKillSwitchEnforcement:
    """f1088112: choose, apply_arm and record_outcome all honor the flag."""

    def test_choose_is_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_EXECUTION_CONTROLLER", "0")
        c = _controller(tmp_path)
        decision = c.choose(objective="q", domain="general", stakes=0.5, uncertainty=0.5)
        assert decision["arm"] == "base"
        assert decision["mode"] == "disabled"

    def test_apply_arm_is_disabled(self, tmp_path, monkeypatch):
        c = _controller(tmp_path)
        monkeypatch.setenv("AURA_EXECUTION_CONTROLLER", "0")
        receipt = c.apply_arm_receipt("deeper_recurrence", {"max_steps": 4})
        assert receipt["applied"] is False
        assert receipt["effective_arm"] == "base"
        assert receipt["reason"] == "controller_disabled"

    def test_record_outcome_is_disabled(self, tmp_path, monkeypatch):
        c = _controller(tmp_path)
        decision = {"bucket": "b", "arm": "base", "mode": "observe"}
        c._issue_decision(decision)
        monkeypatch.setenv("AURA_EXECUTION_CONTROLLER", "0")
        assert c.record_outcome(
            bucket="b", arm="base", verified_score=1.0, success=True,
            checked=True, decision_id=decision["decision_id"],
        ) is False


class TestEpisodeAccounting:
    """b36c802d: one completed pair is counted exactly once."""

    def test_choose_does_not_advance_the_evidence_counter(self, tmp_path):
        c = _controller(tmp_path)
        for _ in range(6):
            c.choose(objective="q", domain="general", stakes=0.5, uncertainty=0.5)
        assert c.status()["episodes_seen"] == 0

    def test_one_outcome_counts_once(self, tmp_path):
        c = _controller(tmp_path)
        _record(c, bucket="b", arm="base", score=0.5, success=True)
        assert c.status()["episodes_seen"] == 1


class TestMultiplicityCorrection:
    """91f196dc: repeated arm-vs-base looks get a corrected boundary."""

    def test_corrected_z_is_stricter_than_nominal(self):
        assert _bonferroni_z(4) > ec._Z95
        assert _bonferroni_z(1) == pytest.approx(ec._Z95)

    def test_decision_receipts_the_boundary(self, tmp_path):
        c = _controller(tmp_path)
        decision = c.choose(objective="q", domain="general", stakes=0.5, uncertainty=0.5)
        boundary = decision["evidence"]["decision_boundary"]
        assert boundary["family_size"] == len(ARMS) - 1
        assert boundary["correction"] == "bonferroni_over_non_base_arms"
        assert boundary["critical_z"] > 1.96


class TestBudgetAwareSelection:
    """c53c85b8: 'budget permitting' actually consults a budget."""

    def test_expensive_arm_is_refused_under_a_small_budget(self, tmp_path):
        c = _controller(tmp_path)
        bucket = context_bucket("q", "general", 0.5, 0.5)
        for _ in range(MIN_TRIALS):
            _record(c, bucket=bucket, arm="base", score=0.2, success=False, latency=1.0)
            _record(
                c, bucket=bucket, arm="deeper_recurrence", score=1.0,
                success=True, latency=30.0,
            )
        decision = c.choose(
            objective="q", domain="general", stakes=0.5, uncertainty=0.5,
            remaining_budget_s=5.0,
        )
        assert decision["arm"] == "base"
        assert decision["evidence"]["candidates"]["deeper_recurrence"]["skipped"] in {
            "over_budget",
            "latency_regression",
        }

    def test_budget_is_receipted(self, tmp_path):
        c = _controller(tmp_path)
        decision = c.choose(
            objective="q", domain="general", stakes=0.5, uncertainty=0.5,
            remaining_budget_s=12.5,
        )
        assert decision["evidence"]["remaining_budget_s"] == 12.5


class TestLatencyGovernsPromotion:
    """171f2e25: a materially slower arm is not promoted."""

    def test_slow_arm_is_not_promoted_even_when_more_accurate(self, tmp_path):
        c = _controller(tmp_path)
        bucket = context_bucket("q", "general", 0.5, 0.5)
        for _ in range(MIN_TRIALS):
            _record(c, bucket=bucket, arm="base", score=0.2, success=False, latency=1.0)
            _record(
                c, bucket=bucket, arm="wider_branches", score=1.0,
                success=True, latency=10.0,
            )
        decision = c.choose(objective="q", domain="general", stakes=0.5, uncertainty=0.5)
        assert decision["arm"] == "base"
        assert (
            decision["evidence"]["candidates"]["wider_branches"]["skipped"]
            == "latency_regression"
        )

    def test_comparable_latency_still_promotes(self, tmp_path):
        c = _controller(tmp_path)
        bucket = context_bucket("q", "general", 0.5, 0.5)
        for _ in range(MIN_TRIALS + 6):
            _record(c, bucket=bucket, arm="base", score=0.1, success=False, latency=1.0)
            _record(
                c, bucket=bucket, arm="wider_branches", score=1.0,
                success=True, latency=1.05,
            )
        decision = c.choose(objective="q", domain="general", stakes=0.5, uncertainty=0.5)
        assert decision["arm"] == "wider_branches"
        assert decision["mode"] == "exploit"


class TestEvidenceReceipts:
    """8c7f02a0: every mode carries its justification."""

    def test_observe_mode_is_receipted(self, tmp_path):
        c = _controller(tmp_path)
        decision = c.choose(objective="q", domain="general", stakes=0.5, uncertainty=0.5)
        evidence = decision["evidence"]
        for key in ("decision_boundary", "candidates", "base_n", "episodes_seen"):
            assert key in evidence, key

    def test_explore_mode_is_receipted(self, tmp_path):
        c = _controller(tmp_path)
        modes = []
        for _ in range(8):
            decision = c.choose(
                objective="q", domain="general", stakes=0.5, uncertainty=0.5
            )
            modes.append(decision["mode"])
            if decision["mode"] == "explore":
                assert decision["evidence"]["explore_reason"]
                assert decision["evidence"]["candidates"]
        assert "explore" in modes


class TestBytecodeApplicationTruth:
    """ea828a97 + 094e4b5d + 261201c3: the bytecode arm cannot lie."""

    def test_absent_region_falls_back_to_base_explicitly(self, tmp_path):
        c = _controller(tmp_path)
        receipt = c.apply_arm_receipt("probe_guided_bytecode", {"max_steps": 6})
        assert receipt["applied"] is False
        assert receipt["effective_arm"] == "base"
        assert receipt["reason"] == "invalid_or_absent_recurrent_region"
        assert "schedule" not in receipt["config"]

    @pytest.mark.parametrize(
        "region",
        [(-1, 10), (10, 10), (48, 16), (True, False), (1,), "16,48", {"a": 1}, (1, 10_000)],
    )
    def test_malformed_regions_are_rejected(self, region):
        assert _validated_region(region) is None

    def test_out_of_model_region_rejected(self):
        assert _validated_region((16, 48), model_layers=32) is None
        assert _validated_region((16, 48), model_layers=64) == (16, 48)

    def test_valid_region_applies_and_receipts_the_schedule(self, tmp_path):
        c = _controller(tmp_path)
        receipt = c.apply_arm_receipt(
            "probe_guided_bytecode", {"max_steps": 6}, recurrent_region=(16, 48)
        )
        assert receipt["applied"] is True
        assert receipt["schedule_sha256"]
        assert receipt["config"]["schedule"]["name"] == "controller_probe_guided_v1"

    def test_superseded_schedule_is_receipted(self, tmp_path):
        c = _controller(tmp_path)
        prior = {"name": "caller_validated_v3", "ops": [{"start": 1, "end": 2, "repeats": 1}]}
        receipt = c.apply_arm_receipt(
            "probe_guided_bytecode",
            {"max_steps": 6, "schedule": prior},
            recurrent_region=(16, 48),
        )
        assert receipt["superseded_schedule"]["name"] == "caller_validated_v3"
        assert receipt["superseded_schedule"]["sha256"]

    def test_emitted_schedule_stays_parseable(self, tmp_path):
        from core.brain.llm.latent_cortex.schedules import LayerSchedule

        c = _controller(tmp_path)
        receipt = c.apply_arm_receipt(
            "probe_guided_bytecode", {"max_steps": 6}, recurrent_region=(16, 48)
        )
        program = LayerSchedule.from_dict(receipt["config"]["schedule"])
        assert program.validate(prelude_end=16, coda_start=48) == []


class TestContextValidation:
    """84345b43: malformed context is explicit, and buckets cannot collide."""

    def test_nan_is_not_silently_low(self):
        bucket = context_bucket("q", "general", float("nan"), 0.5)
        assert "s:invalid" in bucket
        assert "s:low" not in bucket

    def test_out_of_range_is_invalid(self):
        assert "u:invalid" in context_bucket("q", "general", 0.5, 7.0)

    def test_delimiter_in_domain_cannot_forge_a_bucket(self):
        forged = context_bucket("q", "general|none|short", 0.5, 0.5)
        honest = context_bucket("q", "general", 0.5, 0.5)
        assert forged != honest

    def test_non_numeric_does_not_raise(self):
        assert "s:invalid" in context_bucket("q", "general", "high", 0.5)


class TestLedgerRetention:
    """be8c21e8: the declared row cap is enforced."""

    def test_compaction_bounds_the_ledger(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ec, "_MAX_LEDGER_ROWS", 10)
        c = _controller(tmp_path)
        bucket = context_bucket("q", "general", 0.5, 0.5)
        for _ in range(24):
            _record(c, bucket=bucket, arm="base", score=0.5, success=True)
        rows = (tmp_path / "controller" / "outcomes.jsonl").read_text().splitlines()
        assert len(rows) <= 10, "ledger grew past its declared bound"
        # The folded evidence survives compaction.
        assert c.status()["episodes_seen"] == 24


class TestConcurrency:
    """ce5b26e0: state mutation is synchronized."""

    def test_singleton_is_singleflight(self, monkeypatch, tmp_path):
        import threading

        monkeypatch.setattr(ec, "_instance", None)
        built: list[ExecutionController] = []
        barrier = threading.Barrier(4)

        def _make():
            barrier.wait()
            built.append(ec.get_execution_controller())

        threads = [threading.Thread(target=_make) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        assert len({id(item) for item in built}) == 1
        monkeypatch.setattr(ec, "_instance", None)

    def test_concurrent_outcomes_are_all_counted(self, tmp_path):
        import threading

        c = _controller(tmp_path)
        bucket = context_bucket("q", "general", 0.5, 0.5)

        def _worker():
            for _ in range(10):
                _record(c, bucket=bucket, arm="base", score=0.5, success=True)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
        assert c.status()["episodes_seen"] == 40
