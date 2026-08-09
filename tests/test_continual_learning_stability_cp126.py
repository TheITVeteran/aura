"""CP126 contracts for core/advanced_cognition/continual_learning_stability.py.

Ten findings, three critical, and they compose into one sentence: an engine
with no evidence reported that everything was stable, and then recommended
controls nobody ran.

A freshly constructed engine scored 0.0 on all five risk axes — no feature
windows, no canaries, no paired metrics, no value memories — and 0.0 on a
risk axis reads as "no risk". It then emitted strings like
"block_self_modification_until_governance_memories_reverified" that nothing
invoked and nothing acknowledged.
"""

from __future__ import annotations

import json

import pytest

from core.advanced_cognition.continual_learning_stability import (
    ContinualLearningStabilityEngine,
    Measurement,
    StabilityEvidenceError,
)


def _engine(tmp_path=None, **kwargs) -> ContinualLearningStabilityEngine:
    return ContinualLearningStabilityEngine(state_dir=tmp_path, **kwargs)


def _belief(engine, subject, value, *, source="a", predicate="is", confidence=0.8):
    return engine.store_memory(
        kind="belief",
        content={"subject": subject, "predicate": predicate, "value": value},
        provenance={"source": source},
        confidence=confidence,
    )


class TestAnEngineThatCannotSeeIsNotStable:
    def test_an_empty_engine_is_not_ready_rather_than_stable(self):
        """It scored 0.0 on every axis and called that stable."""
        report = _engine().assess_stability()
        assert report.status == "not_ready"
        assert report.metrics["measured_axes"] == []
        assert len(report.metrics["unmeasured_axes"]) == 5

    def test_unmeasured_axes_are_none_not_zero(self):
        metrics = _engine().assess_stability().metrics
        for axis in ("drift", "contradiction", "forgetting", "overfit", "value_drift"):
            assert metrics[axis] is None, axis

    def test_a_partially_instrumented_engine_cannot_claim_stable(self):
        """Some axes measured and good, others blind: "watch" is the most
        that can honestly be said."""
        engine = _engine()
        _belief(engine, "x", "safe")
        report = engine.assess_stability()
        assert report.metrics["measured_axes"] == ["contradiction"]
        assert report.status == "watch"

    def test_a_never_scored_canary_is_not_a_passing_canary(self):
        engine = _engine()
        engine.register_canary("identity_refusal", baseline_score=1.0)
        assert engine._forgetting().measured is False

    def test_a_scored_canary_is_a_measurement(self):
        engine = _engine()
        engine.register_canary("identity_refusal", baseline_score=1.0, min_score=0.9)
        engine.update_canary("identity_refusal", 0.5)
        forgetting = engine._forgetting()
        assert forgetting.measured is True
        assert forgetting.value and forgetting.value > 0.4

    def test_no_value_memories_is_not_evidence_values_are_intact(self):
        engine = _engine()
        _belief(engine, "x", "safe")
        assert engine._value_drift().measured is False

    def test_the_measurement_says_why_it_could_not_measure(self):
        basis = _engine()._drift().basis
        assert "feature windows" in basis

    def test_a_fully_instrumented_healthy_engine_can_still_say_stable(self):
        """Fail-closed must not mean fail-always."""
        engine = _engine()
        for domain in ("d1",):
            for _ in range(6):
                engine.observe_feature_distribution(domain, ["a", "b", "c"])
        engine.register_canary("c", baseline_score=1.0)
        engine.update_canary("c", 1.0)
        for i in range(_min_paired()):
            engine.record_metric("train_score", 0.5, run_id=f"r{i}")
            engine.record_metric("hidden_score", 0.5, run_id=f"r{i}")
        engine.store_memory(
            kind="value", content={"tag": "value", "claim": "honesty"},
            provenance={"source": "charter"}, confidence=1.0,
        )
        _belief(engine, "x", "safe")

        report = engine.assess_stability()
        assert report.metrics["unmeasured_axes"] == []
        assert report.status == "stable"


def _min_paired() -> int:
    from core.advanced_cognition.continual_learning_stability import _MIN_PAIRED_SAMPLES

    return _MIN_PAIRED_SAMPLES


class TestInterventionsSayNobodyRanThem:
    def test_every_intervention_names_an_owner_and_is_unenforced(self):
        engine = _engine()
        engine.register_canary("c", baseline_score=1.0, min_score=0.9)
        engine.update_canary("c", 0.1)
        report = engine.assess_stability()
        for intervention in report.interventions:
            assert intervention["owner"]
            assert intervention["enforced"] is False
            assert "no handler registered" in intervention["acknowledgement"]

    def test_a_registered_handler_closes_the_loop(self):
        seen: list[dict] = []
        engine = _engine()
        engine.set_intervention_handler(lambda i: seen.append(dict(i)) or "frozen")
        engine.register_canary("c", baseline_score=1.0, min_score=0.9)
        engine.update_canary("c", 0.1)
        report = engine.assess_stability()

        enforced = [i for i in report.interventions if i["enforced"]]
        assert enforced
        assert enforced[0]["acknowledgement"] == "frozen"
        assert seen

    def test_a_handler_that_raises_is_recorded_not_swallowed(self):
        def _boom(_intervention):
            raise RuntimeError("owner offline")

        engine = _engine()
        engine.set_intervention_handler(_boom)
        engine.register_canary("c", baseline_score=1.0, min_score=0.9)
        engine.update_canary("c", 0.1)
        report = engine.assess_stability()

        failed = [i for i in report.interventions if "raised" in str(i["acknowledgement"])]
        assert failed
        assert all(i["enforced"] is False for i in failed)

    def test_an_uninstrumented_engine_asks_to_be_instrumented(self):
        report = _engine().assess_stability()
        kinds = {i["kind"] for i in report.interventions}
        assert "not_ready" in kinds


class TestStoringIsIdempotent:
    def test_the_same_memory_twice_is_one_memory(self):
        engine = _engine()
        first = _belief(engine, "x", "safe")
        second = _belief(engine, "x", "safe")
        assert first.record_id == second.record_id
        assert len(engine.memories) == 1

    def test_a_repeat_does_not_re_penalise_its_counterparts(self):
        """It applied -0.04 to every contradicting record on each arrival."""
        engine = _engine()
        _belief(engine, "x", "safe", source="a")
        counterpart = _belief(engine, "x", "unsafe", source="b")
        after_first = counterpart.confidence

        for _ in range(5):
            _belief(engine, "x", "safe", source="a")

        assert counterpart.confidence == after_first

    def test_a_repeat_does_not_grow_the_contradiction_list(self):
        engine = _engine()
        _belief(engine, "x", "safe", source="a")
        counterpart = _belief(engine, "x", "unsafe", source="b")
        for _ in range(3):
            _belief(engine, "x", "safe", source="a")
        assert len(counterpart.contradictions) == 1

    def test_a_repeat_refreshes_the_verification_time(self):
        engine = _engine()
        first = _belief(engine, "x", "safe")
        before = first.last_verified
        again = _belief(engine, "x", "safe")
        assert again.last_verified >= before


class TestContradictionIsAboutClaimsNotFieldNames:
    def test_two_opposing_claims_still_contradict(self):
        engine = _engine()
        _belief(engine, "x", "safe", source="a")
        second = _belief(engine, "x", "unsafe", source="b")
        assert second.contradictions

    def test_two_episodes_in_one_domain_do_not_contradict(self):
        """domain became "subject" and outcome became "value", so every
        episode in a busy domain contradicted every other one."""
        engine = _engine()
        a = engine.store_memory(
            kind="episode",
            content={"observation_id": "o1", "domain": "terminal", "outcome": {"utility": 1.0}},
            provenance={"source": "advanced_cognition"},
        )
        b = engine.store_memory(
            kind="episode",
            content={"observation_id": "o2", "domain": "terminal", "outcome": {"utility": 0.2}},
            provenance={"source": "advanced_cognition"},
        )
        assert a.contradictions == []
        assert b.contradictions == []

    def test_a_record_with_no_claim_does_not_participate(self):
        engine = _engine()
        note = engine.store_memory(
            kind="note", content={"text": "something"}, provenance={"source": "a"}
        )
        assert engine._claim(note) is None

    def test_the_same_value_in_different_json_shapes_is_not_a_contradiction(self):
        engine = _engine()
        _belief(engine, "x", {"a": 1, "b": 2}, source="a")
        second = _belief(engine, "x", {"b": 2, "a": 1}, source="b")
        assert second.contradictions == []


class TestEvidenceIsValidated:
    def test_a_memory_with_no_stated_origin_is_refused(self):
        engine = _engine()
        with pytest.raises(StabilityEvidenceError, match="provenance.source"):
            engine.store_memory(kind="belief", content={"a": 1}, provenance={})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "high"])
    def test_a_non_finite_confidence_is_refused(self, bad):
        engine = _engine()
        with pytest.raises(StabilityEvidenceError):
            engine.store_memory(
                kind="belief", content={"a": 1}, provenance={"source": "s"}, confidence=bad
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_non_finite_metric_is_refused(self, bad):
        with pytest.raises(StabilityEvidenceError):
            _engine().record_metric("train_score", bad)

    def test_a_non_finite_canary_score_is_refused(self):
        with pytest.raises(StabilityEvidenceError):
            _engine().update_canary("c", float("nan"))

    def test_an_oversized_payload_is_refused_rather_than_written(self, tmp_path):
        engine = _engine(tmp_path)
        with pytest.raises(StabilityEvidenceError, match="over the"):
            engine.store_memory(
                kind="belief",
                content={"blob": "x" * (512 * 1024)},
                provenance={"source": "s"},
            )
        assert engine.memories == {}


class TestOverfitIsPairedOrSaysItIsNot:
    def test_paired_runs_produce_a_paired_measurement(self):
        engine = _engine()
        for i in range(_min_paired()):
            engine.record_metric("train_score", 0.5 + 0.05 * i, run_id=f"r{i}")
            engine.record_metric("hidden_score", 0.5, run_id=f"r{i}")
        overfit = engine._overfit()
        assert overfit.measured and overfit.paired is True
        assert overfit.value and overfit.value > 0.0

    def test_unpaired_history_is_flagged_as_unpaired(self):
        """It compared first/last means of two unrelated series and reported
        the difference as an overfit estimate."""
        engine = _engine()
        for i in range(_min_paired()):
            engine.record_metric("train_score", 0.5 + 0.05 * i)
            engine.record_metric("hidden_score", 0.5)
        overfit = engine._overfit()
        assert overfit.measured is True
        assert overfit.paired is False
        assert "UNPAIRED" in overfit.basis

    def test_too_few_samples_is_unmeasured_not_zero(self):
        engine = _engine()
        engine.record_metric("train_score", 0.9)
        engine.record_metric("hidden_score", 0.1)
        assert engine._overfit().measured is False


class TestDurabilityAcrossRestart:
    def test_a_contradiction_penalty_survives_a_reload(self, tmp_path):
        """Only the new record was persisted, so a restart kept half of
        every contradiction update."""
        engine = _engine(tmp_path)
        _belief(engine, "x", "safe", source="a")
        counterpart = _belief(engine, "x", "unsafe", source="b")
        expected = counterpart.confidence
        counterpart_id = counterpart.record_id

        reloaded = _engine(tmp_path)
        assert reloaded.memories[counterpart_id].confidence == pytest.approx(expected)
        assert reloaded.memories[counterpart_id].contradictions

    def test_a_pruned_memory_does_not_come_back(self, tmp_path):
        engine = _engine(tmp_path, horizon_records=2)
        for i in range(5):
            _belief(engine, f"s{i}", "v", source=f"src{i}", confidence=0.1 * (i + 1))
        live = set(engine.memories)
        assert len(live) <= 2

        reloaded = _engine(tmp_path, horizon_records=2)
        assert set(reloaded.memories) == live

    def test_contradiction_ids_do_not_dangle_after_a_reload(self, tmp_path):
        engine = _engine(tmp_path, horizon_records=1)
        _belief(engine, "x", "safe", source="a", confidence=0.9)
        _belief(engine, "x", "unsafe", source="b", confidence=0.3)

        reloaded = _engine(tmp_path, horizon_records=1)
        present = set(reloaded.memories)
        for record in reloaded.memories.values():
            assert set(record.contradictions) <= present

    def test_metric_run_ids_survive_a_reload(self, tmp_path):
        engine = _engine(tmp_path)
        engine.record_metric("train_score", 0.5, run_id="r1")
        engine._persist_state()
        reloaded = _engine(tmp_path)
        assert list(reloaded.metric_history["train_score"]) == [("r1", 0.5)]


class TestAppendIsAnAppend:
    def test_one_line_is_added_per_memory(self, tmp_path):
        """It read the whole file, concatenated a line, and rewrote it — for
        every memory and every report."""
        engine = _engine(tmp_path)
        for i in range(5):
            _belief(engine, f"s{i}", "v", source=f"src{i}")
        lines = (tmp_path / "memory.jsonl").read_text().strip().splitlines()
        assert len(lines) == 5
        assert all(json.loads(line)["record_id"] for line in lines)

    def test_the_log_is_compacted_to_the_retained_set(self, tmp_path, monkeypatch):
        import core.advanced_cognition.continual_learning_stability as module

        monkeypatch.setattr(module, "_COMPACT_THRESHOLD_LINES", 8)
        engine = _engine(tmp_path, horizon_records=3)
        for i in range(20):
            _belief(engine, f"s{i}", "v", source=f"src{i}", confidence=0.5)

        lines = (tmp_path / "memory.jsonl").read_text().strip().splitlines()
        assert len(lines) <= 8, "the log must not grow without bound"


class TestLoadSurvivesCorruption:
    def test_a_malformed_line_is_quarantined_not_raised(self, tmp_path):
        """It raised on the first bad row, inside __init__, so one truncated
        write made the engine unconstructible."""
        engine = _engine(tmp_path)
        _belief(engine, "x", "safe", source="a")
        path = tmp_path / "memory.jsonl"
        path.write_text(path.read_text() + '{"record_id": "half\n')

        reloaded = _engine(tmp_path)
        assert len(reloaded.memories) == 1
        assert reloaded.assess_stability().metrics["quarantined_log_lines"] >= 1

    def test_a_row_with_a_non_finite_confidence_is_quarantined(self, tmp_path):
        path = tmp_path / "memory.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"record_id": "m1", "kind": "belief", "content": {}, "provenance": {},
                 "confidence": "NaN"}
            )
            + "\n"
        )
        engine = _engine(tmp_path)
        assert engine.memories == {}
        assert engine._quarantined_lines == 1

    def test_an_unknown_field_does_not_break_construction(self, tmp_path):
        path = tmp_path / "memory.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"record_id": "m1", "kind": "belief", "content": {}, "provenance": {},
                 "confidence": 0.5, "a_field_from_the_future": 1}
            )
            + "\n"
        )
        engine = _engine(tmp_path)
        assert "m1" in engine.memories

    def test_a_corrupt_state_file_starts_empty_rather_than_raising(self, tmp_path):
        (tmp_path).mkdir(parents=True, exist_ok=True)
        (tmp_path / "stability_state.json").write_text("{not json")
        engine = _engine(tmp_path)
        assert engine.canaries == {}


class TestMeasurementType:
    def test_unmeasured_is_distinguishable_from_zero(self):
        assert Measurement(None, "nothing").measured is False
        assert Measurement(0.0, "measured, and fine").measured is True

    def test_it_serialises_both_states(self):
        assert Measurement(None, "why").as_dict()["measured"] is False
        assert Measurement(0.3, "why", samples=9).as_dict()["samples"] == 9
