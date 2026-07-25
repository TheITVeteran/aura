"""CP126: the last eight strong-evidence findings.

* ``128107a8`` — tool execution constructed and registered an ad hoc
  CapabilityEngine when the real one was missing, publishing a partially
  wired authority under the canonical name during a fault.
* ``8d7a39ac`` — stop() emptied the phase list and nothing else; the engine
  kept thinking, streaming, generating and publishing afterwards.
* ``6b3e534c`` — deep-deliberation health was the literal True.
* ``e7bc7faa`` — schedule status summed mutable records outside the lock, so
  the reported counts need not describe any state that ever existed.
* ``74987822`` — SearchResult could not reproduce or attribute its own scores.
* ``a3f8d861`` — a compound answer cleared objective coverage with two
  matched terms and 8%.
* ``78632859`` — several runners coerced malformed solver outcomes into
  evidence with bool()/int()/float().
* ``51654706`` — long experiment runs had no durable receipt or resume.
"""
from __future__ import annotations

import inspect
import math
import pathlib
import tempfile

import pytest


class TestToolExecutionRefusesAdHocAuthority:
    def test_no_engine_is_constructed_on_the_fault_path(self):
        from core import capability_engine

        source = inspect.getsource(capability_engine.execute_tool)
        assert "engine = CapabilityEngine()" not in source
        assert "capability_engine_unavailable" in source

    def test_the_refusal_carries_a_readiness_receipt(self):
        from core import capability_engine

        source = inspect.getsource(capability_engine.execute_tool)
        assert '"readiness"' in source
        assert '"registered": False' in source

    @pytest.mark.asyncio
    async def test_missing_router_returns_not_ready(self, monkeypatch):
        from core import capability_engine

        monkeypatch.setattr(
            "core.runtime.service_access.optional_service",
            lambda *a, **k: None,
        )
        result = await capability_engine.execute_tool("file_operation", {})
        assert result["ok"] is False
        assert result["reason"] == "capability_engine_unavailable"
        assert result["readiness"]["registered"] is False


class TestStopActuallyStops:
    def _engine(self):
        from core.brain.cognitive_engine import CognitiveEngine

        engine = CognitiveEngine.__new__(CognitiveEngine)
        engine._stopped = False
        engine._active_tasks = set()
        engine._phases = ["phase-a", "phase-b"]
        return engine

    def test_a_running_engine_is_not_stopped(self):
        assert self._engine().stopped is False

    def test_stop_marks_the_engine_stopped(self):
        engine = self._engine()
        engine.stop()
        assert engine.stopped is True
        assert engine._phases == []

    def test_a_stopped_engine_refuses_cognitive_work(self):
        engine = self._engine()
        engine.stop()
        for operation in ("think", "think_stream", "generate"):
            with pytest.raises(RuntimeError, match="cognitive_engine_stopped"):
                engine._refuse_if_stopped(operation)

    def test_stop_cancels_in_flight_work(self):
        class _Task:
            def __init__(self):
                self.cancelled = False

            def done(self):
                return False

            def cancel(self):
                self.cancelled = True

        engine = self._engine()
        task = _Task()
        engine._active_tasks = {task}
        engine.stop()
        assert task.cancelled is True
        assert engine._active_tasks == set()

    def test_the_entry_points_consult_the_flag(self):
        from core.brain import cognitive_engine

        for name in ("think", "think_stream", "generate"):
            source = inspect.getsource(getattr(cognitive_engine.CognitiveEngine, name))
            assert "_refuse_if_stopped" in source, name


class TestDeliberationHealthIsDerived:
    def _engine(self):
        from core.brain.deep_deliberation import DeepDeliberationEngine

        engine = DeepDeliberationEngine.__new__(DeepDeliberationEngine)
        engine._deliberations = 0
        engine._model_backed = 0
        engine._unbacked = 0
        engine._failures = 0
        engine._consecutive_failures = 0
        engine._last_latency_s = 0.0
        engine._last_completed_at = 0.0
        return engine

    def test_health_is_not_a_constant(self):
        from core.brain import deep_deliberation

        source = inspect.getsource(deep_deliberation.DeepDeliberationEngine.get_status)
        assert '"healthy": True' not in source

    def test_untested_is_distinct_from_well(self):
        status = self._engine().get_status()
        assert status["state"] == "untested"
        assert status["healthy"] is True

    def test_a_streak_of_unbacked_deliberations_is_degraded(self):
        from core.brain import deep_deliberation

        engine = self._engine()
        engine._unbacked = deep_deliberation._UNHEALTHY_FAILURE_STREAK
        engine._consecutive_failures = deep_deliberation._UNHEALTHY_FAILURE_STREAK
        status = engine.get_status()
        assert status["healthy"] is False
        assert status["state"] == "degraded"
        assert status["unhealthy_reasons"]

    def test_model_backed_work_is_healthy(self):
        engine = self._engine()
        engine._model_backed = 9
        engine._unbacked = 1
        engine._consecutive_failures = 1
        status = engine.get_status()
        assert status["healthy"] is True
        assert status["model_backed_rate"] == pytest.approx(0.9)

    def test_a_low_model_backed_rate_is_degraded(self):
        engine = self._engine()
        engine._model_backed = 1
        engine._unbacked = 9
        engine._consecutive_failures = 1
        assert engine.get_status()["healthy"] is False


class TestScheduleStatusIsOneSnapshot:
    def test_every_field_is_read_under_the_lock(self):
        from core.brain.llm.latent_cortex import schedules

        source = inspect.getsource(schedules.ScheduleLibrary.status)
        body = source.split("with self._lock:", 1)[1]
        # The counts must be computed before the lock is released, i.e. no
        # summation over records outside the with-block.
        assert "observations += int(record.trials)" in body
        assert "sum(record.trials" not in source


class TestSearchResultIsReproducible:
    def _result(self, **kwargs):
        from core.brain.llm.latent_cortex.schedules import LayerSchedule, SearchResult

        return SearchResult(
            best=LayerSchedule.__new__(LayerSchedule),
            best_score=1.0,
            evaluated=10,
            **kwargs,
        )

    def test_a_bare_result_names_every_gap(self):
        gaps = self._result().reproduction_gaps()
        for expected in (
            "seed_absent",
            "evaluator_unidentified",
            "topology_unrecorded",
            "budget_unrecorded",
            "per_evaluation_receipts_absent",
        ):
            assert expected in gaps

    def test_a_complete_result_has_no_gaps(self):
        result = self._result(
            seed=7,
            evaluator_id="verifier-a",
            topology={"layers": 64},
            budget_evaluations=100,
            evaluation_receipts=[{"score": 1.0}],
        )
        assert result.reproduction_gaps() == []

    def test_budget_exhaustion_is_recorded(self):
        assert self._result().budget_exhausted is False
        assert self._result(budget_exhausted=True).budget_exhausted is True


class TestObjectiveCoverageScales:
    def test_the_floor_is_no_longer_two_terms_at_eight_percent(self):
        from core.brain.llm.latent_cortex import output_quality

        source = inspect.getsource(output_quality)
        assert "len(matched_objective_terms) < 2 or objective_coverage < 0.08" not in source

    def test_the_requirement_scales_with_the_objective(self):
        from core.brain.llm.latent_cortex import output_quality

        assert output_quality._MIN_OBJECTIVE_TERM_MATCHES >= 3
        assert output_quality._OBJECTIVE_TERM_MATCH_RATIO > 0.0
        assert output_quality._MIN_OBJECTIVE_COVERAGE > 0.08

    def test_more_of_the_objective_is_examined(self):
        from core.brain.llm.latent_cortex import output_quality

        assert output_quality._MAX_OBJECTIVE_TERMS > 32

    def test_truncation_is_disclosed(self):
        from core.brain.llm.latent_cortex import output_quality

        source = inspect.getsource(output_quality)
        assert '"objective_terms_truncated"' in source


class TestSolverOutcomesAreStrict:
    def test_the_factorial_runner_uses_the_contract(self):
        from core.brain.llm.latent_cortex import experiments

        source = inspect.getsource(experiments.run_factorial_ablations)
        assert "_coerce_solver_outcome(solve_arm(task, arm))" in source
        assert "int(bool(success))" not in source

    def test_the_role_runner_validates_all_three_fields(self):
        from core.brain.llm.latent_cortex.experiments import _coerce_role_outcome

        assert _coerce_role_outcome((True, 3, 0.5)) == (True, 3, 0.5)
        assert _coerce_role_outcome((True, 3, None)) == (True, 3, None)

    @pytest.mark.parametrize(
        "bad",
        [
            ("yes", 3, 0.5),          # non-empty string was a success
            (True, -1, 0.5),          # negative cost reached reports
            # NOTE: NaN is NOT here. It is this contract's documented
            # sentinel for "no exchange telemetry", every consumer filters
            # it with math.isfinite before aggregating, and refusing it
            # broke conjecture-without-telemetry. Rejecting NaN reflexively
            # was this campaign's error, caught by
            # test_run_role_lesion_conjectures_without_telemetry.
            (True, 3, float("inf")),
            (True, 3, -1.0),          # a distance cannot be negative
            (True, 3, "x"),
            (True, 3),                # wrong arity
            (True, 3.5, 0.5),         # fractional cost was truncated
        ],
    )
    def test_malformed_outcomes_are_refused(self, bad):
        from core.brain.llm.latent_cortex.experiments import _coerce_role_outcome

        with pytest.raises(ValueError):
            _coerce_role_outcome(bad)


class TestTrialJournalMakesRunsResumable:
    def _journal(self, tmp, manifest=None):
        from core.brain.llm.latent_cortex.trial_journal import TrialJournal

        return TrialJournal(
            pathlib.Path(tmp) / "run.jsonl",
            manifest=manifest if manifest is not None else {"arms": ["a"], "tasks": 2},
        ).open()

    def test_trials_are_durable_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = self._journal(tmp)
            journal.record("t1", ok=True, payload={"score": 1})
            reopened = self._journal(tmp)
            assert reopened.is_complete("t1")
            assert reopened.resumed is True

    def test_a_failing_trial_does_not_destroy_completed_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = self._journal(tmp)
            journal.run_trial("t1", lambda: {"score": 1})
            record = journal.run_trial(
                "t2", lambda: (_ for _ in ()).throw(ValueError("boom")),
            )
            assert record.ok is False
            assert "boom" in record.error
            summary = journal.summary()
            assert summary["succeeded"] == 1
            assert summary["failed"] == 1

    def test_completed_trials_are_not_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = self._journal(tmp)
            journal.run_trial("t1", lambda: {"score": 1})
            resumed = self._journal(tmp)
            ran = []
            resumed.run_trial("t1", lambda: ran.append("rerun"))
            assert ran == []

    def test_a_different_manifest_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._journal(tmp)
            with pytest.raises(ValueError, match="manifest_mismatch"):
                self._journal(tmp, manifest={"arms": ["a", "b"], "tasks": 2})

    def test_a_truncated_final_line_does_not_lose_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = self._journal(tmp)
            journal.record("t1", ok=True)
            with journal.path.open("a", encoding="utf-8") as handle:
                handle.write('{"key": "t2", "ok"')
            resumed = self._journal(tmp)
            assert resumed.completed_keys() == {"t1"}
            assert resumed.skipped_corrupt_lines == 1

    def test_the_factorial_runner_resumes(self):
        from core.brain.llm.latent_cortex.experiments import (
            Task,
            run_factorial_ablations,
        )

        def task(seed):
            return Task(prompt="p", answer="a", family="fam", depth=2, seed=seed)

        tasks = {"fam": [task(1), task(2)]}
        calls: list[tuple] = []

        def solve(t, arm):
            calls.append((t.seed, arm))
            return (arm != "vanilla", 3)

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "j.jsonl"
            run_factorial_ablations(solve, tasks, arms=("recurrence",), journal_path=path)
            first = len(calls)
            assert first == 4
            calls.clear()
            run_factorial_ablations(solve, tasks, arms=("recurrence",), journal_path=path)
            assert calls == []

    def test_no_journal_keeps_the_original_behaviour(self):
        from core.brain.llm.latent_cortex.experiments import (
            Task,
            run_factorial_ablations,
        )

        tasks = {
            "fam": [Task(prompt="p", answer="a", family="fam", depth=2, seed=1)],
        }
        result = run_factorial_ablations(
            lambda t, arm: (arm != "vanilla", 2), tasks, arms=("recurrence",),
        )
        assert "claims" in result

    def test_the_manifest_digest_is_order_stable(self):
        from core.brain.llm.latent_cortex.trial_journal import manifest_digest

        assert manifest_digest({"a": 1, "b": 2}) == manifest_digest({"b": 2, "a": 1})
        assert manifest_digest({"a": 1}) != manifest_digest({"a": 2})

    def test_a_trial_payload_survives_the_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = self._journal(tmp)
            journal.record("t1", ok=True, payload={"success": True, "cost": 7})
            resumed = self._journal(tmp)
            record = resumed.get("t1")
            assert record is not None
            assert record.payload["cost"] == 7
            assert math.isclose(float(record.payload["cost"]), 7.0)
