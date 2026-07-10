"""Contracts for the Practice Director — failure-directed deliberate practice.

The learning stack had proven muscle with no self-direction: the flywheel
practiced all domains uniformly while the live model's own sealed eval sat at
0/5 on program_output and 5/5 on four others. The director turns real failure
receipts into a ranked curriculum and causally steers (a) the flywheel's
practice mix and (b) the specialist scheduler's next-domain choice — with
honesty rails (mastery zeroing, exploration floors, receipt-pinned evidence)
and uniform fallback whenever it is disabled, evidence-free, or broken.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from core.learning.deliberate_practice import (
    PracticeDirector,
    get_practice_director,
    set_practice_director_for_test,
)

pytestmark = pytest.mark.unit

DOMAINS = (
    "arithmetic_chain",
    "linear_equation",
    "modular",
    "sequence",
    "string_transform",
    "program_output",
    "date_arithmetic",
    "unit_conversion",
)


@pytest.fixture(autouse=True)
def _isolate_singleton():
    set_practice_director_for_test(None)
    yield
    set_practice_director_for_test(None)


def _director(tmp_path, *, now=None) -> PracticeDirector:
    if now is None:
        return PracticeDirector(tmp_path)
    return PracticeDirector(tmp_path, now=now)


def _feed(director: PracticeDirector, domain: str, attempts: int, correct: int,
          receipt: str = "r1") -> None:
    director.observe(domain=domain, attempts=attempts, correct=correct,
                     source="test", receipt=receipt)


# ---------------------------------------------------------------------------
# Ranking honesty
# ---------------------------------------------------------------------------

class TestCurriculumRanking:
    def test_failing_domain_outranks_passing_domain(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "program_output", attempts=10, correct=0)
        _feed(director, "sequence", attempts=10, correct=8)
        ranked = director.curriculum()
        assert ranked[0].domain == "program_output"
        assert ranked[0].need > 0.5

    def test_mastered_domain_has_zero_need_despite_ancient_failures(self, tmp_path):
        clock = {"now": 1_000_000.0}
        director = _director(tmp_path, now=lambda: clock["now"])
        _feed(director, "modular", attempts=10, correct=0)  # old failures
        clock["now"] += 30 * 86400.0  # a month later
        _feed(director, "modular", attempts=20, correct=20)  # now solid
        need = next(n for n in director.curriculum() if n.domain == "modular")
        assert need.need == 0.0
        assert need.accuracy is not None and need.accuracy >= 0.95
        assert "maintain" in need.reason

    def test_unobserved_domain_gets_exploration_floor_not_fabricated_score(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "program_output", attempts=10, correct=2)
        unobserved = next(
            n for n in director.curriculum() if n.domain == "unit_conversion"
        )
        assert unobserved.accuracy is None
        assert 0.0 < unobserved.need < 0.5
        assert "explore" in unobserved.reason

    def test_old_failures_decay(self, tmp_path):
        clock = {"now": 1_000_000.0}
        director = _director(tmp_path, now=lambda: clock["now"])
        _feed(director, "modular", attempts=10, correct=0)
        fresh_need = next(n for n in director.curriculum() if n.domain == "modular").need
        clock["now"] += 28 * 86400.0  # four half-lives later
        aged = next(n for n in director.curriculum() if n.domain == "modular")
        # the decayed mass has dropped below the evidence floor — the domain
        # honestly returns to "unobserved" rather than trading on stale news
        assert aged.accuracy is None or aged.need < fresh_need

    def test_every_ranked_need_carries_its_receipts(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "program_output", 10, 1, receipt="/receipts/eval_a.json")
        _feed(director, "program_output", 10, 2, receipt="/receipts/eval_b.json")
        need = director.curriculum()[0]
        assert need.domain == "program_output"
        assert "/receipts/eval_b.json" in need.evidence  # newest first
        assert need.evidence[0] == "/receipts/eval_b.json"


# ---------------------------------------------------------------------------
# Harvest: real receipt shapes, idempotent
# ---------------------------------------------------------------------------

class TestHarvest:
    def _write_eval_report(self, root: Path, run: str, per_domain: dict) -> Path:
        run_dir = root / "compounding" / "runs" / run
        run_dir.mkdir(parents=True)
        path = run_dir / "incumbent_eval.json"
        path.write_text(
            json.dumps(
                {
                    "created_at": 1_000_000.0,
                    "result": {"per_domain": per_domain},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_compounding_eval_reports_become_observations(self, tmp_path):
        path = self._write_eval_report(
            tmp_path,
            "g0000-1",
            {
                "program_output": {"correct": 0, "total": 5},
                "sequence": {"correct": 5, "total": 5},
            },
        )
        director = _director(tmp_path, now=lambda: 1_000_100.0)
        assert director.harvest() == 2
        ranked = director.curriculum()
        assert ranked[0].domain == "program_output"
        assert str(path) in ranked[0].evidence

    def test_harvest_is_idempotent_per_receipt(self, tmp_path):
        self._write_eval_report(
            tmp_path, "g0000-1", {"modular": {"correct": 1, "total": 5}}
        )
        director = _director(tmp_path, now=lambda: 1_000_100.0)
        assert director.harvest() == 1
        assert director.harvest() == 0

    def test_idempotence_survives_restart(self, tmp_path):
        self._write_eval_report(
            tmp_path, "g0000-1", {"modular": {"correct": 1, "total": 5}}
        )
        first = _director(tmp_path, now=lambda: 1_000_100.0)
        assert first.harvest() == 1
        second = _director(tmp_path, now=lambda: 1_000_200.0)
        assert second.harvest() == 0

    def test_corrupt_report_is_skipped_and_marked_seen(self, tmp_path):
        run_dir = tmp_path / "compounding" / "runs" / "g-bad"
        run_dir.mkdir(parents=True)
        (run_dir / "candidate_eval.json").write_text("{not json", encoding="utf-8")
        director = _director(tmp_path)
        assert director.harvest() == 0
        assert director.harvest() == 0  # not retried forever


# ---------------------------------------------------------------------------
# Causal consumers
# ---------------------------------------------------------------------------

class TestFocusedBattery:
    def test_battery_concentrates_on_top_need(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "program_output", attempts=20, correct=1)
        _feed(director, "string_transform", attempts=20, correct=4)
        tasks = director.focused_battery(seed=7, size=12)
        assert len(tasks) == 12
        counts = Counter(task.domain for task in tasks)
        assert counts["program_output"] == 6  # half the burst
        assert counts["string_transform"] >= 3  # a quarter
        assert len(set(task.task_id for task in tasks)) == 12  # no dupes

    def test_battery_is_deterministic(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "modular", attempts=20, correct=2)
        first = [t.task_id for t in director.focused_battery(seed=11, size=8)]
        second = [t.task_id for t in director.focused_battery(seed=11, size=8)]
        assert first == second

    def test_no_evidence_falls_back_to_uniform(self, tmp_path):
        from core.learning.heldout_battery import BatterySpec, generate_battery

        director = _director(tmp_path)
        tasks = director.focused_battery(seed=5, size=8)
        uniform = generate_battery(BatterySpec(seed=5, size=8))
        assert [t.task_id for t in tasks] == [t.task_id for t in uniform]

    def test_kill_switch_restores_uniform(self, tmp_path, monkeypatch):
        from core.learning.heldout_battery import BatterySpec, generate_battery

        monkeypatch.setenv("AURA_DELIBERATE_PRACTICE", "0")
        director = _director(tmp_path)
        _feed(director, "program_output", attempts=20, correct=0)
        tasks = director.focused_battery(seed=5, size=8)
        uniform = generate_battery(BatterySpec(seed=5, size=8))
        assert [t.task_id for t in tasks] == [t.task_id for t in uniform]


class TestFocusDomainChoice:
    def test_picks_highest_need_eligible(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "program_output", attempts=20, correct=1)
        _feed(director, "modular", attempts=20, correct=10)
        assert director.choose_focus_domain(["modular", "program_output"]) == "program_output"

    def test_respects_eligibility(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "program_output", attempts=20, correct=1)
        _feed(director, "modular", attempts=20, correct=10)
        assert director.choose_focus_domain(["modular"]) == "modular"

    def test_none_without_real_evidence_so_caller_keeps_lrt(self, tmp_path):
        director = _director(tmp_path)
        assert director.choose_focus_domain(["modular", "sequence"]) is None

    def test_none_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_DELIBERATE_PRACTICE", "0")
        director = _director(tmp_path)
        _feed(director, "modular", attempts=20, correct=0)
        assert director.choose_focus_domain(["modular"]) is None


# ---------------------------------------------------------------------------
# Live wiring: the flywheel feeds and consults the director
# ---------------------------------------------------------------------------

class TestFlywheelIntegration:
    @pytest.mark.asyncio
    async def test_burst_feeds_per_domain_outcomes_and_uses_focus(
        self, tmp_path, monkeypatch
    ):
        from core.learning.selfplay_flywheel import SelfPlayFlywheel

        director = _director(tmp_path)
        _feed(director, "program_output", attempts=20, correct=1)
        monkeypatch.setattr(
            "core.learning.selfplay_flywheel._resolve_practice_director",
            lambda: director,
        )

        class _Router:
            async def think(self, prompt, **kwargs):
                return "Answer: definitely-wrong"

        monkeypatch.setattr(
            "core.runtime.service_access.resolve_llm_router",
            lambda default=None: _Router(),
        )
        flywheel = SelfPlayFlywheel()
        monkeypatch.setattr(flywheel, "_still_allowed", lambda: True)
        monkeypatch.setattr(flywheel, "_state_path", lambda: tmp_path / "fw.json")
        flywheel.burst_tasks = 8
        flywheel.attempts_per_task = 1

        stats = await flywheel._burst()
        assert stats["attempts"] == 8

        with director._lock:
            observed_domains = {o.domain for o in director._observations if o.source == "selfplay_flywheel"}
        assert "program_output" in observed_domains, "burst outcomes must feed the curriculum"
        # the focused battery concentrated half the burst on the top need
        with director._lock:
            program_attempts = sum(
                o.attempts
                for o in director._observations
                if o.source == "selfplay_flywheel" and o.domain == "program_output"
            )
        assert program_attempts == 4
        # and the outcomes were persisted with a receipt
        ledger = (tmp_path / "practice_curriculum.jsonl").read_text(encoding="utf-8")
        assert "selfplay_flywheel" in ledger and "#seed" in ledger

    @pytest.mark.asyncio
    async def test_broken_director_never_breaks_practice(self, tmp_path, monkeypatch):
        from core.learning.selfplay_flywheel import SelfPlayFlywheel

        class _Exploding(PracticeDirector):
            def focused_battery(self, **kwargs):  # pragma: no cover - guard
                raise RuntimeError("director on fire")

        exploding = _Exploding(tmp_path)
        monkeypatch.setattr(
            "core.learning.selfplay_flywheel._resolve_practice_director",
            lambda: exploding,
        )

        class _Router:
            async def think(self, prompt, **kwargs):
                return "Answer: 0"

        monkeypatch.setattr(
            "core.runtime.service_access.resolve_llm_router",
            lambda default=None: _Router(),
        )
        flywheel = SelfPlayFlywheel()
        monkeypatch.setattr(flywheel, "_still_allowed", lambda: True)
        monkeypatch.setattr(flywheel, "_state_path", lambda: tmp_path / "fw.json")
        flywheel.burst_tasks = 4
        flywheel.attempts_per_task = 1
        stats = await flywheel._burst()
        assert stats["attempts"] == 4, "uniform fallback must keep practice alive"


# ---------------------------------------------------------------------------
# Persistence + self-knowledge
# ---------------------------------------------------------------------------

class TestPersistenceAndSelfKnowledge:
    def test_observations_survive_restart(self, tmp_path):
        first = _director(tmp_path)
        _feed(first, "program_output", attempts=10, correct=1, receipt="/r/e.json")
        first.flush()
        second = _director(tmp_path)
        ranked = second.curriculum()
        assert ranked[0].domain == "program_output"
        assert "/r/e.json" in ranked[0].evidence

    def test_corrupt_ledger_lines_are_skipped(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "modular", attempts=10, correct=1)
        director.flush()
        ledger = tmp_path / "practice_curriculum.jsonl"
        ledger.write_text(
            ledger.read_text(encoding="utf-8") + "{broken\n", encoding="utf-8"
        )
        fresh = _director(tmp_path)
        need = next(n for n in fresh.curriculum() if n.domain == "modular")
        assert need.accuracy is not None  # the good line survived

    def test_ledger_stays_bounded(self, tmp_path):
        director = _director(tmp_path)
        for index in range(60):
            _feed(director, DOMAINS[index % len(DOMAINS)], attempts=4, correct=1,
                  receipt=f"/r/{index}")
            director.flush()
        # force the compact path
        import core.learning.deliberate_practice as module

        original = module._LEDGER_MAX_LINES
        try:
            module._LEDGER_MAX_LINES = 40
            _feed(director, "modular", attempts=4, correct=1)
            director.flush()
        finally:
            module._LEDGER_MAX_LINES = original
        lines = [
            line
            for line in (tmp_path / "practice_curriculum.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert len(lines) <= 40

    def test_why_names_the_direction_with_receipts(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "program_output", attempts=20, correct=1,
              receipt="/receipts/incumbent_eval.json")
        why = director.why()
        assert "program_output" in why
        assert "incumbent_eval.json" in why
        assert "failure-directed" in why

    def test_why_is_honest_when_evidence_free(self, tmp_path):
        director = _director(tmp_path)
        assert "uniformly" in director.why()

    def test_status_is_schema_stable(self, tmp_path):
        director = _director(tmp_path)
        _feed(director, "modular", attempts=10, correct=2)
        status = director.get_status()
        assert status["schema"] == "aura.practice_director.v1"
        assert status["enabled"] is True
        assert status["observations"] == 1
        assert status["top_needs"][0]["domain"] == "modular"
        assert status["top_needs"][0]["reason"]
        assert "direction" in status

    def test_selfreport_carries_the_direction(self, tmp_path, monkeypatch):
        director = _director(tmp_path)
        _feed(director, "program_output", attempts=20, correct=1)
        monkeypatch.setattr(
            "core.runtime.service_access.resolve_practice_director",
            lambda default=None: director,
        )
        from core.learning.learning_selfreport import LearningSelfReport

        lines = LearningSelfReport()._direction_lines()
        assert lines and "program_output" in lines[0]

    def test_selfreport_is_silent_without_a_registered_director(self):
        """No spine registration → no direction line — and crucially no
        implicit singleton reading the real machine's ledger."""
        from core.learning.learning_selfreport import LearningSelfReport

        assert LearningSelfReport()._direction_lines() == []

    def test_singleton_roundtrip(self, tmp_path):
        director = _director(tmp_path)
        set_practice_director_for_test(director)
        assert get_practice_director() is director
