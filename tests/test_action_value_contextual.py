"""Q(s,a), continuous refresh, and the closed chunk-correctness loop.

Three deficiencies this pins, all of them the same shape — a mechanism that
existed but was not actually connected to anything that would keep it true:

* the value model was indexed by action identity alone, so it could learn
  "opening Notes tends to succeed" but never "opening Notes succeeds while
  writing and fails while debugging";
* the singleton loaded its statistics at construction and nothing refreshed it,
  so a long-running process ranked forever on the evidence that existed at boot;
* ``ChunkStore.record_outcome`` had no caller, so ``p_correct`` stayed pinned at
  its optimistic default and an over-general chunk could never accumulate the
  negative expected value that retracts it.

All three are now driven by the outcome ledger's resolution stream.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from core.cognition.impasse import ChunkStore, ImpasseLearner, classify
from core.cognition.outcome_ledger import OutcomeLedger
from core.reasoning.action_value import ActionValueModel, on_outcome_resolved
from core.reasoning.native_system2 import NativeSystem2Engine

pytestmark = pytest.mark.unit


def _ledger(tmp: str) -> OutcomeLedger:
    return OutcomeLedger(db_path=os.path.join(tmp, "ledger.db"))


# --------------------------------------------------------------------------
# The regression that hid all of this: statistics that silently returned {}
# --------------------------------------------------------------------------


def test_measured_statistics_actually_come_back():
    """``repeat_count`` was a dataclass field with no column and no migration.

    The statistics query named it, raised ``no such column``, was swallowed at
    debug severity, and returned an empty dict — so every learned action value
    silently degraded to the global prior while the unit tests passed by
    injecting statistics directly. This is the end-to-end check that the
    evidence base is reachable through the database at all.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ledger = _ledger(tmp)
        for action, observed in (("alpha", 0.9), ("alpha", 0.7), ("beta", 0.2)):
            rid = ledger.open(action, 0.5, category="deliberation")
            ledger.resolve(rid, observed)

        stats = ledger.measured_action_stats()
        assert set(stats) == {"alpha", "beta"}
        assert stats["alpha"]["n"] == pytest.approx(2.0)
        assert stats["alpha"]["mean"] == pytest.approx(0.8, abs=1e-6)


def test_repeat_count_survives_a_restart():
    """It was incremented in memory and never persisted."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.db")
        first = OutcomeLedger(db_path=path)
        rid = first.open("repeated", 0.5, category="deliberation")
        first.resolve(rid, 0.9)
        # A second ledger over the same file must read the column back.
        second = OutcomeLedger(db_path=path)
        assert second.measured_action_stats()["repeated"]["n"] >= 1.0


def test_contextual_statistics_are_grouped_by_state():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = _ledger(tmp)
        for state, observed in (
            ("writing", 0.9),
            ("writing", 0.95),
            ("debug", 0.1),
            ("debug", 0.05),
        ):
            rid = ledger.open(
                "open notes", 0.5, category="deliberation", context={"state": state}
            )
            ledger.resolve(rid, observed)

        marginal = ledger.measured_action_stats()
        contextual = ledger.measured_action_stats(by_state=True)
        assert marginal["open notes"]["mean"] == pytest.approx(0.5, abs=0.01)
        assert contextual["writing|open notes"]["mean"] == pytest.approx(0.925, abs=0.01)
        assert contextual["debug|open notes"]["mean"] == pytest.approx(0.075, abs=0.01)


def test_pending_collapse_is_scoped_by_context_and_survives_restart():
    """Two situations are two facts; an identical retry remains one fact."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.db")
        first = OutcomeLedger(db_path=path)
        writing = first.open(
            "open notes", 0.7, category="deliberation", context={"state": "writing"}
        )
        assert first.open(
            "open notes", 0.7, category="deliberation", context={"state": "writing"}
        ) == writing
        debugging = first.open(
            "open notes", 0.7, category="deliberation", context={"state": "debug"}
        )
        assert debugging != writing

        restarted = OutcomeLedger(db_path=path)
        assert restarted.open(
            "open notes", 0.7, category="deliberation", context={"state": "writing"}
        ) == writing
        pending = {row["receipt_id"]: row for row in restarted.pending()}
        assert pending[writing]["repeat_count"] == 2


def test_receipts_without_a_state_stay_out_of_the_contextual_table():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = _ledger(tmp)
        rid = ledger.open("stateless", 0.5, category="deliberation")
        ledger.resolve(rid, 0.9)
        assert ledger.measured_action_stats(by_state=True) == {}
        assert "stateless" in ledger.measured_action_stats()


# --------------------------------------------------------------------------
# The closed loop
# --------------------------------------------------------------------------


def test_a_resolved_outcome_grades_the_chunk_that_produced_it():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = _ledger(tmp)
        learner = ImpasseLearner(ChunkStore())
        impasse = classify(["a", "b"], scores={"a": 0.5, "b": 0.5}, context={"c": 1})
        assert impasse is not None
        learner.learn(impasse, "a", cost_saved_per_use=0.3, match_cost=2e-6)

        import core.cognition.impasse as impasse_module

        previous = impasse_module._learner
        impasse_module._learner = learner
        try:
            ledger.add_resolution_observer(on_outcome_resolved)

            good = ledger.open(
                "a", 0.8, category="deliberation",
                context={"state": "s1", "chunk_signature": impasse.signature},
            )
            ledger.resolve(good, 0.95)
            chunk = learner._store.chunks()[0]
            assert (chunk.correct, chunk.incorrect) == (1, 0)

            bad = ledger.open(
                "a", 0.8, category="deliberation",
                context={"state": "s2", "chunk_signature": impasse.signature},
            )
            ledger.resolve(bad, 0.10)
            chunk = learner._store.chunks()[0]
            assert (chunk.correct, chunk.incorrect) == (1, 1)
            assert chunk.p_correct == pytest.approx(0.5)
        finally:
            impasse_module._learner = previous


def test_a_persistently_wrong_chunk_becomes_negative_and_is_retracted():
    """The loop's purpose: grading is the only thing that can retire a chunk."""
    learner = ImpasseLearner(ChunkStore())
    impasse = classify(["a", "b"], scores={"a": 0.5, "b": 0.5}, context={"c": 2})
    assert impasse is not None
    learner.learn(impasse, "a", cost_saved_per_use=0.01, match_cost=0.005)

    for _ in range(9):
        learner.record_outcome(impasse.signature, correct=False)
    learner.record_outcome(impasse.signature, correct=True)

    assert learner.prune_now(), "a chunk wrong 90% of the time was retained"
    assert learner.report()["chunks"] == 0


def test_an_unrelated_receipt_does_not_grade_anything():
    learner = ImpasseLearner(ChunkStore())
    impasse = classify(["a", "b"], scores={"a": 0.5, "b": 0.5}, context={"c": 3})
    assert impasse is not None
    learner.learn(impasse, "a", cost_saved_per_use=0.3, match_cost=2e-6)

    import core.cognition.impasse as impasse_module

    previous = impasse_module._learner
    impasse_module._learner = learner
    try:

        class Receipt:
            context: dict = {}
            observed = 0.1
            expected = 0.9

        on_outcome_resolved(Receipt())
        chunk = learner._store.chunks()[0]
        assert (chunk.correct, chunk.incorrect) == (0, 0)
    finally:
        impasse_module._learner = previous


def test_the_observer_survives_a_malformed_receipt():
    class Nonsense:
        context = {"chunk_signature": "x"}
        observed = "not a number"
        expected = None

    on_outcome_resolved(Nonsense())  # must not raise


def test_resolution_observers_are_isolated_from_each_other():
    """One broken observer must not stop the others or fail the resolve."""
    with tempfile.TemporaryDirectory() as tmp:
        ledger = _ledger(tmp)
        seen: list[str] = []

        def broken(_receipt):
            raise RuntimeError("observer exploded")

        def working(receipt):
            seen.append(receipt.action)

        ledger.add_resolution_observer(broken)
        ledger.add_resolution_observer(working)
        rid = ledger.open("resilient", 0.5, category="deliberation")
        assert ledger.resolve(rid, 0.9) is not None
        assert seen == ["resilient"]


def test_sweeping_does_not_notify_observers():
    """An expired receipt's zero is a convention, not an observation."""
    with tempfile.TemporaryDirectory() as tmp:
        ledger = _ledger(tmp)
        seen: list[str] = []
        ledger.add_resolution_observer(lambda r: seen.append(r.action))
        ledger.open("never watched", 0.5, category="deliberation", horizon_s=0.0)
        ledger.sweep()
        assert seen == [], "a swept receipt taught a learner that the action failed"


def test_the_model_learns_context_from_the_real_ledger():
    """End to end: receipts in, Q(s,a) out."""
    with tempfile.TemporaryDirectory() as tmp:
        ledger = _ledger(tmp)
        # Writers store the DIGEST, readers pass the raw situation and the
        # model digests it. Exactly one hashing site: storing raw here and
        # digesting on read is the mismatch that made every contextual lookup
        # miss and fall back to V(a).
        for state, observed in (
            ("writing", 0.95),
            ("writing", 0.9),
            ("debug", 0.05),
            ("debug", 0.1),
        ):
            rid = ledger.open(
                "open notes",
                0.5,
                category="deliberation",
                context={"state": ActionValueModel.state_key(state)},
            )
            ledger.resolve(rid, observed)
        for observed in (0.6, 0.4):
            rid = ledger.open("other action", 0.5, category="deliberation")
            ledger.resolve(rid, observed)

        model = ActionValueModel()
        model.refresh(ledger=ledger)
        good = model.value_for("open notes", state="writing")
        bad = model.value_for("open notes", state="debug")
        assert good.evidence == "learned_contextual"
        assert good.value > bad.value


def test_real_acceptance_and_resolution_close_the_native_system2_loop(monkeypatch):
    """Simulation emits provenance; only accepted execution opens evidence."""
    with tempfile.TemporaryDirectory() as tmp:
        ledger = _ledger(tmp)
        monkeypatch.setattr(
            "core.cognition.outcome_ledger.get_outcome_ledger", lambda: ledger
        )

        import asyncio

        engine = NativeSystem2Engine()
        result = asyncio.run(
            engine.rank_actions(
                context="Writing a note for Bryan",
                actions=[
                    {"name": "OPEN NOTES", "score_hint": 0.9},
                    {"name": "do nothing", "score_hint": 0.1},
                ],
            )
        )
        assert ledger.pending() == [], "a simulated ranking created a phantom action"
        assert result.receipt.outcome_state_key == ActionValueModel.state_key(
            "Writing a note for Bryan"
        )
        assert result.receipt.outcome_action == "open notes"

        outcome_id = engine.open_outcome_receipt(result.search_id)
        assert outcome_id
        assert engine.open_outcome_receipt(result.search_id) == outcome_id
        assert len(ledger.pending()) == 1
        assert engine.resolve_outcome_receipt(outcome_id, 1.0, note="verified")

        contextual = ledger.measured_action_stats(by_state=True)
        key = f"{result.receipt.outcome_state_key}|open notes"
        assert contextual[key]["mean"] == pytest.approx(1.0)


def test_value_lookup_never_refreshes_sqlite_synchronously():
    model = ActionValueModel({"known": {"n": 2.0, "mean": 0.8, "m2": 0.1}})
    model.mark_stale()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("value_for performed database refresh")

    model.refresh = fail_if_called  # type: ignore[method-assign]
    assert model.value_for("known").value == pytest.approx(0.8)


def test_native_receipt_cache_is_bounded_and_reports_eviction():
    import asyncio

    engine = NativeSystem2Engine()
    engine.MAX_RECEIPTS = 2
    for index in range(3):
        asyncio.run(
            engine.rank_actions(
                context=f"bounded-{index}",
                actions=[
                    {"name": "a", "score_hint": 0.8},
                    {"name": "b", "score_hint": 0.2},
                ],
            )
        )
    status = engine.get_status()
    assert status["receipts"] == 2
    assert status["receipt_evictions"] == 1
