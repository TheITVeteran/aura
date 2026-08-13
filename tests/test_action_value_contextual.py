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
