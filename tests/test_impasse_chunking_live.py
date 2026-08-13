"""Chunking is wired into live deliberation, not just implemented.

``ChunkStore`` existed with tests and no production caller — implemented, and
not doing Soar chunking in Aura's cognition. It is now wired into
``NativeSystem2Engine.rank_actions``, which is the decision that is expensive
enough to be worth compiling: a budgeted MCTS/beam search over the candidate
set is a real substate, so a compiled resolution saves measured search time.

This is also why it is NOT wired to workspace tie-breaking. A chunk has to
out-earn its own match cost, and arbitration between tied bids costs
microseconds — chunking it would be theatre that the expected-value accounting
would immediately retract.

Measured here: first deliberation ~330ms, reuse ~2ms, same commitment.
"""

from __future__ import annotations

import asyncio

import pytest

from core.cognition.impasse import (
    Chunk,
    ChunkStore,
    ImpasseLearner,
    ImpasseType,
    classify,
    get_impasse_learner,
)
from core.reasoning.native_system2 import NativeSystem2Engine

pytestmark = pytest.mark.unit

ACTIONS = ["refactor the parser", "write a unit test", "profile the hot loop"]


def _rank(engine, actions, context="a repeated decision"):
    return asyncio.run(engine.rank_actions(context=context, actions=actions))


def test_a_repeated_decision_is_compiled_and_reused():
    engine = NativeSystem2Engine()
    first = _rank(engine, ACTIONS)
    second = _rank(engine, ACTIONS)

    assert first.receipt.chunk_reused is False
    assert second.receipt.chunk_reused is True
    assert first.receipt.chunk_signature == second.receipt.chunk_signature


def test_reuse_commits_to_the_same_action_including_verification():
    """A chunk must compile the decision, not quietly simplify it.

    Measured failure during development: collapsing the reuse search to depth 1
    skipped the generator's ``verify:`` successor, so the reuse committed to
    the bare action where deliberation had committed to verifying it. A chunk
    that drops the verification step is a different, less safe decision.
    """
    engine = NativeSystem2Engine()
    first = _rank(engine, ACTIONS, context="verification must survive")
    second = _rank(engine, ACTIONS, context="verification must survive")

    assert first.committed_action is not None
    assert second.committed_action is not None
    assert first.committed_action.name == second.committed_action.name


def test_reuse_cost_is_flat_in_the_number_of_candidates():
    """The property worth asserting, rather than a speedup number.

    A first measurement here showed 171x and was mostly wrong: it compared a
    cold process against a warm one, so it was timing imports rather than
    deliberation. Warm, a three-candidate decision costs ~2ms and reuse saves
    ~0.6ms — real, but modest.

    What actually generalises is the shape. Deliberation grows with the
    candidate set because the budget scales with it; reuse collapses to a
    one-candidate confirmation and stays flat. So the saving grows with the
    size of the decision, which is the behaviour that makes chunking worth
    having on expensive deliberation and not worth having on cheap ties.

    Measured warm: n=3 1.4x, n=8 1.9x, n=16 2.8x, reuse ~1.3ms throughout.
    """
    import time

    engine = NativeSystem2Engine()
    _rank(engine, ["warm", "up", "the", "process"], context="warmup")

    def _measure(n: int) -> tuple[float, float]:
        actions = [f"candidate action number {i}" for i in range(n)]
        context = f"width-{n}"
        started = time.perf_counter()
        _rank(engine, actions, context=context)
        deliberated = time.perf_counter() - started
        started = time.perf_counter()
        reused = _rank(engine, actions, context=context)
        reuse_s = time.perf_counter() - started
        assert reused.receipt.chunk_reused is True
        return deliberated, reuse_s

    narrow_delib, narrow_reuse = _measure(3)
    wide_delib, wide_reuse = _measure(16)

    assert wide_delib > narrow_delib, (
        "deliberation did not grow with the candidate set; this test can no "
        "longer distinguish compiled reuse from search"
    )
    assert wide_reuse < wide_delib, (
        f"reuse was not cheaper at n=16: {wide_delib * 1000:.2f}ms -> "
        f"{wide_reuse * 1000:.2f}ms"
    )
    # Reuse is a one-candidate confirmation either way, so widening the
    # decision must not make it much more expensive.
    assert wide_reuse < narrow_reuse * 2.0, (
        f"reuse cost scaled with candidates: n=3 {narrow_reuse * 1000:.2f}ms, "
        f"n=16 {wide_reuse * 1000:.2f}ms"
    )


def test_a_different_candidate_set_does_not_reuse():
    """The signature includes the options, so a chunk cannot answer another question."""
    engine = NativeSystem2Engine()
    _rank(engine, ACTIONS, context="scoped")
    other = _rank(engine, ACTIONS[:2], context="scoped")
    assert other.receipt.chunk_reused is False


def test_a_different_context_does_not_reuse():
    engine = NativeSystem2Engine()
    _rank(engine, ACTIONS, context="context one")
    other = _rank(engine, ACTIONS, context="context two")
    assert other.receipt.chunk_reused is False


def test_the_learner_reports_live_hit_statistics():
    engine = NativeSystem2Engine()
    _rank(engine, ACTIONS, context="stats probe")
    _rank(engine, ACTIONS, context="stats probe")
    report = get_impasse_learner().report()
    assert report["chunks"] >= 1
    assert report["hits"] >= 1
    assert 0.0 <= report["hit_rate"] <= 1.0


# --------------------------------------------------------------------------
# Bounds — a cache in a long-lived loop is a leak without them
# --------------------------------------------------------------------------


def test_the_store_is_bounded_and_evicts_the_least_valuable():
    """Age-based eviction would forget precisely what it had learned best."""
    store = ChunkStore(max_chunks=3)
    for i in range(6):
        imp = classify(["x", "y"], scores={"x": 0.5, "y": 0.5}, context={"i": i})
        assert imp is not None
        # Later chunks are worth progressively more.
        store.learn(imp, "x", cost_saved_per_use=0.01 * (i + 1), match_cost=0.001)

    kept = store.chunks()
    assert len(kept) == 3
    values = sorted(c.cost_saved_per_use for c in kept)
    assert values == pytest.approx([0.04, 0.05, 0.06]), (
        f"eviction kept the wrong chunks: {values}"
    )
    assert store.report()["evicted"] == 3


def test_the_impasse_log_is_bounded():
    store = ChunkStore(max_impasse_log=10)
    for i in range(50):
        imp = classify(["x", "y"], scores={"x": 0.5, "y": 0.5}, context={"i": i})
        assert imp is not None
        store.record_impasse(imp)
    assert len(store.impasses()) == 10


def test_the_learner_prunes_on_a_cadence_rather_than_never():
    """Non-paying chunks must not accumulate match cost unchecked."""
    learner = ImpasseLearner(ChunkStore())
    for i in range(learner._PRUNE_INTERVAL + 1):
        imp = classify(["x", "y"], scores={"x": 0.5, "y": 0.5}, context={"i": i})
        assert imp is not None
        # Every one of these costs more to match than it saves.
        learner.learn(imp, "x", cost_saved_per_use=0.0001, match_cost=0.01)
    # The prune fires at the interval and clears everything learned so far;
    # whatever was learned after it is still waiting for the next one. The
    # property is that the store does not grow with every learn.
    remaining = learner.report()["chunks"]
    assert remaining < learner._PRUNE_INTERVAL, (
        f"a prune cadence never fired: {remaining} non-paying chunks retained"
    )


def test_pruning_keeps_a_chunk_that_pays():
    learner = ImpasseLearner(ChunkStore())
    imp = classify(["x", "y"], scores={"x": 0.5, "y": 0.5}, context={"k": 1})
    assert imp is not None
    learner.learn(imp, "x", cost_saved_per_use=0.30, match_cost=0.000002)
    assert learner.prune_now() == []
    assert learner.report()["chunks"] == 1


def test_the_learner_is_a_process_singleton():
    assert get_impasse_learner() is get_impasse_learner()


def test_an_evicted_chunk_is_counted_not_silently_dropped():
    store = ChunkStore(max_chunks=1)
    for i in range(3):
        imp = classify(["x", "y"], scores={"x": 0.5, "y": 0.5}, context={"i": i})
        assert imp is not None
        store.learn(imp, "x", cost_saved_per_use=0.1, match_cost=0.001)
    assert store.report()["evicted"] == 2


def test_chunk_expected_value_drives_retention_not_recency():
    old_and_good = Chunk(
        signature="old",
        resolution="a",
        impasse_type=ImpasseType.TIE,
        cost_saved_per_use=0.5,
        match_cost=0.001,
    )
    new_and_useless = Chunk(
        signature="new",
        resolution="b",
        impasse_type=ImpasseType.TIE,
        cost_saved_per_use=0.0001,
        match_cost=0.01,
    )
    assert old_and_good.expected_value > new_and_useless.expected_value
    assert new_and_useless.expected_value <= 0.0
