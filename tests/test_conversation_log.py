"""Forgetting is recorded, and the window it produces is byte-for-byte the old one."""

from __future__ import annotations

import pytest

from core.context.condenser import (
    AmortizedForgettingCondenser,
    Condensation,
    ContextEvent,
    LLMSummarizingCondenser,
    View,
)
from core.context.conversation_log import ConversationLog, get_conversation_log


def _msgs(n: int, *, start: int = 0, role: str = "user") -> list[dict]:
    return [{"role": role, "content": f"message {i}"} for i in range(start, start + n)]


def _prune_like_live(history: list[dict], keep: int = 30) -> list[dict]:
    """What context_pruner actually does: summary in front, recent tail behind."""
    return [{"role": "system", "content": "[CONSOLIDATED MEMORY]: things happened"}] + history[-keep:]


# ── the property the wiring exists for ───────────────────────────────────


def test_the_window_is_unchanged_by_being_recorded():
    log = ConversationLog()
    before = _msgs(80)
    after = _prune_like_live(before)
    log.record_pruning(before, after)
    assert log.live_history() == after


def test_an_earlier_window_can_be_reconstructed():
    log = ConversationLog()
    before = _msgs(80)
    after = _prune_like_live(before)
    log.record_pruning(before, after)

    original = log.reconstruct(before_condensation=0)
    assert original == before, "the question that could not be asked before"
    assert log.reconstruct(before_condensation=1) == after


def test_successive_prunings_each_reconstruct():
    log = ConversationLog()
    first = _msgs(80)
    second_input = _prune_like_live(first)
    log.record_pruning(first, second_input)

    grown = second_input + _msgs(70, start=100)
    third = _prune_like_live(grown)
    log.record_pruning(grown, third)

    assert log.reconstruct(before_condensation=0) == first
    assert log.live_history() == third
    assert len(log.condensations()) == 2


def test_reconstruction_excludes_what_was_said_afterwards():
    """Otherwise "the window from three turns ago" arrives carrying every message since."""
    log = ConversationLog()
    first = _msgs(80)
    after_first = _prune_like_live(first)
    log.record_pruning(first, after_first)

    later = after_first + _msgs(70, start=500)
    log.record_pruning(later, _prune_like_live(later))

    recovered = log.reconstruct(before_condensation=0)
    assert recovered == first
    assert {m["content"] for m in recovered}.isdisjoint(
        {m["content"] for m in _msgs(70, start=500)}
    )


def test_a_condensation_names_what_left_and_what_replaced_it():
    log = ConversationLog()
    before = _msgs(80)
    after = _prune_like_live(before)
    condensation = log.record_pruning(before, after, reason="context pruner")

    assert condensation is not None
    assert len(condensation.forgotten_ids) == 50
    assert "CONSOLIDATED MEMORY" in condensation.summary
    assert condensation.summary_offset == 0
    assert condensation.reason == "context pruner"
    assert condensation.tokens_reclaimed > 0


def test_a_plain_tail_truncation_is_recorded_with_no_summary():
    log = ConversationLog()
    before = _msgs(80)
    after = before[-50:]
    condensation = log.record_pruning(before, after, reason="bounded tail")

    assert condensation is not None
    assert len(condensation.forgotten_ids) == 30
    assert condensation.summary == ""
    assert log.live_history() == after
    assert log.reconstruct(before_condensation=0) == before


def test_a_prune_that_changed_nothing_is_not_an_event():
    log = ConversationLog()
    before = _msgs(20)
    assert log.record_pruning(before, list(before)) is None
    assert log.condensations() == []


# ── alignment ────────────────────────────────────────────────────────────


def test_incidental_keys_do_not_break_identity():
    """The pruner passes messages through and may drop routing metadata."""
    log = ConversationLog()
    before = [{"role": "user", "content": f"m{i}", "ts": i} for i in range(80)]
    after = [{"role": "user", "content": f"m{i}"} for i in range(50, 80)]
    condensation = log.record_pruning(before, after)
    assert condensation is not None and len(condensation.forgotten_ids) == 50


def test_repeated_content_aligns_by_position_not_by_first_match():
    log = ConversationLog()
    before = [{"role": "user", "content": "same"} for _ in range(10)]
    after = before[-4:]
    condensation = log.record_pruning(before, after)
    assert condensation is not None
    assert len(condensation.forgotten_ids) == 6
    assert len(log.live_history()) == 4


def test_a_reordered_result_is_a_break_not_a_guess():
    """A record that quietly drifts out of step reads as evidence while being wrong."""
    log = ConversationLog()
    before = _msgs(20)
    after = list(reversed(before[-5:]))
    assert log.record_pruning(before, after) is None
    assert log.report()["alignment_breaks"] == 1


def test_the_log_resynchronises_after_a_break():
    log = ConversationLog()
    before = _msgs(20)
    log.record_pruning(before, list(reversed(before[-5:])))
    assert log.live_history() == list(reversed(before[-5:]))

    grown = log.live_history() + _msgs(60, start=100)
    after = _prune_like_live(grown)
    assert log.record_pruning(grown, after) is not None
    assert log.live_history() == after


def test_scattered_insertions_are_refused():
    log = ConversationLog()
    before = _msgs(10)
    after = [
        {"role": "system", "content": "new A"},
        *before[5:7],
        {"role": "system", "content": "new B"},
        *before[7:],
    ]
    assert log.record_pruning(before, after) is None
    assert log.report()["alignment_breaks"] == 1


# ── bounds ───────────────────────────────────────────────────────────────


def test_the_log_is_bounded():
    log = ConversationLog(capacity=100)
    history: list[dict] = []
    for round_index in range(12):
        history = history + _msgs(40, start=round_index * 40)
        pruned = _prune_like_live(history, keep=20)
        log.record_pruning(history, pruned)
        history = pruned
    assert log.report()["messages_retained"] <= 100
    assert log.report()["evicted_from_log"] > 0


def test_eviction_never_takes_a_live_message():
    """Otherwise the window stops being reconstructible from its own record."""
    log = ConversationLog(capacity=10)
    history = _msgs(60)
    after = _prune_like_live(history, keep=30)
    log.record_pruning(history, after)
    assert log.live_history() == after


def test_condensations_are_bounded():
    log = ConversationLog(condensation_capacity=5)
    history = _msgs(60)
    for round_index in range(9):
        pruned = _prune_like_live(history, keep=20)
        log.record_pruning(history, pruned)
        history = pruned + _msgs(40, start=1000 + round_index * 40)
    assert len(log.condensations()) <= 5


def test_the_report_counts_what_was_forgotten():
    log = ConversationLog()
    before = _msgs(80)
    log.record_pruning(before, _prune_like_live(before))
    report = log.report()
    assert report["condensations"] == 1
    assert report["messages_forgotten"] == 50
    assert report["alignment_breaks"] == 0


def test_the_singleton_is_stable():
    assert get_conversation_log() is get_conversation_log()


# ── the condenser extension this required ────────────────────────────────


def test_two_summaries_no_longer_share_an_identity():
    a = Condensation(forgotten_ids=(1,), summary="first", summary_offset=0, summary_id=-1)
    b = Condensation(forgotten_ids=(2,), summary="second", summary_offset=0, summary_id=-2)
    assert a.summary_event.event_id != b.summary_event.event_id


def test_a_summary_id_may_not_collide_with_a_source_event():
    with pytest.raises(ValueError, match="must be negative"):
        Condensation(forgotten_ids=(), summary="s", summary_offset=0, summary_id=7)


def test_a_later_condensation_can_subsume_an_earlier_summary():
    events = [ContextEvent(event_id=i, kind="user", content=f"m{i}") for i in range(4)]
    first = Condensation(
        forgotten_ids=(0, 1), summary="summary one", summary_offset=0, summary_id=-1
    )
    second = Condensation(
        forgotten_ids=(-1, 2), summary="summary two", summary_offset=0, summary_id=-2
    )
    view = View.from_events(events, (first, second))
    contents = [e.content for e in view.events]
    assert contents == ["summary two", "m3"], "the first summary was folded into the second"


def test_summaries_are_immortal_unless_subsumption_is_asked_for():
    condenser = LLMSummarizingCondenser(summarize=lambda events: "s", max_size=4, keep_first=1)
    view = View(
        events=(
            ContextEvent(event_id=1, kind="user", content="head"),
            ContextEvent(event_id=-1, kind="condensation", content="old summary"),
            *[ContextEvent(event_id=i, kind="user", content=f"m{i}") for i in range(2, 8)],
        )
    )
    result = condenser.condense(view)
    assert isinstance(result, Condensation)
    assert -1 not in result.forgotten_ids


def test_subsumption_lets_a_summarizer_fold_the_previous_snapshot():
    seen: list[str] = []

    def summarize(events):
        seen.extend(e.content for e in events)
        return "integrated"

    condenser = LLMSummarizingCondenser(
        summarize=summarize, max_size=4, keep_first=1, voice_anchors=0, subsume_summaries=True
    )
    view = View(
        events=(
            ContextEvent(event_id=1, kind="user", content="head"),
            ContextEvent(event_id=-1, kind="condensation", content="old summary"),
            *[ContextEvent(event_id=i, kind="user", content=f"m{i}") for i in range(2, 8)],
        )
    )
    result = condenser.condense(view)
    assert isinstance(result, Condensation)
    assert -1 in result.forgotten_ids
    assert "old summary" in seen, "the summarizer must see what it is folding in"


def test_amortized_forgetting_refuses_subsumption():
    """It writes no summary, so subsuming one would delete the span outright."""
    with pytest.raises(ValueError, match="writes no summary"):
        AmortizedForgettingCondenser(max_size=10, subsume_summaries=True)


# ── the live path actually records ───────────────────────────────────────


def _pruned_target(history):
    """A real mixin instance — the point is to exercise the live method."""
    from core.orchestrator.mixins.context_streaming import ContextStreamingMixin

    class Target(ContextStreamingMixin):
        def __init__(self, messages):
            self.conversation_history = list(messages)
            self.cognitive_engine = object()

    return Target(history)


@pytest.mark.asyncio
async def test_the_live_pruning_path_records_the_forgetting(monkeypatch):
    from core.context import conversation_log as cl
    log = ConversationLog()
    monkeypatch.setattr(cl, "_log", log)

    class Pruner:
        async def prune_history(self, history, _engine):
            return _prune_like_live(history)

    monkeypatch.setattr("core.memory.context_pruner.context_pruner", Pruner())

    target = _pruned_target(_msgs(80))
    before = list(target.conversation_history)
    await target._prune_history_async()

    assert target.conversation_history == log.live_history()
    assert log.reconstruct(before_condensation=0) == before
    assert log.report()["messages_forgotten"] == 50


@pytest.mark.asyncio
async def test_a_rejected_pruner_result_is_still_recorded(monkeypatch):
    """The bounded-tail fallback shortens the window too, so it is a forgetting."""
    from core.context import conversation_log as cl
    log = ConversationLog()
    monkeypatch.setattr(cl, "_log", log)

    class Pruner:
        async def prune_history(self, history, _engine):
            return history[:2]  # suspiciously short; the mixin rejects it

    monkeypatch.setattr("core.memory.context_pruner.context_pruner", Pruner())

    target = _pruned_target(_msgs(120))
    await target._prune_history_async()

    assert log.condensations(), "the fallback dropped messages and said nothing before"
    assert target.conversation_history == log.live_history()


@pytest.mark.asyncio
async def test_a_broken_recorder_never_blocks_the_pruning(monkeypatch):
    """An audit trail is not worth an out-of-memory conversation."""
    from core.context import conversation_log as cl
    class Exploding:
        def record_pruning(self, *_args, **_kwargs):
            raise RuntimeError("recorder is broken")

    monkeypatch.setattr(cl, "get_conversation_log", lambda: Exploding())

    class Pruner:
        async def prune_history(self, history, _engine):
            return _prune_like_live(history)

    monkeypatch.setattr("core.memory.context_pruner.context_pruner", Pruner())

    target = _pruned_target(_msgs(80))
    await target._prune_history_async()
    assert len(target.conversation_history) == 31
