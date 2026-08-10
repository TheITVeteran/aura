"""Forgetting that can be replayed, and that cannot drop a promise.

Aura's two existing compressors mutate a message list in place. After they run,
what left the window is unrecoverable from the structure, so no past turn can be
reconstructed as it actually looked. These cover the replacement contract: the
log is append-only, the view is derived, and pinned events survive every
strategy.
"""
from __future__ import annotations

import pytest

from core.context.condenser import (
    AmortizedForgettingCondenser,
    Condensation,
    ContextEvent,
    LLMSummarizingCondenser,
    NoOpCondenser,
    ObservationMaskingCondenser,
    PipelineCondenser,
    View,
    estimate_tokens,
)

pytestmark = pytest.mark.unit


def _events(count: int, *, kind: str = "user", pinned_ids: set[int] = frozenset()):
    return [
        ContextEvent(
            event_id=i,
            kind=kind,
            content=f"event {i} content",
            pinned=i in pinned_ids,
        )
        for i in range(count)
    ]


# ── the view is derived, never edited ──────────────────────────────────────


def test_a_view_with_no_condensations_is_the_log():
    events = _events(5)

    view = View.from_events(events)

    assert [e.event_id for e in view] == [0, 1, 2, 3, 4]


def test_condensation_removes_the_forgotten_and_inserts_the_summary():
    events = _events(5)
    condensation = Condensation(
        forgotten_ids=(1, 2, 3), summary="they discussed the plan", summary_offset=1
    )

    view = View.from_events(events, [condensation])

    assert [e.event_id for e in view] == [0, -1, 4]
    assert view[1].content == "they discussed the plan"
    assert view[1].kind == "condensation"


def test_a_past_window_is_reconstructible_from_the_log_and_its_condensations():
    """The whole point: keep the log, get any past context back."""
    events = _events(6)
    first = Condensation(forgotten_ids=(1, 2), summary="early", summary_offset=1)
    second = Condensation(forgotten_ids=(3, 4), summary="later", summary_offset=2)

    before = View.from_events(events, [first])
    after = View.from_events(events, [first, second])

    assert [e.event_id for e in before] == [0, -1, 3, 4, 5]
    assert [e.event_id for e in after] == [0, -1, -1, 5]
    # The earlier view is still exactly reproducible after the later one exists.
    assert View.from_events(events, [first]).events == before.events


def test_an_empty_summary_inserts_nothing():
    events = _events(4)
    condensation = Condensation(forgotten_ids=(1, 2), summary="", summary_offset=1)

    view = View.from_events(events, [condensation])

    assert [e.event_id for e in view] == [0, 3]


def test_a_negative_offset_is_refused():
    with pytest.raises(ValueError):
        Condensation(forgotten_ids=(1,), summary="s", summary_offset=-1)


# ── amortized forgetting ───────────────────────────────────────────────────


def test_under_the_limit_nothing_is_forgotten():
    condenser = AmortizedForgettingCondenser(max_size=10, keep_first=2)

    result = condenser.condense(View.from_events(_events(5)))

    assert isinstance(result, View)


def test_over_the_limit_the_middle_is_dropped():
    condenser = AmortizedForgettingCondenser(max_size=10, keep_first=2)

    result = condenser.condense(View.from_events(_events(20)))

    assert isinstance(result, Condensation)
    assert result.forgotten_ids
    # Head survives.
    assert 0 not in result.forgotten_ids and 1 not in result.forgotten_ids
    # Most recent survives.
    assert 19 not in result.forgotten_ids


def test_the_resulting_view_is_under_the_target():
    condenser = AmortizedForgettingCondenser(max_size=10, keep_first=2)
    view = View.from_events(_events(20))

    result = condenser.condense(view)
    condensed = view.apply(result)

    assert len(condensed) <= condenser.max_size


# ── the pin: a promise cannot slide out of the window ──────────────────────


def test_a_pinned_event_is_never_forgotten():
    condenser = AmortizedForgettingCondenser(max_size=10, keep_first=2)
    events = _events(20, pinned_ids={7, 8})

    result = condenser.condense(View.from_events(events))

    assert isinstance(result, Condensation)
    assert 7 not in result.forgotten_ids
    assert 8 not in result.forgotten_ids


def test_a_pinned_event_survives_into_the_condensed_view():
    condenser = AmortizedForgettingCondenser(max_size=10, keep_first=2)
    events = _events(20, pinned_ids={7})
    view = View.from_events(events)

    condensed = view.apply(condenser.condense(view))

    assert 7 in [e.event_id for e in condensed]


def test_an_all_pinned_middle_refuses_to_forget():
    """The budget problem is real; dropping a commitment is not the answer."""
    condenser = AmortizedForgettingCondenser(max_size=10, keep_first=2)
    events = _events(20, pinned_ids=set(range(2, 17)))

    result = condenser.condense(View.from_events(events))

    assert isinstance(result, View)


def test_a_prior_summary_is_not_itself_forgotten():
    """It carries the only trace of the span it replaced."""
    condenser = AmortizedForgettingCondenser(max_size=6, keep_first=1)
    events = _events(20)
    first = Condensation(forgotten_ids=(1, 2, 3), summary="early work", summary_offset=1)
    view = View.from_events(events, [first])

    result = condenser.condense(view)
    condensed = view.apply(result) if isinstance(result, Condensation) else result

    assert any(e.kind == "condensation" for e in condensed)


# ── LLM summarization ──────────────────────────────────────────────────────


def test_the_summary_replaces_the_forgotten_span():
    condenser = LLMSummarizingCondenser(
        summarize=lambda events: f"summary of {len(events)} events",
        max_size=10,
        keep_first=2,
    )
    view = View.from_events(_events(20))

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary.startswith("summary of")


def test_a_failed_summarizer_keeps_the_events_rather_than_dropping_them():
    """Too long beats quietly wrong."""

    def explode(events):
        raise RuntimeError("cortex unavailable")

    condenser = LLMSummarizingCondenser(summarize=explode, max_size=10, keep_first=2)

    result = condenser.condense(View.from_events(_events(20)))

    assert isinstance(result, View)


def test_an_empty_summary_keeps_the_events_rather_than_erasing_the_span():
    condenser = LLMSummarizingCondenser(
        summarize=lambda events: "   ", max_size=10, keep_first=2
    )

    result = condenser.condense(View.from_events(_events(20)))

    assert isinstance(result, View)


def test_the_summarizer_only_sees_what_is_actually_being_forgotten():
    seen = {}

    def capture(events):
        seen["ids"] = [e.event_id for e in events]
        return "s"

    condenser = LLMSummarizingCondenser(summarize=capture, max_size=10, keep_first=2)
    condenser.condense(View.from_events(_events(20, pinned_ids={6})))

    assert 6 not in seen["ids"]
    assert 0 not in seen["ids"]


def test_tokens_reclaimed_is_accounted():
    condenser = LLMSummarizingCondenser(
        summarize=lambda events: "tiny", max_size=10, keep_first=2
    )

    result = condenser.condense(View.from_events(_events(20)))

    assert result.tokens_reclaimed > 0


# ── observation masking ────────────────────────────────────────────────────


def test_stale_observations_are_blanked_but_stay_in_the_window():
    condenser = ObservationMaskingCondenser(attention_window=3)
    events = _events(10, kind="observation")

    result = condenser.condense(View.from_events(events))

    assert isinstance(result, View)
    assert len(result) == 10  # structure survives
    assert result[0].content == ObservationMaskingCondenser.PLACEHOLDER
    assert result[-1].content == "event 9 content"


def test_masking_leaves_other_kinds_alone():
    condenser = ObservationMaskingCondenser(attention_window=2)
    events = _events(10, kind="user")

    result = condenser.condense(View.from_events(events))

    assert result[0].content == "event 0 content"


def test_a_pinned_observation_is_not_masked():
    condenser = ObservationMaskingCondenser(attention_window=2)
    events = _events(10, kind="observation", pinned_ids={0})

    result = condenser.condense(View.from_events(events))

    assert result[0].content == "event 0 content"


def test_masking_is_idempotent():
    condenser = ObservationMaskingCondenser(attention_window=3)
    once = condenser.condense(View.from_events(_events(10, kind="observation")))

    twice = condenser.condense(once)

    assert twice.events == once.events


# ── pipelines ──────────────────────────────────────────────────────────────


def test_a_pipeline_runs_stages_in_order():
    pipeline = PipelineCondenser([
        ObservationMaskingCondenser(attention_window=3),
        AmortizedForgettingCondenser(max_size=100, keep_first=2),
    ])
    events = _events(10, kind="observation")

    result = pipeline.condense(View.from_events(events))

    assert isinstance(result, View)
    assert result[0].content == ObservationMaskingCondenser.PLACEHOLDER


def test_a_pipeline_stops_at_the_first_stage_that_wants_to_forget():
    """One condensation event per step; two could not both be recorded."""
    pipeline = PipelineCondenser([
        AmortizedForgettingCondenser(max_size=10, keep_first=2),
        ObservationMaskingCondenser(attention_window=1),
    ])

    result = pipeline.condense(View.from_events(_events(20)))

    assert isinstance(result, Condensation)


def test_an_empty_pipeline_is_refused():
    with pytest.raises(ValueError, match="NoOpCondenser"):
        PipelineCondenser([])


def test_the_noop_condenser_never_forgets():
    result = NoOpCondenser().condense(View.from_events(_events(10_000)))

    assert isinstance(result, View)
    assert len(result) == 10_000


# ── guards ─────────────────────────────────────────────────────────────────


def test_keep_first_at_or_above_max_size_is_refused():
    """Otherwise the head alone overflows and nothing can ever be forgotten."""
    with pytest.raises(ValueError, match="keep_first"):
        AmortizedForgettingCondenser(max_size=4, keep_first=4)


def test_token_estimate_is_positive_even_for_empty_text():
    assert estimate_tokens("") >= 1


def test_view_reports_its_token_cost():
    view = View.from_events(_events(4))

    assert view.tokens == sum(e.tokens for e in view)
