"""The standing disposition: a view is hers to give without loading anything."""
import asyncio

from core.epistemics.opinion_engine import standing_disposition
from core.knowledge.source_comprehension import (
    SourceComprehension,
    comprehend_source,
    reading_disposition,
    record_reading_opinion,
)


def test_disposition_exists_with_no_engine_no_store_no_topic():
    text = standing_disposition("")
    assert text
    assert "allowed to think things" in text
    assert "survey" in text


def test_held_position_wins_over_the_standing_permission():
    held = "[MY EXISTING POSITION on x]: I think X is wrong."
    assert standing_disposition(held) == held
    assert standing_disposition(held, compact=True) == held


def test_compact_variant_is_shorter_and_still_a_permission():
    assert len(standing_disposition("", compact=True)) < len(standing_disposition(""))
    assert "views" in standing_disposition("", compact=True)


def test_reading_does_not_own_a_second_opinion_system():
    """Opinion formation from a reading lands in the general store."""
    import core.knowledge.source_comprehension as sc

    calls = []

    class _Engine:
        async def form_opinion(self, topic, context=""):
            calls.append((topic, context))
            return "stored"

    record = comprehend_source(
        url="https://example.com/a",
        title="A claim",
        text="Researchers say the effect is large. Everyone knows it is obvious.",
    )
    import core.runtime.service_access as sa

    real = sa.optional_service
    sa.optional_service = lambda *a, **k: _Engine()
    try:
        out = asyncio.run(record_reading_opinion(record))
    finally:
        sa.optional_service = real
    assert out == "stored"
    assert calls and calls[0][0]
    assert "Where I land:" in calls[0][1]
    assert not hasattr(sc, "form_opinion"), "reading must not export its own form_opinion"


def test_from_dict_round_trips_so_a_stored_reading_still_yields_a_view():
    record = comprehend_source(
        url="https://example.com/b", title="T", text="Studies show the drug works."
    )
    again = SourceComprehension.from_dict(record.to_dict())
    assert again.claim == record.claim
    assert reading_disposition(again).disposition == reading_disposition(record).disposition


def test_unreadable_source_forms_no_opinion():
    empty = SourceComprehension(url="https://x", title="")
    assert asyncio.run(record_reading_opinion(empty)) is None


def test_reply_path_carries_the_disposition_with_no_opinion_service():
    """The live shape of the bug: no stored opinion meant no view was offered."""
    from core.brain.inference_gate import InferenceGate

    gate = InferenceGate()
    compact = asyncio.run(
        gate._build_compact_living_mind_context("what do you think about tabs vs spaces", "api")
    )
    full = asyncio.run(
        gate._build_living_mind_context("what do you think about tabs vs spaces", "api")
    )
    assert "HOLDING A VIEW" in compact
    assert "HOLDING A VIEW" in full


def test_background_origins_do_not_pay_for_the_disposition():
    from core.brain.inference_gate import InferenceGate

    gate = InferenceGate()
    compact = asyncio.run(
        gate._build_compact_living_mind_context("summarize this log", "maintenance")
    )
    assert "HOLDING A VIEW" not in compact
