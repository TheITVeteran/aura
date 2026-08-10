"""Consolidation that runs while she is awake, without eating the live turn.

SleepManager is stop-the-world: by the time low energy triggers it, the
conversation that produced the correction is over. This is the other half —
and its entire difficulty is that two writers now share one block.

The scenario these exist for: the user says "no, my sister's name is Tanya",
the live turn writes it, and a background pass that read the block ten seconds
earlier writes its summary over the top. Under last-write-wins the correction
is gone with nothing logged.
"""
from __future__ import annotations

import pytest

from core.memory.memory_blocks import MemoryBlock, MemoryBlockSet
from core.sleep.sleeptime_consolidation import (
    FAILED_CONFLICT,
    FAILED_OVERFLOW,
    FAILED_SUMMARIZER,
    REWRITTEN,
    SKIPPED_BELOW_PRESSURE,
    SKIPPED_IMMUTABLE,
    UNCHANGED,
    SleepTimeConsolidator,
)

pytestmark = pytest.mark.unit


def _blocks(**kwargs):
    return MemoryBlockSet([
        MemoryBlock(label="human", value="x" * 90, limit=100, **kwargs),
        MemoryBlock(label="quiet", value="x" * 10, limit=100),
        MemoryBlock(label="heartstone", value="x" * 95, limit=100, immutable=True),
    ])


def _consolidator(blocks, summarize=None, **kwargs):
    return SleepTimeConsolidator(
        blocks, summarize=summarize or (lambda b: "compacted"), **kwargs
    )


# ── selection by pressure ──────────────────────────────────────────────────


def test_only_blocks_under_pressure_are_targeted():
    blocks = _blocks()

    assert _consolidator(blocks).under_pressure() == ["human"]


def test_fullest_blocks_come_first():
    blocks = MemoryBlockSet([
        MemoryBlock(label="a", value="x" * 80, limit=100),
        MemoryBlock(label="b", value="x" * 95, limit=100),
    ])

    assert _consolidator(blocks).under_pressure() == ["b", "a"]


def test_immutable_blocks_are_never_targeted_however_full():
    blocks = _blocks()

    assert "heartstone" not in _consolidator(blocks).under_pressure()


def test_a_low_pressure_block_is_skipped_not_rewritten():
    """Age is not a criterion: a stale block at 20% is costing nothing."""
    blocks = _blocks()

    result = _consolidator(blocks).consolidate("quiet")

    assert result.outcome == SKIPPED_BELOW_PRESSURE
    assert blocks.get("quiet").value == "x" * 10


def test_an_immutable_block_is_skipped_rather_than_attempted():
    blocks = _blocks()

    result = _consolidator(blocks).consolidate("heartstone")

    assert result.outcome == SKIPPED_IMMUTABLE
    assert result.ok


# ── the happy path ─────────────────────────────────────────────────────────


def test_a_pressured_block_is_rewritten():
    blocks = _blocks()

    result = _consolidator(blocks).consolidate("human")

    assert result.outcome == REWRITTEN
    assert blocks.get("human").value == "compacted"


def test_consolidation_reclaims_room():
    blocks = _blocks()

    result = _consolidator(blocks).consolidate("human")

    assert result.before_utilization == 0.9
    assert result.after_utilization < result.before_utilization
    assert result.reclaimed > 0


def test_the_edit_is_attributed_to_the_background_pass():
    """The history must distinguish what a live turn learned from what a
    background pass inferred; they warrant different trust when they disagree."""
    blocks = _blocks()

    _consolidator(blocks, author="dreamer").consolidate("human")

    edit = blocks.history("human")[-1]
    assert edit.author == "dreamer"
    assert "consolidation" in edit.reason


def test_an_identical_summary_is_not_a_write():
    blocks = _blocks()
    current = blocks.get("human").value

    result = _consolidator(blocks, summarize=lambda b: current).consolidate("human")

    assert result.outcome == UNCHANGED
    assert blocks.history("human") == []


# ── the race ───────────────────────────────────────────────────────────────


def test_a_concurrent_live_write_is_never_overwritten():
    """The scenario this module exists for."""
    blocks = _blocks()

    def summarize_but_the_user_speaks_first(block):
        # The live turn lands between the read and the write, exactly once.
        if blocks.get("human").version == 0:
            blocks.rewrite("human", "her sister is Tanya", author="live-turn")
        return "compacted"

    result = _consolidator(
        blocks, summarize=summarize_but_the_user_speaks_first
    ).consolidate("human")

    assert result.attempts == 2  # lost once, re-derived, won
    assert "Tanya" in blocks.history("human")[0].after


def test_the_re_derivation_sees_what_the_live_turn_wrote():
    """Summarizing a superseded value is how a corrected fact comes back."""
    blocks = _blocks()
    seen = []

    def summarize(block):
        seen.append(block.value)
        if blocks.get("human").version == 0:
            blocks.rewrite("human", "her sister is Tanya", author="live-turn")
        return "compacted"

    _consolidator(blocks, summarize=summarize).consolidate("human")

    assert seen[-1] == "her sister is Tanya"


def test_a_permanently_contended_block_is_left_to_the_live_writer():
    blocks = _blocks()
    counter = {"n": 0}

    def always_loses(block):
        counter["n"] += 1
        blocks.rewrite("human", f"live write {counter['n']}", author="live-turn")
        return "compacted"

    result = _consolidator(
        blocks, summarize=always_loses, max_attempts=3
    ).consolidate("human")

    assert result.outcome == FAILED_CONFLICT
    assert result.attempts == 3
    assert blocks.get("human").value == "live write 3"  # the live turn kept it


def test_contention_is_reported_as_attempts_not_hidden():
    """Repeated conflicts are a tuning fact about the schedule."""
    blocks = _blocks()

    def always_loses(block):
        blocks.rewrite("human", "live", author="live-turn")
        return "compacted"

    consolidator = _consolidator(blocks, summarize=always_loses, max_attempts=2)

    assert consolidator.consolidate("human").attempts == 2


# ── failure is conservative ────────────────────────────────────────────────


def test_an_overflowing_summary_leaves_the_block_alone():
    """Stale is a smaller loss than absent."""
    blocks = _blocks()
    consolidator = _consolidator(blocks, summarize=lambda b: "y" * 200)

    result = consolidator.consolidate("human")

    assert result.outcome == FAILED_OVERFLOW
    assert blocks.get("human").value == "x" * 90


def test_a_failing_summarizer_leaves_the_block_alone():
    def explode(block):
        raise RuntimeError("cortex unavailable")

    blocks = _blocks()

    result = _consolidator(blocks, summarize=explode).consolidate("human")

    assert result.outcome == FAILED_SUMMARIZER
    assert blocks.get("human").value == "x" * 90


def test_an_empty_summary_is_a_failure_not_an_erasure():
    blocks = _blocks()

    result = _consolidator(blocks, summarize=lambda b: "   ").consolidate("human")

    assert result.outcome == FAILED_SUMMARIZER
    assert blocks.get("human").value == "x" * 90


def test_an_unknown_block_does_not_raise_into_the_background_loop():
    blocks = _blocks()

    result = _consolidator(blocks).consolidate("nope")

    assert not result.ok


# ── a pass ─────────────────────────────────────────────────────────────────


def test_run_once_covers_everything_under_pressure():
    blocks = MemoryBlockSet([
        MemoryBlock(label="a", value="x" * 90, limit=100),
        MemoryBlock(label="b", value="x" * 95, limit=100),
        MemoryBlock(label="c", value="x" * 10, limit=100),
    ])

    results = _consolidator(blocks).run_once()

    assert {r.label for r in results} == {"a", "b"}
    assert all(r.outcome == REWRITTEN for r in results)


def test_run_once_can_be_pointed_at_named_blocks():
    blocks = _blocks()

    results = _consolidator(blocks).run_once(["quiet"])

    assert [r.label for r in results] == ["quiet"]


def test_a_pass_over_nothing_is_not_an_error():
    blocks = MemoryBlockSet([MemoryBlock(label="a", value="x", limit=100)])

    assert _consolidator(blocks).run_once() == []


# ── guards ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kwargs", [
    {"pressure_threshold": 1.5},
    {"pressure_threshold": -0.1},
    {"max_attempts": 0},
])
def test_incoherent_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        _consolidator(_blocks(), **kwargs)


# ── the scheduler surface ──────────────────────────────────────────────────


def test_the_scheduled_task_is_a_valid_spec():
    """TaskSpec validates its own interval; a bad one would register healthy
    and then never run."""
    spec = _consolidator(_blocks()).scheduled_task(interval_s=60.0)

    assert spec.name == "sleeptime_consolidation"
    assert spec.tick_interval == 60.0
    assert spec.critical is False


@pytest.mark.asyncio
async def test_the_scheduled_tick_runs_a_pass_off_the_loop():
    blocks = _blocks()
    spec = _consolidator(blocks).scheduled_task()

    await spec.coro()

    assert blocks.get("human").value == "compacted"


def test_a_nonsense_interval_is_refused_by_the_spec():
    with pytest.raises(ValueError):
        _consolidator(_blocks()).scheduled_task(interval_s=-1.0)


def test_the_threshold_is_configurable():
    blocks = _blocks()

    consolidator = _consolidator(blocks, pressure_threshold=0.05)

    assert set(consolidator.under_pressure()) == {"human", "quiet"}
