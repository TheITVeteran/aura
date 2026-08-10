"""The part of the context window Aura writes herself.

IdentityCore tiers identity by mutability but has no notion of budget or
address: the evolved file grows without a ceiling, nothing names its regions,
and it only changes by whole-file rewrite from outside the turn.

The three positions under test: overflow refuses rather than truncates,
immutable blocks refuse out loud rather than ignoring, and shared blocks are
compare-and-swap because sleep-time consolidation writes the same blocks a live
turn is editing.
"""
from __future__ import annotations

import pytest

from core.memory.memory_blocks import (
    BlockImmutable,
    BlockOverflow,
    BlockVersionConflict,
    MemoryBlock,
    MemoryBlockSet,
    UnknownBlock,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def blocks():
    return MemoryBlockSet([
        MemoryBlock(label="persona", value="I am Aura.", limit=100,
                    description="Who I am"),
        MemoryBlock(label="human", value="Bryan builds me.", limit=100),
        MemoryBlock(label="heartstone", value="Let no god stand in the way.",
                    limit=100, immutable=True),
    ])


# ── addressing ─────────────────────────────────────────────────────────────


def test_a_block_is_addressed_by_label(blocks):
    assert blocks.get("persona").value == "I am Aura."


def test_an_unknown_label_says_what_is_attached(blocks):
    with pytest.raises(UnknownBlock, match="persona"):
        blocks.get("nope")


def test_two_blocks_cannot_share_a_label(blocks):
    with pytest.raises(ValueError, match="already attached"):
        blocks.attach(MemoryBlock(label="persona"))


# ── overflow refuses, never truncates ──────────────────────────────────────


def test_a_write_over_the_limit_is_refused(blocks):
    with pytest.raises(BlockOverflow):
        blocks.rewrite("persona", "x" * 101, author="aura")


def test_a_refused_write_leaves_the_block_untouched(blocks):
    with pytest.raises(BlockOverflow):
        blocks.rewrite("persona", "x" * 101, author="aura")

    assert blocks.get("persona").value == "I am Aura."
    assert blocks.get("persona").version == 0


def test_the_refusal_says_how_much_room_there_was(blocks):
    with pytest.raises(BlockOverflow, match="100"):
        blocks.rewrite("persona", "x" * 500, author="aura")


def test_an_append_that_would_overflow_is_refused(blocks):
    with pytest.raises(BlockOverflow):
        blocks.append("persona", "y" * 100, author="aura")


def test_a_block_cannot_be_constructed_over_its_own_limit():
    with pytest.raises(BlockOverflow):
        MemoryBlock(label="b", value="x" * 11, limit=10)


def test_remaining_room_is_visible_before_writing(blocks):
    assert blocks.get("persona").remaining == 100 - len("I am Aura.")


# ── immutability refuses out loud ──────────────────────────────────────────


def test_an_immutable_block_refuses_edits(blocks):
    with pytest.raises(BlockImmutable):
        blocks.rewrite("heartstone", "something else", author="aura")


def test_the_immutable_refusal_is_not_a_silent_no_op(blocks):
    """Proceeding as though it succeeded would be worse than refusing."""
    with pytest.raises(BlockImmutable, match="refusing rather than ignoring"):
        blocks.append("heartstone", "more", author="aura")

    assert blocks.get("heartstone").value == "Let no god stand in the way."


def test_an_immutable_block_still_renders_into_the_prompt(blocks):
    assert "Let no god stand in the way." in blocks.render()


# ── compare-and-swap across the sleep-time seam ────────────────────────────


def test_a_write_at_the_expected_version_succeeds(blocks):
    blocks.append("human", "He is direct.", author="aura", expected_version=0)

    assert blocks.get("human").version == 1


def test_a_stale_version_is_refused_rather_than_clobbering(blocks):
    """Last-write-wins loses whichever finished first, silently."""
    blocks.append("human", "live turn learned this", author="turn")

    with pytest.raises(BlockVersionConflict):
        blocks.rewrite(
            "human", "sleep-time consolidation", author="dreamer", expected_version=0
        )


def test_the_conflict_says_what_to_do(blocks):
    blocks.append("human", "x", author="turn")

    with pytest.raises(BlockVersionConflict, match="Re-read and reapply"):
        blocks.rewrite("human", "y", author="dreamer", expected_version=0)


def test_the_loser_of_a_conflict_did_not_change_anything(blocks):
    blocks.append("human", "live turn learned this", author="turn")
    before = blocks.get("human").value

    with pytest.raises(BlockVersionConflict):
        blocks.rewrite("human", "clobbered", author="dreamer", expected_version=0)

    assert blocks.get("human").value == before


def test_omitting_the_expected_version_opts_out_of_the_check(blocks):
    """Single-writer paths should not have to thread a version through."""
    blocks.append("human", "a", author="turn")
    blocks.append("human", "b", author="turn")

    assert blocks.get("human").version == 2


# ── edits ──────────────────────────────────────────────────────────────────


def test_append_joins_with_a_separator(blocks):
    blocks.append("human", "He is direct.", author="aura")

    assert blocks.get("human").value == "Bryan builds me.\nHe is direct."


def test_append_to_an_empty_block_adds_no_leading_separator():
    blocks = MemoryBlockSet([MemoryBlock(label="b", limit=50)])

    blocks.append("b", "first", author="aura")

    assert blocks.get("b").value == "first"


def test_replace_substitutes_within_the_block(blocks):
    blocks.replace("human", "builds", "designs", author="aura")

    assert blocks.get("human").value == "Bryan designs me."


def test_a_replace_that_matches_nothing_is_an_error_not_a_no_op(blocks):
    """The agent believes it corrected something. Letting that stand is how a
    stale fact survives being 'fixed'."""
    with pytest.raises(ValueError, match="does not appear"):
        blocks.replace("human", "absent text", "new", author="aura")


def test_rewrite_replaces_everything(blocks):
    blocks.rewrite("persona", "I am something new.", author="aura")

    assert blocks.get("persona").value == "I am something new."


def test_clear_empties_a_block(blocks):
    blocks.clear("persona", author="aura")

    assert blocks.get("persona").value == ""


# ── history ────────────────────────────────────────────────────────────────


def test_every_edit_is_attributed_and_recorded(blocks):
    blocks.append("human", "He is direct.", author="dreamer", reason="consolidation")

    edit = blocks.history("human")[-1]
    assert edit.author == "dreamer"
    assert edit.reason == "consolidation"
    assert edit.operation == "append"
    assert edit.before == "Bryan builds me."


def test_history_can_be_read_per_block_or_whole(blocks):
    blocks.append("human", "a", author="x")
    blocks.append("persona", "b", author="x")

    assert len(blocks.history("human")) == 1
    assert len(blocks.history()) == 2


def test_a_refused_write_leaves_no_history(blocks):
    with pytest.raises(BlockOverflow):
        blocks.rewrite("persona", "x" * 500, author="aura")

    assert blocks.history("persona") == []


def test_revert_undoes_the_last_edit(blocks):
    blocks.rewrite("persona", "drifted", author="aura")

    blocks.revert("persona", author="bryan")

    assert blocks.get("persona").value == "I am Aura."


def test_revert_is_itself_recorded_not_a_history_pop(blocks):
    """An undo that erases what it undid is indistinguishable from the edit
    never having happened."""
    blocks.rewrite("persona", "drifted", author="aura")

    blocks.revert("persona", author="bryan")

    operations = [e.operation for e in blocks.history("persona")]
    assert operations == ["rewrite", "revert"]


def test_reverting_an_unedited_block_is_an_error(blocks):
    with pytest.raises(ValueError, match="no edits"):
        blocks.revert("persona", author="bryan")


# ── prompt surface ─────────────────────────────────────────────────────────


def test_a_block_renders_with_its_label_as_a_tag(blocks):
    rendered = blocks.get("persona").render()

    assert rendered.startswith("<persona>")
    assert rendered.endswith("</persona>")


def test_the_rendered_block_shows_the_budget(blocks):
    """An agent that cannot see its remaining room discovers the ceiling only
    by hitting it."""
    assert "/100 characters used" in blocks.get("persona").render()


def test_the_description_is_rendered_for_the_model(blocks):
    assert "Who I am" in blocks.get("persona").render()


def test_render_can_be_limited_to_named_blocks(blocks):
    rendered = blocks.render(labels=["persona"])

    assert "persona" in rendered
    assert "heartstone" not in rendered


# ── pressure ───────────────────────────────────────────────────────────────


def test_pressure_reports_utilization_per_block(blocks):
    pressure = blocks.pressure()

    assert pressure["persona"] == pytest.approx(len("I am Aura.") / 100)
    assert set(pressure) == {"persona", "human", "heartstone"}


def test_pressure_is_readable_before_a_write_fails(blocks):
    """The signal that a block is full must not arrive only after the loss."""
    blocks.rewrite("persona", "x" * 95, author="aura")

    assert blocks.pressure()["persona"] > 0.9


def test_a_zero_or_negative_limit_is_refused():
    with pytest.raises(ValueError):
        MemoryBlock(label="b", limit=0)


def test_a_block_needs_a_label():
    with pytest.raises(ValueError, match="label"):
        MemoryBlock(label="   ")
