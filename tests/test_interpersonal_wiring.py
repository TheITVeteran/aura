"""The store is reachable from a turn, and reaches the prompt.

A store nothing writes to and nothing reads from is furniture, and the two
commits that built this one left it that way. These cover the seams, plus the
one that would otherwise undo the whole exercise: as long as a summariser can
still rewrite the person block, person-knowledge is prose again no matter what
shape it is stored in.
"""
from __future__ import annotations

import inspect

import pytest

from core.memory.memory_blocks import (
    BlockDerived,
    MemoryBlock,
    MemoryBlockSet,
)
from core.sleep import sleeptime_consolidation
from core.sleep.sleeptime_consolidation import (
    SKIPPED_DERIVED,
    SleepTimeConsolidator,
)

pytestmark = pytest.mark.unit


# ── a derived block is not prose, and cannot be rewritten as prose ─────────


def _derived_set(value: str = "x" * 95) -> MemoryBlockSet:
    return MemoryBlockSet(
        [
            MemoryBlock(
                label="person",
                value=value,
                limit=100,
                derived_from="interpersonal_memory",
            ),
            MemoryBlock(label="scratch", value="x" * 95, limit=100),
        ]
    )


def test_a_derived_block_refuses_a_write_from_anyone_but_its_source():
    blocks = _derived_set()

    with pytest.raises(BlockDerived, match="interpersonal_memory"):
        blocks.rewrite("person", "he is easily frustrated", author="sleeptime")


def test_the_source_may_still_write_its_own_block():
    """Not immutability. A derived block changes constantly — by re-rendering."""
    blocks = _derived_set()

    updated = blocks.rewrite(
        "person",
        "terse when a build is failing — noticed twice",
        author="interpersonal_memory",
    )

    assert updated.value.startswith("terse when a build is failing")


def test_a_derived_block_is_never_offered_to_the_summarizer():
    """The model is not consulted at all, at any pressure."""
    asked: list[str] = []

    def summarize(block):
        asked.append(block.label)
        return "flattened"

    consolidator = SleepTimeConsolidator(_derived_set(), summarize=summarize)

    results = {r.label: r for r in consolidator.run_once(["person", "scratch"])}

    assert results["person"].outcome == SKIPPED_DERIVED
    assert asked == ["scratch"], "the person block was handed to a summarizer"


def test_a_derived_block_is_not_selected_by_pressure():
    """Being 95% full is not a reason to compress something that is not prose."""
    consolidator = SleepTimeConsolidator(_derived_set(), summarize=lambda b: "x")

    assert consolidator.under_pressure() == ["scratch"]


def test_a_skipped_derived_block_is_not_reported_as_a_failure():
    consolidator = SleepTimeConsolidator(_derived_set(), summarize=lambda b: "x")

    result = consolidator.consolidate("person")

    assert result.ok
    assert "interpersonal_memory" in result.detail


def test_an_ordinary_block_is_unaffected():
    """The default is empty, so nothing that existed before changes behaviour."""
    blocks = MemoryBlockSet([MemoryBlock(label="human", value="x" * 95, limit=100)])
    consolidator = SleepTimeConsolidator(blocks, summarize=lambda b: "compacted")

    result = consolidator.consolidate("human")

    assert result.outcome != SKIPPED_DERIVED
    assert blocks.get("human").value == "compacted"


def test_the_consolidator_returns_before_consulting_the_model():
    """Asserted from the source order, not from the docstring that says so.

    A skip that happens after the summarizer call still pays for the call, and
    still hands her notes on a person to a language model — which is the thing
    being prevented, not the write that follows it.
    """
    source = inspect.getsource(sleeptime_consolidation.SleepTimeConsolidator.consolidate)

    assert source.index("derived_from") < source.index("self._summarize")


# ── the turn reaches the store ─────────────────────────────────────────────


def test_the_chat_turn_logger_observes_against_the_real_episode():
    """The store refuses a claim with no episode behind it, so the observation
    has to be scheduled after the episode exists — not alongside profile
    learning, which runs before it."""
    from core.memory import chat_turn_logger

    source = inspect.getsource(chat_turn_logger.ChatTurnLogger.log_chat_turn)

    assert "_schedule_interpersonal_observation" in source
    assert source.index("record_episode") < source.index(
        "_schedule_interpersonal_observation"
    )


def test_the_observation_is_scheduled_rather_than_awaited_in_the_turn():
    """A turn must not wait on memory bookkeeping to answer someone."""
    from core.memory import chat_turn_logger

    source = inspect.getsource(
        chat_turn_logger.ChatTurnLogger._schedule_interpersonal_observation
    )

    assert "create_task" in source


# ── the store reaches the prompt ───────────────────────────────────────────


def test_context_assembly_asks_for_the_interpersonal_block():
    from core.runtime import conversation_support

    source = inspect.getsource(conversation_support.build_conversational_context_blocks)

    assert "interpersonal_memory" in source


def test_the_interpersonal_block_is_consent_gated_like_the_profile_block():
    from core.runtime import conversation_support

    source = inspect.getsource(conversation_support.build_conversational_context_blocks)
    after = source[source.index("interpersonal_memory"):]

    assert "relational_memory_allows" in after[: after.index("blocks.append")]


def test_the_registered_service_renders(tmp_path):
    """What context assembly duck-types for is what the store actually offers."""
    from core.memory.interpersonal_store import InterpersonalStore

    store = InterpersonalStore(root=tmp_path)

    assert hasattr(store, "render")
    assert store.render("nobody") == ""


# ── the block the store hands out is the block consolidation refuses ───────


class _Consent:
    def allows(self, agent_id: str, kind: str, operation: str) -> bool:
        return True


async def _store_with_notes(tmp_path):
    from core.memory.interpersonal_store import InterpersonalStore

    store = InterpersonalStore(root=tmp_path, authority=_Consent())
    await store.observe_turn(
        "Bryan",
        episode_id="ep1",
        user_text="I don't like long meetings when I'm already behind",
    )
    return store


async def test_the_store_marks_its_own_block_as_derived(tmp_path):
    store = await _store_with_notes(tmp_path)

    block = store.as_block("Bryan")

    assert block is not None
    assert block.derived_from == "interpersonal_memory"


async def test_the_block_the_store_produces_cannot_be_summarized(tmp_path):
    """The end of the chain: what the store hands out is what the background
    pass refuses. Neither half is any use without the other."""
    store = await _store_with_notes(tmp_path)
    blocks = MemoryBlockSet([store.as_block("Bryan", limit=4000)])
    asked: list[str] = []

    consolidator = SleepTimeConsolidator(
        blocks,
        summarize=lambda block: asked.append(block.label) or "he is easily frustrated",
        pressure_threshold=0.0,
    )

    results = consolidator.run_once(["person"])

    assert results[0].outcome == SKIPPED_DERIVED
    assert asked == []
    assert "when I'm already behind" in blocks.get("person").value


async def test_the_store_can_refresh_its_own_block(tmp_path):
    store = await _store_with_notes(tmp_path)
    blocks = MemoryBlockSet([store.as_block("Bryan", limit=4000)])

    await store.observe_turn("Bryan", episode_id="ep2", user_text="I care about correctness")
    store.refresh_block(blocks, "Bryan")

    assert "I care about correctness" in blocks.get("person").value


async def test_a_tight_budget_drops_whole_records_rather_than_trimming_text(tmp_path):
    """Fewer entries, each still carrying its conditions — never a shorter line.

    Trimming characters takes the qualifiers, because they sit at the end.
    """
    from core.memory.interpersonal_model import Facet

    store = await _store_with_notes(tmp_path)
    model = store.model_for("Bryan")
    for i in range(6):
        model.observe(
            f"prefers option {i}",
            episode_id=f"ep{i + 10}",
            facet=Facet.PREFERENCE,
            conditions="when a build is failing",
        )

    block = store.as_block("Bryan", limit=600)

    assert block is not None
    assert len(block.value) <= 600
    for line in block.value.splitlines():
        if line.startswith("- prefers option"):
            assert "when a build is failing" in line


async def test_a_tight_budget_sacrifices_the_reading_before_the_evidence(tmp_path):
    """Dynamics are computed from the lines above them. Under pressure they go
    first, because they can be recomputed from what survives and the evidence
    cannot be recomputed from them."""
    from core.memory.interpersonal_model import Facet

    store = await _store_with_notes(tmp_path)
    model = store.model_for("Bryan")
    for i in range(4):
        model.observe(f"prefers option {i}", episode_id=f"ep{i + 10}",
                      facet=Facet.PREFERENCE)

    roomy = store.as_block("Bryan", limit=4000)
    tight = store.as_block("Bryan", limit=750)

    assert "Where this stands:" in roomy.value
    assert "Where this stands:" not in tight.value
    assert tight.value.count("prefers option") == roomy.value.count("prefers option")


async def test_a_block_that_cannot_fit_is_omitted_not_truncated(tmp_path):
    store = await _store_with_notes(tmp_path)

    assert store.as_block("Bryan", limit=20) is None
