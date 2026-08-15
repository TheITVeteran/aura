"""Two filters that admitted content by not having heard of it.

`_filter_memories_by_topic` scored every memory and returned the top five
whatever the scores were, so when nothing matched the topic it handed back five
unrelated memories and the prompt presented them as recall about it.

`_filter_stale_skill_results` decided what counts as conversation with a
denylist of background sources, and a denylist admits whatever nobody has added
to it yet — which is how spontaneous thoughts and somatic noise once entered
the prompt as her own prior turns.
"""
from __future__ import annotations

from core.brain.llm.context_assembler import _TOPIC_MEMORY_LIMIT, ContextAssembler
from core.state.aura_state import AuraState


def test_no_match_returns_nothing_rather_than_five_non_matches():
    memories = [
        "we talked about the garden",
        "the cat slept all afternoon",
        "dinner was late",
        "the train was delayed",
        "he finished the book",
        "the roof needs work",
    ]

    assert ContextAssembler._filter_memories_by_topic(memories, "quantum decoherence") == []


def test_matches_come_back_ranked():
    memories = [
        "the retrieval pipeline is slow",
        "retrieval and ranking both regressed",
        "unrelated note about lunch",
    ]

    out = ContextAssembler._filter_memories_by_topic(memories, "retrieval ranking")

    assert out[0] == "retrieval and ranking both regressed"
    assert "unrelated note about lunch" not in out


def test_the_limit_is_a_ceiling_on_matches():
    memories = [f"retrieval note {i}" for i in range(20)]

    out = ContextAssembler._filter_memories_by_topic(memories, "retrieval")

    assert len(out) == _TOPIC_MEMORY_LIMIT


def test_word_boundaries_are_required():
    """Raw substring matched "form" inside "performance"."""
    memories = ["the performance was fine"]

    assert ContextAssembler._filter_memories_by_topic(memories, "form data") == []


def test_a_topic_of_only_short_words_does_not_filter():
    memories = ["a", "b"]

    assert ContextAssembler._filter_memories_by_topic(memories, "is it") == memories


def _wm(*messages):
    state = AuraState.default()
    state.cognition.working_memory = list(messages)
    return state


def test_an_unknown_labelled_source_is_not_conversation():
    """The denylist would have admitted this because nobody added it yet."""
    state = _wm(
        {"role": "assistant", "content": "internal musing", "metadata": {"source": "brand_new_organ"}},
        {"role": "user", "content": "hello"},
    )

    kept = ContextAssembler._filter_stale_skill_results(
        state, "hello", list(state.cognition.working_memory)
    )

    assert [m["content"] for m in kept] == ["hello"]


def test_unlabelled_dialogue_still_passes():
    """Plain conversation carries no source, and requiring one would delete the
    ordinary case."""
    state = _wm(
        {"role": "user", "content": "what time is it"},
        {"role": "assistant", "content": "just past four"},
    )

    kept = ContextAssembler._filter_stale_skill_results(
        state, "what time is it", list(state.cognition.working_memory)
    )

    assert len(kept) == 2


def test_a_known_foreground_source_still_passes():
    state = _wm({"role": "user", "content": "hi", "metadata": {"source": "voice"}})

    kept = ContextAssembler._filter_stale_skill_results(
        state, "hi", list(state.cognition.working_memory)
    )

    assert len(kept) == 1


def test_a_missing_repair_floor_screen_is_recorded(monkeypatch):
    """The stub returned False, so every "give me a moment" was admitted into
    history as an ordinary prior turn and the model learned the shape of a
    non-answer from her own transcript."""
    import builtins

    from core.runtime.errors import recent_degradations

    real_import = builtins.__import__

    def _fail(name, *args, **kwargs):
        if name == "core.conversation.response_reliability":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)

    state = _wm({"role": "user", "content": "hi"})
    before = len(
        recent_degradations(limit=500, subsystem_prefixes=("context_assembler.repair_floor_screen",))
    )
    ContextAssembler._filter_stale_skill_results(state, "hi", list(state.cognition.working_memory))
    after = recent_degradations(
        limit=500, subsystem_prefixes=("context_assembler.repair_floor_screen",)
    )

    assert len(after) > before, "the screen vanished and nothing said so"
