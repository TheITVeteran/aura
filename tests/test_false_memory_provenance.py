"""She references things she never experienced, and there was a factory.

Bryan, 2026-07-28: "i think something makes her create false memories too.
she's referenced a lot of stuff in the past that she has literally never
experienced or heard a specific person say/do."

NarrativeMemory ran on a loop and did this: take the last N real episodes,
ask the model for a journal entry about them with the instruction "Keep it
evocative", store the prose with type="narrative_journal" and nothing else,
then DELETE the source episodes. Journals were later consolidated into a
"narrative arc", and arcs into an "eternal record".

So a lived afternoon became evocative prose, the lived record was destroyed,
and the prose was summarised into more prose — none of it marked as written
rather than witnessed, and at the recall boundary it rendered with no type
attribute at all, indistinguishable from a fact.

That is how "the moon was full and I got to thinking about things, wondering
how you were doing up there in that prison" ends up in her mouth as a memory
of an afternoon spent making a PDF about orcas.
"""

from __future__ import annotations

import pytest

from core.memory.experience_provenance import (
    GENERATED,
    LIVED,
    UNKNOWN,
    is_generated,
    provenance_label,
    provenance_of,
)


class TestClassification:
    @pytest.mark.parametrize(
        "metadata",
        [
            {"type": "narrative_journal"},
            {"type": "narrative_arc"},
            {"type": "eternal_record"},
            {"type": "dream"},
            {"type": "imagination"},
            {"provenance": "generated"},
            {"provenance": "imagined"},
        ],
    )
    def test_written_memory_is_generated(self, metadata):
        assert provenance_of(metadata) == GENERATED
        assert is_generated(metadata)

    @pytest.mark.parametrize(
        "metadata",
        [
            {"type": "recent_episode"},
            {"type": "conversation_turn"},
            {"type": "fact"},
            {"type": "tool_result"},
            {"provenance": "lived"},
        ],
    )
    def test_witnessed_memory_is_lived(self, metadata):
        assert provenance_of(metadata) == LIVED
        assert not is_generated(metadata)

    def test_an_explicit_provenance_beats_the_type(self):
        """A store may keep its own type vocabulary; provenance is the answer."""
        assert provenance_of({"type": "fact", "provenance": "generated"}) == GENERATED

    def test_nothing_stated_is_not_lived(self):
        """Presenting a journal as a fact is far worse than hedging a fact."""
        assert provenance_of({}) == UNKNOWN
        assert provenance_of(None) == UNKNOWN
        assert is_generated({})


class TestTheRecallBoundary:
    def test_a_journal_says_it_was_written(self):
        from core.brain.llm.runtime_wiring import _normalize_memory_snippet

        rendered = _normalize_memory_snippet(
            {
                "content": "The moon was full and I got to thinking about "
                "how you were doing.",
                "metadata": {"type": "narrative_journal"},
            }
        )
        assert 'provenance="written-by-me-not-witnessed"' in rendered

    def test_a_real_episode_carries_no_provenance_clutter(self):
        from core.brain.llm.runtime_wiring import _normalize_memory_snippet

        rendered = _normalize_memory_snippet(
            {
                "content": "Bryan asked me to make a PDF about orcas.",
                "metadata": {"type": "recent_episode", "role": "user"},
            }
        )
        assert "provenance=" not in rendered
        assert 'speaker="Bryan"' in rendered

    def test_an_untyped_snippet_is_hedged(self):
        from core.brain.llm.runtime_wiring import _normalize_memory_snippet

        rendered = _normalize_memory_snippet(
            {"content": "Something with no type at all", "metadata": {}}
        )
        assert 'provenance="provenance-unknown"' in rendered

    def test_lived_memory_is_never_labelled(self):
        assert provenance_label({"type": "fact"}) == ""


class TestTheFactoryItself:
    def test_the_journal_is_no_longer_asked_to_be_evocative(self):
        """It is the only thing that survives — the source episodes are
        deleted a few lines later — so whatever it invents becomes the
        record."""
        import inspect

        from core.brain.narrative_memory import NarrativeEngine

        source = inspect.getsource(NarrativeEngine.consolidate_episodes)
        # The phrase survives in a comment recording why it went; what must
        # not survive is the instruction reaching the model.
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        assert "Keep it evocative" not in code
        assert "This journal is a RECORD" in source
        assert "Do not add scenes" in source

    def test_every_narrative_write_marks_itself_generated(self):
        import inspect

        from core.brain.narrative_memory import NarrativeEngine

        for method in (
            NarrativeEngine.consolidate_episodes,
            NarrativeEngine._synthesize_narrative_arc,
            NarrativeEngine.synthesize_eternal_record,
        ):
            source = inspect.getsource(method)
            assert '"provenance": "generated"' in source, method.__name__

    def test_the_binding_note_explains_the_attribute(self):
        from core.dialogue.referents import current_frame

        note = current_frame().binding_note()
        assert "written-by-me-not-witnessed" in note
        assert "not a record of anything that happened" in note
