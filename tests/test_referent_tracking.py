"""Who "I" is, who "you" is, and who said the sentence being remembered.

Measured live on 2026-07-28. Bryan typed:

    It was the notes thing. I was trying to get you to write one about
    yourself in your own words. Like about who you are

Six turns later Aura said, nearly verbatim:

    I was trying to get you to write a paragraph about yourself in your own
    words. Like an actual summary of who you are

She was quoting him. The quote had lost its speaker on the way through
memory, and in her own context an unlabelled "I" is hers. She then told
Bryan what "Bryan asked for" as though he were a third party, offered to
write his self-summary, and when he said "I am Bryan" replied "I know that"
while still holding the swapped frame.
"""

from __future__ import annotations

import pytest

from core.dialogue.referents import (
    AURA,
    OWNER,
    UNATTRIBUTED,
    ReferentFrame,
    attribute,
    current_frame,
    has_person_reference,
    resolve_second_person,
    speaker_of,
)

#: The literal sentence that started it.
THE_SENTENCE = (
    "It was the notes thing. I was trying to get you to write one about "
    "yourself in your own words. Like about who you are"
)


class TestTheFrame:
    def test_i_and_you_swap_with_the_speaker(self):
        frame = ReferentFrame(owner_name="Bryan")
        assert frame.first_person_of(OWNER) == "Bryan"
        assert frame.second_person_of(OWNER) == "Aura"
        assert frame.first_person_of(AURA) == "Aura"
        assert frame.second_person_of(AURA) == "Bryan"

    def test_an_unknown_speaker_is_named_as_unknown(self):
        """Not defaulted to either party — that is the whole bug."""
        frame = ReferentFrame()
        assert frame.first_person_of(UNATTRIBUTED) == "someone unidentified"

    def test_the_binding_note_says_what_the_attribute_means(self):
        note = current_frame().binding_note()
        assert "Bryan" in note and "Aura" in note
        assert "unattributed" in note

    def test_identity_is_the_role_not_the_name(self):
        """An unknown name must not collapse the frame."""
        frame = ReferentFrame(owner_name="")
        assert frame.second_person_of(AURA) == "the person I am talking to"
        assert frame.second_person_of(OWNER) == "Aura"


class TestRecoveringTheSpeaker:
    @pytest.mark.parametrize(
        "metadata,expected",
        [
            ({"role": "user"}, OWNER),
            ({"role": "assistant"}, AURA),
            ({"speaker": "Bryan"}, OWNER),
            ({"author": "aura"}, AURA),
            ({"type": "user_message"}, OWNER),
            ({"type": "aura_response"}, AURA),
            ({"type": "recent_episode"}, UNATTRIBUTED),
            ({}, UNATTRIBUTED),
            (None, UNATTRIBUTED),
        ],
    )
    def test_every_store_spells_it_differently(self, metadata, expected):
        assert speaker_of(metadata) == expected


class TestWhatTravelsWithARememberedSentence:
    def test_a_pronoun_free_fact_needs_no_label(self):
        """"The folder is on the Desktop" means the same from any mouth."""
        assert not has_person_reference("The Orca Demo folder is in Documents")
        assert attribute("The Orca Demo folder is in Documents") == ""

    def test_a_first_person_sentence_always_carries_one(self):
        assert has_person_reference(THE_SENTENCE)
        assert attribute(THE_SENTENCE, OWNER) == "Bryan"
        assert attribute(THE_SENTENCE, AURA) == "Aura"

    def test_an_unknown_speaker_is_stated_not_assumed(self):
        assert attribute(THE_SENTENCE) == UNATTRIBUTED


class TestTheRecallBoundary:
    """The exact place the attribution was lost."""

    def test_the_measured_snippet_now_names_bryan(self):
        from core.brain.llm.runtime_wiring import _normalize_memory_snippet

        rendered = _normalize_memory_snippet(
            {
                "content": THE_SENTENCE,
                "metadata": {"type": "recent_episode", "role": "user"},
            }
        )
        assert 'speaker="Bryan"' in rendered
        assert THE_SENTENCE in rendered

    def test_an_unattributed_first_person_snippet_says_so(self):
        from core.brain.llm.runtime_wiring import _normalize_memory_snippet

        rendered = _normalize_memory_snippet(
            {"content": THE_SENTENCE, "metadata": {"type": "recent_episode"}}
        )
        assert 'speaker="unattributed"' in rendered

    def test_a_pronoun_free_snippet_is_not_cluttered(self):
        from core.brain.llm.runtime_wiring import _normalize_memory_snippet

        rendered = _normalize_memory_snippet(
            {"content": "The Orca Demo folder is in Documents", "metadata": {}}
        )
        assert "speaker=" not in rendered


class TestTheEpisodeItself:
    """Attribution at encoding, so recall has something to recover."""

    def test_a_conversation_episode_names_both_voices(self):
        from core.memory.episodic_memory import Episode

        episode = Episode(
            id="e",
            timestamp=0.0,
            context=THE_SENTENCE,
            action="conversation_reply",
            outcome="I opened Notes and wrote a paragraph describing myself.",
        )
        rendered = episode.full_description
        assert rendered.startswith("User: ")
        assert "\nAura: " in rendered
        assert " | " not in rendered

    def test_a_tool_episode_is_untouched(self):
        from core.memory.episodic_memory import Episode

        episode = Episode(
            id="e",
            timestamp=0.0,
            context="make a folder",
            action="execute_tool(desktop_task)",
            outcome="created ~/Documents/Orca Demo",
        )
        assert episode.full_description == (
            "make a folder | execute_tool(desktop_task) | created ~/Documents/Orca Demo"
        )


class TestARequestAboutHerself:
    @pytest.mark.parametrize(
        "asked",
        [
            "open the Notes app and write a note where you write a paragraph "
            "describing yourself",
            "write a note about yourself in your own words",
            "tell me who you are",
        ],
    )
    def test_yourself_from_the_owner_means_aura(self, asked):
        assert resolve_second_person(asked) == "Aura"

    def test_a_request_naming_no_one_resolves_to_no_one(self):
        assert resolve_second_person("summarize the orca articles") == ""

    def test_the_authoring_topic_is_bound_before_it_reaches_the_model(self):
        """"yourself" is a pronoun with no antecedent inside the prompt."""
        from core.skills.desktop_task import DesktopTaskSkill

        raw = DesktopTaskSkill._extract_requested_writing_topic(
            "write a note where you write a paragraph describing yourself"
        )
        assert raw == "yourself"
        assert resolve_second_person(
            "write a note where you write a paragraph describing yourself"
        ) == "Aura"
