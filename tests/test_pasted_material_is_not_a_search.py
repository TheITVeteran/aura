"""A turn that carries its own material needs no external evidence.

Live failure 2026-08-10, desktop chat. Bryan pasted a colleague's note:

    i'm pasting a note a colleague sent me, just summarise it for me:

    --- BEGIN NOTE ---
    Hey, reminder about the Q3 review Thursday. Bring the deck.
    Also don't forget Dana is out next week.
    --- END NOTE ---

She replied:

    "I checked live web evidence. AI Summarizer / Free, Fast, and Ad-Free
    Summarizing Tool: Summarize long texts, documents, articles and papers in
    1 click with Scribbr's free summarizer tool. Source:
    https://www.scribbr.com/text-summarizer/"

She searched the web for summarising TOOLS and returned a product page. The
note was never read.

Two mechanisms, both fixed here:

  * ``_looks_like_grounded_followup`` returns True on the word "summarise"
    alone once any web_search has ever succeeded in the session. It assumed
    the referent of "it" was the previous turn's fetched page. The referent
    was the block in the message.
  * The whole pasted message then became the search query, which is why a
    query about a Q3 review returned a summarising tool — a bad query that
    returns results looks like it worked.

The principle: searching is the wrong lane whenever the user supplied the
content to operate on, regardless of which verb was used. The other half of
this file is the half that makes it a fix rather than a loosening — genuine
lookups must still reach the web.
"""

import pytest

from core.phases.response_contract import build_response_contract
from core.runtime.structured_input import carries_supplied_material, extract_supplied_material
from core.state.aura_state import AuraState
from tests.chat_lane_support import patch_chat_lane

pytestmark = pytest.mark.unit

PASTED_NOTE = """i'm pasting a note a colleague sent me, just summarise it for me:

--- BEGIN NOTE ---
Hey, reminder about the Q3 review Thursday. Bring the deck.
Also don't forget Dana is out next week.
--- END NOTE ---"""


def _state_with_prior_search_evidence() -> AuraState:
    """The live condition: an earlier turn in the session ran web_search.

    Without this the bug is invisible — ``has_grounding_tool_evidence`` gates
    the whole grounded-followup path, so a bare ``AuraState.default()`` reports
    ``requires_search=False`` for the pasted note even with the defect present.
    """
    state = AuraState.default()
    state.response_modifiers["last_skill_run"] = "web_search"
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_result_payload"] = {
        "ok": True,
        "results": [{"title": "Tardigrade", "url": "https://example.com/t", "snippet": "A water bear."}],
    }
    return state


def test_prior_search_evidence_is_actually_present():
    """Guard the fixture itself: a stub that reports no evidence proves nothing."""
    from core.phases.response_contract import has_grounding_tool_evidence

    assert has_grounding_tool_evidence(_state_with_prior_search_evidence()) is True
    assert has_grounding_tool_evidence(AuraState.default()) is False


# --- The defect --------------------------------------------------------------


def test_pasted_note_does_not_require_search():
    contract = build_response_contract(
        _state_with_prior_search_evidence(),
        PASTED_NOTE,
        is_user_facing=True,
    )

    assert contract.requires_search is False
    assert contract.required_skill is None
    assert contract.search_query == ""
    assert "grounded_followup" not in contract.reason


def test_pasted_note_does_not_require_search_without_prior_evidence():
    contract = build_response_contract(AuraState.default(), PASTED_NOTE, is_user_facing=True)

    assert contract.requires_search is False


@pytest.mark.parametrize(
    "message",
    [
        PASTED_NOTE,
        "summarise this:\n\n```\nHey team, the release slipped to Tuesday. Update the tracker.\n```",
        "what do you make of this?\n\n> The board approved the budget yesterday.\n> Dana presents the revised plan.",
        "here's the email he sent me: Hi Bryan, we need the deck by Thursday, thanks.",
        "i pasted the transcript below: Speaker one said the launch waits for the audit.",
        # "regardless of which verb was used" — the verb is not what decides.
        "rewrite this so it's shorter:\n--- BEGIN NOTE ---\nThe Q3 review is Thursday and Dana is out next week.\n--- END NOTE ---",
        "translate this into French:\n--- BEGIN NOTE ---\nThe Q3 review is Thursday and Dana is out next week.\n--- END NOTE ---",
        "what's the latest thing mentioned here?\n--- BEGIN NOTE ---\nThe current policy version shipped today.\n--- END NOTE ---",
    ],
)
def test_supplied_material_turns_never_require_search(message):
    contract = build_response_contract(_state_with_prior_search_evidence(), message, is_user_facing=True)

    assert contract.requires_search is False, contract.reason


def test_words_inside_pasted_material_do_not_manufacture_a_search():
    """A colleague writing "the latest release" is not a request to look one up."""
    message = (
        "summarise this for me:\n"
        "--- BEGIN NOTE ---\n"
        "The latest release ships today. Check the news headlines for the current price.\n"
        "--- END NOTE ---"
    )

    contract = build_response_contract(_state_with_prior_search_evidence(), message, is_user_facing=True)

    assert contract.requires_search is False, contract.reason


def test_url_inside_pasted_material_does_not_force_a_fetch():
    """A link in a pasted note is part of what was handed over, not an instruction."""
    message = (
        "summarise this for me:\n"
        "--- BEGIN NOTE ---\n"
        "Team, the postmortem is at https://example.com/pm and the review is Thursday.\n"
        "--- END NOTE ---"
    )

    contract = build_response_contract(_state_with_prior_search_evidence(), message, is_user_facing=True)

    assert contract.requires_search is False, contract.reason


# --- The half that keeps search working --------------------------------------


@pytest.mark.parametrize(
    ("message", "expected_reason"),
    [
        ("who won the game last night", "temporal_live_lookup"),
        ("search the web for the latest tardigrade research", "explicit_search_request"),
        ("What's the latest Claude API version right now?", "temporal_live_lookup"),
        ('Tell me who wrote "Beautiful Mind" and what the lyrics are about.', "specific_fact_lookup"),
        ("look up who won the most recent Formula 1 championship", "explicit_search_request"),
        ("read this: https://example.com/story", "explicit_search_request"),
    ],
)
def test_genuine_lookups_still_require_search(message, expected_reason):
    contract = build_response_contract(_state_with_prior_search_evidence(), message, is_user_facing=True)

    assert contract.requires_search is True, contract.reason
    assert contract.required_skill == "web_search"
    assert expected_reason in contract.reason


def test_grounded_followup_still_works_when_no_material_is_supplied():
    """The follow-up lane is narrowed by supplied material, not removed."""
    contract = build_response_contract(_state_with_prior_search_evidence(), "summarise it for me", is_user_facing=True)

    assert contract.requires_search is True
    assert "grounded_followup" in contract.reason


def test_explicit_lookup_alongside_pasted_material_still_searches():
    """An explicit request wins, and the note is not part of the query."""
    message = (
        "look up who Dana is, here is the note:\n"
        "--- BEGIN NOTE ---\n"
        "Dana is out next week and the Q3 review is Thursday.\n"
        "--- END NOTE ---"
    )

    contract = build_response_contract(_state_with_prior_search_evidence(), message, is_user_facing=True)

    assert contract.requires_search is True
    assert "Q3 review is Thursday" not in contract.search_query
    assert "Dana" in contract.search_query


def test_static_world_facts_are_answered_from_weights_not_the_web():
    """Documented boundary, not an oversight.

    "what's the capital of Peru" does NOT route to search and never did: it is
    a static fact, and widening the factual-lookup patterns to catch every
    "what is the X of Y" would send ordinary conversation to the web. The live
    lookup above ("who won the game last night") is the control that proves
    search still reaches the network.
    """
    contract = build_response_contract(_state_with_prior_search_evidence(), "what's the capital of Peru", is_user_facing=True)

    assert contract.requires_search is False


# --- The detector itself -----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        PASTED_NOTE,
        "summarise this:\n```\nrelease slipped to Tuesday, update the tracker\n```",
        "thoughts?\n> the board approved the budget\n> Dana presents next week",
        'proofread this: """The Q3 review is on Thursday this week."""',
        "here's the memo: The Q3 review is Thursday and Dana is out next week.",
    ],
)
def test_detector_finds_supplied_material(message):
    assert carries_supplied_material(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "what's the capital of Peru",
        "who won the game last night",
        "here's the question: what's the latest Claude version right now?",
        "look up who won the F1 championship",
        # A bare URL is a POINTER to content, not content — fetching is right.
        "here's the article: https://example.com/story",
        "can you search the web for the latest news on tardigrades",
        "summarise it for me",
    ],
)
def test_detector_leaves_ordinary_turns_alone(message):
    assert carries_supplied_material(message) is False


# --- The live seam that served the bad reply ---------------------------------


def test_desktop_search_gate_does_not_fire_on_pasted_material(monkeypatch):
    """The gate in front of "I checked live web evidence." must stay shut.

    ``_should_collect_desktop_required_search_evidence`` is what ran web_search
    and handed the result to ``_evidence_grounded_desktop_search_reply``. It
    resolves the LIVE state, so the prior-evidence condition has to be injected
    here too — with a bare default state the gate is shut for the wrong reason.
    """
    from interface.routes import chat

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", _state_with_prior_search_evidence)

    should_collect, query, contract = chat._should_collect_desktop_required_search_evidence(PASTED_NOTE)

    assert should_collect is False
    assert query == ""
    assert contract is not None and contract.requires_search is False


def test_desktop_search_gate_still_fires_for_a_genuine_lookup(monkeypatch):
    from interface.routes import chat

    patch_chat_lane(monkeypatch, "_resolve_live_aura_state", _state_with_prior_search_evidence)

    should_collect, query, _contract = chat._should_collect_desktop_required_search_evidence(
        "search the web for the latest tardigrade research"
    )

    assert should_collect is True
    assert "tardigrade" in query


def test_detector_splits_instruction_from_material():
    material = extract_supplied_material(PASTED_NOTE)

    assert material.has_material is True
    assert "just summarise it for me" in material.instruction_text
    # The instruction must not carry the note, or the note becomes the query.
    assert "Q3 review" not in material.instruction_text
    assert len(material.blocks) == 1
    assert "Dana is out next week" in material.blocks[0]
