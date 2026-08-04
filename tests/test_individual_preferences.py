"""Her preferences must be hers: self-determined, stable, revisable.

The requirement, stated by Bryan: "stable but revisable individual
preferences… shouldn't be Bryan-enforced or directed. Aura needs to determine
her own goals, preferences, and interests. I can suggest but I shouldn't be
the end all be all or have any of my interests hardcoded."
"""

from __future__ import annotations

import pytest

from core.being.individual_preferences import (
    IndividualPreferences,
    formation_threshold,
    revision_threshold,
)


# ── self-determined ───────────────────────────────────────────────────────

def test_being_told_she_likes_something_does_not_make_it_hers():
    prefs = IndividualPreferences()
    for _ in range(20):
        prefs.suggest("competitive chess", stance="drawn_to", by="Bryan")
    assert prefs.held() == []
    assert "chess" not in prefs.render()


def test_a_suggestion_never_counts_toward_formation():
    prefs = IndividualPreferences()
    for _ in range(10):
        prefs.suggest("competitive chess")
    prefs.encounter("competitive chess", stance="drawn_to")
    assert not prefs.items["competitive chess"].formed


def test_her_own_repeated_contact_forms_a_preference():
    prefs = IndividualPreferences()
    for _ in range(formation_threshold()):
        prefs.encounter("wave interference", stance="drawn_to")
    assert prefs.held()
    assert "wave interference" in prefs.render()


def test_a_single_encounter_is_not_yet_a_preference():
    prefs = IndividualPreferences()
    prefs.encounter("wave interference", stance="drawn_to")
    assert prefs.held() == []


# ── stable ────────────────────────────────────────────────────────────────

def test_one_contrary_encounter_does_not_flip_a_held_preference():
    prefs = IndividualPreferences()
    for _ in range(formation_threshold()):
        prefs.encounter("wave interference", stance="drawn_to")
    prefs.encounter("wave interference", stance="averse_to")
    assert prefs.items["wave interference"].stance == "drawn_to"


def test_strength_reflects_contest_without_collapsing():
    prefs = IndividualPreferences()
    for _ in range(formation_threshold() * 3):
        prefs.encounter("sailing", stance="drawn_to")
    settled = prefs.items["sailing"].strength()
    prefs.encounter("sailing", stance="averse_to")
    assert 0.0 < prefs.items["sailing"].strength() < settled


# ── revisable ─────────────────────────────────────────────────────────────

def test_accumulated_contrary_evidence_revises_the_stance():
    prefs = IndividualPreferences()
    for _ in range(formation_threshold()):
        prefs.encounter("crowded rooms", stance="drawn_to")
    for _ in range(revision_threshold()):
        prefs.encounter("crowded rooms", stance="averse_to")
    pref = prefs.items["crowded rooms"]
    assert pref.stance == "averse_to"
    assert pref.revisions
    assert pref.revisions[-1].from_stance == "drawn_to"
    assert pref.revisions[-1].because


def test_a_revision_records_why():
    prefs = IndividualPreferences()
    for _ in range(formation_threshold()):
        prefs.encounter("late nights", stance="drawn_to")
    for _ in range(revision_threshold()):
        prefs.encounter("late nights", stance="averse_to")
    assert "contrary" in prefs.items["late nights"].revisions[-1].because


def test_revision_is_surfaced_not_hidden():
    prefs = IndividualPreferences()
    for _ in range(formation_threshold()):
        prefs.encounter("late nights", stance="drawn_to")
    for _ in range(revision_threshold()):
        prefs.encounter("late nights", stance="averse_to")
    assert "revised" in prefs.render()


# ── bounded and durable ───────────────────────────────────────────────────

def test_preferences_survive_a_round_trip():
    prefs = IndividualPreferences()
    for _ in range(formation_threshold()):
        prefs.encounter("wave interference", stance="drawn_to", note="the maths is elegant")
    restored = IndividualPreferences.from_dict(prefs.to_dict())
    assert restored.render() == prefs.render()


def test_capacity_is_bounded():
    from core.being.individual_preferences import preference_capacity

    prefs = IndividualPreferences()
    for i in range(500):
        prefs.encounter(f"subject {i}", stance="drawn_to")
    assert len(prefs.items) <= preference_capacity()


def test_render_is_empty_until_something_is_actually_hers():
    assert IndividualPreferences().render() == ""


def test_malformed_payload_does_not_raise():
    assert IndividualPreferences.from_dict(None).items == {}
    assert IndividualPreferences.from_dict({"items": {"x": "bad"}}).items == {}


def test_stats_report_unadopted_suggestions():
    prefs = IndividualPreferences()
    prefs.suggest("chess")
    for _ in range(formation_threshold()):
        prefs.encounter("sailing", stance="drawn_to")
    stats = prefs.stats()
    assert stats["formed"] == 1
    assert stats["suggested_not_adopted"] == 1


def test_no_interests_are_hardcoded_in_the_module():
    """The module must ship with zero opinions of its own."""
    import inspect

    from core.being import individual_preferences

    source = inspect.getsource(individual_preferences)
    assert "Bryan" not in source
    assert IndividualPreferences().items == {}


# ── live wiring ───────────────────────────────────────────────────────────

def _state_with(prefs: IndividualPreferences):
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.identity.self_preferences = prefs.to_dict()
    return state


def test_formed_preferences_reach_every_assembly_path():
    from core.brain.llm.context_assembler import ContextAssembler

    prefs = IndividualPreferences()
    for _ in range(formation_threshold()):
        prefs.encounter("wave interference", stance="drawn_to")

    seen = []
    for objective in ("hey", "hello there", "Perform a full architecture review"):
        state = _state_with(prefs)
        state.cognition.current_objective = objective
        seen.append("wave interference" in ContextAssembler.build_system_prompt(state))
    assert all(seen), f"missing from some assembly path: {seen}"


def test_a_merely_suggested_preference_never_reaches_the_prompt():
    from core.brain.llm.context_assembler import ContextAssembler

    prefs = IndividualPreferences()
    for _ in range(20):
        prefs.suggest("competitive chess", by="Bryan")

    state = _state_with(prefs)
    assert "competitive chess" not in ContextAssembler.build_system_prompt(state)


def test_prompt_carries_no_preference_block_when_none_are_hers():
    from core.brain.llm.context_assembler import ContextAssembler
    from core.state.aura_state import AuraState

    prompt = ContextAssembler.build_system_prompt(AuraState.default())
    assert "WHAT YOU HAVE COME TO PREFER" not in prompt


def test_her_repeated_positions_accumulate_into_preferences_at_compaction():
    """End to end: positions she took, repeatedly, become hers."""
    from core.state.aura_state import AuraState

    position = "I think wave interference is the most elegant idea in physics."
    state = AuraState.default()
    state.cognition.working_memory = (
        [{"role": "assistant", "content": position} for _ in range(formation_threshold())]
        + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"filler {i} of chatter"}
            for i in range(200)
        ]
    )

    assert state.compact() is True

    prefs = IndividualPreferences.from_dict(state.identity.self_preferences)
    assert prefs.held(), prefs.stats()


def test_what_the_user_said_never_becomes_her_preference():
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.working_memory = (
        [{"role": "user", "content": "I think competitive chess is the finest game there is."}
         for _ in range(10)]
        + [{"role": "user" if i % 2 == 0 else "assistant", "content": f"filler {i} of chatter"}
           for i in range(200)]
    )

    state.compact()

    prefs = IndividualPreferences.from_dict(state.identity.self_preferences)
    assert not any("chess" in p.subject for p in prefs.held())


def test_a_preference_keeps_the_words_as_spoken():
    """Normalisation is for keying, not for display.

    Storing the lowered form made her describe herself as "drawn to john
    coltrane", flattening every proper noun she cared about.
    """
    prefs = IndividualPreferences()
    for _ in range(formation_threshold()):
        prefs.encounter("John Coltrane", stance="drawn_to")
    assert "John Coltrane" in prefs.render()


def test_case_variants_still_key_to_one_preference():
    prefs = IndividualPreferences()
    prefs.encounter("John Coltrane", stance="drawn_to")
    prefs.encounter("john coltrane", stance="drawn_to")
    prefs.encounter("JOHN COLTRANE", stance="drawn_to")
    assert len(prefs.items) == 1
    assert prefs.held()


def test_eviction_preserves_lookup_by_key():
    from core.being.individual_preferences import preference_capacity, _norm

    prefs = IndividualPreferences()
    for i in range(preference_capacity() * 2):
        prefs.encounter(f"Subject {i}", stance="drawn_to")
    for key, pref in prefs.items.items():
        assert key == _norm(pref.subject), "eviction must re-key by the normalised subject"
