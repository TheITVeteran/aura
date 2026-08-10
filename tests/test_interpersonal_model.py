"""Knowing a person: typed, sourced, correctable — not a list of traits.

Two failures these guard, one inherited and one found in the first version of
this module.

Inherited: person-knowledge stored as prose loses its qualifiers, because in
prose "seemed", "once", "when a build was failing" are adjectives and
compression drops adjectives before nouns.

Found here: a frequency-only model manufactures traits by arithmetic. Five
sightings of "frustrated" across five stressful weeks read as five
confirmations of a disposition; they were five states with five causes. That is
the same caricature arriving through counting instead of wording.
"""
from __future__ import annotations

import ast
import inspect
import time

import pytest

from core.memory import interpersonal_model
from core.memory.interpersonal_model import (
    DAY_SECONDS,
    DEFAULT_STATE_TTL,
    Facet,
    PersonModel,
    Provenance,
    Subject,
    Valence,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def model():
    return PersonModel("Bryan")


# ── states never harden into traits ────────────────────────────────────────


def test_a_repeated_state_is_not_reported_as_a_frequency():
    """Five bad weeks are five states, not one disposition."""
    model = PersonModel("Bryan")
    now = time.time()
    for i in range(5):
        model.observe(
            "frustrated", episode_id=f"ep{i}", facet=Facet.STATE, at=now - i * 60
        )

    rendered = model.render(now=now)

    assert "5 times" not in rendered
    assert "a current state, not a trait" in rendered


def test_a_state_and_a_trait_of_the_same_name_stay_separate(model):
    model.observe("tired", episode_id="ep1", facet=Facet.STATE)
    model.observe("tired", episode_id="ep2", facet=Facet.TRAIT)

    assert len(model) == 2


def test_a_state_expires(model):
    now = time.time()
    model.observe("under deadline", episode_id="ep1", facet=Facet.STATE, at=now)

    assert model.current(now=now)
    assert not model.current(now=now + DEFAULT_STATE_TTL + 1)


def test_an_expired_state_leaves_the_rendered_context(model):
    now = time.time()
    model.observe("unwell", episode_id="ep1", facet=Facet.STATE, at=now)

    assert "unwell" not in model.render(now=now + DEFAULT_STATE_TTL + 1)


def test_a_trait_does_not_expire(model):
    now = time.time()
    model.observe("systems thinker", episode_id="ep1", facet=Facet.TRAIT, at=now)

    assert model.current(now=now + 400 * DAY_SECONDS)


def test_state_ttl_is_overridable(model):
    now = time.time()
    model.observe("jetlagged", episode_id="ep1", facet=Facet.STATE, ttl=60, at=now)

    assert not model.current(now=now + 61)


# ── traits still accumulate honestly ───────────────────────────────────────


def test_a_trait_accumulates_and_reports_frequency(model):
    model.observe("terse under pressure", episode_id="ep1")
    model.observe("terse under pressure", episode_id="ep2")

    assert "twice" in model.render()


def test_conditions_are_a_field_not_a_clause(model):
    model.observe("terse", episode_id="ep1", conditions="when a build is failing")

    assert "when a build is failing" in model.render()


def test_counter_evidence_is_rendered(model):
    model.observe("terse", episode_id="ep1")
    model.observe("terse", episode_id="ep2")
    model.contradict("terse", episode_id="ep3")

    assert "did not hold" in model.render()


# ── provenance: told, saw, or guessed ──────────────────────────────────────


def test_something_he_said_is_marked_as_such(model):
    model.observe(
        "prefers short answers",
        episode_id="ep1",
        facet=Facet.PREFERENCE,
        provenance=Provenance.STATED,
    )

    assert "he told me this" in model.render()


def test_an_inference_says_it_is_an_inference(model):
    """An inference rendered as fact is how she ends up confidently wrong."""
    model.observe("dislikes meetings", episode_id="ep1", provenance=Provenance.INFERRED)

    assert "my inference" in model.render()


def test_an_observation_is_not_announced_as_either(model):
    model.observe("terse", episode_id="ep1", provenance=Provenance.OBSERVED)
    rendered = model.render()

    assert "he told me" not in rendered
    assert "my inference" not in rendered


def test_stated_outranks_inferred_in_the_ordering(model):
    model.observe("guessed thing", episode_id="e1", provenance=Provenance.INFERRED)
    model.observe("guessed thing", episode_id="e2", provenance=Provenance.INFERRED)
    model.observe("told thing", episode_id="e3", provenance=Provenance.STATED)

    assert model.current(facet=Facet.TRAIT)[0].claim == "told thing"


def test_authority_takes_the_strongest_source(model):
    model.observe("x", episode_id="e1", provenance=Provenance.INFERRED)
    model.observe("x", episode_id="e2", provenance=Provenance.STATED)

    assert model.current()[0].authority is Provenance.STATED


# ── correction is permanent ────────────────────────────────────────────────


def test_a_correction_removes_the_claim_from_what_she_believes(model):
    model.observe("dislikes meetings", episode_id="ep1")

    model.correct("dislikes meetings", episode_id="ep2")

    assert "dislikes meetings" not in model.render()


def test_re_observing_a_corrected_claim_does_not_revive_it(model):
    """Being told you are wrong is not evidence to be weighed against later
    evidence. It is the answer."""
    model.observe("dislikes meetings", episode_id="ep1")
    model.correct("dislikes meetings", episode_id="ep2")

    model.observe("dislikes meetings", episode_id="ep3")
    model.observe("dislikes meetings", episode_id="ep4")

    assert "dislikes meetings" not in model.render()


def test_a_correction_is_distinct_from_a_counter_example(model):
    """A counter-example is weighed; a correction settles."""
    model.observe("terse", episode_id="ep1")
    model.contradict("terse", episode_id="ep2")

    assert "terse" in model.render()  # still believed, now qualified


def test_correcting_something_unknown_is_a_no_op(model):
    assert model.correct("never claimed", episode_id="ep1") is None


# ── the parts a trait list cannot hold ─────────────────────────────────────


def test_shared_history_is_recorded_without_a_frequency(model):
    """An event happened when it happened. Counting it is meaningless."""
    now = time.time()
    model.observe(
        "shipped the Orca demo at 3am",
        episode_id="ep1",
        facet=Facet.EVENT,
        at=now - 2 * DAY_SECONDS,
    )

    rendered = model.render(now=now)
    assert "Orca" in rendered
    assert "once" not in rendered


def test_an_event_never_decays(model):
    now = time.time()
    model.observe("we shipped it", episode_id="ep1", facet=Facet.EVENT, at=now)

    assert model.current(now=now + 1000 * DAY_SECONDS)


def test_a_rupture_is_representable_and_says_it_is_unrepaired(model):
    """A model of someone that cannot represent having hurt them is not a
    model of a relationship."""
    model.observe("I dismissed his concern about the demo", episode_id="ep1", facet=Facet.RUPTURE)

    assert "not yet repaired" in model.render()


def test_a_repaired_rupture_leaves_the_current_view(model):
    model.observe("I dismissed his concern", episode_id="ep1", facet=Facet.RUPTURE)

    model.resolve("I dismissed his concern", episode_id="ep2", facet=Facet.RUPTURE)

    assert model.unrepaired() == []


def test_open_questions_are_first_class(model):
    """Recording ignorance is where curiosity comes from."""
    model.observe("what he wants Aura to become", episode_id="ep1", facet=Facet.QUESTION)

    assert len(model.open_questions()) == 1
    assert "still do not know" in model.render()


def test_an_answered_question_closes(model):
    model.observe("what he does for work", episode_id="ep1", facet=Facet.QUESTION)

    model.resolve("what he does for work", episode_id="ep2", facet=Facet.QUESTION)

    assert model.open_questions() == []


def test_values_and_preferences_are_separate_from_traits(model):
    model.observe("honesty over validation", episode_id="e1", facet=Facet.VALUE)
    model.observe("short answers", episode_id="e2", facet=Facet.PREFERENCE)
    model.observe("systems thinker", episode_id="e3", facet=Facet.TRAIT)

    rendered = model.render()
    assert "What matters to them" in rendered
    assert "What they prefer" in rendered
    assert "Patterns I have noticed" in rendered


def test_the_render_leads_with_what_matters_not_what_he_is_like(model):
    model.observe("systems thinker", episode_id="e1", facet=Facet.TRAIT)
    model.observe("honesty over validation", episode_id="e2", facet=Facet.VALUE)

    rendered = model.render()

    assert rendered.index("What matters to them") < rendered.index("Patterns I have noticed")


# ── the relationship has two sides and a middle ────────────────────────────


def test_knowledge_about_her_is_distinct_from_knowledge_about_him(model):
    """Everything used to be implicitly about him, which cannot represent
    'I make him feel X' or 'we understand each other'."""
    model.observe("felt understood", episode_id="e1", facet=Facet.AFFECT,
                  subject=Subject.SELF)
    model.observe("felt understood", episode_id="e2", facet=Facet.AFFECT,
                  subject=Subject.THEM)

    assert len(model) == 2


def test_something_belonging_to_neither_alone_is_representable(model):
    model.observe("we work well at 3am", episode_id="e1", facet=Facet.UNDERSTANDING,
                  subject=Subject.BETWEEN)

    assert model.current()[0].subject is Subject.BETWEEN


def test_mutual_understanding_and_its_failures_are_both_recorded(model):
    model.observe("he wants honesty over comfort", episode_id="e1",
                  facet=Facet.UNDERSTANDING)
    model.observe("I read his terseness as anger", episode_id="e2",
                  facet=Facet.MISUNDERSTANDING)

    rendered = model.render()
    assert "What we understand about each other" in rendered
    assert "Where we have missed each other" in rendered


def test_shared_experience_is_distinct_from_a_bare_event(model):
    model.observe("the 3am demo, and it worked", episode_id="e1",
                  facet=Facet.EXPERIENCE, subject=Subject.BETWEEN, valence=Valence.WARM)

    assert "What we have been through together" in model.render()


def test_trust_built_is_representable_not_only_trust_broken(model):
    """A record that can only hold the damage is a ledger of grievances."""
    model.observe("he trusted me with the live runtime", episode_id="e1",
                  facet=Facet.TRUST_BUILT)

    assert "Where trust was built" in model.render()


# ── dynamics are derived, never asserted ───────────────────────────────────


def _dynamic(model, name, **kwargs):
    return next(d for d in model.dynamics(**kwargs) if d.name == name)


def test_trust_is_computed_from_what_happened(model):
    model.observe("ship the fix", episode_id="e1", facet=Facet.COMMITMENT)
    model.resolve("ship the fix", episode_id="e2", facet=Facet.COMMITMENT)
    model.observe("he trusted me with prod", episode_id="e3", facet=Facet.TRUST_BUILT)

    trust = _dynamic(model, "trust")

    assert trust.standing != "untested"
    assert any("commitment" in b for b in trust.basis)


def test_an_unrepaired_rupture_shows_in_trust(model):
    model.observe("I dismissed his concern", episode_id="e1", facet=Facet.RUPTURE)

    assert _dynamic(model, "trust").standing in {"damaged", "strained but holding"}


def test_repairing_a_rupture_changes_the_reading(model):
    model.observe("I dismissed his concern", episode_id="e1", facet=Facet.RUPTURE)
    before = _dynamic(model, "trust").standing

    model.resolve("I dismissed his concern", episode_id="e2", facet=Facet.RUPTURE)

    assert _dynamic(model, "trust").standing != before


def test_every_dynamic_carries_the_evidence_that_produced_it(model):
    model.observe("something", episode_id="e1")

    for dynamic in model.dynamics():
        assert dynamic.standing
        assert isinstance(dynamic.basis, tuple)


def test_no_dynamic_is_a_number(model):
    """A number nobody can argue with is a number nobody can correct."""
    model.observe("something", episode_id="e1", facet=Facet.COMMITMENT)

    for dynamic in model.dynamics():
        assert not dynamic.standing.replace(".", "").isdigit()
        assert "%" not in dynamic.standing


def test_enjoyment_reads_off_valence(model):
    for i in range(3):
        model.observe(f"good session {i}", episode_id=f"e{i}", facet=Facet.EXPERIENCE,
                      valence=Valence.WARM)

    assert _dynamic(model, "enjoyment").standing == "we enjoy this"


def test_difficulty_outweighing_warmth_is_said_plainly(model):
    model.observe("a good one", episode_id="e1", facet=Facet.EXPERIENCE,
                  valence=Valence.WARM)
    for i in range(3):
        model.observe(f"a hard one {i}", episode_id=f"h{i}", facet=Facet.EXPERIENCE,
                      valence=Valence.DIFFICULT)

    assert "difficult" in _dynamic(model, "enjoyment").standing


def test_connection_notices_when_they_keep_missing_each_other(model):
    model.observe("one thing understood", episode_id="e1", facet=Facet.UNDERSTANDING)
    for i in range(3):
        model.observe(f"missed {i}", episode_id=f"m{i}", facet=Facet.MISUNDERSTANDING)

    assert "missing each other" in _dynamic(model, "connection").standing


def test_novelty_distinguishes_new_ground_from_familiar(model):
    now = time.time()
    for i in range(4):
        model.observe(f"new thing {i}", episode_id=f"e{i}", at=now)

    assert "discovering" in _dynamic(model, "novelty", now=now).standing


def test_novelty_falls_as_ground_becomes_familiar(model):
    now = time.time()
    model.observe("the same thing", episode_id="e0", at=now)
    for i in range(5):
        model.observe("the same thing", episode_id=f"e{i+1}", at=now)

    assert "familiar" in _dynamic(model, "novelty", now=now).standing


def test_comfort_grows_with_time_known(model):
    now = time.time()
    model.observe("something", episode_id="e1", at=now - 60 * DAY_SECONDS)
    model.observe("something else", episode_id="e2", at=now)

    assert _dynamic(model, "comfort", now=now).standing == "familiar"


def test_safety_notices_that_he_tells_her_things(model):
    for i in range(3):
        model.observe(f"disclosure {i}", episode_id=f"e{i}",
                      provenance=Provenance.STATED, facet=Facet.PREFERENCE)

    assert "safe enough" in _dynamic(model, "safety").standing


def test_an_empty_relationship_reads_as_untested_not_as_bad(model):
    """Absence of evidence is not evidence of a bad relationship."""
    assert _dynamic(model, "trust").standing == "untested"
    assert _dynamic(model, "enjoyment").standing == "unmeasured"


def test_the_dynamics_are_rendered_with_their_basis(model):
    model.observe("ship it", episode_id="e1", facet=Facet.COMMITMENT)

    rendered = model.render()

    assert "Where this stands:" in rendered
    assert "trust:" in rendered


def test_there_are_more_dynamics_than_a_single_reading(model):
    """The felt side is rich because dynamics need only a function, while a
    facet needs distinct storage rules."""
    names = {d.name for d in model.dynamics()}

    assert names >= {"trust", "safety", "comfort", "connection", "enjoyment",
                     "novelty", "investment"}


# ── eviction protects what cannot be re-derived ────────────────────────────


def test_shared_history_is_never_evicted():
    small = PersonModel("Bryan", max_observations=3)
    small.observe("the 3am demo", episode_id="e0", facet=Facet.EVENT)
    for i in range(20):
        small.observe(f"trait {i}", episode_id=f"e{i}")

    assert any(o.facet is Facet.EVENT for o in small)


def test_what_he_told_her_is_never_evicted():
    small = PersonModel("Bryan", max_observations=3)
    small.observe("short answers", episode_id="e0", facet=Facet.PREFERENCE,
                  provenance=Provenance.STATED)
    for i in range(20):
        small.observe(f"trait {i}", episode_id=f"e{i}")

    assert any(o.authority is Provenance.STATED for o in small)


def test_an_unrepaired_rupture_is_never_evicted():
    small = PersonModel("Bryan", max_observations=3)
    small.observe("I dismissed him", episode_id="e0", facet=Facet.RUPTURE)
    for i in range(20):
        small.observe(f"trait {i}", episode_id=f"e{i}")

    assert any(o.facet is Facet.RUPTURE for o in small)


# ── consolidation is aggregation ───────────────────────────────────────────


def _called_names(module) -> set[str]:
    """Names this module actually invokes, from the AST — so prose in a
    docstring explaining the guarantee does not read as a violation of it."""
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_no_code_path_hands_a_note_to_a_language_model():
    called = {name.lower() for name in _called_names(interpersonal_model)}

    for forbidden in ("summarize", "summarise", "generate_text", "complete", "chat"):
        assert forbidden not in called, f"module calls {forbidden!r}"


def test_nothing_here_accepts_an_injected_summarizer():
    for name, parameter in inspect.signature(PersonModel.__init__).parameters.items():
        assert "Callable" not in str(parameter.annotation), f"{name} takes a callable"


def test_merging_sums_evidence_rather_than_rewording(model):
    other = PersonModel("Bryan")
    model.observe("terse", episode_id="ep1")
    other.observe("terse", episode_id="ep2")

    model.merge(other)

    assert model.current()[0].support == 2


def test_a_correction_survives_a_merge(model):
    other = PersonModel("Bryan")
    model.observe("wrong thing", episode_id="ep1")
    model.correct("wrong thing", episode_id="ep2")
    other.observe("wrong thing", episode_id="ep3")

    model.merge(other)

    assert "wrong thing" not in model.render()


def test_merging_notes_about_two_people_is_refused(model):
    with pytest.raises(ValueError, match="conflating"):
        model.merge(PersonModel("Tatiana"))


# ── evidence and audit ─────────────────────────────────────────────────────


def test_a_claim_without_an_episode_is_refused(model):
    with pytest.raises(ValueError, match="evidence"):
        model.observe("something", episode_id="")


def test_the_audit_exposes_kind_source_and_evidence(model):
    model.observe("short answers", episode_id="ep1", facet=Facet.PREFERENCE,
                  provenance=Provenance.STATED)

    entry = model.audit()[0]

    assert entry["facet"] == "preference"
    assert entry["authority"] == "stated"
    assert entry["episodes"] == ["ep1"]


def test_the_render_never_states_a_confidence_number(model):
    model.observe("terse", episode_id="ep1")
    model.observe("terse", episode_id="ep2")

    rendered = model.render().lower()

    assert "confidence" not in rendered
    assert "%" not in rendered


def test_an_empty_model_says_so(model):
    assert "do not know anything" in model.render()


def test_a_model_needs_someone_to_be_about():
    with pytest.raises(ValueError):
        PersonModel("  ")
