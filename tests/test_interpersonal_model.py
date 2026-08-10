"""Qualifiers, frequency, conditions and counter-evidence must survive.

The failure this replaces: notes on a person kept as prose and periodically
summarised. In prose the load-bearing parts of knowing someone are adjectives —
"seemed", "once", "when a build was failing", "though usually not" — and
compressing prose drops adjectives before nouns. So "Bryan seemed frustrated
once during a failing deploy" walks to "Bryan gets frustrated" to "Bryan is
easily frustrated", no step unreasonable, nobody deciding.

The first test is the one that matters: the drift is not merely unlikely here,
it is unrepresentable, because frequency is an integer field and not a word.
"""
from __future__ import annotations

import ast
import inspect
import time

import pytest

from core.memory import interpersonal_model
from core.memory.interpersonal_model import DAY_SECONDS, PersonModel

pytestmark = pytest.mark.unit


@pytest.fixture
def model():
    return PersonModel("Bryan")


# ── the drift is unrepresentable ───────────────────────────────────────────


def test_one_sighting_renders_as_once_not_as_a_trait(model):
    model.observe("frustrated", episode_id="ep1", conditions="a failing deploy")

    rendered = model.render()

    assert "once" in rendered
    assert "a failing deploy" in rendered


def test_a_second_sighting_increments_rather_than_rewording(model):
    model.observe("frustrated", episode_id="ep1")
    model.observe("frustrated", episode_id="ep2")

    observation = model.strongest()[0]

    assert observation.support == 2
    assert observation.claim == "frustrated"  # wording untouched


def test_frequency_cannot_be_inflated_by_rewording(model):
    """Strength lives in the count, not the adjectives, so there is no wording
    change that can make one sighting look like a pattern."""
    model.observe("frustrated", episode_id="ep1")

    assert "once" in model.render()
    assert "3 times" not in model.render()


def test_conditions_are_a_field_not_a_clause(model):
    """A clause can be dropped as a stylistic choice; a field cannot."""
    model.observe("terse", episode_id="ep1", conditions="when a build is failing")

    assert model.strongest()[0].conditions == "when a build is failing"
    assert "when a build is failing" in model.render()


def test_the_same_claim_under_different_conditions_stays_separate(model):
    """Collapsing them is exactly how a conditional becomes a trait."""
    model.observe("terse", episode_id="ep1", conditions="when a build is failing")
    model.observe("terse", episode_id="ep2", conditions="in the morning")

    assert len(model) == 2


def _called_names(module) -> set[str]:
    """Every function/method name this module actually invokes.

    Parsed from the AST rather than grepped from the source, so prose in a
    docstring explaining the guarantee does not read as a violation of it.
    """
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
    """The structural guarantee. Consolidation is aggregation; if any path
    could hand a note to a model to rewrite, every other property here is
    decoration."""
    called = {name.lower() for name in _called_names(interpersonal_model)}

    for forbidden in ("summarize", "summarise", "generate_text", "complete", "chat"):
        assert forbidden not in called, f"module calls {forbidden!r}"


def test_nothing_here_accepts_an_injected_summarizer():
    """The other way a rewrite path arrives: a callable passed in at
    construction, the way LLMSummarizingCondenser takes one."""
    signature = inspect.signature(PersonModel.__init__)

    for name, parameter in signature.parameters.items():
        annotation = str(parameter.annotation)
        assert "Callable" not in annotation, f"{name} accepts a callable"


# ── evidence is required ───────────────────────────────────────────────────


def test_an_observation_without_an_episode_is_refused(model):
    with pytest.raises(ValueError, match="evidence"):
        model.observe("frustrated", episode_id="")


def test_every_claim_can_be_traced_to_its_episodes(model):
    model.observe("frustrated", episode_id="ep1")
    model.observe("frustrated", episode_id="ep2")

    assert model.strongest()[0].episodes() == ["ep1", "ep2"]


def test_the_audit_exposes_the_evidence_for_a_human(model):
    model.observe("terse", episode_id="ep1", conditions="failing build")

    entry = model.audit()[0]

    assert entry["claim"] == "terse"
    assert entry["episodes"] == ["ep1"]
    assert entry["conditions"] == "failing build"


def test_an_empty_claim_is_refused(model):
    with pytest.raises(ValueError):
        model.observe("   ", episode_id="ep1")


# ── counter-evidence is first-class ────────────────────────────────────────


def test_a_counter_example_is_recorded_not_discarded(model):
    model.observe("frustrated", episode_id="ep1")

    model.contradict("frustrated", episode_id="ep2")

    assert model.strongest()[0].contradictions == 1


def test_counter_evidence_is_rendered(model):
    """Prose summarisation never keeps it. A view that only accumulates
    confirmations is not a model, it is a grudge."""
    model.observe("frustrated", episode_id="ep1")
    model.observe("frustrated", episode_id="ep2")
    model.contradict("frustrated", episode_id="ep3")

    assert "did not hold" in model.render()


def test_counter_evidence_weakens_standing(model):
    model.observe("a", episode_id="e1")
    model.observe("a", episode_id="e2")
    model.observe("b", episode_id="e3")
    model.observe("b", episode_id="e4")
    model.contradict("b", episode_id="e5")

    assert model.strongest()[0].claim == "a"


def test_contradicting_something_never_observed_is_a_no_op(model):
    assert model.contradict("never seen", episode_id="ep1") is None


# ── recency is stated, not baked in ────────────────────────────────────────


def test_recency_is_rendered_in_words_derived_from_the_timestamp(model):
    now = time.time()
    model.observe("frustrated", episode_id="ep1", at=now - 3 * DAY_SECONDS)

    assert "3 days ago" in model.render(now=now)


def test_a_stale_observation_says_so(model):
    now = time.time()
    model.observe("frustrated", episode_id="ep1", at=now - 200 * DAY_SECONDS)

    assert "months ago" in model.render(now=now)


def test_today_is_rendered_as_today(model):
    now = time.time()
    model.observe("frustrated", episode_id="ep1", at=now - 60)

    assert "today" in model.render(now=now)


# ── consolidation is aggregation ───────────────────────────────────────────


def test_merging_sums_evidence_rather_than_reconciling_wording(model):
    other = PersonModel("Bryan")
    model.observe("terse", episode_id="ep1")
    other.observe("terse", episode_id="ep2")

    model.merge(other)

    assert model.strongest()[0].support == 2


def test_merging_is_idempotent(model):
    other = PersonModel("Bryan")
    model.observe("terse", episode_id="ep1")
    other.observe("terse", episode_id="ep1", at=model.strongest()[0].occurrences[0].at)

    model.merge(other)
    model.merge(other)

    assert model.strongest()[0].support == 1


def test_merging_carries_counter_evidence_across(model):
    other = PersonModel("Bryan")
    model.observe("terse", episode_id="ep1")
    other.observe("terse", episode_id="ep2")
    other.contradict("terse", episode_id="ep3")

    model.merge(other)

    assert model.strongest()[0].contradictions == 1


def test_merging_notes_about_two_different_people_is_refused(model):
    with pytest.raises(ValueError, match="conflating"):
        model.merge(PersonModel("Tatiana"))


# ── bounded, and bounded by evidence ───────────────────────────────────────


def test_the_model_is_bounded(model):
    small = PersonModel("Bryan", max_observations=3)

    for i in range(10):
        small.observe(f"claim {i}", episode_id=f"ep{i}")

    assert len(small) == 3


def test_eviction_drops_the_weakest_evidence_not_the_oldest():
    """A standing pattern that happens to be old is worth more than a thing
    noticed once and never again."""
    small = PersonModel("Bryan", max_observations=2)
    now = time.time()
    small.observe("old pattern", episode_id="e1", at=now - 100 * DAY_SECONDS)
    small.observe("old pattern", episode_id="e2", at=now - 99 * DAY_SECONDS)
    small.observe("old pattern", episode_id="e3", at=now - 98 * DAY_SECONDS)
    small.observe("recent one-off", episode_id="e4", at=now)
    small.observe("another one-off", episode_id="e5", at=now)

    claims = {o.claim for o in small}
    assert "old pattern" in claims


def test_a_correction_removes_an_observation_outright(model):
    model.observe("frustrated", episode_id="ep1")

    assert model.forget("frustrated") is True
    assert len(model) == 0


def test_forgetting_something_absent_reports_so(model):
    assert model.forget("never seen") is False


# ── rendering ──────────────────────────────────────────────────────────────


def test_an_empty_model_renders_honestly(model):
    assert "No observations" in model.render()


def test_the_render_is_capped(model):
    for i in range(20):
        model.observe(f"claim {i}", episode_id=f"ep{i}")

    assert len(model.render(limit=5).splitlines()) == 6  # header + 5


def test_the_render_never_states_a_confidence_number(model):
    """A manufactured score would be a summary of the evidence — the same lossy
    move one level up, and one nobody could audit."""
    model.observe("frustrated", episode_id="ep1")
    model.observe("frustrated", episode_id="ep2")

    rendered = model.render().lower()

    assert "confidence" not in rendered
    assert "%" not in rendered


def test_a_model_needs_someone_to_be_about():
    with pytest.raises(ValueError):
        PersonModel("  ")
