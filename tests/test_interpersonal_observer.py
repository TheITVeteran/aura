"""Noticing things about a person, without inventing them.

`PersonModel` guarantees that stored knowledge cannot lose its qualifiers. Every
one of those guarantees is downstream of what the observer admits, so this is
where a caricature actually gets manufactured — not by summarising a note, but
by writing a note that was never true.

Three failures these guard, all found by running the detectors on ordinary
phrasing rather than by reading them:

* A paraphrase at the door. Rebuilding "I don't like long meetings" as "does not
  want like long meetings" filed a sentence he never said as something he told
  her — at STATED authority, which is protected from eviction and rendered as
  "he told me this".
* A figure of speech read as a standing instruction. "don't worry about it"
  became a permanent preference.
* Agreement read as a rupture. "no, it works fine now" became a
  MISUNDERSTANDING, and misunderstandings feed the connection dynamic, so
  enough of them render "we keep missing each other" out of him saying things
  are fine.
"""
from __future__ import annotations

import ast
import inspect
import time

import pytest

from core.memory.interpersonal_model import (
    Facet,
    PersonModel,
    Provenance,
    Subject,
    Valence,
)
from core.memory.interpersonal_observer import (
    Exchange,
    InterpersonalObserver,
    Proposal,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def model():
    return PersonModel("Bryan")


@pytest.fixture
def observer(model):
    return InterpersonalObserver(model)


def _facets(proposals) -> set[Facet]:
    return {p.facet for p in proposals}


def _claims(proposals) -> list[str]:
    return [p.claim for p in proposals]


# ── a disposition is never manufactured from one exchange ──────────────────


def test_a_single_exchange_never_produces_a_trait(observer):
    """No amount of trait-shaped phrasing yields a trait. Dispositions need time."""
    proposals = observer.notice(
        Exchange(
            episode_id="ep1",
            user_text="I hate flaky tests and this is frustrating, you keep doing it",
        )
    )

    assert proposals, "the exchange should be noticed at all"
    assert Facet.TRAIT not in _facets(proposals)


def test_a_proposer_that_returns_a_trait_is_refused(model):
    def propose(exchange):
        return [
            Proposal(
                claim="is easily frustrated",
                facet=Facet.TRAIT,
                provenance=Provenance.INFERRED,
                episode_id=exchange.episode_id,
            )
        ]

    observer = InterpersonalObserver(model, propose=propose)

    proposals = observer.notice(Exchange(episode_id="ep1", user_text="this is annoying"))

    assert "is easily frustrated" not in _claims(proposals)


def test_admit_refuses_a_trait_handed_to_it_directly(observer, model):
    """The gate is on the write, not only on the noticing.

    A caller that assembles its own proposals — a future extraction pass, a
    replay, a test — must not be able to route around the one rule that keeps
    a single conversation from becoming a personality.
    """
    written = observer.admit(
        [
            Proposal(
                claim="is easily frustrated",
                facet=Facet.TRAIT,
                provenance=Provenance.OBSERVED,
                episode_id="ep1",
            )
        ]
    )

    assert written == []
    assert len(model) == 0


def test_the_notice_path_and_the_write_path_share_one_gate():
    """Asserted from the AST, so a docstring promising it is not the evidence."""
    source = ast.parse(inspect.getsource(InterpersonalObserver))
    gated = {
        node.name: {
            call.func.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        }
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef)
    }

    assert "_admissible" in gated["notice"]
    assert "_admissible" in gated["admit"], (
        "admit() must re-check; filtering only in notice() leaves the write "
        "path open to any caller that builds its own proposals"
    )


def test_a_stated_claim_must_be_about_him(observer, model):
    """STATED means he said it about himself. It cannot be claimed for her."""
    written = observer.admit(
        [
            Proposal(
                claim="I am patient",
                facet=Facet.PREFERENCE,
                provenance=Provenance.STATED,
                subject=Subject.SELF,
                episode_id="ep1",
            )
        ]
    )

    assert written == []
    assert len(model) == 0


# ── a proposer may propose, never promote itself ───────────────────────────


def test_a_model_proposal_arrives_as_an_inference(model):
    """Whatever a proposer claims to know, it inferred."""

    def propose(exchange):
        return [
            Proposal(
                claim="he is under deadline pressure",
                facet=Facet.STATE,
                provenance=Provenance.STATED,
                episode_id=exchange.episode_id,
            )
        ]

    observer = InterpersonalObserver(model, propose=propose)

    proposals = observer.notice(Exchange(episode_id="ep1", user_text="anything"))

    assert [p.provenance for p in proposals] == [Provenance.INFERRED]


def test_an_inferred_claim_says_so_when_rendered(model):
    """An inference rendered as fact is how she ends up confidently wrong."""

    def propose(exchange):
        return [
            Proposal(
                claim="he is under deadline pressure",
                facet=Facet.STATE,
                provenance=Provenance.INFERRED,
                episode_id=exchange.episode_id,
            )
        ]

    observer = InterpersonalObserver(model, propose=propose)
    observer.observe_exchange(Exchange(episode_id="ep1", user_text="anything"))

    assert "my inference, not something I was told or saw" in model.render()


def test_a_failing_proposer_does_not_break_the_turn(model):
    def propose(exchange):
        raise RuntimeError("the small model fell over")

    observer = InterpersonalObserver(model, propose=propose)

    proposals = observer.notice(
        Exchange(episode_id="ep1", user_text="I prefer terse answers")
    )

    assert _claims(proposals) == ["I prefer terse answers"]


def test_the_proposer_is_never_handed_anything_already_stored(model):
    """It extracts from raw conversation. It does not get to revisit beliefs.

    Handing a proposer what she already believes is how "aggregation, never
    summarisation" is lost: the model would be re-wording stored notes, one
    call removed from the thing the store forbids outright.
    """
    seen: list[object] = []

    def propose(exchange):
        seen.append(exchange)
        return []

    model.observe("terse", episode_id="old", facet=Facet.TRAIT)
    observer = InterpersonalObserver(model, propose=propose)

    exchange = Exchange(episode_id="ep1", user_text="hello")
    observer.notice(exchange)

    assert seen == [exchange]
    assert not hasattr(seen[0], "model")
    signature = inspect.signature(InterpersonalObserver.__init__)
    assert "Exchange" in str(signature.parameters["propose"].annotation)
    assert "PersonModel" not in str(signature.parameters["propose"].annotation)


# ── noticing is not recording ──────────────────────────────────────────────


def test_notice_writes_nothing(observer, model):
    proposals = observer.notice(
        Exchange(episode_id="ep1", user_text="I prefer terse answers")
    )

    assert proposals
    assert len(model) == 0


def test_admit_is_what_writes(observer, model):
    observer.admit(
        observer.notice(Exchange(episode_id="ep1", user_text="I prefer terse answers"))
    )

    assert [o.claim for o in model.current()] == ["I prefer terse answers"]


# ── his words are kept as his words ────────────────────────────────────────


def test_one_statement_becomes_one_record(observer):
    """A general pattern must not re-read what a specific one already read."""
    proposals = observer.notice(
        Exchange(episode_id="ep1", user_text="I don't like long meetings")
    )

    assert _claims(proposals) == ["I don't like long meetings"]


def test_a_statement_is_never_paraphrased(observer, model):
    """The claim is what he typed.

    A module whose thesis is that rewording person-knowledge loses it should
    not begin by rewording it. The paraphrase this replaced produced "does not
    want like long meetings" and filed it as something he said.
    """
    observer.observe_exchange(
        Exchange(episode_id="ep1", user_text="I don't like long meetings")
    )

    rendered = model.render()

    assert "I don't like long meetings" in rendered
    assert "does not want" not in rendered


def test_a_figure_of_speech_is_not_a_standing_preference(observer):
    """Reassurance is not an instruction, and a bare negative cannot tell them apart."""
    for phrase in ("don't worry about it", "never mind", "don't get me wrong"):
        proposals = observer.notice(Exchange(episode_id="ep1", user_text=phrase))
        assert proposals == [], f"{phrase!r} was read as a preference"


def test_an_emphatic_instruction_is_still_caught(observer):
    """The narrowing must not cost the case it exists for."""
    proposals = observer.notice(
        Exchange(episode_id="ep1", user_text="please don't ever use emojis")
    )

    assert _claims(proposals) == ["please don't ever use emojis"]
    assert [p.provenance for p in proposals] == [Provenance.STATED]


def test_a_claim_stops_at_the_sentence_it_came_from(observer):
    """Two statements are two records, not one that swallowed the other."""
    proposals = observer.notice(
        Exchange(
            episode_id="ep1",
            user_text="I care about correctness. Also I hate flaky tests.",
        )
    )

    assert sorted(_claims(proposals)) == [
        "I care about correctness",
        "I hate flaky tests",
    ]
    assert _facets(proposals) == {Facet.VALUE, Facet.PREFERENCE}


def test_the_same_statement_twice_increments_rather_than_duplicates(observer, model):
    for episode in ("ep1", "ep2"):
        observer.observe_exchange(
            Exchange(episode_id=episode, user_text="I prefer terse answers")
        )

    assert len(model) == 1
    assert model.current()[0].support == 2


# ── agreement is not a rupture ─────────────────────────────────────────────


def test_agreement_is_not_recorded_as_a_misunderstanding(observer, model):
    observer.observe_exchange(
        Exchange(episode_id="ep1", user_text="no, it works fine now")
    )

    assert model.current(facet=Facet.MISUNDERSTANDING) == []


def test_a_false_misunderstanding_would_have_moved_the_connection_reading(observer, model):
    """Why the loose pattern mattered: it does not just sit in the record."""
    for episode in ("ep1", "ep2", "ep3"):
        observer.observe_exchange(
            Exchange(episode_id=episode, user_text="no, it works fine now")
        )

    connection = next(d for d in model.dynamics() if d.name == "connection")

    assert connection.standing != "we keep missing each other"


def test_an_explicit_repair_is_recorded(observer, model):
    observer.observe_exchange(
        Exchange(episode_id="ep1", user_text="no, that's not what I said")
    )

    missed = model.current(facet=Facet.MISUNDERSTANDING)

    assert len(missed) == 1
    assert missed[0].subject is Subject.BETWEEN
    assert missed[0].valence is Valence.DIFFICULT


# ── commitments are hers, not his ──────────────────────────────────────────


def test_his_undertaking_is_not_recorded_as_her_commitment(observer, model):
    observer.observe_exchange(
        Exchange(episode_id="ep1", user_text="I'll rerun the suite tonight")
    )

    assert model.current(facet=Facet.COMMITMENT) == []


def test_her_undertaking_is_recorded(observer, model):
    observer.observe_exchange(
        Exchange(
            episode_id="ep1",
            assistant_text="I'll rerun the suite before you wake up",
        )
    )

    commitments = model.current(facet=Facet.COMMITMENT)

    assert len(commitments) == 1
    assert commitments[0].subject is Subject.BETWEEN


# ── how it landed ──────────────────────────────────────────────────────────


def test_warmth_and_difficulty_are_recorded_with_their_valence(observer, model):
    observer.observe_exchange(
        Exchange(episode_id="ep1", user_text="thank you, that's exactly right")
    )
    observer.observe_exchange(
        Exchange(episode_id="ep2", user_text="this is frustrating")
    )

    valences = {o.valence for o in model.current(facet=Facet.AFFECT)}

    assert valences == {Valence.WARM, Valence.DIFFICULT}


# ── evidence ───────────────────────────────────────────────────────────────


def test_every_admitted_claim_carries_the_episode_that_justifies_it(observer, model):
    observer.observe_exchange(
        Exchange(episode_id="ep-42", user_text="I prefer terse answers")
    )

    assert all(entry["episodes"] == ["ep-42"] for entry in model.audit())


def test_the_evidence_note_is_the_sentence_the_claim_came_from(observer, model):
    """A claim shown beside the fragment that produced it is checkable only
    against itself; audit() exists so a human can see the context."""
    observer.observe_exchange(
        Exchange(
            episode_id="ep1",
            user_text="After the outage I decided I prefer terse answers",
        )
    )

    note = model.current()[0].occurrences[0].note

    assert note == "After the outage I decided I prefer terse answers"


def test_an_exchange_with_no_episode_id_is_refused(observer, model):
    with pytest.raises(ValueError, match="episode"):
        observer.observe_exchange(
            Exchange(episode_id="", user_text="I prefer terse answers")
        )


# ── the property the whole module exists for ───────────────────────────────


def test_a_hedged_statement_survives_the_whole_round_trip(model):
    """Exchange to store to prompt text, with the qualifier still attached.

    This is the end-to-end version of the original complaint: "seemed
    frustrated once during a failing deploy" must not arrive as "is easily
    frustrated" at the other end.
    """
    observer = InterpersonalObserver(model)
    observer.observe_exchange(
        Exchange(
            episode_id="ep1",
            user_text="I don't like long meetings when I'm already behind",
            at=time.time(),
        )
    )

    rendered = model.render()

    assert "when I'm already behind" in rendered
    assert "he told me this" in rendered
