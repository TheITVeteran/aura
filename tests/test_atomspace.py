"""Contract tests for the AtomSpace (Hyperon fusion).

Covers the metagraph store, PLN truth-value algebra, unification queries with
grounded predicates, the ECAN attention economy, economic forward chaining,
and the live fusion with the belief revision engine.
"""
from __future__ import annotations

import pytest

from core.knowledge.atomspace import (
    EVALUATION,
    GROUNDED_PREDICATE,
    IMPLICATION,
    LIST,
    AtomSpace,
    Link,
    Node,
    TruthValue,
    Variable,
    assert_claim,
    concept,
    deduction_tv,
    evaluation,
    implication,
    predicate,
    reset_atomspace_for_test,
    substitute,
    unify,
)


@pytest.fixture(autouse=True)
def _fresh_space():
    reset_atomspace_for_test()
    yield
    reset_atomspace_for_test()


# ── truth values ──────────────────────────────────────────────────────────

def test_revision_is_evidence_weighted_and_convergent():
    prior = TruthValue(0.9, 10.0)
    weak_contrary = TruthValue(0.1, 1.0)
    revised = prior.revise(weak_contrary)
    # Ten units of 0.9-evidence vs one of 0.1: barely moves.
    assert 0.8 < revised.strength < 0.9
    assert revised.count == pytest.approx(11.0)
    # Symmetric: revision order does not matter.
    assert weak_contrary.revise(prior).strength == pytest.approx(revised.strength)


def test_revision_confidence_grows_with_count():
    tv = TruthValue(0.8, 1.0)
    for _ in range(9):
        tv = tv.revise(TruthValue(0.8, 1.0))
    assert tv.count == pytest.approx(10.0)
    assert tv.confidence > TruthValue(0.8, 1.0).confidence


def test_deduction_formula_bounds_and_degeneracy():
    ab = TruthValue(0.9, 10.0)
    bc = TruthValue(0.8, 10.0)
    tv = deduction_tv(ab, bc, TruthValue(0.5, 1), TruthValue(0.5, 1))
    assert 0.0 <= tv.strength <= 1.0
    assert tv.count < min(ab.count, bc.count)  # deduction discount
    # Perfect premises with sB≈1 degenerate to multiplication.
    tv2 = deduction_tv(TruthValue(1.0, 10), TruthValue(1.0, 10), TruthValue(1.0, 10), TruthValue(0.5, 1))
    assert tv2.strength == pytest.approx(1.0)


# ── store semantics ───────────────────────────────────────────────────────

def test_add_is_deduplicating_and_revising():
    space = AtomSpace()
    a = concept("rain")
    tv1 = space.add(a, TruthValue(1.0, 1.0))
    tv2 = space.add(a, TruthValue(0.0, 1.0))
    assert len(space) == 1
    assert tv1.strength == 1.0
    assert tv2.strength == pytest.approx(0.5)
    assert tv2.count == pytest.approx(2.0)


def test_links_can_point_at_links_metagraph():
    space = AtomSpace()
    inner = implication(concept("a"), concept("b"))
    outer = evaluation(predicate("derived_from"), inner, concept("observation"))
    space.add(outer, TruthValue(1.0, 1.0))
    assert inner in space
    assert outer in space
    # The incoming index chains through the nesting: inner sits inside the
    # List link, which sits inside the Evaluation link.
    lists = space.incoming(inner)
    assert lists and lists[0].atype == LIST
    assert any(link.atype == EVALUATION for link in space.incoming(lists[0]))


def test_cannot_add_patterns():
    space = AtomSpace()
    with pytest.raises(ValueError):
        space.add(Link(IMPLICATION, (Variable("x"), concept("b"))))


# ── unification and queries ───────────────────────────────────────────────

def test_unify_binds_consistently():
    pattern = implication(Variable("x"), Variable("x"))
    assert unify(pattern, implication(concept("a"), concept("a"))) is not None
    assert unify(pattern, implication(concept("a"), concept("b"))) is None


def test_substitute_instantiates():
    pattern = implication(Variable("x"), concept("wet"))
    out = substitute(pattern, {"x": concept("rain")})
    assert out == implication(concept("rain"), concept("wet"))


def test_conjunctive_query_joins_on_shared_variables():
    space = AtomSpace()
    space.add(implication(concept("rain"), concept("wet")), TruthValue(0.9, 5))
    space.add(implication(concept("wet"), concept("slippery")), TruthValue(0.8, 5))
    space.add(implication(concept("sunny"), concept("dry")), TruthValue(0.9, 5))
    results = space.query([
        Link(IMPLICATION, (Variable("a"), Variable("b"))),
        Link(IMPLICATION, (Variable("b"), Variable("c"))),
    ])
    chains = {(b["a"].name, b["b"].name, b["c"].name) for b in results}
    assert ("rain", "wet", "slippery") in chains
    assert all(mid != "dry" for _, mid, _ in chains)


def test_grounded_predicate_filters_query():
    space = AtomSpace()
    for name in ("alpha", "beta", "gamma"):
        space.add(concept(name), TruthValue(1.0, 1.0))
    space.register_grounded("starts_with_a", lambda atom: isinstance(atom, Node) and atom.name.startswith("a"))
    results = space.query([
        Variable("x"),
        Link(EVALUATION, (Node(GROUNDED_PREDICATE, "starts_with_a"), Link(LIST, (Variable("x"),)))),
    ])
    names = {b["x"].name for b in results if isinstance(b["x"], Node) and b["x"].atype == "Concept"}
    assert names == {"alpha"}


# ── ECAN economy ──────────────────────────────────────────────────────────

def test_stimulation_is_fund_bounded():
    space = AtomSpace(sti_fund=30.0, stimulus_size=20.0)
    a, b = concept("a"), concept("b")
    space.add(a)
    space.add(b)
    assert space.stimulate(a) == pytest.approx(20.0)
    # Fund only has 10 left — the economy cannot print attention.
    assert space.stimulate(b) == pytest.approx(10.0)
    assert space.stimulate(b) == 0.0


def test_rent_returns_sti_to_fund():
    space = AtomSpace(sti_fund=100.0, stimulus_size=50.0, rent_rate=0.1)
    a = concept("a")
    space.add(a)
    space.stimulate(a)
    before = space.get_av(a).sti
    collected = space.collect_rent()
    assert collected > 0
    assert space.get_av(a).sti < before


def test_importance_spreads_along_structure():
    space = AtomSpace(sti_fund=1000.0, stimulus_size=100.0, spread_fraction=0.5)
    rain, wet = concept("rain"), concept("wet")
    link = implication(rain, wet)
    space.add(link)
    space.stimulate(rain)
    assert space.get_av(wet).sti == 0.0
    space.spread_importance()
    # Attention reached the sibling through the link.
    assert space.get_av(wet).sti > 0.0
    assert space.get_av(link).sti > 0.0


def test_forgetting_evicts_only_unreferenced_low_importance_atoms():
    space = AtomSpace(max_atoms=4, sti_fund=1000.0)
    rain, wet = concept("rain"), concept("wet")
    link = implication(rain, wet)
    space.add(link)                             # 3 atoms; rain/wet referenced by link
    space.stimulate(link, 5.0)                  # attended structure outranks junk
    junk = [concept(f"junk{i}") for i in range(3)]
    for j in junk:
        space.add(j)
    keeper = concept("keeper")
    space.add(keeper)
    space.set_vlti(keeper)
    evicted = space.forget()
    assert evicted, "expected eviction over capacity"
    assert keeper in space                      # VLTI protected
    assert rain in space and wet in space       # referenced by the link
    assert link in space                        # attended, so kept
    assert set(evicted) == set(junk)


def test_attentional_focus_ranks_by_sti():
    space = AtomSpace(sti_fund=1000.0)
    a, b = concept("hot"), concept("cold")
    space.add(a)
    space.add(b)
    space.stimulate(a, 50.0)
    space.stimulate(b, 5.0)
    focus = space.attentional_focus()
    assert focus[0][0] == a


# ── economic forward chaining ─────────────────────────────────────────────

def test_forward_chain_derives_transitive_implication():
    space = AtomSpace(sti_fund=1000.0)
    space.add(implication(concept("rain"), concept("wet")), TruthValue(0.9, 10))
    space.add(implication(concept("wet"), concept("slippery")), TruthValue(0.8, 10))
    space.stimulate(concept("wet"), 50.0)
    derived = space.forward_chain(max_derivations=4, focus_only=True)
    ac = implication(concept("rain"), concept("slippery"))
    assert ac in [d for d in derived]
    tv = space.get_tv(ac)
    assert tv is not None and 0.0 < tv.strength <= 1.0 and tv.count > 0


def test_forward_chain_respects_attention_gate():
    space = AtomSpace(sti_fund=1000.0)
    space.add(implication(concept("rain"), concept("wet")), TruthValue(0.9, 10))
    space.add(implication(concept("wet"), concept("slippery")), TruthValue(0.8, 10))
    # Nothing stimulated → nothing in focus → economic chainer stays idle.
    assert space.forward_chain(focus_only=True) == []
    # Full scan still works when explicitly asked.
    assert space.forward_chain(focus_only=False) != []


def test_forward_chain_is_budget_bounded():
    space = AtomSpace(sti_fund=10_000.0)
    for i in range(12):
        space.add(
            implication(concept(f"n{i}"), concept(f"n{i + 1}")), TruthValue(0.9, 10)
        )
        space.stimulate(concept(f"n{i}"), 20.0)
    derived = space.forward_chain(max_derivations=5, focus_only=False)
    assert len(derived) <= 5


# ── claim bridge + live belief fusion ─────────────────────────────────────

def test_assert_claim_encodes_implications():
    space = AtomSpace(sti_fund=1000.0)
    atom, tv = assert_claim(space, "if it rains then the ground is wet", TruthValue(0.8, 1.0))
    assert isinstance(atom, Link) and atom.atype == IMPLICATION
    assert tv.strength == pytest.approx(0.8)


def test_assert_claim_negation_inverts_strength():
    space = AtomSpace(sti_fund=1000.0)
    atom, tv = assert_claim(space, "the system is not stable", TruthValue(0.9, 1.0))
    assert isinstance(atom, Node)
    assert tv.strength == pytest.approx(0.1)


def test_assert_claim_repeated_assertion_revises():
    space = AtomSpace(sti_fund=1000.0)
    _, tv1 = assert_claim(space, "the sky is blue", TruthValue(1.0, 1.0))
    _, tv2 = assert_claim(space, "the sky is blue", TruthValue(1.0, 1.0))
    assert tv2.count > tv1.count


@pytest.mark.asyncio
async def test_belief_engine_uses_pln_revision_and_mirrors(tmp_path):
    from core.epistemics.belief_revision import BeliefRevisionEngine
    from core.knowledge.atomspace import get_atomspace

    engine = BeliefRevisionEngine(db_path=str(tmp_path / "beliefs.json"))
    created = await engine.process_new_claim("the reactor is stable", "world", "tool_result", 0.9)
    assert created["ok"]
    belief = next(b for b in engine.beliefs if b.id == created["belief_id"])
    first_conf, first_count = belief.confidence, belief.evidence_count

    updated = await engine.process_new_claim("the reactor is stable", "world", "tool_result", 0.9)
    assert updated["updated"]
    assert belief.evidence_count > first_count          # evidence accumulates
    assert 0.0 <= belief.confidence <= 1.0
    assert belief.confidence == pytest.approx(first_conf, abs=0.05)

    # The claim is mirrored into the atomspace with attention.
    space = get_atomspace()
    assert len(space) > 0
    assert space.attentional_focus(), "claim atoms should hold attention"


def test_atomspace_service_registered_name():
    from core.service_names import ServiceNames

    assert ServiceNames.ATOMSPACE == "atomspace"
