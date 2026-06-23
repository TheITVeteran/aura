"""Tests for the natural-deduction prover (Pantheon image-4 deduction engine).

Covers the board's named rules (axiom, contradiction/ex-falso, ¬¬-elimination,
∧-elimination, ∨ split) plus classic propositional soundness: valid sequents must
prove, invalid ones must fail with a genuine countermodel.
"""
from __future__ import annotations

import pytest

from core.reasoning.natural_deduction import (
    And,
    Atom,
    Implies,
    Not,
    Or,
    entails,
    find_contradiction,
    is_consistent,
    parse,
    prove,
    prove_text,
)

A, B, C = Atom("A"), Atom("B"), Atom("C")


# ── parser ────────────────────────────────────────────────────────────────

def test_parser_precedence_and_associativity():
    assert parse("~A & B | C") == Or(And(Not(A), B), C)
    assert parse("A -> B -> C") == Implies(A, Implies(B, C))      # right-assoc
    assert parse("A <-> B") == And(Implies(A, B), Implies(B, A))
    assert parse("(A | B) & C") == And(Or(A, B), C)


# ── the board's named rules ───────────────────────────────────────────────

def test_axiom_membership():
    assert prove([A], A).provable                                  # G ∈ Γ


def test_contradiction_ex_falso_proves_anything():
    # {A, ¬A} ⊆ Γ  →  prove any goal
    assert prove([A, Not(A)], B).provable
    assert not is_consistent([A, Not(A)])


def test_double_negation_elimination():
    assert prove([Not(Not(A))], A).provable                        # ¬¬A ⊢ A


def test_conjunction_elimination():
    assert prove([And(A, B)], A).provable                          # A∧B ⊢ A
    assert prove([And(A, B)], B).provable                          # A∧B ⊢ B


def test_disjunction_split():
    assert prove([Or(A, B), Not(A)], B).provable                   # disjunctive syllogism


# ── classic valid sequents ────────────────────────────────────────────────

@pytest.mark.parametrize("premises,goal", [
    (["A", "A -> B"], "B"),                       # modus ponens
    (["A -> B", "~B"], "~A"),                     # modus tollens
    (["A -> B", "B -> C"], "A -> C"),             # hypothetical syllogism
    (["A -> B", "C -> B", "A | C"], "B"),         # constructive dilemma
    (["A & B"], "B & A"),                         # ∧ commutes
    (["~(A & B)"], "~A | ~B"),                    # De Morgan
    (["~A | ~B"], "~(A & B)"),                    # De Morgan (other way)
    (["A <-> B", "A"], "B"),                      # iff elimination
    ([], "A | ~A"),                               # excluded middle
    ([], "A -> A"),                               # identity
    ([], "~(A & ~A)"),                            # non-contradiction
    ([], "(A -> B) | (B -> A)"),                  # a propositional tautology
])
def test_valid_sequents_prove(premises, goal):
    assert prove_text(premises, goal).provable, f"{premises} ⊢ {goal} should hold"


# ── invalid sequents must fail with a real countermodel ───────────────────

@pytest.mark.parametrize("premises,goal", [
    (["A -> B"], "B -> A"),            # converse error
    (["A -> B", "B"], "A"),            # affirming the consequent
    (["A | B"], "A"),                  # disjunction doesn't give a disjunct
    (["A"], "B"),                      # non sequitur
])
def test_invalid_sequents_fail_with_countermodel(premises, goal):
    p = prove_text(premises, goal)
    assert not p.provable
    assert p.countermodel is not None


def test_countermodel_actually_refutes():
    # {A→B} ⊬ B→A: the countermodel must satisfy the premise and falsify the goal.
    p = prove_text(["A -> B"], "B -> A")
    m = p.countermodel
    # premise A→B true, goal B→A false  ⇒  B true, A false
    assert m.get("B") is True
    assert m.get("A") is False


# ── consistency + contradiction detection ─────────────────────────────────

def test_is_consistent():
    assert is_consistent([A, Implies(A, B), B])
    assert not is_consistent([A, Not(A)])
    assert not is_consistent([And(A, B), Not(A)])


def test_find_contradiction_returns_minimal_core():
    core = find_contradiction([A, B, Not(A), C])
    assert core is not None
    names = {str(f) for f in core}
    assert names == {"A", "¬A"}                                    # minimal conflicting pair


def test_find_contradiction_none_when_consistent():
    assert find_contradiction([A, B, Implies(A, C)]) is None


def test_entails_matches_prove():
    assert entails([A, Implies(A, B)], B)
    assert not entails([Implies(A, B)], A)


def test_implication_chain_contradiction():
    # A, A→B, B→¬A is inconsistent (A forces B forces ¬A)
    core = find_contradiction([A, parse("A -> B"), parse("B -> ~A")])
    assert core is not None
