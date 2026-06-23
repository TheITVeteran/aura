"""Natural-deduction proof search — the *Pantheon* whiteboard's deduction engine.

The fourth board panel is not plasticity: it is a symbolic decision procedure,
``PROCEDURE Hp FOR DEDUCTION IN Hc`` over a hypothesis set Γ, with the rules

    1. if G is an axiom or G ∈ Γ            → return ⟨G⟩
    2. if {¬A, A} ⊆ Γ                       → contradiction (ex falso: prove anything)
    3. if ¬¬A ∈ Γ                           → reduce with A           (¬¬-elimination)
    4. if A∧B ∈ Γ                           → reduce with A, B        (∧-elimination)
       … and the dual goal/▽-rules and the ∨ case-split (``SIMPL(H₁∨H₂, Γ)``).

This module makes that a real, sound, terminating prover for **propositional
logic**. Under the hood it runs an analytic tableau (semantic decision procedure)
— Γ ⊢ G iff Γ ∪ {¬G} is unsatisfiable — which is exactly the board's structure:
the closure test ``{A,¬A} ⊆ branch`` is rule 2, and the α-rules (¬¬A→A, A∧B→A,B,
De Morgan) are rules 3–4 and ``SIMPL``. It is complete and decidable for
propositional logic, so it always terminates with a proof or a countermodel.

The engine is pure and side-effect free; it is wired into ``SymbolicBridge`` as an
exact solver, into belief contradiction-detection, and to a governance signal in
``core/reasoning/deduction_governance.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# ── Formula AST ───────────────────────────────────────────────────────────

class Formula:
    """Base class for propositional formulas (immutable, hashable)."""

    __slots__ = ()

    def __and__(self, other: "Formula") -> "And":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True)
class Atom(Formula):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Bot(Formula):
    """Falsum (⊥ / □) — the always-false proposition."""

    def __str__(self) -> str:
        return "⊥"


@dataclass(frozen=True)
class Not(Formula):
    f: Formula

    def __str__(self) -> str:
        return f"¬{_wrap(self.f)}"


@dataclass(frozen=True)
class And(Formula):
    a: Formula
    b: Formula

    def __str__(self) -> str:
        return f"({_wrap(self.a)} ∧ {_wrap(self.b)})"


@dataclass(frozen=True)
class Or(Formula):
    a: Formula
    b: Formula

    def __str__(self) -> str:
        return f"({_wrap(self.a)} ∨ {_wrap(self.b)})"


@dataclass(frozen=True)
class Implies(Formula):
    a: Formula
    b: Formula

    def __str__(self) -> str:
        return f"({_wrap(self.a)} → {_wrap(self.b)})"


def _wrap(f: Formula) -> str:
    return str(f)


# Convenience constructors
def implies(a: Formula, b: Formula) -> Implies:
    return Implies(a, b)


# ── Parser (ergonomics for beliefs/tests) ─────────────────────────────────
# Grammar (low→high precedence): <-> , -> , | , & , ~ , atoms / parens.
# Accepts ~ ! ¬ for not; & ∧ for and; | ∨ for or; -> → for implies; <-> ↔ iff.

_TOKEN_RE = re.compile(r"\s*(<->|↔|->|→|<-|[()&|~!¬∧∨]|[A-Za-z_][A-Za-z0-9_]*)")


def parse(text: str) -> Formula:
    """Parse a propositional formula string into a Formula AST."""
    tokens = _tokenize(text)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def eat(expected: str | None = None) -> str:
        nonlocal pos
        tok = tokens[pos]
        if expected is not None and tok != expected:
            raise ValueError(f"expected {expected!r}, got {tok!r}")
        pos += 1
        return tok

    def parse_iff() -> Formula:
        left = parse_implies()
        while peek() in ("<->", "↔"):
            eat()
            right = parse_implies()
            # A ↔ B  ≡  (A→B) ∧ (B→A)
            left = And(Implies(left, right), Implies(right, left))
        return left

    def parse_implies() -> Formula:
        left = parse_or()
        if peek() in ("->", "→"):
            eat()
            return Implies(left, parse_implies())   # right-associative
        return left

    def parse_or() -> Formula:
        left = parse_and()
        while peek() in ("|", "∨"):
            eat()
            left = Or(left, parse_and())
        return left

    def parse_and() -> Formula:
        left = parse_not()
        while peek() in ("&", "∧"):
            eat()
            left = And(left, parse_not())
        return left

    def parse_not() -> Formula:
        if peek() in ("~", "!", "¬"):
            eat()
            return Not(parse_not())
        return parse_atom()

    def parse_atom() -> Formula:
        tok = peek()
        if tok == "(":
            eat("(")
            inner = parse_iff()
            eat(")")
            return inner
        if tok in ("false", "False", "⊥", "_|_"):
            eat()
            return Bot()
        if tok is None or not re.match(r"[A-Za-z_]", tok):
            raise ValueError(f"unexpected token {tok!r} in {text!r}")
        eat()
        if tok in ("true", "True"):
            # ⊤ ≡ ¬⊥
            return Not(Bot())
        return Atom(tok)

    result = parse_iff()
    if pos != len(tokens):
        raise ValueError(f"trailing tokens in {text!r}: {tokens[pos:]}")
    return result


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ValueError(f"cannot tokenize {text[pos:]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


# ── Tableau decision procedure ────────────────────────────────────────────

@dataclass
class Proof:
    """Result of a proof search."""

    goal: str
    premises: list[str]
    provable: bool
    method: str = "analytic_tableau"
    trace: list[str] = field(default_factory=list)
    countermodel: dict[str, bool] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "premises": self.premises,
            "provable": self.provable,
            "method": self.method,
            "trace": self.trace,
            "countermodel": self.countermodel,
        }


def _negate(f: Formula) -> Formula:
    return f.f if isinstance(f, Not) else Not(f)


def _expand(f: Formula) -> tuple[str, list[list[Formula]]]:
    """Return (rule_name, branches) for a non-literal formula.

    Each inner list is conjunctive (all added to the same branch); the outer list
    is disjunctive (a branch split). α-rules return one branch; β-rules two.
    """
    if isinstance(f, Not) and isinstance(f.f, Not):                  # ¬¬A → A
        return "¬¬-elimination", [[f.f.f]]
    if isinstance(f, And):                                            # A∧B → A, B
        return "∧-elimination", [[f.a, f.b]]
    if isinstance(f, Not) and isinstance(f.f, Or):                    # ¬(A∨B) → ¬A, ¬B
        return "De Morgan(¬∨)", [[_negate(f.f.a), _negate(f.f.b)]]
    if isinstance(f, Not) and isinstance(f.f, Implies):              # ¬(A→B) → A, ¬B
        return "¬→", [[f.f.a, _negate(f.f.b)]]
    if isinstance(f, Or):                                             # A∨B → A | B  (SIMPL split)
        return "∨-split", [[f.a], [f.b]]
    if isinstance(f, Not) and isinstance(f.f, And):                  # ¬(A∧B) → ¬A | ¬B
        return "De Morgan(¬∧)", [[_negate(f.f.a)], [_negate(f.f.b)]]
    if isinstance(f, Implies):                                        # A→B → ¬A | B
        return "→-split", [[_negate(f.a)], [f.b]]
    return "", []


def _literal_conflict(branch: set[Formula]) -> tuple[Formula, Formula] | None:
    if Bot() in branch:
        return Bot(), Bot()
    for f in branch:
        if isinstance(f, Not) and f.f in branch:                     # {A, ¬A} ⊆ Γ
            return f.f, f
    return None


def _saturate(formulas: Iterable[Formula], trace: list[str] | None = None) -> dict[str, bool] | None:
    """Analytic tableau: return a satisfying model (open branch) or None (closed).

    Each branch is processed to completion; non-literals are decomposed by α/β
    rules (smaller subformulas → termination). A branch closes on {A,¬A} or ⊥;
    an open saturated branch yields a countermodel.
    """
    stack: list[set[Formula]] = [set(formulas)]
    while stack:
        branch = stack.pop()
        conflict = _literal_conflict(branch)
        if conflict is not None:
            if trace is not None:
                trace.append(f"closed: {conflict[0]} ∧ ¬{conflict[0]}" if conflict[0] != Bot() else "closed: ⊥")
            continue
        # find a non-literal to expand
        target = None
        for f in branch:
            if not (isinstance(f, Atom) or isinstance(f, Bot)
                    or (isinstance(f, Not) and isinstance(f.f, Atom))):
                target = f
                break
        if target is None:
            # saturated open branch → countermodel
            model: dict[str, bool] = {}
            for f in branch:
                if isinstance(f, Atom):
                    model[f.name] = True
                elif isinstance(f, Not) and isinstance(f.f, Atom):
                    model.setdefault(f.f.name, False)
            return model
        rule, branches = _expand(target)
        if trace is not None and rule:
            trace.append(f"apply {rule} to {target}")
        rest = branch - {target}
        for add in branches:
            stack.append(rest | set(add))
    return None


# ── Public API ────────────────────────────────────────────────────────────

def atoms(formula: Formula) -> set[str]:
    """All atom names appearing in a formula."""
    if isinstance(formula, Atom):
        return {formula.name}
    if isinstance(formula, Bot):
        return set()
    if isinstance(formula, Not):
        return atoms(formula.f)
    if isinstance(formula, (And, Or, Implies)):
        return atoms(formula.a) | atoms(formula.b)
    return set()


def is_consistent(formulas: Iterable[Formula]) -> bool:
    """True iff the set of formulas is satisfiable (has a model)."""
    return _saturate(formulas) is not None


def entails(premises: Iterable[Formula], goal: Formula) -> bool:
    """True iff premises ⊨ goal (Γ ∪ {¬goal} is unsatisfiable)."""
    return _saturate(list(premises) + [_negate(goal)]) is None


def prove(premises: Iterable[Formula], goal: Formula) -> Proof:
    """Hp(Γ, G): search for a proof of ``goal`` from ``premises``.

    Returns a :class:`Proof` whose ``provable`` flag is the soundness verdict,
    with a rule trace when proved or a countermodel when not.
    """
    prem = list(premises)
    trace: list[str] = []
    closed = _saturate(prem + [_negate(goal)], trace)
    if closed is None:
        return Proof(
            goal=str(goal),
            premises=[str(p) for p in prem],
            provable=True,
            trace=trace or [f"{goal} follows from the premises (all tableau branches close)"],
        )
    # not provable → the open branch is a countermodel of Γ ⊨ G
    return Proof(
        goal=str(goal),
        premises=[str(p) for p in prem],
        provable=False,
        trace=[f"countermodel found: {goal} does not follow"],
        countermodel=closed,
    )


def find_contradiction(formulas: Iterable[Formula]) -> list[Formula] | None:
    """If the set is inconsistent, return a minimal unsatisfiable subset; else None."""
    fs = list(dict.fromkeys(formulas))   # dedupe, preserve order
    if is_consistent(fs):
        return None
    # delta-debug to a minimal conflicting core
    minimal = list(fs)
    changed = True
    while changed:
        changed = False
        for f in list(minimal):
            candidate = [g for g in minimal if g is not f]
            if candidate and not is_consistent(candidate):
                minimal = candidate
                changed = True
                break
    return minimal


def prove_text(premises: Iterable[str], goal: str) -> Proof:
    """String-friendly :func:`prove` — parses premises/goal first."""
    return prove([parse(p) for p in premises], parse(goal))
