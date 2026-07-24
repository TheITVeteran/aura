"""AtomSpace — Hyperon-style typed metagraph with economic attention.

OpenCog Hyperon's core organs, fused into Aura's knowledge layer:

- **Metagraph store**: immutable ``Node``/``Link`` atoms where links may point
  at links (not just vertices), typed by plain strings, deduplicated by value.
- **PLN truth values**: every atom carries a ``TruthValue`` (strength, count);
  repeated assertions merge by the PLN *revision* rule — evidence-weighted,
  convergent — replacing ad-hoc confidence blends. Chained implications are
  derived with the PLN *deduction* formula.
- **Unification queries**: MeTTa-style pattern matching with ``Variable``
  atoms, conjunctive multi-clause joins, and grounded predicates (Python
  callables evaluated during the match).
- **ECAN attention economy**: each atom carries STI/LTI attention values paid
  from a fixed fund; stimulation, rent, importance spreading along links,
  an attentional focus, and LTI-based forgetting. Inference is *economic*:
  the forward chainer only expands what is attentionally hot, so compute
  follows salience instead of scanning the whole graph.

Live wiring: ``BeliefRevisionEngine`` asserts every claim here (PLN revision
is now the confidence-update rule), stimulates claim atoms on access, ticks
the attention economy from its revision loop, and publishes derived
implications on the event bus (``atomspace.derived``).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

# ── PLN truth values ──────────────────────────────────────────────────────

# PLN's confidence lookahead: confidence = count / (count + _LOOKAHEAD).
_LOOKAHEAD = 1.0
_MAX_COUNT = 1000.0
# Confidence discount applied by inference rules (derived knowledge is never
# as certain as directly observed evidence).
_DEDUCTION_DISCOUNT = 0.9


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass(frozen=True)
class TruthValue:
    """PLN simple truth value: strength (probability) + count (evidence mass)."""

    strength: float = 0.5
    count: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "strength", _clamp01(float(self.strength)))
        object.__setattr__(self, "count", max(0.0, min(float(self.count), _MAX_COUNT)))

    @property
    def confidence(self) -> float:
        return self.count / (self.count + _LOOKAHEAD)

    def revise(self, other: "TruthValue") -> "TruthValue":
        """PLN revision: merge two independent estimates of the same atom.

        Evidence-weighted mean of strengths; counts add. Convergent — repeated
        confirmation raises confidence, contradictory evidence pulls strength
        toward the middle while the count keeps the disagreement visible.
        """
        total = self.count + other.count
        if total <= 0.0:
            return TruthValue((self.strength + other.strength) / 2.0, 0.0)
        s = (self.strength * self.count + other.strength * other.count) / total
        return TruthValue(s, total)

    def negation(self) -> "TruthValue":
        return TruthValue(1.0 - self.strength, self.count)

    def to_dict(self) -> dict[str, float]:
        return {"strength": self.strength, "count": self.count, "confidence": self.confidence}


def deduction_tv(ab: TruthValue, bc: TruthValue, b: TruthValue, c: TruthValue) -> TruthValue:
    """PLN deduction: from A→B and B→C derive A→C.

    Uses the standard independence-based PLN formula
    ``sAC = sAB·sBC + (1−sAB)·(sC − sB·sBC)/(1−sB)`` with node prevalences
    ``sB``/``sC`` (0.5 when unknown), clamped to [0,1]; the count is the
    weaker premise's count with a deduction discount.
    """
    s_ab, s_bc, s_b, s_c = ab.strength, bc.strength, b.strength, c.strength
    if s_b >= 0.9999:
        s_ac = s_ab * s_bc
    else:
        s_ac = s_ab * s_bc + (1.0 - s_ab) * (s_c - s_b * s_bc) / (1.0 - s_b)
    return TruthValue(_clamp01(s_ac), min(ab.count, bc.count) * _DEDUCTION_DISCOUNT)


# ── Atoms: the typed metagraph vocabulary ─────────────────────────────────

class Atom:
    """Base class for metagraph atoms (immutable, hashable, value-identity)."""

    __slots__ = ()


@dataclass(frozen=True)
class Node(Atom):
    atype: str
    name: str

    def __str__(self) -> str:
        return f"({self.atype} \"{self.name}\")"


@dataclass(frozen=True)
class Variable(Atom):
    """A pattern variable — binds to any atom during unification."""

    name: str

    def __str__(self) -> str:
        return f"${self.name}"


@dataclass(frozen=True)
class Link(Atom):
    atype: str
    outgoing: tuple[Atom, ...]

    def __str__(self) -> str:
        inner = " ".join(str(a) for a in self.outgoing)
        return f"({self.atype} {inner})"


# Conventional type names (plain strings, like Hyperon's symbols).
CONCEPT = "Concept"
PREDICATE = "Predicate"
GROUNDED_PREDICATE = "GroundedPredicate"
IMPLICATION = "Implication"
INHERITANCE = "Inheritance"
EVALUATION = "Evaluation"
LIST = "List"


def concept(name: str) -> Node:
    return Node(CONCEPT, name)


def predicate(name: str) -> Node:
    return Node(PREDICATE, name)


def implication(a: Atom, b: Atom) -> Link:
    return Link(IMPLICATION, (a, b))


def evaluation(pred: Atom, *args: Atom) -> Link:
    return Link(EVALUATION, (pred, Link(LIST, tuple(args))))


# ── Attention values (ECAN) ───────────────────────────────────────────────

@dataclass
class AttentionValue:
    sti: float = 0.0        # short-term importance (what matters *now*)
    lti: float = 0.0        # long-term importance (what has mattered repeatedly)
    vlti: bool = False      # very-long-term: exempt from forgetting

    def to_dict(self) -> dict[str, object]:
        return {"sti": self.sti, "lti": self.lti, "vlti": self.vlti}


@dataclass
class _Record:
    atom: Atom
    tv: TruthValue
    av: AttentionValue = field(default_factory=AttentionValue)
    added_at: float = field(default_factory=time.time)


# ── Unification ───────────────────────────────────────────────────────────

Bindings = dict[str, Atom]


def unify(pattern: Atom, ground: Atom, bindings: Bindings | None = None) -> Bindings | None:
    """Match ``pattern`` (may contain Variables) against a ground atom.

    Returns the extended bindings on success, None on mismatch. Consistent:
    a variable bound earlier must match the same atom on reuse.
    """
    b = dict(bindings) if bindings else {}

    def walk(p: Atom, g: Atom) -> bool:
        if isinstance(p, Variable):
            bound = b.get(p.name)
            if bound is None:
                b[p.name] = g
                return True
            return bound == g
        if isinstance(p, Node):
            return isinstance(g, Node) and p == g
        if isinstance(p, Link):
            if not isinstance(g, Link) or p.atype != g.atype or len(p.outgoing) != len(g.outgoing):
                return False
            return all(walk(pc, gc) for pc, gc in zip(p.outgoing, g.outgoing))
        return False

    return b if walk(pattern, ground) else None


def substitute(pattern: Atom, bindings: Bindings) -> Atom:
    """Instantiate a pattern with bindings (unbound variables stay symbolic)."""
    if isinstance(pattern, Variable):
        return bindings.get(pattern.name, pattern)
    if isinstance(pattern, Link):
        return Link(pattern.atype, tuple(substitute(a, bindings) for a in pattern.outgoing))
    return pattern


def _pattern_is_ground(atom: Atom) -> bool:
    if isinstance(atom, Variable):
        return False
    if isinstance(atom, Link):
        return all(_pattern_is_ground(a) for a in atom.outgoing)
    return True


# ── The AtomSpace ─────────────────────────────────────────────────────────

class AtomSpace:
    """Thread-safe metagraph store with PLN truth values and ECAN attention."""

    def __init__(
        self,
        *,
        sti_fund: float = 10_000.0,
        stimulus_size: float = 20.0,
        rent_rate: float = 0.05,
        spread_fraction: float = 0.2,
        focus_size: int = 20,
        max_atoms: int = 50_000,
    ) -> None:
        self._lock = threading.RLock()
        self._records: dict[Atom, _Record] = {}
        self._by_type: dict[str, set[Atom]] = {}
        self._incoming: dict[Atom, set[Link]] = {}
        self._grounded: dict[str, Callable[..., bool]] = {}
        # ECAN economy parameters
        self._sti_fund = float(sti_fund)
        self._sti_fund_capacity = float(sti_fund)
        self._stimulus_size = float(stimulus_size)
        self._rent_rate = float(rent_rate)
        self._spread_fraction = float(spread_fraction)
        self._focus_size = int(focus_size)
        self._max_atoms = int(max_atoms)
        self._forgotten_total = 0
        self._derived_total = 0

    # ── store ─────────────────────────────────────────────────────────

    def add(self, atom: Atom, tv: TruthValue | None = None) -> TruthValue:
        """Insert an atom (and its recursive outgoing set), revising the TV.

        Re-adding an existing atom merges truth by PLN revision — the Hyperon
        semantics for repeated assertion. Returns the atom's current TV.
        """
        if not _pattern_is_ground(atom):
            raise ValueError("cannot add a pattern (Variable) to the AtomSpace")
        with self._lock:
            return self._add_locked(atom, tv)

    def _add_locked(self, atom: Atom, tv: TruthValue | None) -> TruthValue:
        if isinstance(atom, Link):
            for child in atom.outgoing:
                if child not in self._records:
                    self._add_locked(child, None)
                self._incoming.setdefault(child, set()).add(atom)
        rec = self._records.get(atom)
        if rec is None:
            rec = _Record(atom=atom, tv=tv if tv is not None else TruthValue())
            self._records[atom] = rec
            atype = atom.atype if isinstance(atom, (Node, Link)) else "Unknown"
            self._by_type.setdefault(atype, set()).add(atom)
        elif tv is not None:
            rec.tv = rec.tv.revise(tv)
        return rec.tv

    def get_tv(self, atom: Atom) -> TruthValue | None:
        with self._lock:
            rec = self._records.get(atom)
            return rec.tv if rec else None

    def get_av(self, atom: Atom) -> AttentionValue | None:
        with self._lock:
            rec = self._records.get(atom)
            return AttentionValue(rec.av.sti, rec.av.lti, rec.av.vlti) if rec else None

    def set_vlti(self, atom: Atom, vlti: bool = True) -> None:
        with self._lock:
            rec = self._records.get(atom)
            if rec:
                rec.av.vlti = vlti

    def incoming(self, atom: Atom) -> list[Link]:
        with self._lock:
            return list(self._incoming.get(atom, ()))

    def atoms_of_type(self, atype: str) -> list[Atom]:
        with self._lock:
            return list(self._by_type.get(atype, ()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __contains__(self, atom: Atom) -> bool:
        with self._lock:
            return atom in self._records

    # ── queries ───────────────────────────────────────────────────────

    def register_grounded(self, name: str, fn: Callable[..., bool]) -> None:
        """Register a grounded predicate callable, usable in queries as
        ``evaluation(Node(GROUNDED_PREDICATE, name), args…)``."""
        with self._lock:
            self._grounded[name] = fn

    def match(self, pattern: Atom, bindings: Bindings | None = None) -> list[Bindings]:
        """All bindings under which ``pattern`` unifies with a stored atom."""
        with self._lock:
            if isinstance(pattern, Variable):
                candidates: Iterable[Atom] = list(self._records)
            elif isinstance(pattern, (Node, Link)):
                candidates = list(self._by_type.get(pattern.atype, ()))
            else:
                candidates = ()
            out: list[Bindings] = []
            for cand in candidates:
                b = unify(pattern, cand, bindings)
                if b is not None:
                    out.append(b)
            return out

    def _grounded_clause(self, clause: Atom) -> tuple[str, tuple[Atom, ...]] | None:
        if (
            isinstance(clause, Link)
            and clause.atype == EVALUATION
            and len(clause.outgoing) == 2
            and isinstance(clause.outgoing[0], Node)
            and clause.outgoing[0].atype == GROUNDED_PREDICATE
            and isinstance(clause.outgoing[1], Link)
            and clause.outgoing[1].atype == LIST
        ):
            return clause.outgoing[0].name, clause.outgoing[1].outgoing
        return None

    def query(self, clauses: Sequence[Atom]) -> list[Bindings]:
        """Conjunctive pattern query: join clause matches on shared variables.

        Grounded-predicate clauses are evaluated (not matched) once their
        arguments are bound, acting as filters — MeTTa's grounded atoms.
        """
        results: list[Bindings] = [{}]
        # Evaluate structural clauses first, grounded filters last.
        structural = [c for c in clauses if self._grounded_clause(c) is None]
        grounded = [c for c in clauses if self._grounded_clause(c) is not None]
        for clause in structural:
            next_results: list[Bindings] = []
            for b in results:
                next_results.extend(self.match(substitute(clause, b), b))
            results = next_results
            if not results:
                return []
        for clause in grounded:
            gname, gargs = self._grounded_clause(clause)  # type: ignore[misc]
            with self._lock:
                fn = self._grounded.get(gname)
            if fn is None:
                return []
            kept: list[Bindings] = []
            for b in results:
                args = tuple(substitute(a, b) for a in gargs)
                if any(not _pattern_is_ground(a) for a in args):
                    continue
                try:
                    if fn(*args):
                        kept.append(b)
                except (ValueError, TypeError, ArithmeticError):
                    continue
            results = kept
            if not results:
                return []
        return results

    # ── ECAN: the attention economy ───────────────────────────────────

    def stimulate(self, atom: Atom, amount: float | None = None) -> float:
        """Pay STI to an atom from the fund (bounded by what the fund holds).

        Every stimulation also accrues a sliver of LTI — repeated relevance is
        what long-term importance *is*. Returns the STI actually granted.
        """
        amt = self._stimulus_size if amount is None else float(amount)
        with self._lock:
            rec = self._records.get(atom)
            if rec is None or amt <= 0.0:
                return 0.0
            grant = min(amt, self._sti_fund)
            self._sti_fund -= grant
            rec.av.sti += grant
            rec.av.lti += grant * 0.01
            return grant

    def attentional_focus(self, k: int | None = None) -> list[tuple[Atom, float]]:
        """The top-k atoms by STI with any attention at all — the focus."""
        limit = self._focus_size if k is None else int(k)
        with self._lock:
            ranked = sorted(
                ((rec.atom, rec.av.sti) for rec in self._records.values() if rec.av.sti > 0.0),
                key=lambda pair: pair[1],
                reverse=True,
            )
            return ranked[:limit]

    def _neighbors_locked(self, atom: Atom) -> set[Atom]:
        out: set[Atom] = set()
        if isinstance(atom, Link):
            out.update(atom.outgoing)
        for link in self._incoming.get(atom, ()):
            out.add(link)
            out.update(a for a in link.outgoing if a != atom)
        return out

    def spread_importance(self) -> float:
        """Diffuse a fraction of focus atoms' STI to their graph neighbors.

        This is ECAN's importance spreading: salience flows along structure,
        so what is *related to* the current focus becomes findable next.
        Returns the total STI moved.
        """
        moved = 0.0
        with self._lock:
            focus = [
                self._records[atom]
                for atom, sti in self.attentional_focus()
                if sti > 0 and atom in self._records
            ]
            for rec in focus:
                neighbors = [self._records[n] for n in self._neighbors_locked(rec.atom) if n in self._records]
                if not neighbors:
                    continue
                share = rec.av.sti * self._spread_fraction
                rec.av.sti -= share
                per = share / len(neighbors)
                for n in neighbors:
                    n.av.sti += per
                moved += share
        return moved

    def collect_rent(self) -> float:
        """Charge proportional rent on all STI back into the fund (decay)."""
        collected = 0.0
        with self._lock:
            for rec in self._records.values():
                if rec.av.sti <= 0.0:
                    continue
                rent = rec.av.sti * self._rent_rate
                rec.av.sti -= rent
                collected += rent
                if rec.av.sti < 0.01:
                    collected += rec.av.sti
                    rec.av.sti = 0.0
            self._sti_fund = min(self._sti_fund + collected, self._sti_fund_capacity)
        return collected

    def forget(self) -> list[Atom]:
        """Evict the least-important atoms when over capacity (ECAN forgetting).

        Only atoms with no incoming links (nothing else depends on them), not
        marked VLTI, ranked by (LTI, STI). Never touches the belief store —
        this is working-memory hygiene, not knowledge deletion.
        """
        with self._lock:
            overflow = len(self._records) - self._max_atoms
            if overflow <= 0:
                return []
            candidates = sorted(
                (
                    rec
                    for rec in self._records.values()
                    if not rec.av.vlti and not self._incoming.get(rec.atom)
                ),
                key=lambda r: (r.av.lti, r.av.sti),
            )
            evicted: list[Atom] = []
            for rec in candidates[:overflow]:
                self._evict_locked(rec.atom)
                evicted.append(rec.atom)
            self._forgotten_total += len(evicted)
            return evicted

    def _evict_locked(self, atom: Atom) -> None:
        rec = self._records.pop(atom, None)
        if rec is None:
            return
        atype = atom.atype if isinstance(atom, (Node, Link)) else "Unknown"
        self._by_type.get(atype, set()).discard(atom)
        self._incoming.pop(atom, None)
        if isinstance(atom, Link):
            for child in atom.outgoing:
                inc = self._incoming.get(child)
                if inc:
                    inc.discard(atom)

    def tick(self) -> dict[str, float]:
        """One attention-economy cycle: rent, spreading, forgetting."""
        rent = self.collect_rent()
        moved = self.spread_importance()
        evicted = self.forget()
        return {"rent_collected": rent, "sti_spread": moved, "forgotten": float(len(evicted))}

    # ── PLN forward chaining (economic inference control) ─────────────

    def forward_chain(self, *, max_derivations: int = 16, focus_only: bool = True) -> list[Link]:
        """Derive new implications by PLN deduction: A→B, B→C ⇒ A→C.

        With ``focus_only`` (the ECAN synergy), only implications whose
        midpoint is in the attentional focus are expanded — inference effort
        follows salience instead of scanning the whole graph. Derived links
        enter the space via revision and receive a small stimulus so useful
        derivations can compound across cycles.
        """
        derived: list[Link] = []
        with self._lock:
            implications = [
                a for a in self._by_type.get(IMPLICATION, ())
                if isinstance(a, Link) and len(a.outgoing) == 2
            ]
            if focus_only:
                hot = {atom for atom, _ in self.attentional_focus()}
                candidates = [lk for lk in implications if lk.outgoing[1] in hot or lk in hot]
            else:
                candidates = implications
            by_source: dict[Atom, list[Link]] = {}
            for link in implications:
                by_source.setdefault(link.outgoing[0], []).append(link)
            for ab in candidates:
                if len(derived) >= max_derivations:
                    break
                b_atom = ab.outgoing[1]
                for bc in by_source.get(b_atom, ()):
                    if len(derived) >= max_derivations:
                        break
                    a_atom, c_atom = ab.outgoing[0], bc.outgoing[1]
                    if a_atom == c_atom:
                        continue
                    ac = Link(IMPLICATION, (a_atom, c_atom))
                    tv_ab = self._records[ab].tv
                    tv_bc = self._records[bc].tv
                    tv_b = self._records.get(b_atom, _Record(b_atom, TruthValue())).tv
                    tv_c = self._records.get(c_atom, _Record(c_atom, TruthValue())).tv
                    new_tv = deduction_tv(tv_ab, tv_bc, tv_b, tv_c)
                    if new_tv.count <= 0.0:
                        continue
                    already = self._records.get(ac)
                    if already is not None and already.tv.count >= new_tv.count:
                        continue
                    self._add_locked(ac, new_tv)
                    derived.append(ac)
            self._derived_total += len(derived)
        for link in derived:
            self.stimulate(link, self._stimulus_size * 0.25)
        return derived

    # ── introspection ─────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "atoms": len(self._records),
                "by_type": {t: len(s) for t, s in self._by_type.items() if s},
                "sti_fund": self._sti_fund,
                "focus": [(str(a), round(s, 2)) for a, s in self.attentional_focus(5)],
                "forgotten_total": self._forgotten_total,
                "derived_total": self._derived_total,
            }


# ── Claim bridge: NL beliefs → atoms (shared encoding with the prover) ────

def assert_claim(
    space: AtomSpace,
    claim: str,
    tv: TruthValue,
    *,
    domain: str = "world",
    stimulate: bool = True,
) -> tuple[Atom, TruthValue]:
    """Assert a natural-language claim into the space and return its revised TV.

    Uses the same propositional encoder as the deduction prover
    (``core/reasoning/belief_consistency.encode_belief``) so the atom
    namespace and the prover's atom namespace are one: implication-shaped
    claims become ``Implication`` links (fuel for PLN deduction), negated
    claims assert the concept with inverted strength (PLN negation).
    """
    from core.reasoning.belief_consistency import encode_belief
    from core.reasoning.natural_deduction import Implies, Not, Atom as PropAtom

    encoded = encode_belief(claim)
    formula = encoded.formula
    if isinstance(formula, Implies):
        def _leaf(f: Any) -> tuple[str, bool]:
            if isinstance(f, Not) and isinstance(f.f, PropAtom):
                return f.f.name, True
            if isinstance(f, PropAtom):
                return f.name, False
            return str(f), False

        ante, ante_neg = _leaf(formula.a)
        cons, cons_neg = _leaf(formula.b)
        # Negated endpoints keep their polarity in the concept name so the
        # implication link is faithful to the claim's logical content.
        a_node = concept(("¬" if ante_neg else "") + ante)
        c_node = concept(("¬" if cons_neg else "") + cons)
        atom: Atom = implication(a_node, c_node)
        out_tv = space.add(atom, tv)
    else:
        atom = concept(encoded.core_key)
        out_tv = space.add(atom, tv.negation() if encoded.negated else tv)
    ev = evaluation(predicate("claim_domain"), atom, concept(domain))
    space.add(ev, TruthValue(1.0, 1.0))
    if stimulate:
        space.stimulate(atom)
    return atom, out_tv


# ── Singleton ─────────────────────────────────────────────────────────────

_space_lock = threading.Lock()
_space: AtomSpace | None = None


def get_atomspace() -> AtomSpace:
    global _space
    with _space_lock:
        if _space is None:
            _space = AtomSpace()
        return _space


def reset_atomspace_for_test(**kwargs: Any) -> AtomSpace:
    global _space
    with _space_lock:
        _space = AtomSpace(**kwargs)
        return _space


__all__ = [
    "Atom",
    "AtomSpace",
    "AttentionValue",
    "Bindings",
    "CONCEPT",
    "EVALUATION",
    "GROUNDED_PREDICATE",
    "IMPLICATION",
    "INHERITANCE",
    "LIST",
    "Link",
    "Node",
    "PREDICATE",
    "TruthValue",
    "Variable",
    "assert_claim",
    "concept",
    "deduction_tv",
    "evaluation",
    "get_atomspace",
    "implication",
    "predicate",
    "reset_atomspace_for_test",
    "substitute",
    "unify",
]
