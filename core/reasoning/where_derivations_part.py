"""Where two derivations of one question part, and the check that settles that point.

A wrong answer to an open question usually turns on one point: a fact
misremembered, a number slipped, a case missed, a word read two ways. Two
derivations of the same question made independently (her greedy one and one
sampled) either agree, and the answer stands, or they part. Where they part,
her model names the crux: what each asserts on the point that decides between
them, and one question whose answer settles it.

The crux is then settled by the strongest check that applies:

* a quantity, by a program in the sandbox computing the settling question
  (core/brain/executable_reasoning.py), matched to the side whose number it is;
* a search told as a story, by the whole question written as constraints and
  solved by her engine (core/reasoning/written_as_constraints.py): a unique
  solution that is one derivation's answer settles the point for that side;
* anything else, first by her corpus (core/knowledge/local_corpus.py): the
  passages found for the settling question are read for which side they
  support;
* failing both, by two derivations of the settling question alone, a smaller
  question than the whole.

What settled it, and how, is written as runtime evidence for her final
derivation of the whole question, the channel a kept procedure's answer uses.
A crux nothing settles leaves her final derivation without evidence, and the
greedy answer stands.

The stages are pure functions, so a benchmark harness can batch each stage
across many questions (tools/run_general_reasoning.py) and her runtime can run
them one question at a time (``settle``). Nothing here knows a benchmark: the
caller says when two answers are the same and supplies the engines.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

__all__ = [
    "Crux", "Derivation", "Settled", "crux_request", "evidence_text", "kind_of", "parse_crux", "parse_reading",
    "reading_request", "settle", "side_of_value",
]

SIDES = ("first", "second")
_LINE = re.compile(r"^\s*(FIRST|SECOND|QUESTION)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?(?![\w.])")


@dataclass
class Derivation:
    """One derivation of a question: its normalized answer ("" when it gave none), its reply, its tokens."""

    answer: str
    reply: str
    tokens: int = 0


@dataclass
class Crux:
    """What each derivation asserts on the deciding point, the question that settles it, and how it was settled."""

    first: str
    second: str
    question: str
    kind: str = ""
    settled: str = ""
    how: str = "unsettled"
    detail: str = ""

    @property
    def held(self) -> str:
        """The claim that held, or "" when nothing settled the crux."""
        return {"first": self.first, "second": self.second}.get(self.settled, "")


@dataclass
class Settled:
    answer: str
    agreed: bool
    derivations: list[Derivation]
    crux: Crux | None = None
    final: Derivation | None = None
    stages: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(d.tokens for d in self.derivations) + (self.final.tokens if self.final else 0)


def crux_request(question: str, first_reply: str, second_reply: str) -> str:
    """What her model is asked when two derivations disagree."""
    return (
        "Two answers to the same question disagree.\n\n"
        f"Question:\n{question.strip()}\n\nFirst answer:\n{first_reply.strip()}\n\n"
        f"Second answer:\n{second_reply.strip()}\n\n"
        "Name the one point they disagree on that decides between them. Reply with three lines:\n"
        "FIRST: what the first answer asserts on that point\n"
        "SECOND: what the second answer asserts on that point\n"
        "QUESTION: one question, complete in itself, whose answer settles that point"
    )


def parse_crux(text: str) -> Crux | None:
    """The crux her model named, or None when the reply does not name both sides and a question."""
    found = {label.lower(): value for label, value in _LINE.findall(str(text or ""))}
    if not all(found.get(k) for k in ("first", "second", "question")):
        return None
    if " ".join(found["first"].split()).casefold() == " ".join(found["second"].split()).casefold():
        return None
    return Crux(first=found["first"], second=found["second"], question=found["question"])


def _numbers(text: str) -> list[Fraction]:
    out = []
    for token in _NUMBER.findall(str(text or "")):
        try:
            out.append(Fraction(token.replace(",", "")))
        except (ValueError, ZeroDivisionError):
            continue
    return out


def kind_of(crux: Crux) -> str:
    """"quantity" when the two sides state different numbers, else "claim"."""
    first, second = set(_numbers(crux.first)), set(_numbers(crux.second))
    return "quantity" if first and second and first != second else "claim"


def side_of_value(value: str, crux: Crux) -> str | None:
    """The side whose numbers include the computed value and the other's do not; None otherwise."""
    computed = _numbers(value)
    if len(computed) != 1:
        return None
    holds = [side for side, claim in zip(SIDES, (crux.first, crux.second), strict=True)
             if computed[0] in set(_numbers(claim))]
    return holds[0] if len(holds) == 1 else None


def reading_request(passages: Sequence[str], crux: Crux) -> str:
    """What her model is asked about passages found for the settling question."""
    shown = "\n\n".join(f"[{i + 1}] {p.strip()}" for i, p in enumerate(passages))
    return (
        f"Passages:\n{shown}\n\nWhich of these do the passages support?\n"
        f"FIRST: {crux.first}\nSECOND: {crux.second}\n"
        "Reply with one word: FIRST, SECOND or NEITHER."
    )


def parse_reading(text: str) -> str | None:
    """"first" or "second" when the reply names exactly one; None for NEITHER or anything else."""
    words = {w.lower() for w in re.findall(r"\b(FIRST|SECOND|NEITHER)\b", str(text or ""), re.IGNORECASE)}
    picked = words & set(SIDES)
    return picked.pop() if len(picked) == 1 and "neither" not in words else None


def evidence_text(crux: Crux) -> str:
    """What her final derivation is given: the claim that held and how it was settled; "" when none held."""
    if not crux.held:
        return ""
    return (f"Checked before answering ({crux.how}): {crux.held}\n"
            f"The point was: {crux.question}\n{crux.detail}".rstrip())


Derive = Callable[[str, str, bool], Awaitable[Derivation]]
Ask = Callable[[str], Awaitable[str]]
Compute = Callable[[str], Awaitable[str | None]]
Lookup = Callable[[str], Awaitable[list[str]]]
Same = Callable[[str, str], bool]
Formalize = Callable[[str], Awaitable[str | None]]


async def _no_formalization(question: str) -> str | None:
    return None


async def settle(question: str, *, derive: Derive, ask: Ask, compute: Compute, lookup: Lookup,
                 same: Same, formalize: Formalize = _no_formalization) -> Settled:
    """One question, one stage at a time: derive twice; on disagreement name the crux, settle it, derive again.

    ``derive(question, evidence, sampled)`` is her derivation; ``ask`` a short
    closed-channel request; ``compute`` the sandbox's value for a question;
    ``lookup`` her corpus's passages; ``same`` whether two answers agree;
    ``formalize`` the engine's unique answer to the whole question written as
    constraints, in the derivations' answer form, or None.
    """
    greedy = await derive(question, "", False)
    sampled = await derive(question, "", True)
    result = Settled(answer=greedy.answer, agreed=False, derivations=[greedy, sampled], stages=["derived_twice"])
    if greedy.answer and same(greedy.answer, sampled.answer):
        result.agreed = True
        return result
    crux = parse_crux(await ask(crux_request(question, greedy.reply, sampled.reply)))
    result.stages.append("crux_named" if crux else "no_crux")
    if crux is None:
        return result
    result.crux = crux
    crux.kind = kind_of(crux)
    if crux.kind == "quantity":
        value = await compute(crux.question)
        side = side_of_value(value or "", crux)
        if side:
            crux.settled, crux.how, crux.detail = side, "computed in the sandbox", f"The computed value: {value}"
    if not crux.settled:
        solved = await formalize(question)
        side = ("first" if solved and same(solved, greedy.answer)
                else "second" if solved and same(solved, sampled.answer) else None)
        if side:
            crux.settled, crux.how = side, "solved as constraints by her engine"
            crux.detail = f"Written as constraints, the problem has exactly one solution, whose answer is {solved}."
    if not crux.settled:
        passages = await lookup(crux.question)
        if passages:
            side = parse_reading(await ask(reading_request(passages, crux)))
            if side:
                crux.settled, crux.how = side, "found in her corpus"
                crux.detail = "From her corpus: " + " … ".join(p.strip() for p in passages)
    if not crux.settled:
        alone = [await derive(crux.question, "", False), await derive(crux.question, "", True)]
        result.derivations.extend(alone)
        # The settling question is open, so the two are compared by the side each supports.
        sides = [parse_reading(await ask(reading_request([d.reply], crux))) for d in alone]
        if sides[0] and sides[0] == sides[1]:
            crux.settled, crux.how = sides[0], "derived again on the point alone, twice in agreement"
    result.stages.append(f"settled:{crux.how}" if crux.settled else "unsettled")
    if crux.settled:
        result.final = await derive(question, evidence_text(crux), False)
        result.answer = result.final.answer or greedy.answer
    return result
