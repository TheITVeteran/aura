"""Live deductive self-audit — catch non-sequiturs in natural-language reasoning.

This is the prover acting on *active thought* rather than stored beliefs. It pulls
explicit deductive structure out of text ("X, therefore Y" / "since X, Y"),
formalizes it through the same propositional encoder used for beliefs, and asks the
natural-deduction prover whether the conclusion actually follows. When it does not —
a fallacy like *affirming the consequent* — that is surfaced as a non-sequitur.

Correctness over coverage. The audit is **conservative**: it only renders a verdict
when the conclusion shares propositional structure with the premises (so the link is
actually formalizable). If the reasoning can't be captured in propositional logic it
returns ``UNDECIDABLE`` and stays silent — it never flags valid or unformalizable
reasoning as wrong. This makes it safe to run on Aura's own draft replies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from core.reasoning.belief_consistency import encode_belief
from core.reasoning.natural_deduction import Implies, atoms, entails, is_consistent

Status = Literal["valid", "invalid", "undecidable"]

# Conclusion markers: premises precede, conclusion follows.
_CONCL_MARKERS = [
    "it follows that", "which means that", "which means", "as a result",
    "therefore", "consequently", "hence", "thus", "ergo", "so",
]
# Premise-first markers: "since/because <premise>, <conclusion>".
_PREMISE_FIRST_RE = re.compile(
    r"^\s*(?:since|because|given that)\b(?P<prem>.+?),(?P<concl>.+)$",
    re.IGNORECASE,
)
_CONCL_RE = re.compile(
    r"^(?P<prem>.+?)\b(?P<marker>" + "|".join(re.escape(m) for m in _CONCL_MARKERS) + r")\b(?P<concl>.+)$",
    re.IGNORECASE,
)
_PREMISE_SPLIT_RE = re.compile(r"\s*(?:;|,\s*and\b|\band\b|,)\s*", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Inference:
    premises: tuple[str, ...]
    conclusion: str
    marker: str
    raw: str


@dataclass
class InferenceVerdict:
    status: Status
    inference: Inference
    countermodel: dict[str, bool] | None = None
    explanation: str = ""

    @property
    def is_non_sequitur(self) -> bool:
        return self.status == "invalid"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "premises": list(self.inference.premises),
            "conclusion": self.inference.conclusion,
            "marker": self.inference.marker,
            "countermodel": self.countermodel,
            "explanation": self.explanation,
        }


def _split_premises(text: str) -> list[str]:
    parts = [p.strip(" .,;") for p in _PREMISE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def extract_inferences(text: str) -> list[Inference]:
    """Pull deductive (premises → conclusion) structures out of free text."""
    out: list[Inference] = []
    for sentence in _SENTENCE_SPLIT_RE.split(str(text or "").strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        pm = _PREMISE_FIRST_RE.match(sentence)
        if pm and pm.group("prem").strip() and pm.group("concl").strip():
            out.append(Inference(
                premises=tuple(_split_premises(pm.group("prem"))),
                conclusion=pm.group("concl").strip(" .,;"),
                marker="since/because",
                raw=sentence,
            ))
            continue
        cm = _CONCL_RE.match(sentence)
        if cm and cm.group("prem").strip() and cm.group("concl").strip():
            out.append(Inference(
                premises=tuple(_split_premises(cm.group("prem"))),
                conclusion=cm.group("concl").strip(" .,;"),
                marker=cm.group("marker").lower(),
                raw=sentence,
            ))
    return out


def audit_inference(inf: Inference) -> InferenceVerdict:
    """Verdict for one inference, guarded so unformalizable reasoning stays silent."""
    try:
        prem_f = [encode_belief(p).formula for p in inf.premises if p.strip()]
        concl_f = encode_belief(inf.conclusion).formula
    except (ValueError, TypeError, AttributeError):
        return InferenceVerdict("undecidable", inf, explanation="could not formalize")
    if not prem_f:
        return InferenceVerdict("undecidable", inf, explanation="no premises")

    prem_atoms: set[str] = set()
    for f in prem_f:
        prem_atoms |= atoms(f)
    concl_atoms = atoms(concl_f)
    # Only judge when the conclusion is actually about something the premises
    # constrain — otherwise the propositional encoding can't capture the link.
    if not (concl_atoms & prem_atoms):
        return InferenceVerdict("undecidable", inf, explanation="no shared propositional structure")
    if not is_consistent(prem_f):
        return InferenceVerdict("undecidable", inf, explanation="premises inconsistent (ex falso)")

    if entails(prem_f, concl_f):
        return InferenceVerdict("valid", inf, explanation="conclusion follows from the premises")

    from core.reasoning.natural_deduction import prove

    proof = prove(prem_f, concl_f)
    return InferenceVerdict(
        "invalid",
        inf,
        countermodel=proof.countermodel,
        explanation="conclusion does not follow — the premises can hold while it is false",
    )


def audit_text(text: str) -> list[InferenceVerdict]:
    """Audit every inference found in the text."""
    return [audit_inference(inf) for inf in extract_inferences(text)]


def find_non_sequiturs(text: str) -> list[InferenceVerdict]:
    """Only the confidently-invalid inferences (formalizable non-sequiturs)."""
    return [v for v in audit_text(text) if v.is_non_sequitur]


def verify(premises: list[str], conclusion: str) -> InferenceVerdict:
    """NL-friendly direct check: does ``conclusion`` follow from ``premises``?"""
    return audit_inference(Inference(tuple(premises), conclusion, "verify", " ".join(premises)))
