"""Courtroom — adversarial multi-role reasoning over a single local model.

You make a smaller model act smarter than itself by forcing *adversarial* cognition:
the same Cortex plays isolated roles that do not see each other's drafts until after
they have each spoken independently, then a judge rules on the assembled record.

    Solver          produces candidate answer(s) with its reasoning
    Skeptic         independently lists how answers like this usually go wrong,
                    then attacks the Solver's actual answer
    Evidence clerk  assembles the grounding (supplied evidence + retrieved facts)
    Verifier        runs the deterministic truth engines (not an LLM opinion)
    Simplifier      reduces the winning answer to its minimal correct form
    Judge           rules on candidates + objections + verifier verdict + evidence

This is slower but local, and it is gated behind the amplifier's DEEP/EXTREME modes
so it only runs when the stakes justify the spend.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Courtroom")

GenerateFn = Callable[[str, float], Awaitable[str]]


@dataclass
class CourtroomVerdict:
    answer: str
    confidence: float
    winning_role: str = "judge"
    candidates: list[str] = field(default_factory=list)
    objections: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    verifier_ok: bool = True
    verifier_issues: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer[:240],
            "confidence": round(self.confidence, 3),
            "winning_role": self.winning_role,
            "n_candidates": len(self.candidates),
            "n_objections": len(self.objections),
            "verifier_ok": self.verifier_ok,
            "verifier_issues": self.verifier_issues[:6],
            "unresolved": self.unresolved[:6],
        }


_SOLVER_SYS = (
    "You are the Solver. Answer the question as correctly as you can. Show the key "
    "reasoning steps, then end with a line beginning 'Answer:'. Be concrete."
)
_SKEPTIC_PRIOR_SYS = (
    "You are the Skeptic. WITHOUT seeing any proposed answer, list the 2-4 most common "
    "ways an answer to this question goes wrong, and what evidence would distinguish a "
    "right answer from a plausible-sounding wrong one. Be specific."
)
_SKEPTIC_ATTACK_SYS = (
    "You are the Skeptic. Find concrete flaws, unstated assumptions, or missing evidence "
    "in the proposed answer. If it is actually sound, say so plainly. Do not invent flaws."
)
_CLERK_SYS = (
    "You are the Evidence Clerk. From the supplied material only, list the facts that "
    "bear on this question as short bullet points. If the material does not settle it, "
    "say which fact is missing. Do not speculate beyond the material."
)
_SIMPLIFIER_SYS = (
    "You are the Simplifier. Restate the chosen answer in its shortest correct form, "
    "preserving every load-bearing qualifier. Do not add new claims."
)
_JUDGE_SYS = (
    "You are the Judge. You are given candidate answers, the Skeptic's objections, an "
    "evidence summary, and a mechanical verifier verdict. Decide the single best answer. "
    "Honor the verifier: if it found a provable error, do not certify that candidate. "
    "If uncertainty remains, state it honestly. End with a line beginning 'Verdict:'."
)


def _extract_answer(text: str) -> str:
    for marker in ("Verdict:", "Answer:"):
        idx = text.rfind(marker)
        if idx >= 0:
            return text[idx + len(marker):].strip().splitlines()[0].strip() or text.strip()
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else str(text or "").strip()


class Courtroom:
    """Run the adversarial role pipeline for one question."""

    def __init__(self, generate: GenerateFn, *, verifier: Any | None = None) -> None:
        self._generate = generate
        self._verifier = verifier  # VerifierRegistry-like; optional

    async def _ask(self, system: str, user: str, temperature: float) -> str:
        try:
            prompt = f"[ROLE]\n{system}\n\n[TASK]\n{user}"
            out = await self._generate(prompt, temperature)
            return str(out or "").strip()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("courtroom_role", exc)
            return ""

    async def deliberate(
        self,
        question: str,
        *,
        evidence: list[str] | None = None,
        task_type: str | None = None,
        n_candidates: int = 2,
    ) -> CourtroomVerdict:
        evidence = evidence or []
        ev_blob = "\n".join(f"- {e}" for e in evidence) if evidence else "(no external material supplied)"

        # Phase 1 — independent generation (roles do not see each other yet).
        solver_task = [
            self._ask(_SOLVER_SYS, question, 0.3 + 0.2 * i) for i in range(max(1, n_candidates))
        ]
        skeptic_prior_task = self._ask(_SKEPTIC_PRIOR_SYS, question, 0.6)
        clerk_task = self._ask(_CLERK_SYS, f"Question:\n{question}\n\nMaterial:\n{ev_blob}", 0.2)
        gathered = await asyncio.gather(*solver_task, skeptic_prior_task, clerk_task)
        candidates = [c for c in gathered[: len(solver_task)] if c]
        skeptic_prior = gathered[-2]
        clerk = gathered[-1]
        if not candidates:
            return CourtroomVerdict(answer="", confidence=0.0, winning_role="solver", unresolved=["solver produced nothing"])

        primary = candidates[0]

        # Phase 2 — adversarial + mechanical review of the primary candidate.
        attack_prompt = f"Question:\n{question}\n\nProposed answer:\n{primary}"
        skeptic_attack_task = self._ask(_SKEPTIC_ATTACK_SYS, attack_prompt, 0.5)
        verify_task = self._verify(primary, task_type, evidence)
        skeptic_attack, verdict = await asyncio.gather(skeptic_attack_task, verify_task)

        objections = [o for o in (skeptic_prior, skeptic_attack) if o]
        verifier_ok = bool(getattr(verdict, "ok", True))
        verifier_issues = list(getattr(verdict, "issues", []) or [])

        # Phase 3 — judge rules on the full record.
        dossier = (
            f"Question:\n{question}\n\n"
            + "\n\n".join(f"Candidate {i+1}:\n{c}" for i, c in enumerate(candidates))
            + f"\n\nSkeptic objections:\n{chr(10).join(objections) or '(none)'}"
            + f"\n\nEvidence summary:\n{clerk or '(none)'}"
            + f"\n\nMechanical verifier verdict: {'PASS' if verifier_ok else 'FAIL'}"
            + (f" — issues: {'; '.join(verifier_issues[:4])}" if verifier_issues else "")
        )
        judgment = await self._ask(_JUDGE_SYS, dossier, 0.2)
        if not judgment:
            judgment = primary
        final = _extract_answer(judgment)

        # Phase 4 — minimalise (optional, cheap).
        simplified = await self._ask(_SIMPLIFIER_SYS, f"Question:\n{question}\n\nChosen answer:\n{final}", 0.2)
        answer = _extract_answer(simplified) if simplified else final

        confidence = self._score(candidates, objections, verifier_ok, bool(evidence))
        unresolved = self._unresolved(objections, verifier_issues, verifier_ok)
        logger.info(
            "⚖️ [Courtroom] %d candidates, %d objections, verifier=%s → conf %.2f",
            len(candidates), len(objections), "PASS" if verifier_ok else "FAIL", confidence,
        )
        return CourtroomVerdict(
            answer=answer or final,
            confidence=confidence,
            winning_role="judge",
            candidates=candidates,
            objections=objections,
            evidence=[clerk] if clerk else [],
            verifier_ok=verifier_ok,
            verifier_issues=verifier_issues,
            unresolved=unresolved,
        )

    async def _verify(self, candidate: str, task_type: str | None, evidence: list[str]) -> Any:
        if self._verifier is None:
            return None
        try:
            return await self._verifier.verify(
                candidate, task_type=task_type, context={"evidence": evidence}
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("courtroom_verify", exc)
            return None

    @staticmethod
    def _score(candidates: list[str], objections: list[str], verifier_ok: bool, has_evidence: bool) -> float:
        base = 0.55
        if verifier_ok:
            base += 0.2
        else:
            base -= 0.25
        if has_evidence:
            base += 0.1
        # Strong objections that were not resolved erode confidence a little.
        substantive = sum(1 for o in objections if len(o) > 40 and "sound" not in o.lower())
        base -= 0.04 * min(3, substantive)
        return round(max(0.05, min(0.97, base)), 4)

    @staticmethod
    def _unresolved(objections: list[str], verifier_issues: list[str], verifier_ok: bool) -> list[str]:
        out = list(verifier_issues[:3])
        if not verifier_ok:
            out.append("verifier flagged a provable issue in the leading candidate")
        return out
