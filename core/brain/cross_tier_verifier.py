"""Cross-tier verification — weak generator, strong verifier.

A real frontier technique: it is far cheaper to *verify* an answer than to produce one,
so let the bigger model (Aura's 72B Solver) check and, if needed, correct the answer the
cheaper 32B Cortex deliberated to. Verification catches the subtle factual/semantic
mistakes the symbolic deduction engine can't (it only proves logic/arithmetic), so the
two verifiers are complementary: the prover for formal validity, the strong tier for
everything else.

``strong_generate`` is injected (the 72B call), so this is deterministically testable and
wires to the live Solver tier in production. Because the MLX hot-swap makes loading the
72B expensive, this is meant for the hardest / highest-stakes turns — adaptive compute,
not every reply.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CrossTier")

StrongGenerate = Callable[[str], Awaitable[str]]

_VERDICT_RE = re.compile(r"verdict\s*[:\-]\s*(correct|incorrect)", re.IGNORECASE)
_CORRECTED_RE = re.compile(r"corrected\s*[:\-]\s*(.+)", re.IGNORECASE)


@dataclass
class CrossTierVerdict:
    ok: bool
    answer: str
    corrected: bool = False
    critique: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "answer": self.answer[:200], "corrected": self.corrected}


class CrossTierVerifier:
    """Use a stronger model tier to verify/correct a cheaper tier's answer."""

    def __init__(self, strong_generate: StrongGenerate | None = None) -> None:
        self._strong = strong_generate

    async def _generate(self, prompt: str) -> str:
        if self._strong is not None:
            return await self._strong(prompt)
        # Production default: route to the Solver (72B) tier via the inference gate.
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate is None or not hasattr(gate, "generate_response"):
                return ""
            return await gate.generate_response(
                prompt, tier="solver", origin="cross_tier_verify", max_tokens=400
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("cross_tier_verifier", exc)
            return ""

    async def verify(self, question: str, answer: str, *, reasoning: str = "") -> CrossTierVerdict:
        """Have the strong tier judge ``answer``; return its verdict and any correction."""
        prompt = (
            "You are a careful expert verifier. Judge whether the proposed answer is "
            "correct for the question. Be strict.\n\n"
            f"Question: {question}\n"
            f"Proposed answer: {answer}\n"
            + (f"Reasoning: {reasoning}\n" if reasoning else "")
            + "\nRespond in exactly this form:\n"
            "VERDICT: CORRECT or INCORRECT\n"
            "If INCORRECT, then a second line: CORRECTED: <the correct answer>"
        )
        try:
            resp = await self._generate(prompt)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("cross_tier_verifier", exc)
            resp = ""
        if not resp:
            # Strong tier unavailable → don't block; keep the original answer unverified.
            return CrossTierVerdict(ok=True, answer=answer, corrected=False, critique="strong tier unavailable")

        m = _VERDICT_RE.search(resp)
        verdict_correct = bool(m and m.group(1).lower() == "correct")
        if verdict_correct:
            return CrossTierVerdict(ok=True, answer=answer, corrected=False, critique="verified by strong tier")

        cm = _CORRECTED_RE.search(resp)
        corrected_answer = cm.group(1).strip() if cm else ""
        if corrected_answer:
            logger.info("🔬 [CrossTier] strong tier corrected the answer.")
            return CrossTierVerdict(ok=True, answer=corrected_answer, corrected=True, critique=resp[:300])
        # Flagged incorrect but no correction offered → surface the doubt.
        return CrossTierVerdict(ok=False, answer=answer, corrected=False, critique=resp[:300])


_instance: CrossTierVerifier | None = None


def get_cross_tier_verifier() -> CrossTierVerifier:
    global _instance
    if _instance is None:
        _instance = CrossTierVerifier()
    return _instance
