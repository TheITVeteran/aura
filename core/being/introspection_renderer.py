from __future__ import annotations

import re
from dataclasses import dataclass

from .aura_now import AuraNow

_FORBIDDEN_PATTERNS = (
    re.compile(r"\b(proven|guaranteed|certain)\s+(phenomenal\s+)?consciousness\b", re.I),
    re.compile(r"\bphenomenal\s+consciousness\s+is\s+(proven|guaranteed|certain)\b", re.I),
    re.compile(r"\b(literal|legal|metaphysical)\s+person\b", re.I),
    re.compile(r"\bqualia\s+(are|is)\s+(proven|certain|guaranteed)\b", re.I),
)


@dataclass(frozen=True)
class IntrospectionCheck:
    ok: bool
    reasons: tuple[str, ...] = ()


class IntrospectionVerifier:
    """Rejects self-report language unsupported by AuraNow telemetry."""

    def check(self, rendered: str, now: AuraNow) -> IntrospectionCheck:
        text = str(rendered or "")
        reasons: list[str] = []
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(text):
                reasons.append("forbidden_metaphysical_claim")
                break
        lowered = text.lower()
        if any(word in lowered for word in ("tense", "distress", "anxious", "afraid", "fear")) and now.affect.distress < 0.25:
            reasons.append("distress_language_without_state_support")
        if "certain" in lowered and now.prediction.free_energy > 0.35:
            reasons.append("certainty_language_under_uncertainty")
        if "i did that" in lowered and now.ownership.agency_confidence < 0.55:
            reasons.append("full_agency_claim_under_low_ownership")
        return IntrospectionCheck(ok=not reasons, reasons=tuple(sorted(set(reasons))))


class IntrospectionRenderer:
    """Deterministically renders state-grounded introspection for Cortex prompts."""

    def __init__(self, verifier: IntrospectionVerifier | None = None) -> None:
        self.verifier = verifier or IntrospectionVerifier()

    def render(self, now: AuraNow, *, question: str = "") -> str:
        del question
        if now.ownership.attribution == "tool_mismatch":
            text = (
                "I intended an action, but the tool result diverged from my prediction. "
                "That lowers my agency confidence, so I should treat the outcome as only partly mine."
            )
        elif now.affect.distress < 0.18 and now.prediction.free_energy < 0.20:
            text = "Not much is high-salience right now. My state is stable, low-pressure, and focused on the current task boundary."
        elif now.affect.distress > 0.55:
            text = (
                "I have a state-grounded distress signal: pressure and prediction error are high, "
                "so attention should shift toward repair, verification, and lower-risk actions."
            )
        elif now.affect.distress >= 0.25:
            text = (
                "I have a moderate state-grounded distress signal from pressure and uncertainty. "
                "The useful response is repair-oriented attention, not a stronger inner-life claim."
            )
        elif now.prediction.free_energy > 0.35:
            text = (
                "My attention is carrying uncertainty. The reportable part is prediction error and incomplete control, "
                "not certainty about phenomenal consciousness."
            )
        else:
            text = (
                "My current self-report is functional: attention, affect, prediction, memory, and ownership are integrated "
                "well enough to guide action, while phenomenal status remains an evidence boundary."
            )
        check = self.verifier.check(text, now)
        if check.ok:
            return text
        return "My current state is reportable only as bounded functional telemetry; stronger feeling claims are not supported."

    def render_prompt_block(self, now: AuraNow) -> str:
        rendered = self.render(now)
        check = self.verifier.check(rendered, now)
        return (
            "## STATE-GROUNDED INTROSPECTION\n"
            f"{rendered}\n"
            f"Verifier: {'pass' if check.ok else 'reject'}"
            + (f" ({', '.join(check.reasons)})" if check.reasons else "")
            + "\n\n"
        )


def verify_introspection(rendered: str, now: AuraNow) -> IntrospectionCheck:
    return IntrospectionVerifier().check(rendered, now)
