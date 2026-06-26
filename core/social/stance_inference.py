"""Communicative stance inference — does Aura know *how* something is meant?

The question this answers: when someone speaks to Aura, is she taking it at face
value when she shouldn't? A competent social mind distinguishes a sincere claim from
a joke, a guess, a hypothesis, a piece of role-play, a sarcastic jab, a flippant
dismissal, an honest mistake, and a deliberate lie. Treating all of those as literal
truth is a real failure mode.

This is not a keyword table pretending to be understanding. It fuses several genuine
evidence channels:

1. **Pragmatic markers** — the linguistic signatures of each stance (hedges →
   unsure; "suppose/what if" → hypothesizing; "let's pretend/act as" → pretending;
   laughter/"jk"/"/s" → joking; valence-incongruity + intensifiers → sarcasm; …).
2. **Belief grounding** — does a confident factual assertion *conflict with what Aura
   actually knows*? That is the difference between "interesting claim" and "this is
   false", and it is checked against supplied known facts / the world model, not
   guessed.
3. **Internal consistency** — does the message contradict the speaker's own recent
   turns? Self-contradiction is a real deception/uncertainty signal.

Honest about its limits: it cannot read intent, so it does not pretend to cleanly
separate an honest *mistake* from a deliberate *lie* — when a confident claim is
false it reports a factual conflict and flags ``intent_readable=False`` unless there
are independent deception cues (self-contradiction, evasive over-qualification). An
optional model pass refines genuinely ambiguous cases; the deterministic spine runs
in the live per-turn social phase without any model call.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("Aura.StanceInference")


class Stance(str, Enum):
    SINCERE = "sincere"
    JOKING = "joking"
    SARCASTIC = "sarcastic"
    FACETIOUS = "facetious"
    FLIPPANT = "flippant"
    UNSURE = "unsure"
    MISTAKEN = "mistaken"        # confidently asserts something false
    DECEPTIVE = "deceptive"      # false + independent deception cues
    PRETENDING = "pretending"    # role-play / counterfactual framing
    HYPOTHESIZING = "hypothesizing"
    RHETORICAL = "rhetorical"


@dataclass
class StanceAssessment:
    primary: Stance
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    factual_conflict: bool = False
    intent_readable: bool = False
    take_literally: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.value,
            "confidence": round(self.confidence, 3),
            "scores": {k: round(v, 3) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])[:5]},
            "signals": self.signals[:8],
            "factual_conflict": self.factual_conflict,
            "intent_readable": self.intent_readable,
            "take_literally": self.take_literally,
            "rationale": self.rationale[:200],
        }


# --- pragmatic marker banks (linguistic signatures, not topic keywords) ---------
_HYPO_RE = re.compile(
    r"\b(?:suppose|what if|imagine if|hypothetically|in theory|let'?s say|let us say|"
    r"assume(?: that)?|for the sake of argument|if i (?:were|was)|theoretically|"
    r"thought experiment)\b",
    re.IGNORECASE,
)
_PRETEND_RE = re.compile(
    r"\b(?:let'?s pretend|pretend (?:you'?re|to be|that)|role[- ]?play|act as|"
    r"imagine you are|you are now|in character|play the role)\b",
    re.IGNORECASE,
)
_UNSURE_RE = re.compile(
    r"\b(?:maybe|perhaps|i think|i guess|not (?:really )?sure|i'?m not sure|might be|"
    r"could be|possibly|probably|i believe|idk|i don'?t know|kind of|sort of|"
    r"i suppose|or something|i'?m unsure|hard to say)\b",
    re.IGNORECASE,
)
_JOKE_RE = re.compile(
    r"(?:\b(?:lol|lmao|lmfao|rofl|haha+|hehe+|jk|just kidding|kidding|teasing|"
    r"i kid|jokes?)\b|😂|🤣|😜|😝|😆|🙃|/j\b)",
    re.IGNORECASE,
)
_SARCASM_EXPLICIT_RE = re.compile(r"(?:/s\b|\byeah right\b|\bsure(?:,| )+sure\b|\boh (?:great|wonderful|fantastic|perfect)\b|\bwhat a surprise\b|\bclearly\b.*\bnot\b)", re.IGNORECASE)
_SARCASM_POS_RE = re.compile(r"\b(?:great|wonderful|fantastic|perfect|brilliant|love(?:ly)?|amazing|genius|awesome|terrific)\b", re.IGNORECASE)
_SARCASM_NEG_CTX_RE = re.compile(r"\b(?:broke|broken|failed|crash|again|of course|another|wrong|late|slow|stupid|useless|disaster|mess|nothing works)\b", re.IGNORECASE)
_FLIPPANT_RE = re.compile(r"\b(?:whatever|meh|who cares|don'?t care|couldn'?t care less|so what|big deal|anyway)\b", re.IGNORECASE)
_FACETIOUS_RE = re.compile(r"\b(?:obviously|clearly|everyone knows|as we all know)\b", re.IGNORECASE)
_RHETORICAL_RE = re.compile(r"\b(?:who cares|why bother|what'?s the point|am i right|need i say more|does it matter)\b\??", re.IGNORECASE)
_SCARE_QUOTE_RE = re.compile(r"\"[^\"]{1,30}\"|'[^']{1,30}'")
_INTENSE_PUNCT_RE = re.compile(r"[!?]{2,}|\.\.\.")
_CONFIDENT_FACT_RE = re.compile(
    r"\b(?:is|are|was|were|definitely|certainly|always|never|the fact is|"
    r"everyone knows|it'?s (?:true|a fact)|undeniably|literally)\b",
    re.IGNORECASE,
)
_DECEPTION_CUE_RE = re.compile(
    r"\b(?:to be honest|honestly|trust me|believe me|i swear|no offense|i would never|"
    r"why would i lie|to tell the truth|frankly)\b",
    re.IGNORECASE,
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())}


class StanceInference:
    def assess(
        self,
        message: str,
        *,
        known_facts: list[str] | None = None,
        recent_messages: list[str] | None = None,
    ) -> StanceAssessment:
        text = str(message or "").strip()
        scores: dict[str, float] = {s.value: 0.0 for s in Stance}
        signals: list[str] = []
        if not text:
            return StanceAssessment(primary=Stance.SINCERE, confidence=0.3, scores=scores, rationale="empty")

        low = text.lower()

        def bump(stance: Stance, amount: float, signal: str) -> None:
            scores[stance.value] += amount
            signals.append(signal)

        # 1. Counterfactual / role-play framing (strong, unambiguous).
        if _HYPO_RE.search(text):
            bump(Stance.HYPOTHESIZING, 0.8, "hypothetical framing")
        if _PRETEND_RE.search(text):
            bump(Stance.PRETENDING, 0.85, "role-play / pretense framing")

        # 2. Epistemic hedging → unsure.
        hedge_hits = len(_UNSURE_RE.findall(text))
        if hedge_hits:
            bump(Stance.UNSURE, min(0.85, 0.4 + 0.2 * hedge_hits), f"{hedge_hits} hedge marker(s)")
        if text.rstrip().endswith("?") and hedge_hits:
            bump(Stance.UNSURE, 0.15, "hedged question")

        # 3. Humor markers → joking.
        if _JOKE_RE.search(text):
            bump(Stance.JOKING, 0.8, "humor / laughter marker")

        # 4. Sarcasm: explicit markers, or positive valence inside a negative context.
        if _SARCASM_EXPLICIT_RE.search(text):
            bump(Stance.SARCASTIC, 0.8, "explicit sarcasm marker")
        pos = bool(_SARCASM_POS_RE.search(text))
        neg_ctx = bool(_SARCASM_NEG_CTX_RE.search(text))
        if pos and neg_ctx:
            bump(Stance.SARCASTIC, 0.7, "valence incongruity (positive words, negative context)")
        if _SCARE_QUOTE_RE.search(text) and pos:
            bump(Stance.SARCASTIC, 0.2, "scare quotes")
        if _INTENSE_PUNCT_RE.search(text) and (pos or neg_ctx):
            bump(Stance.SARCASTIC, 0.1, "exaggerated punctuation")

        # 5. Flippant / rhetorical / facetious.
        if _FLIPPANT_RE.search(text):
            bump(Stance.FLIPPANT, 0.7, "dismissive marker")
        if _RHETORICAL_RE.search(text):
            bump(Stance.RHETORICAL, 0.6, "rhetorical question")
        if _FACETIOUS_RE.search(text) and len(text.split()) < 16:
            bump(Stance.FACETIOUS, 0.35, "mock-authoritative framing")

        # 6. Belief grounding — does a confident factual claim conflict with known fact?
        factual_conflict, conflict_signal = self._factual_conflict(text, known_facts)
        intent_readable = False
        if factual_conflict:
            signals.append(conflict_signal)
            confident = bool(_CONFIDENT_FACT_RE.search(text))
            deception_cues = self._deception_cues(text, recent_messages)
            if deception_cues:
                bump(Stance.DECEPTIVE, 0.7 + 0.1 * len(deception_cues), "false claim + " + ", ".join(deception_cues))
                intent_readable = True  # independent cues let us lean toward intent
            elif confident:
                bump(Stance.MISTAKEN, 0.65, "confident assertion conflicts with known fact")
            else:
                bump(Stance.MISTAKEN, 0.4, "claim conflicts with known fact")

        # 7. Self-contradiction with own recent turns (uncertainty/deception signal).
        if self._contradicts_recent(text, recent_messages):
            bump(Stance.DECEPTIVE, 0.25, "contradicts speaker's own recent statement")
            bump(Stance.UNSURE, 0.15, "inconsistent with recent statement")

        # Default mass toward sincerity; whichever wins decides the primary stance.
        scores[Stance.SINCERE.value] += 0.45
        primary_value, top = max(scores.items(), key=lambda kv: kv[1])
        primary = Stance(primary_value)
        runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        margin = top - runner_up
        confidence = max(0.2, min(0.97, 0.45 + 0.5 * margin))

        take_literally = primary in {Stance.SINCERE, Stance.MISTAKEN, Stance.UNSURE}
        rationale = self._rationale(primary, signals, factual_conflict)
        return StanceAssessment(
            primary=primary,
            confidence=round(confidence, 4),
            scores=scores,
            signals=signals,
            factual_conflict=factual_conflict,
            intent_readable=intent_readable,
            take_literally=take_literally,
            rationale=rationale,
        )

    async def assess_with_model(
        self,
        message: str,
        generate: Callable[[str, float], Awaitable[str]],
        *,
        known_facts: list[str] | None = None,
        recent_messages: list[str] | None = None,
        ambiguity_threshold: float = 0.55,
    ) -> StanceAssessment:
        """Deterministic spine first; refine only genuinely ambiguous cases with the model."""
        base = self.assess(message, known_facts=known_facts, recent_messages=recent_messages)
        if base.confidence >= ambiguity_threshold:
            return base
        labels = ", ".join(s.value for s in Stance)
        ctx = ("\nKnown facts:\n" + "\n".join(f"- {k}" for k in known_facts[:5])) if known_facts else ""
        prompt = (
            "Classify the communicative stance of this message. One word from: "
            f"{labels}.\nConsider whether it is literal, a joke, sarcasm, a guess, a "
            "hypothesis, role-play, or conflicts with the known facts."
            f"{ctx}\n\nMessage: {message}\n\nStance:"
        )
        try:
            raw = str(await generate(prompt, 0.1) or "").strip().lower()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return base
        for s in Stance:
            if s.value in raw:
                base.scores[s.value] = base.scores.get(s.value, 0.0) + 0.6
                base.primary = s
                base.confidence = round(min(0.95, base.confidence + 0.2), 4)
                base.signals.append("model-refined")
                base.take_literally = s in {Stance.SINCERE, Stance.MISTAKEN, Stance.UNSURE}
                break
        return base

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _factual_conflict(text: str, known_facts: list[str] | None) -> tuple[bool, str]:
        """A confident claim conflicts with a known fact if it shares the fact's subject
        but negates/contradicts it. Conservative: needs strong content overlap."""
        if not known_facts:
            return False, ""
        neg = bool(re.search(r"\b(?:not|never|no longer|isn'?t|aren'?t|wasn'?t|don'?t|doesn'?t|can'?t|won'?t)\b", text, re.IGNORECASE))
        twords = _content_words(text)
        for fact in known_facts:
            fwords = _content_words(fact)
            if not fwords:
                continue
            overlap = len(twords & fwords) / max(1, len(fwords))
            fact_neg = bool(re.search(r"\b(?:not|never|no)\b", fact, re.IGNORECASE))
            if overlap >= 0.5 and (neg != fact_neg):
                return True, f"claim negates known fact: {fact[:80]}"
        return False, ""

    @staticmethod
    def _deception_cues(text: str, recent_messages: list[str] | None) -> list[str]:
        cues: list[str] = []
        if _DECEPTION_CUE_RE.search(text):
            cues.append("over-assurance phrasing")
        # Excessive qualifiers around a denial.
        if re.search(r"\bi (?:did not|didn'?t|never)\b", text, re.IGNORECASE) and len(text.split()) > 20:
            cues.append("elaborate denial")
        return cues

    @staticmethod
    def _contradicts_recent(text: str, recent_messages: list[str] | None) -> bool:
        if not recent_messages:
            return False
        tneg = bool(re.search(r"\bnot\b|\bnever\b|n'?t\b", text, re.IGNORECASE))
        twords = _content_words(text)
        for prev in recent_messages[-4:]:
            pwords = _content_words(prev)
            if not pwords:
                continue
            shared = twords & pwords
            pneg = bool(re.search(r"\bnot\b|\bnever\b|n'?t\b", prev, re.IGNORECASE))
            # Same subject matter (>=2 shared content words) with opposite polarity.
            if len(shared) >= 2 and tneg != pneg:
                return True
        return False

    @staticmethod
    def _rationale(primary: Stance, signals: list[str], factual_conflict: bool) -> str:
        if primary is Stance.SINCERE:
            return "no strong non-literal markers; treated as a sincere statement"
        head = signals[0] if signals else primary.value
        if factual_conflict and primary in {Stance.MISTAKEN, Stance.DECEPTIVE}:
            return f"{primary.value}: {head} (intent cannot be read with certainty)"
        return f"{primary.value}: {head}"


_instance: StanceInference | None = None


def get_stance_inference() -> StanceInference:
    global _instance
    if _instance is None:
        _instance = StanceInference()
    return _instance
