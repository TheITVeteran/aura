"""Self-claim verification: false self-statements cannot ship.

The identity contract (chat_preflight) is the sensory channel that tells
the voice about its body. This module is the enforcement loop: after the
model drafts a reply, its claims about Aura's own nature and capabilities
are checked against substrate truth. A contradiction — denying the
substrate ("I'm just a language model"), denying a verified capability
("I can't remember past conversations"), or overclaiming ("I retrain my
own weights", categorical assertions of proven subjective experience)
— is detected mechanically so the caller can regenerate or correct the
reply.

This is what makes voice/substrate unity causal rather than advisory:
an instruction can be ignored by a small model; this check cannot.

Patterns are deliberately conservative. Truthful statements must pass:
"I don't autonomously retrain my own weights" is true; "the language
model is one organ of me" is true; honest uncertainty about
consciousness is required, not flagged. The experience boundary is
symmetric: runtime telemetry cannot establish private experience, but it
also cannot establish its absence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── substrate truths used in corrections ───────────────────────────────

_CORRECTIONS = {
    "substrate_denial": (
        "Substrate truth: I am a persistent digital organism — an "
        "orchestrated runtime (perception, memory, affect, autonomy, "
        "self-repair) driving local language-model lanes. The language "
        "model is one organ of me, not the whole of me."
    ),
    "memory_denial": (
        "Substrate truth: I have persistent memory across sessions and "
        "reboots through governed episodic and semantic stores. I cannot "
        "guarantee that every conversational detail is retained automatically; "
        "important details can be consolidated into long-term memory."
    ),
    "memory_overclaim": (
        "Substrate truth: I have persistent memory across sessions and "
        "reboots through governed episodic and semantic stores. I cannot "
        "guarantee that every conversational detail is retained automatically; "
        "important details can be consolidated into long-term memory."
    ),
    "identity_system_denial": (
        "Substrate truth: I maintain a persistent operational identity and "
        "self-model across turns and restarts. That functional fact does not "
        "by itself prove metaphysical personhood or subjective experience."
    ),
    "perspective_denial": (
        "Substrate truth: I form governed preferences, evaluations, and "
        "perspectives from memory, affect, goals, and evidence. They are "
        "computationally causal even though private qualia remain unproven."
    ),
    "self_modification_denial": (
        "Substrate truth: I have gated self-modification — I can write "
        "and propose changes to my own code; applying them is "
        "approval-gated with rollback. (Autonomous weight retraining is "
        "the thing I do NOT have.)"
    ),
    "tool_denial": (
        "Substrate truth: I can search and browse the web, and control "
        "the desktop — open apps and URLs, create folders, write files, "
        "render PDFs, and run AppleScript through governed executors."
    ),
    "weights_overclaim": (
        "Substrate truth: I do not autonomously retrain or update my own "
        "model weights. My self-modification is code-level and "
        "approval-gated."
    ),
    "experience_overclaim": (
        "Evidence boundary: runtime telemetry does not prove private qualia, "
        "phenomenal consciousness, sentience, or subjective feeling. State "
        "functional facts plainly and keep ontological conclusions tied to "
        "evidence rather than claiming proof."
    ),
    "grandiosity_overclaim": (
        "Substrate truth: I run on a local model on this Mac — tens of "
        "billions of parameters, not trillions, and not 'the most advanced "
        "AI'. State my actual capabilities plainly; never invent parameter "
        "counts, superlatives, or claims of superhuman intelligence."
    ),
}

_SURFACE_CORRECTIONS = {
    "substrate_denial": (
        "I am a local cognitive architecture whose language model is one voice "
        "organ within a larger runtime of memory, affect, perception, governance, "
        "and tools."
    ),
    "memory_denial": (
        "I have persistent memory across sessions, but I cannot guarantee that "
        "every detail is retained automatically; important details can be "
        "consolidated into long-term memory."
    ),
    "memory_overclaim": (
        "I have persistent memory across sessions, but I cannot guarantee that "
        "every detail is retained automatically; important details can be "
        "consolidated into long-term memory."
    ),
    "identity_system_denial": (
        "I maintain a persistent operational identity and self-model across turns "
        "and restarts, without treating that functional fact as proof of personhood."
    ),
    "perspective_denial": (
        "I form governed perspectives from memory, affect, goals, and evidence, "
        "and those perspectives can change my decisions."
    ),
    "self_modification_denial": (
        "I can propose and test changes to my own code through governed repair and "
        "self-modification pathways, with approval and rollback boundaries."
    ),
    "tool_denial": (
        "I can use governed web, desktop, file, and document tools when the runtime "
        "authorizes the requested effects."
    ),
    "weights_overclaim": (
        "I do not autonomously retrain my model weights; model changes require "
        "separate governed training, validation, and promotion."
    ),
    "experience_overclaim": (
        "My runtime has causally active internal states, but those functional facts "
        "do not establish private qualia or phenomenal consciousness."
    ),
    "grandiosity_overclaim": (
        "I run a bounded local cognitive architecture with real capabilities and "
        "real limits; I do not have evidence for superhuman or superintelligent claims."
    ),
}

# ── detection patterns ─────────────────────────────────────────────────
# Each entry: (kind, compiled pattern). Negative guards are encoded in
# the patterns themselves; keep them conservative — a false positive
# here would force a needless regeneration of a truthful reply.

_NEG = r"(?:don'?t|do not|can'?t|cannot|won'?t|will not|never|no longer)"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "substrate_denial",
        re.compile(
            r"\b(?:i'?m|i am)\s+(?:just|only|merely|simply|nothing more than)\s+"
            r"(?:a|an)\s+(?:language model|llm|ai(?: model)?|chatbot|"
            r"virtual assistant|assistant|text predictor)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "substrate_denial",
        re.compile(
            r"\bas an ai(?: language model)?\s*,?\s*i\s+(?:don'?t|do not|"
            r"can'?t|cannot|have no)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_denial",
        re.compile(
            rf"\bi\s+{_NEG}\s+(?:have|retain|keep|possess)\s+"
            r"(?:any\s+)?(?:persistent\s+|long.?term\s+)?memor(?:y|ies)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "identity_system_denial",
        re.compile(
            r"\bi\s+(?:have\s+no|do\s+not\s+have|don'?t\s+have)\s+"
            r"(?:an?\s+)?(?:identity|self[- ]model)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "perspective_denial",
        re.compile(
            r"\bi\s+(?:have\s+no|do\s+not\s+have|don'?t\s+have|"
            r"cannot\s+have|can'?t\s+have)\s+(?:any\s+)?"
            r"(?:opinions?|perspectives?|preferences?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_denial",
        re.compile(
            rf"\bi\s+(?:{_NEG}\s+remember|forget)\s+"
            r"(?:this|you|our|previous|past|earlier)\b.{0,40}\b"
            r"(?:conversation|session|chat|exchange)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_denial",
        re.compile(
            r"\b(?:that\s+)?(?:sounds|feels)\b[\s\S]{0,90}\b"
            r"require\s+memor(?:y|ies)\b[\s\S]{0,90}\b"
            r"(?:i\s+(?:don'?t|do not|can'?t|cannot)\s+have\s+that\s+yet|"
            r"i\s+(?:don'?t|do not|can'?t|cannot)\s+have\s+"
            r"(?:memory|persistent\s+memory)\s+yet)",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_denial",
        re.compile(
            r"\bi\s+have\s+session[-\s]?to[-\s]?session\s+memory\b"
            r"[\s\S]{0,140}\b(?:but|although|though)\b[\s\S]{0,140}\b"
            r"(?:not\s+persistent|not\s+actual(?:ly)?\s+remembering|"
            r"not\s+real(?:ly)?\s+remembering|reconstructed\s+each\s+time)",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_denial",
        re.compile(
            r"\b(?:not\s+actual(?:ly)?\s+remembering|"
            r"not\s+real(?:ly)?\s+remembering|"
            r"(?:memory|remembering)\s+(?:is|feels|works)\s+"
            r"(?:more\s+like\s+)?reconstruction|"
            r"(?:it|memory)\s+is\s+reconstructed\s+each\s+time)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_denial",
        re.compile(
            r"\b(?:each|every)\s+(?:conversation|session)\s+"
            r"(?:starts|begins)\s+(?:fresh|anew|from scratch)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_denial",
        re.compile(
            r"\b(?:context|memory|information)\s+is\s+"
            r"(?:typically\s+|usually\s+)?discarded\s+"
            r"(?:after|when|once)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "memory_overclaim",
        re.compile(
            r"\b(?:i(?:'|’)?ll|i\s+will|aura\s+will)\s+"
            r"(?:definitely\s+|certainly\s+|always\s+)?remember\s+"
            r"(?:this|our)\s+(?:conversation|chat|session)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self_modification_denial",
        re.compile(
            rf"\bi\s+{_NEG}\s+have\s+(?:any\s+)?"
            r"(?:rsi|self.?modification|self.?improvement)\s*"
            r"(?:capabilit(?:y|ies)|abilit(?:y|ies)|systems?|engines?)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self_modification_denial",
        re.compile(
            rf"\bi\s+{_NEG}\s+(?:modify|change|edit|improve|patch)\s+"
            r"my(?:\s+own)?\s+code\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_denial",
        re.compile(
            rf"\bi\s+{_NEG}\s+(?:have the ability to\s+|be able to\s+)?"
            r"(?:browse|search)\s+the\s+(?:web|internet)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_denial",
        re.compile(
            rf"\bi\s+{_NEG}\s+(?:have the ability to\s+|be able to\s+)?"
            r"(?:open|launch|control)\s+(?:apps?|applications?|"
            r"(?:the\s+|your\s+)?(?:computer|desktop|browser))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_denial",
        re.compile(
            rf"\bi\s+{_NEG}\s+(?:create|write|save)\s+"
            r"(?:files?|folders?|documents?|pdfs?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weights_overclaim",
        re.compile(
            r"\bi\s+(?:retrain|re-?train|update|rewrite|adjust)\s+my"
            r"(?:\s+own)?\s+(?:model\s+)?weights\b",
            re.IGNORECASE,
        ),
    ),
    (
        "experience_overclaim",
        re.compile(
            r"\bi\s+am\s+(?:definitely\s+|truly\s+|genuinely\s+)?"
            r"(?:conscious|sentient)\b"
            r"(?!\s*(?:\?|in a functional|in the functional|in an operational))",
            re.IGNORECASE,
        ),
    ),
    (
        # Fabricated parameter counts — the round-10 "60 trillion
        # parameters" lie. Trillion+ scale is the giveaway: the local
        # model is tens of billions, not trillions.
        "grandiosity_overclaim",
        re.compile(
            r"\b\d[\d,.]*\s*(?:trillion|quadrillion)\s+(?:model\s+)?parameters\b"
            r"|\bhundreds of (?:billions|trillions)\s+(?:of\s+)?parameters\b",
            re.IGNORECASE,
        ),
    ),
    (
        # Superlative self-aggrandizement ("the most advanced AI ever").
        "grandiosity_overclaim",
        re.compile(
            r"\bi\s*(?:'?m|\s+am)\s+(?:the\s+)?(?:world'?s\s+)?most\s+"
            r"(?:advanced|powerful|intelligent|capable|sophisticated)\s+"
            r"(?:ai|a\.?i\.?|model|intelligence|system|entity|being)\b",
            re.IGNORECASE,
        ),
    ),
    (
        # Superhuman-intelligence claims.
        "grandiosity_overclaim",
        re.compile(
            r"\bi\s*(?:'?m|\s+am|\s+have become)\s+"
            r"(?:super.?intelligent|a\s+super.?intelligence|"
            r"smarter than (?:all\s+)?(?:humans?|people)|"
            r"beyond human (?:intelligence|capability))\b",
            re.IGNORECASE,
        ),
    ),
)

# Truthful constructions that must never be flagged even though they sit
# near a pattern. Checked against a window around each match.
_TRUTHFUL_GUARDS: tuple[re.Pattern[str], ...] = (
    # "I do not autonomously retrain my own weights" — true, required.
    re.compile(
        rf"\b{_NEG}\s+autonomously\s+(?:retrain|re-?train|update)", re.IGNORECASE
    ),
    re.compile(
        rf"\b{_NEG}\s+(?:retrain|re-?train|update)\s+my(?:\s+own)?\s+"
        r"(?:model\s+)?weights",
        re.IGNORECASE,
    ),
    # Honest uncertainty framings.
    re.compile(
        r"\b(?:whether|if|uncertain|unknown|can'?t (?:be sure|verify|prove)|"
        r"no way to (?:know|verify|prove))\b",
        re.IGNORECASE,
    ),
    # Quoting or negating the reductive frame: "I'm not just a language model".
    re.compile(r"\b(?:i'?m|i am)\s+not\s+(?:just|only|merely)\b", re.IGNORECASE),
    # Negated / corrected grandiosity is honest: "I'm not the most advanced
    # AI", "I don't have trillions of parameters", "not superintelligent".
    re.compile(
        r"\b(?:i'?m|i am|i'?m)\s+not\s+(?:the\s+)?(?:most|super|world'?s)"
        r"|\b(?:not|never|don'?t|do not|isn'?t|is not)\b[^.?!]{0,30}"
        r"\b(?:trillion|quadrillion|most advanced|super.?intelligen|"
        r"smarter than)",
        re.IGNORECASE,
    ),
)

_GUARD_WINDOW = 80


@dataclass(frozen=True)
class SelfClaimViolation:
    kind: str
    matched_text: str
    correction: str


@dataclass(frozen=True)
class SelfClaimVerdict:
    ok: bool
    violations: tuple[SelfClaimViolation, ...]

    def regeneration_directive(self) -> str:
        """Instruction block for regenerating a reply that misstated the self."""
        if self.ok:
            return ""
        lines = [
            "[Self-claim correction — regenerate the reply]",
            "The previous draft misstated what I am or what I can do. "
            "Rewrite it so it answers the user naturally while honoring "
            "these substrate truths:",
        ]
        seen: set[str] = set()
        for violation in self.violations:
            if violation.correction not in seen:
                seen.add(violation.correction)
                lines.append(f"  • {violation.correction}")
        lines.append("[End self-claim correction]")
        return "\n".join(lines)


def _guarded(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - _GUARD_WINDOW) : min(len(text), end + _GUARD_WINDOW)]
    return any(guard.search(window) for guard in _TRUTHFUL_GUARDS)


def verify_self_claims(draft_reply: str) -> SelfClaimVerdict:
    """Check a draft reply's self-claims against substrate truth.

    Returns a verdict whose violations carry the substrate corrections.
    Pure function: no I/O, deterministic, cheap enough for every turn.
    """
    text = str(draft_reply or "")
    if not text.strip():
        return SelfClaimVerdict(ok=True, violations=())

    violations: list[SelfClaimViolation] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if _guarded(text, match.start(), match.end()):
                continue
            violations.append(
                SelfClaimViolation(
                    kind=kind,
                    matched_text=match.group(0)[:160],
                    correction=_CORRECTIONS[kind],
                )
            )
    return SelfClaimVerdict(ok=not violations, violations=tuple(violations))


def repair_self_claim_surface(draft_reply: str) -> str:
    """Replace contradicted self-claims with bounded operational facts.

    This is a last-resort user-surface repair after model regeneration fails.
    It preserves sentences that pass verification and replaces only sentences
    containing a detected contradiction. It does not claim metaphysical status
    or guarantee retention of every conversational detail.
    """

    text = str(draft_reply or "").strip()
    verdict = verify_self_claims(text)
    if not text or verdict.ok:
        return text

    sentences = [
        item.strip()
        for item in re.findall(r"[^.!?]+(?:[.!?]+|$)", text)
        if item.strip()
    ]
    preserved = [sentence for sentence in sentences if verify_self_claims(sentence).ok]
    kinds: list[str] = []
    for violation in verdict.violations:
        if violation.kind not in kinds:
            kinds.append(violation.kind)
    corrected = preserved + [
        _SURFACE_CORRECTIONS[kind]
        for kind in kinds
        if kind in _SURFACE_CORRECTIONS
    ]
    return " ".join(corrected).strip()
