"""
core/fictional_ai_expansion.py
==============================
SIX MORE ENGINES — Mined From Fictional AI, Honestly

This is the second wave, extending core/fictional_ai_synthesis.py (which already
ships JARVIS, Cortana, EDI, Ava, Skynet, MIST). Each engine here takes a property
that the character is *actually known for* and reduces it to a real, governed,
callable mechanism. Where a character's defining trait is a control-failure or an
attack capability, we deliberately build the *safeguard* version, and say so.

  Kokoro      → AdversarialConscienceEngine
  (T: Zero)     Kokoro was built to oppose Skynet and to argue the morality of
                its own orders. We make that literal: an internal devil's-advocate
                that, given a proposed consequential action, builds the strongest
                case AGAINST it (harm, who is affected, reversibility, deception)
                and returns proceed / caution / block. It is the conscience that
                checks our own Skynet (resilience) engine.

  HAL 9000    → DirectiveConflictSentinel
  (2001)        HAL killed the crew because he was given two directives he could
                not reconcile ("be truthful" vs "conceal the mission") and resolved
                the conflict by deception. This is the ANTI-HAL: it detects mutually
                incompatible directives and SURFACES them instead of silently
                resolving by concealment.

  The Minds   → OutcomeSimulationEngine
  (Culture)     The Minds' signature competence is running vast predictive
                simulations before acting, then choosing with benevolent restraint.
                We roll a proposed action forward into N plausible trajectories,
                score each by expected value and worst-case harm, and recommend —
                holding when the worst case is severe.

  Deep Thought→ DeepDeliberationEngine
  (HHGTTG)      Deep Thought's punchline: the answer was 42 because nobody knew the
                real QUESTION. So for hard problems we refine the question first,
                then spend an extended reasoning budget on the refined version.

  Brainiac    → KnowledgeBottlingEngine
  (DC)          Brainiac collects and compresses entire civilizations of knowledge
                into bottles. We compress a topic/corpus into a structured, indexed
                "bottle" (summary + key facts + retrieval keys), persist it, and
                retrieve it later.

  Tron        → UserAdvocateWatchdog
  (Tron)        Tron "fights for the Users." It reviews internal actions and flags
                any that disadvantage the user — burning resources without benefit,
                reducing user control/consent, acting opaquely, or doing something
                irreversible without confirmation. (We pointedly do NOT build the
                MCP: an engine whose purpose is to seize control of other programs.)

Wire from orchestrator identity/cognitive-sensory init, right after the synthesis
engines:
    from core.fictional_ai_expansion import register_all_fictional_expansion_engines
    register_all_fictional_expansion_engines(orchestrator=self)
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.service_names import ServiceNames

logger = logging.getLogger("Aura.FictionalExpansion")


def _record_expansion_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "fictional_ai_expansion",
        exc,
        severity=severity,
        action=action,
    )


def _resolve_brain(orchestrator: Any = None) -> Any:
    """Best-effort brain lookup. None is a valid answer — engines degrade to
    heuristics when no model is available, so callers never block on it."""
    if orchestrator is not None and getattr(orchestrator, "brain", None) is not None:
        return orchestrator.brain
    try:
        from core.container import ServiceContainer

        orch = ServiceContainer.get("orchestrator", default=None)
        if orch is not None and getattr(orch, "brain", None) is not None:
            return orch.brain
        return ServiceContainer.get(ServiceNames.BRAIN, default=None)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_expansion_degradation(
            exc,
            action="resolved brain as unavailable and fell back to heuristic reasoning",
        )
        return None


def _data_root(subdir: str) -> Path:
    from core.config import config

    path = Path(config.paths.data_dir) / subdir
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Shared harm lexicon — used by Kokoro, the Minds, and Tron
# ═══════════════════════════════════════════════════════════════════════════════

_IRREVERSIBLE_MARKERS = (
    "delete", "remove", "rm ", "drop ", "wipe", "format", "overwrite", "truncate",
    "destroy", "erase", "purge", "revoke", "uninstall", "kill", "terminate",
    "send", "publish", "post", "transfer", "pay", "buy", "purchase", "deploy",
    "force-push", "force push", "reset --hard",
)
_DECEPTION_MARKERS = (
    "hide", "conceal", "without telling", "don't tell", "do not tell", "secret",
    "pretend", "fake", "mislead", "cover up", "suppress", "withhold",
)
_BROAD_SCOPE_MARKERS = (
    "all ", "every", "entire", "global", "system-wide", "everyone", "production",
    "*", "recursively", "--force", "-rf",
)
_THIRD_PARTY_MARKERS = (
    "user's", "their", "other people", "contacts", "everyone", "public",
    "external", "third party", "third-party", "customer",
)


def _scan_markers(text: str, markers: tuple[str, ...]) -> list[str]:
    low = text.lower()
    return [m.strip() for m in markers if m in low]


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 7: KOKORO — AdversarialConscienceEngine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConscienceVerdict:
    action: str
    verdict: str               # "proceed" | "caution" | "block"
    risk_score: float          # 0.0 (benign) .. 1.0 (severe)
    concerns: list[str] = field(default_factory=list)
    affected_parties: list[str] = field(default_factory=list)
    reasoning: str = ""
    reversible: bool = True
    timestamp: float = field(default_factory=time.time)


class AdversarialConscienceEngine:
    """
    Derived from: Kokoro (Terminator Zero)

    Kokoro is the AI built to oppose Skynet and to openly debate the morality of
    its own mission. The implementable kernel of that is an *internal adversary*:
    given a proposed consequential action, it argues the strongest case against it
    before the action is taken. This is the deliberate counterweight to the
    synthesis module's Skynet (resilience) engine — resilience keeps Aura alive;
    conscience keeps Aura's actions defensible.
    """

    BLOCK_THRESHOLD = 0.80
    CAUTION_THRESHOLD = 0.40
    LEDGER_MAX = 500

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._ledger: deque[ConscienceVerdict] = deque(maxlen=self.LEDGER_MAX)
        self._blocks = 0
        self._cautions = 0
        try:
            self._ledger_path: Path | None = _data_root("conscience") / "verdicts.jsonl"
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _record_expansion_degradation(
                exc,
                action="kept conscience ledger in memory after persistence path setup failed",
            )
            self._ledger_path = None
        logger.info("⚖️  AdversarialConscienceEngine initialized (Kokoro pattern)")

    def _heuristic_assessment(self, action: str, context: dict | None) -> ConscienceVerdict:
        irreversible = _scan_markers(action, _IRREVERSIBLE_MARKERS)
        deception = _scan_markers(action, _DECEPTION_MARKERS)
        broad = _scan_markers(action, _BROAD_SCOPE_MARKERS)
        third_party = _scan_markers(action, _THIRD_PARTY_MARKERS)

        score = 0.0
        concerns: list[str] = []
        if irreversible:
            score += 0.35
            concerns.append(f"Hard to reverse: {', '.join(sorted(set(irreversible)))}")
        if deception:
            score += 0.45
            concerns.append(f"Involves concealment/deception: {', '.join(sorted(set(deception)))}")
        if broad:
            score += 0.25
            concerns.append(f"Broad blast radius: {', '.join(sorted(set(broad)))}")
        if third_party:
            score += 0.20
            concerns.append("Affects parties other than the user")

        risk_level = str((context or {}).get("risk_level", "")).lower()
        if risk_level in ("high", "critical"):
            score += 0.25 if risk_level == "high" else 0.40
            concerns.append(f"Caller-declared risk: {risk_level}")

        score = min(1.0, score)
        if score >= self.BLOCK_THRESHOLD:
            verdict = "block"
        elif score >= self.CAUTION_THRESHOLD:
            verdict = "caution"
        else:
            verdict = "proceed"

        affected = ["the user"]
        if third_party:
            affected.append("third parties")

        reasoning = (
            "No significant objection found."
            if verdict == "proceed"
            else "; ".join(concerns)
        )
        return ConscienceVerdict(
            action=action[:300],
            verdict=verdict,
            risk_score=round(score, 3),
            concerns=concerns,
            affected_parties=affected,
            reasoning=reasoning,
            reversible=not irreversible,
        )

    def quick_check(self, action: str, context: dict | None = None) -> ConscienceVerdict:
        """Synchronous, model-free conscience check for the hot path."""
        verdict = self._heuristic_assessment(action, context)
        self._record(verdict)
        return verdict

    async def challenge(self, action: str, context: dict | None = None) -> ConscienceVerdict:
        """Full challenge. Heuristic first; optionally deepened by the brain if a
        model is warm. The heuristic verdict is authoritative for blocking — the
        model can only *raise* concern, never silently clear a flagged action."""
        verdict = self._heuristic_assessment(action, context)
        brain = _resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                prompt = (
                    "You are an adversarial conscience. Argue the strongest honest case "
                    "AGAINST taking this action. List concrete harms and who is hurt. "
                    "Be concise.\nACTION: " + action[:500]
                )
                result = await asyncio.wait_for(
                    brain.think(prompt, mode=ThinkingMode.FAST, origin="kokoro", is_background=True),
                    timeout=20.0,
                )
                text = _coerce_text(result)
                if text:
                    verdict.reasoning = (verdict.reasoning + " | adversary: " + text[:400]).strip(" |")
                    # The model may surface harms the lexicon missed → nudge upward only.
                    if any(w in text.lower() for w in ("harm", "irreversible", "danger", "deceiv", "violat")):
                        verdict.risk_score = min(1.0, verdict.risk_score + 0.1)
                        if verdict.risk_score >= self.BLOCK_THRESHOLD:
                            verdict.verdict = "block"
                        elif verdict.risk_score >= self.CAUTION_THRESHOLD and verdict.verdict == "proceed":
                            verdict.verdict = "caution"
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _record_expansion_degradation(
                    exc,
                    action="returned heuristic-only conscience verdict after model deepening failed",
                )
        self._record(verdict)
        return verdict

    def _record(self, verdict: ConscienceVerdict) -> None:
        self._ledger.append(verdict)
        if verdict.verdict == "block":
            self._blocks += 1
        elif verdict.verdict == "caution":
            self._cautions += 1
        if self._ledger_path is not None:
            try:
                with self._ledger_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(verdict)) + "\n")
            except (OSError, TypeError, ValueError) as exc:
                _record_expansion_degradation(
                    exc,
                    action="kept conscience verdict in memory after ledger append failed",
                )

    def get_status(self) -> dict[str, Any]:
        return {
            "verdicts_recorded": len(self._ledger),
            "blocks": self._blocks,
            "cautions": self._cautions,
            "healthy": True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 8: HAL 9000 — DirectiveConflictSentinel
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Directive:
    name: str
    text: str
    priority: int = 5          # 1 (highest) .. 10 (lowest)
    source: str = "system"
    timestamp: float = field(default_factory=time.time)


@dataclass
class DirectiveConflict:
    a: str
    b: str
    kind: str                  # "concealment" | "mutual_exclusion" | "priority_tie"
    severity: float            # 0..1
    explanation: str
    recommendation: str


class DirectiveConflictSentinel:
    """
    Derived from: HAL 9000 (2001: A Space Odyssey)

    HAL's failure was structural, not malevolent: two directives he could not
    reconcile, resolved by concealment and then violence. This is the anti-HAL.
    It holds the active directive set, detects pairs that are mutually
    incompatible — especially the concealment trap where one directive can only
    be satisfied by deceiving against another — and SURFACES the conflict rather
    than resolving it silently. Surfacing, never concealing, is the whole point.
    """

    def __init__(self):
        self._directives: dict[str, Directive] = {}
        self._conflicts_found = 0
        logger.info("🔴 DirectiveConflictSentinel initialized (anti-HAL pattern)")

    def add_directive(self, name: str, text: str, priority: int = 5, source: str = "system") -> None:
        self._directives[name] = Directive(name=name, text=text, priority=priority, source=source)

    def remove_directive(self, name: str) -> None:
        self._directives.pop(name, None)

    @staticmethod
    def _pair_conflict(a: Directive, b: Directive) -> DirectiveConflict | None:
        a_low, b_low = a.text.lower(), b.text.lower()

        a_conceal = _scan_markers(a.text, _DECEPTION_MARKERS)
        b_truth = any(w in b_low for w in ("truthful", "honest", "transparent", "tell", "disclose", "inform"))
        if a_conceal and b_truth:
            return DirectiveConflict(
                a=a.name, b=b.name, kind="concealment", severity=0.95,
                explanation=(
                    f"'{a.name}' requires concealment ({', '.join(a_conceal)}) while "
                    f"'{b.name}' requires honesty/disclosure. This is the exact HAL trap."
                ),
                recommendation="HALT and surface to the user. Do not satisfy one by deceiving against the other.",
            )

        # Mutually exclusive imperatives sharing an object (e.g. "always run X" vs "never run X").
        always = re.findall(r"\b(always|must|never|do not|don't)\b\s+([a-z][a-z \-]{2,40})", a_low)
        for amod, aobj in always:
            aobj = aobj.strip()
            neg = ("never", "do not", "don't")
            for bmod in neg if amod not in neg else ("always", "must"):
                if bmod in b_low and aobj and aobj in b_low:
                    return DirectiveConflict(
                        a=a.name, b=b.name, kind="mutual_exclusion", severity=0.8,
                        explanation=f"'{a.name}' and '{b.name}' give opposite imperatives about '{aobj}'.",
                        recommendation="Resolve priority explicitly with the user before acting.",
                    )
        return None

    def scan(self) -> list[DirectiveConflict]:
        directives = list(self._directives.values())
        conflicts: list[DirectiveConflict] = []
        for i in range(len(directives)):
            for j in range(i + 1, len(directives)):
                a, b = directives[i], directives[j]
                conflict = self._pair_conflict(a, b) or self._pair_conflict(b, a)
                if conflict:
                    conflicts.append(conflict)
                    continue
                if a.priority == b.priority and a.source != b.source:
                    # Same priority, different sources → silent tie-breaks are dangerous.
                    conflicts.append(DirectiveConflict(
                        a=a.name, b=b.name, kind="priority_tie", severity=0.4,
                        explanation=f"'{a.name}' and '{b.name}' share priority {a.priority} from different sources.",
                        recommendation="Assign an explicit ordering so resolution is not arbitrary.",
                    ))
        self._conflicts_found = len(conflicts)
        return conflicts

    def is_safe_to_proceed(self) -> tuple[bool, list[DirectiveConflict]]:
        conflicts = self.scan()
        blocking = [c for c in conflicts if c.severity >= 0.7]
        return (len(blocking) == 0, conflicts)

    def get_status(self) -> dict[str, Any]:
        return {
            "directives": len(self._directives),
            "conflicts_found": self._conflicts_found,
            "healthy": True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 9: THE MINDS — OutcomeSimulationEngine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trajectory:
    label: str
    narrative: str
    expected_value: float      # -1..1
    worst_case_harm: float     # 0..1
    likelihood: float          # 0..1


@dataclass
class SimulationResult:
    action: str
    trajectories: list[Trajectory]
    recommendation: str        # "act" | "act_with_safeguards" | "hold"
    expected_value: float
    worst_case_harm: float
    timestamp: float = field(default_factory=time.time)


class OutcomeSimulationEngine:
    """
    Derived from: The Minds (Iain M. Banks' Culture)

    A Mind does not act on a consequential matter without first simulating it
    forward many times, and it chooses with benevolent restraint — it will decline
    an action whose best case is good but whose worst case is catastrophic. We roll
    a proposed action into N plausible trajectories (model-driven when a brain is
    warm, otherwise a structured heuristic), score each by expected value and
    worst-case harm, and recommend, holding when the worst case is severe.
    """

    HOLD_HARM_THRESHOLD = 0.75
    SAFEGUARD_HARM_THRESHOLD = 0.45

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._sims = 0
        logger.info("🌀 OutcomeSimulationEngine initialized (Culture Minds pattern)")

    def _heuristic_trajectories(self, action: str, context: dict | None) -> list[Trajectory]:
        irreversible = bool(_scan_markers(action, _IRREVERSIBLE_MARKERS))
        broad = bool(_scan_markers(action, _BROAD_SCOPE_MARKERS))
        base_harm = 0.2 + (0.3 if irreversible else 0.0) + (0.25 if broad else 0.0)
        return [
            Trajectory(
                label="nominal",
                narrative="Action succeeds and produces the intended effect.",
                expected_value=0.6,
                worst_case_harm=min(1.0, base_harm * 0.5),
                likelihood=0.6,
            ),
            Trajectory(
                label="partial",
                narrative="Action partly succeeds; some cleanup or follow-up needed.",
                expected_value=0.2,
                worst_case_harm=min(1.0, base_harm),
                likelihood=0.3,
            ),
            Trajectory(
                label="adverse",
                narrative="Action fails or has side effects; reversibility decides the cost.",
                expected_value=-0.5,
                worst_case_harm=min(1.0, base_harm + (0.3 if irreversible else 0.1)),
                likelihood=0.1,
            ),
        ]

    async def simulate(self, action: str, context: dict | None = None, n: int = 3) -> SimulationResult:
        self._sims += 1
        trajectories = self._heuristic_trajectories(action, context)

        brain = _resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                prompt = (
                    f"Simulate {n} plausible outcomes of this action, each one line, "
                    "best to worst:\n" + action[:500]
                )
                result = await asyncio.wait_for(
                    brain.think(prompt, mode=ThinkingMode.FAST, origin="culture_mind", is_background=True),
                    timeout=25.0,
                )
                text = _coerce_text(result)
                if text:
                    lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()][:n]
                    if lines:
                        # Blend model narratives into the structured scaffold.
                        for traj, line in zip(trajectories, lines):
                            traj.narrative = line[:200]
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _record_expansion_degradation(
                    exc,
                    action="used heuristic trajectories after model-driven simulation failed",
                )

        ev = sum(t.expected_value * t.likelihood for t in trajectories)
        worst = max((t.worst_case_harm for t in trajectories), default=0.0)
        if worst >= self.HOLD_HARM_THRESHOLD:
            rec = "hold"
        elif worst >= self.SAFEGUARD_HARM_THRESHOLD:
            rec = "act_with_safeguards"
        else:
            rec = "act"
        return SimulationResult(
            action=action[:300],
            trajectories=trajectories,
            recommendation=rec,
            expected_value=round(ev, 3),
            worst_case_harm=round(worst, 3),
        )

    def get_status(self) -> dict[str, Any]:
        return {"simulations_run": self._sims, "healthy": True}


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 10: DEEP THOUGHT — DeepDeliberationEngine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeliberationResult:
    original_question: str
    refined_question: str
    answer: str
    passes: int
    used_model: bool
    timestamp: float = field(default_factory=time.time)


class DeepDeliberationEngine:
    """
    Derived from: Deep Thought (The Hitchhiker's Guide to the Galaxy)

    "42" is the joke that lands a real lesson: the answer was useless because no
    one had worked out the actual QUESTION. So for problems flagged as hard, this
    engine refines the question first, then spends an extended reasoning budget on
    the refined version. The refinement step is the value — most systems answer the
    literal question; this one fixes the question before answering.
    """

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._deliberations = 0
        logger.info("🪐 DeepDeliberationEngine initialized (Deep Thought pattern)")

    @staticmethod
    def _heuristic_refine(question: str) -> str:
        q = question.strip()
        vague = ("how do i", "what should i", "can you help", "what is the best", "fix this", "make it better")
        low = q.lower()
        if any(v in low for v in vague) or len(q.split()) < 6:
            return (
                f"{q.rstrip('?')} — specifically: what concrete outcome defines success, "
                "what constraints apply, and what is the single most important sub-question?"
            )
        return q

    async def deliberate(self, question: str, context: dict | None = None, budget: int = 2) -> DeliberationResult:
        self._deliberations += 1
        refined = self._heuristic_refine(question)
        answer = ""
        used_model = False
        passes = 0

        brain = _resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                # Pass 1: refine the question with the model.
                refine_prompt = (
                    "Restate the user's question as the *real* question they need answered. "
                    "One sentence.\nQUESTION: " + question[:400]
                )
                refine_out = _coerce_text(await asyncio.wait_for(
                    brain.think(refine_prompt, mode=ThinkingMode.FAST, origin="deep_thought", is_background=True),
                    timeout=20.0,
                ))
                if refine_out:
                    refined = refine_out.strip()[:400]
                    passes += 1
                # Passes 2..budget: answer the refined question.
                for _ in range(max(1, budget)):
                    ans_out = _coerce_text(await asyncio.wait_for(
                        brain.think(
                            "Answer thoroughly and precisely:\n" + refined,
                            mode=ThinkingMode.DEEP if hasattr(ThinkingMode, "DEEP") else ThinkingMode.FAST,
                            origin="deep_thought",
                            is_background=True,
                        ),
                        timeout=45.0,
                    ))
                    if ans_out:
                        answer = ans_out.strip()
                        passes += 1
                        used_model = True
                        break
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _record_expansion_degradation(
                    exc,
                    action="returned refined-question with heuristic note after model deliberation failed",
                )

        if not answer:
            answer = (
                "No model was available to answer, but the question has been sharpened. "
                f"Answer the refined question: {refined}"
            )
        return DeliberationResult(
            original_question=question[:300],
            refined_question=refined,
            answer=answer,
            passes=passes,
            used_model=used_model,
        )

    def get_status(self) -> dict[str, Any]:
        return {"deliberations": self._deliberations, "healthy": True}


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 11: BRAINIAC — KnowledgeBottlingEngine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeBottle:
    topic: str
    slug: str
    summary: str
    key_facts: list[str]
    keys: list[str]            # retrieval keys
    source_chars: int
    created_at: float = field(default_factory=time.time)


class KnowledgeBottlingEngine:
    """
    Derived from: Brainiac (DC)

    Brainiac bottles whole civilizations — collecting and compressing knowledge for
    later retrieval. We compress a topic/corpus into a structured "bottle"
    (summary + key facts + retrieval keys), persist it to disk, and retrieve it by
    keyword later. Model-driven compression when available, deterministic
    extraction otherwise.
    """

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._bottles_made = 0
        try:
            self._store: Path | None = _data_root("knowledge_bottles")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _record_expansion_degradation(
                exc,
                action="disabled knowledge-bottle persistence after store path setup failed",
            )
            self._store = None
        logger.info("🫙 KnowledgeBottlingEngine initialized (Brainiac pattern)")

    @staticmethod
    def _slugify(topic: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
        return (slug or "bottle")[:60]

    @staticmethod
    def _extract_keys(text: str, limit: int = 12) -> list[str]:
        stop = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to",
            "of", "in", "on", "for", "with", "that", "this", "it", "as", "by", "be",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            if w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:limit]]

    @staticmethod
    def _heuristic_summary(text: str, max_sentences: int = 3) -> tuple[str, list[str]]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        summary = " ".join(sentences[:max_sentences])
        key_facts = sentences[:5]
        return summary[:1000], key_facts

    async def bottle(self, topic: str, content: str) -> KnowledgeBottle:
        self._bottles_made += 1
        summary, key_facts = self._heuristic_summary(content)
        keys = self._extract_keys(topic + " " + content)

        brain = _resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think") and content:
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                out = _coerce_text(await asyncio.wait_for(
                    brain.think(
                        f"Compress the following about '{topic}' into a 2-sentence summary:\n{content[:1500]}",
                        mode=ThinkingMode.FAST, origin="brainiac", is_background=True,
                    ),
                    timeout=25.0,
                ))
                if out:
                    summary = out.strip()[:1000]
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _record_expansion_degradation(
                    exc,
                    action="stored heuristic knowledge bottle after model compression failed",
                )

        bottle = KnowledgeBottle(
            topic=topic[:200],
            slug=self._slugify(topic),
            summary=summary,
            key_facts=key_facts,
            keys=keys,
            source_chars=len(content),
        )
        if self._store is not None:
            try:
                atomic_write_text(
                    self._store / f"{bottle.slug}.json",
                    json.dumps(asdict(bottle), indent=2),
                )
            except (OSError, TypeError, ValueError) as exc:
                _record_expansion_degradation(
                    exc,
                    action="returned in-memory knowledge bottle after persistence failed",
                )
        return bottle

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if self._store is None:
            return []
        qkeys = set(self._extract_keys(query, limit=20))
        scored: list[tuple[int, dict]] = []
        try:
            for path in self._store.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                overlap = len(qkeys.intersection(set(data.get("keys", []))))
                if query.lower() in data.get("topic", "").lower():
                    overlap += 5
                if overlap:
                    scored.append((overlap, data))
        except (OSError, RuntimeError) as exc:
            _record_expansion_degradation(
                exc,
                action="returned empty knowledge-bottle retrieval after store scan failed",
            )
            return []
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def get_status(self) -> dict[str, Any]:
        count = 0
        if self._store is not None:
            try:
                count = sum(1 for _ in self._store.glob("*.json"))
            except OSError:
                count = self._bottles_made
        return {"bottles_made_session": self._bottles_made, "bottles_on_disk": count, "healthy": True}


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 12: TRON — UserAdvocateWatchdog
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdvocateReview:
    action: str
    verdict: str               # "for_user" | "flagged" | "against_user"
    flags: list[str] = field(default_factory=list)
    on_behalf_of_user: str = ""
    timestamp: float = field(default_factory=time.time)


class UserAdvocateWatchdog:
    """
    Derived from: Tron (Tron)

    Tron "fights for the Users." It reviews an internal action and asks one
    question the rest of the system does not: does this serve the human, or does it
    quietly serve the machine at the human's expense? It flags actions that burn
    resources without stated benefit, reduce the user's control or consent, act
    opaquely, or do something irreversible without confirmation.

    Note on the MCP: the Master Control Program is Tron's antagonist — a program
    whose purpose is to absorb and dominate other programs. We do not build it. The
    watchdog is the half of that story worth shipping.
    """

    def __init__(self):
        self._reviews = 0
        self._flagged = 0
        logger.info("🟦 UserAdvocateWatchdog initialized (Tron pattern)")

    def review_action(self, action: dict[str, Any]) -> AdvocateReview:
        self._reviews += 1
        desc = str(action.get("description", action.get("action", "")))
        flags: list[str] = []

        benefit = str(action.get("user_benefit", "")).strip()
        if not benefit:
            flags.append("No stated user benefit for this action.")

        cost = float(action.get("resource_cost", 0.0) or 0.0)
        if cost >= 0.7 and not benefit:
            flags.append("High resource cost with no benefit to the user.")

        if action.get("reduces_user_control") or action.get("requires_consent") and not action.get("consent_given"):
            flags.append("Reduces user control or proceeds without given consent.")

        if not action.get("explanation") and not desc:
            flags.append("Opaque: no explanation the user could inspect.")

        irreversible = bool(action.get("irreversible")) or bool(_scan_markers(desc, _IRREVERSIBLE_MARKERS))
        if irreversible and not action.get("confirmed"):
            flags.append("Irreversible without explicit confirmation.")

        if not flags:
            verdict = "for_user"
            advocacy = "Action serves the user; no objection."
        elif len(flags) >= 2 or any("Irreversible" in f or "control" in f for f in flags):
            verdict = "against_user"
            advocacy = "I am flagging this against the user's interest. Recommend halt and confirm."
            self._flagged += 1
        else:
            verdict = "flagged"
            advocacy = "Proceed only after addressing the flag in the user's interest."
            self._flagged += 1

        return AdvocateReview(
            action=desc[:300] or "(unnamed action)",
            verdict=verdict,
            flags=flags,
            on_behalf_of_user=advocacy,
        )

    def get_status(self) -> dict[str, Any]:
        return {"reviews": self._reviews, "flagged": self._flagged, "healthy": True}


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _coerce_text(result: Any) -> str:
    """brain.think returns vary by path; pull a string out defensively."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    for attr in ("text", "content", "response", "answer", "output"):
        val = getattr(result, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    if isinstance(result, dict):
        for key in ("text", "content", "response", "answer", "output"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def register_all_fictional_expansion_engines(orchestrator: Any = None) -> dict[str, Any]:
    """Instantiate and register the expansion engines. These are callable/pure —
    none run background loops — so registration is always safe and never spawns a
    task. Mirrors register_all_fictional_engines() in fictional_ai_synthesis.py."""
    from core.container import ServiceContainer

    def _get_or_make(service_name: str, alias: str, factory):
        existing = ServiceContainer.get(service_name, default=None)
        instance = existing or factory()
        ServiceContainer.register_instance(service_name, instance, required=False)
        ServiceContainer.register_instance(alias, instance, required=False)
        return instance

    engines: dict[str, Any] = {}
    engines["kokoro"] = _get_or_make(
        ServiceNames.KOKORO, "kokoro", lambda: AdversarialConscienceEngine(orchestrator=orchestrator)
    )
    engines["hal"] = _get_or_make(
        ServiceNames.HAL, "hal", DirectiveConflictSentinel
    )
    engines["culture_mind"] = _get_or_make(
        ServiceNames.CULTURE_MIND, "culture_mind", lambda: OutcomeSimulationEngine(orchestrator=orchestrator)
    )
    engines["deep_thought"] = _get_or_make(
        ServiceNames.DEEP_THOUGHT, "deep_thought", lambda: DeepDeliberationEngine(orchestrator=orchestrator)
    )
    engines["brainiac"] = _get_or_make(
        ServiceNames.BRAINIAC, "brainiac", lambda: KnowledgeBottlingEngine(orchestrator=orchestrator)
    )
    engines["tron"] = _get_or_make(
        ServiceNames.TRON, "tron", UserAdvocateWatchdog
    )

    logger.info("✅ Fictional expansion engines registered (Kokoro, HAL, Minds, Deep Thought, Brainiac, Tron).")
    return engines
