"""Reasoning Amplifier v2 — the mandatory hard-task cognition layer.

For hard tasks Aura should almost never do ``prompt → one generation → answer``. She
should do ``prompt → problem model → N attempts → verification → search/repair →
judge → calibrated answer → receipt``. That is how a 32B local model is made to feel
dramatically smarter: not more parameters, but more *time, search and verification*,
and a system that refuses to accept first drafts, unverifiable claims, untested code
or ungrounded plans.

This module is the orchestrator that ties together the organs built around it:

* :mod:`core.brain.verifiers`        — domain truth engines (the hard gate)
* :mod:`core.brain.symbolic_sandbox` — run-and-self-correct code/math
* :mod:`core.brain.courtroom`        — adversarial Solver/Skeptic/Judge (deep modes)
* :mod:`core.brain.calibration_gate` — only assert what survived a check
* :mod:`core.brain.reasoning_memory` — pull failure modes from past episodes
* :func:`core.brain.reasoning_amplifier.amplify` — verifier-filtered self-consistency

The pipeline is budgeted, cancellable and foreground-aware (a wall-clock deadline and
a sample cap), and every answer carries a :class:`ReasoningReceipt`. It is wired into
the live response path via :mod:`core.brain.reasoning_strategies` so it is causal, not
a shelf ornament.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ReasoningAmplifierV2")

GenerateFn = Callable[[str, float], Awaitable[str]]


class ReasoningMode(str, Enum):
    FAST = "fast"        # 1 candidate + calibration (casual / low-stakes)
    NORMAL = "normal"    # 3 candidates + verifier + self-consistency
    DEEP = "deep"        # up to 9 candidates + verifier + courtroom judge
    EXTREME = "extreme"  # courtroom + sandbox repair + memory (high stakes)
    PROOF = "proof"      # refuse to answer unless verifier-clean


_MODE_BUDGET = {
    ReasoningMode.FAST: 1,
    ReasoningMode.NORMAL: 3,
    ReasoningMode.DEEP: 9,
    ReasoningMode.EXTREME: 9,
    ReasoningMode.PROOF: 9,
}
_TEMPS = [0.2, 0.5, 0.7, 0.9, 1.0, 0.4, 0.6, 0.8, 1.1]
_MAX_SAMPLES = len(_TEMPS)

# Φ thresholds mirror core.phases.phi_consciousness so the amplifier's notion of
# "is cognition integrated enough to spend deep compute" matches the kernel's.
PHI_DORMANT = 0.05      # below: fragmented — cap at FAST
PHI_REACTIVE = 0.20     # below: shallow — cap at NORMAL
PHI_DELIBERATE = 0.55   # at/above: full deliberate capacity


def _flag_on(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in {"0", "false", "off", "no"}


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Problem representation — reason from a normalized task model, not raw text.
# ---------------------------------------------------------------------------
@dataclass
class ProblemRepresentation:
    objective: str
    task_type: str
    knowns: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    verification_plan: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective[:200],
            "task_type": self.task_type,
            "constraints": self.constraints[:6],
            "verification_plan": self.verification_plan[:6],
        }


_CODE_HINT = re.compile(r"\b(code|function|bug|patch|compile|traceback|stack ?trace|def |class |import )\b", re.I)
_MATH_HINT = re.compile(
    r"\b(calculate|compute|how many|sum|product|equation|solve|factorial|prime|"
    r"to the power of|raised to|power of|mod(?:ulo)?|gcd|greatest common divisor|"
    r"remainder|divisible|\d+\s*[-+*/^]\s*\d+)\b",
    re.I,
)
_REPO_HINT = re.compile(r"\b(repo|codebase|module|where is|which file|architecture|how does .* work|implemented)\b", re.I)
_PLAN_HINT = re.compile(r"\b(plan|steps|how (?:do|would) (?:i|we|you)|approach|strategy|roadmap)\b", re.I)
_FACT_HINT = re.compile(r"\b(what is|who is|when did|define|explain|fact|true that)\b", re.I)


def classify_task_type(text: str) -> str:
    t = str(text or "")
    if _CODE_HINT.search(t):
        return "code"
    if _MATH_HINT.search(t):
        return "math"
    if _REPO_HINT.search(t):
        return "repo_audit"
    if _PLAN_HINT.search(t):
        return "planning"
    if _FACT_HINT.search(t):
        return "factual"
    return "generic"


_VERIFICATION_PLANS = {
    "code": ["py_compile + AST safety", "ruff static check", "run in symbolic sandbox if assertions present"],
    "math": ["re-check arithmetic exactly", "evaluate with sympy", "run computation in sandbox"],
    "repo_audit": ["confirm every cited path exists", "confirm cited symbols are defined"],
    "architecture": ["require source-span evidence", "confirm cited paths/symbols exist"],
    "factual": ["ground confident claims in supplied evidence", "hedge what cannot be grounded"],
    "planning": ["steps must be actionable", "plan must end with a verification step"],
    "generic": ["check for non-sequiturs", "calibrate confidence to evidence"],
}


def normalize_problem(objective: str, *, task_type: str | None = None, required_evidence: list[str] | None = None) -> ProblemRepresentation:
    """Convert a raw prompt into a structured task model (start from a higher level)."""
    tt = (task_type or classify_task_type(objective)).lower()
    constraints: list[str] = []
    low = objective.lower()
    if "without" in low:
        constraints.append("honor stated 'without' constraint")
    if any(w in low for w in ("must", "only", "exactly", "no ")):
        constraints.append("honor hard constraints in the prompt")
    return ProblemRepresentation(
        objective=str(objective or "").strip(),
        task_type=tt,
        required_evidence=list(required_evidence or []),
        verification_plan=_VERIFICATION_PLANS.get(tt, _VERIFICATION_PLANS["generic"]),
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Budget policy — spend compute where it pays.
# ---------------------------------------------------------------------------
class ReasoningBudgetPolicy:
    @staticmethod
    def choose_mode(task_type: str, *, risk_level: str = "normal", explicit: ReasoningMode | None = None) -> ReasoningMode:
        if explicit is not None:
            return explicit
        risk = (risk_level or "normal").lower()
        if risk == "high":
            return ReasoningMode.EXTREME if task_type in {"architecture", "factual", "self_claim", "repo_audit"} else ReasoningMode.DEEP
        if task_type in {"code", "math"}:
            return ReasoningMode.DEEP        # verifier-search until pass or budget
        if task_type in {"repo_audit", "architecture"}:
            return ReasoningMode.DEEP        # evidence-first cross-check
        if task_type in {"planning", "factual"}:
            return ReasoningMode.NORMAL
        return ReasoningMode.NORMAL


# ---------------------------------------------------------------------------
# Request / receipt / answer.
# ---------------------------------------------------------------------------
@dataclass
class AmplificationRequest:
    objective: str
    task_type: str = ""
    risk_level: str = "normal"
    mode: ReasoningMode | None = None
    time_budget_s: float = 30.0
    sample_budget: int | None = None
    required_evidence: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningReceipt:
    mode: str
    strategy_used: str
    task_type: str
    num_candidates: int
    verifiers_run: list[str]
    valid_candidates: int
    winning_candidate_id: int
    confidence: float
    agreement: float
    epistemic_status: str
    evidence_refs: list[str] = field(default_factory=list)
    known_failures: list[str] = field(default_factory=list)
    budget_used: dict[str, Any] = field(default_factory=dict)
    fallbacks_used: list[str] = field(default_factory=list)
    guards_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "strategy_used": self.strategy_used,
            "task_type": self.task_type,
            "num_candidates": self.num_candidates,
            "verifiers_run": self.verifiers_run,
            "valid_candidates": self.valid_candidates,
            "winning_candidate_id": self.winning_candidate_id,
            "confidence": round(self.confidence, 3),
            "agreement": round(self.agreement, 3),
            "epistemic_status": self.epistemic_status,
            "evidence_refs": self.evidence_refs[:6],
            "known_failures": self.known_failures[:6],
            "budget_used": self.budget_used,
            "fallbacks_used": self.fallbacks_used,
            "guards_applied": self.guards_applied[:6],
        }


@dataclass
class AmplifiedAnswer:
    answer: str
    confidence: float
    verified: bool
    calibrated: bool
    receipt: ReasoningReceipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "verified": self.verified,
            "calibrated": self.calibrated,
            "receipt": self.receipt.to_dict(),
        }


class ReasoningAmplifierV2:
    """The orchestrator. Construct with a ``generate(prompt, temperature)`` callable."""

    def __init__(
        self,
        generate: GenerateFn,
        *,
        verifier: Any | None = None,
        sandbox: Any | None = None,
        calibration: Any | None = None,
        memory: Any | None = None,
        evidence_provider: Any | None = None,
        substrate: Any | None = None,
        courtroom_factory: Callable[[GenerateFn, Any], Any] | None = None,
        escalate_generate: GenerateFn | None = None,
    ) -> None:
        self._generate = generate
        self._verifier = verifier
        self._sandbox = sandbox
        self._calibration = calibration
        self._memory = memory
        self._evidence_provider = evidence_provider
        self._substrate = substrate
        self._courtroom_factory = courtroom_factory
        # Optional stronger-tier generator (72B / api_deep). When set, a turn that
        # finishes verifier-dirty with time left gets one escalation pass. Off by
        # default (None) so the local/bench lane stays honest and surprise cloud
        # calls never happen implicitly.
        self._escalate_generate = escalate_generate

    # --- lazy default wiring (so callers can pass just a generate fn) -------
    def _ensure_verifier(self) -> Any:
        if self._verifier is None:
            from core.brain.verifiers import get_verifier_registry

            self._verifier = get_verifier_registry()
        return self._verifier

    def _ensure_calibration(self) -> Any:
        if self._calibration is None:
            from core.brain.calibration_gate import get_calibration_gate

            self._calibration = get_calibration_gate()
        return self._calibration

    def _ensure_memory(self) -> Any:
        if self._memory is None:
            from core.brain.reasoning_memory import get_reasoning_memory

            self._memory = get_reasoning_memory()
        return self._memory

    def _ensure_sandbox(self) -> Any:
        if self._sandbox is None:
            from core.brain.symbolic_sandbox import get_symbolic_sandbox

            self._sandbox = get_symbolic_sandbox()
        return self._sandbox

    def _ensure_evidence(self) -> Any:
        if self._evidence_provider is None:
            from core.brain.evidence_provider import get_evidence_provider

            self._evidence_provider = get_evidence_provider()
        return self._evidence_provider

    def _make_courtroom(self) -> Any:
        if self._courtroom_factory is not None:
            return self._courtroom_factory(self._generate, self._ensure_verifier())
        from core.brain.courtroom import Courtroom

        return Courtroom(self._generate, verifier=self._ensure_verifier())

    def _read_substrate(self) -> dict[str, float]:
        """Best-effort live read of valence/arousal/uncertainty from the substrate.

        Negative valence (repeated failure) or high uncertainty are the substrate
        telling the reasoner "you're stuck — think harder". We honour that by
        deepening the mode. Read-only and fail-open to neutral.
        """
        sub = self._substrate
        if sub is None:
            try:
                from core.container import ServiceContainer

                sub = (
                    ServiceContainer.get("soma", default=None)
                    or ServiceContainer.get("affect_engine", default=None)
                )
                self._substrate = sub
            except (ImportError, RuntimeError, AttributeError):
                sub = None
        signals: dict[str, float] = {}
        if sub is not None:
            for attr in ("get_substrate_affect", "get_affect", "affect_snapshot"):
                fn = getattr(sub, attr, None)
                if callable(fn):
                    try:
                        data = fn()
                        if isinstance(data, dict):
                            signals = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
                            break
                    except (RuntimeError, AttributeError, TypeError, ValueError):
                        continue
        # Augment with the kernel's live Φ (integration) and free-energy (prediction
        # error) so compute is allocated by how integrated AND how surprised the mind
        # is right now — not just by task type. Read-only, fail-open to whatever the
        # affect snapshot already gave us.
        try:
            from core.kernel.kernel_interface import KernelInterface

            loop = KernelInterface.get_instance().loop_state()
            if isinstance(loop, dict):
                if "phi" in loop and "phi" not in signals:
                    signals["phi"] = float(loop.get("phi") or 0.0)
                if "arousal" in loop and "arousal" not in signals:
                    signals["arousal"] = float(loop.get("arousal") or 0.0)
                if "valence" in loop and "valence" not in signals:
                    signals["valence"] = float(loop.get("valence") or 0.0)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            pass
        if "free_energy" not in signals and "fe" not in signals:
            try:
                from core.container import ServiceContainer

                fe_engine = ServiceContainer.get("free_energy_engine", default=None)
                fe_state = fe_engine.get_current_state() if fe_engine is not None and hasattr(fe_engine, "get_current_state") else None
                if fe_state is not None and hasattr(fe_state, "free_energy"):
                    signals["free_energy"] = float(fe_state.free_energy)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                pass
        return signals

    @staticmethod
    def _should_deepen(affect: dict[str, float]) -> bool:
        if not affect:
            return False
        uncertainty = affect.get("uncertainty", affect.get("entropy", 0.0))
        valence = affect.get("valence", 0.0)
        frustration = affect.get("frustration", 0.0)
        return uncertainty >= 0.75 or valence <= -0.4 or frustration >= 0.6

    @staticmethod
    def _deepen(mode: ReasoningMode, rungs: int = 1) -> ReasoningMode:
        ladder = {
            ReasoningMode.FAST: ReasoningMode.NORMAL,
            ReasoningMode.NORMAL: ReasoningMode.DEEP,
            ReasoningMode.DEEP: ReasoningMode.EXTREME,
        }
        for _ in range(max(0, int(rungs))):
            nxt = ladder.get(mode)
            if nxt is None:
                break
            mode = nxt
        return mode

    @staticmethod
    def _phi_mode_cap(mode: ReasoningMode, phi: float) -> ReasoningMode:
        """Fragmented cognition (low Φ) cannot sustain deep deliberation.

        Mirrors phi_consciousness mode-gating: under-integrated states get capped
        so the amplifier never burns a 9-sample courtroom on a mind that can't hold
        the result together. PROOF is never downgraded — refusal-on-unverified is a
        safety contract, not a compute tier.
        """
        if mode is ReasoningMode.PROOF:
            return mode
        order = [ReasoningMode.FAST, ReasoningMode.NORMAL, ReasoningMode.DEEP, ReasoningMode.EXTREME]
        if mode not in order:
            return mode
        if phi < PHI_DORMANT:
            ceiling = ReasoningMode.FAST
        elif phi < PHI_REACTIVE:
            ceiling = ReasoningMode.NORMAL
        else:
            return mode
        return mode if order.index(mode) <= order.index(ceiling) else ceiling

    @classmethod
    def _cognitive_demand(cls, signals: dict[str, float]) -> float:
        """How hard is *this* thought, from the live mind state → [0, 1].

        High prediction-error (free energy), uncertainty, negative valence (stuck),
        or frustration all say "you are not converging — think harder". Weights sum
        > 1 on purpose so a single strong signal can drive demand high; the result
        is clamped.
        """
        if not signals:
            return 0.0
        fe = _clamp01(signals.get("free_energy", signals.get("fe", 0.0)))
        uncertainty = _clamp01(signals.get("uncertainty", signals.get("entropy", 0.0)))
        neg_valence = _clamp01(-float(signals.get("valence", 0.0) or 0.0))
        frustration = _clamp01(signals.get("frustration", 0.0))
        return _clamp01(0.45 * fe + 0.30 * uncertainty + 0.25 * neg_valence + 0.20 * frustration)

    @staticmethod
    def _phi_capacity(phi: float) -> float:
        """Fraction of deep-compute the current integration level can support → [0,1]."""
        if phi <= PHI_DORMANT:
            return 0.2
        if phi >= PHI_DELIBERATE:
            return 1.0
        # Linear ramp from 0.2 at DORMANT to 1.0 at DELIBERATE.
        span = PHI_DELIBERATE - PHI_DORMANT
        return _clamp01(0.2 + 0.8 * ((phi - PHI_DORMANT) / span))

    @classmethod
    def _resolve_compute_budget(
        cls, base_mode: ReasoningMode, signals: dict[str, float]
    ) -> tuple[ReasoningMode, int, float]:
        """Φ-gated test-time compute allocation.

        Returns ``(mode, sample_budget, time_multiplier)``. Demand (how hard the
        thought is) buys depth; Φ-capacity (how integrated the mind is) gates how
        much of that depth it is allowed to spend. This is the controller that turns
        a fixed local model into a variable-effort reasoner.
        """
        phi = _clamp01(signals.get("phi", PHI_DELIBERATE))  # absent Φ ⇒ full capacity
        demand = cls._cognitive_demand(signals)
        capacity = cls._phi_capacity(phi)
        push = demand * capacity

        rungs = 0
        if push >= 0.66:
            rungs = 2
        elif push >= 0.33:
            rungs = 1
        # Preserve the legacy hard trigger: a clearly stuck mind always deepens once.
        if rungs == 0 and cls._should_deepen(signals):
            rungs = 1

        mode = cls._deepen(base_mode, rungs) if rungs else base_mode
        mode = cls._phi_mode_cap(mode, phi)

        base_samples = _MODE_BUDGET[mode]
        sample_budget = int(round(base_samples * (1.0 + 0.5 * push)))
        sample_budget = max(1, min(_MAX_SAMPLES, sample_budget))

        time_multiplier = round(max(0.6, min(2.5, 1.0 + 1.2 * push)), 3)
        return mode, sample_budget, time_multiplier

    async def amplify(self, request: AmplificationRequest) -> AmplifiedAnswer:
        start = time.monotonic()
        deadline = start + max(2.0, float(request.time_budget_s))
        fallbacks: list[str] = []

        # 1. normalize the problem into a structured task model.
        problem = normalize_problem(
            request.objective,
            task_type=request.task_type or None,
            required_evidence=request.required_evidence,
        )

        # 1b. solved-cache — a verifier-clean derivation of this exact problem is an
        # O(1) lookup instead of a full amplifier run. The lawful free lunch: amortize
        # search cost across re-encounters. Only source-independent task types are
        # cached, so a stale answer can never be served as current truth.
        if _flag_on("AURA_REASONING_CACHE") and not request.context.get("skip_cache"):
            try:
                from core.brain.reasoning_solved_cache import get_reasoning_solved_cache

                cached = get_reasoning_solved_cache().get(problem.objective, problem.task_type)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("amplifier_v2_cache_get", exc)
                cached = None
            if cached is not None and cached.answer:
                logger.info(
                    "🧠 [AmplifyV2] solved-cache HIT task=%s conf=%.2f (hits=%d)",
                    problem.task_type, cached.confidence, cached.hits,
                )
                receipt = ReasoningReceipt(
                    mode=cached.mode or "cached",
                    strategy_used="solved_cache",
                    task_type=problem.task_type,
                    num_candidates=0,
                    verifiers_run=list(cached.verifiers_run),
                    valid_candidates=1,
                    winning_candidate_id=0,
                    confidence=cached.confidence,
                    agreement=1.0,
                    epistemic_status="verified",
                    budget_used={"samples": 0, "time_s": round(time.monotonic() - start, 3), "cache": True},
                    fallbacks_used=["solved_cache_hit"],
                )
                return AmplifiedAnswer(
                    answer=cached.answer,
                    confidence=cached.confidence,
                    verified=True,
                    calibrated=False,
                    receipt=receipt,
                )

        mode = ReasoningBudgetPolicy.choose_mode(
            problem.task_type, risk_level=request.risk_level, explicit=request.mode
        )
        # Couple to the live mind: Φ-gated test-time compute. Demand (free energy,
        # uncertainty, stuck-valence) buys depth; Φ-capacity gates how much depth the
        # current integration level can spend. Explicit caller mode is never overridden.
        affect = self._read_substrate()
        sample_budget = request.sample_budget or _MODE_BUDGET[mode]
        if request.mode is None:
            if _flag_on("AURA_PHI_GATED_COMPUTE"):
                resolved_mode, resolved_samples, time_mult = self._resolve_compute_budget(mode, affect)
                if resolved_mode is not mode:
                    fallbacks.append(f"phi_gated:{mode.value}->{resolved_mode.value}")
                mode = resolved_mode
                if request.sample_budget is None:
                    sample_budget = resolved_samples
                if time_mult != 1.0:
                    deadline = start + max(2.0, float(request.time_budget_s) * time_mult)
                    fallbacks.append(f"phi_time_x{time_mult}")
            elif self._should_deepen(affect):
                mode = self._deepen(mode)
                sample_budget = request.sample_budget or _MODE_BUDGET[mode]
                fallbacks.append("substrate_deepened")

        # 2. retrieve failure-mode guards from prior reasoning.
        guard_text = ""
        guards: list[str] = []
        try:
            guard_text = self._ensure_memory().as_guard_text(problem.objective, task_type=problem.task_type)
            if guard_text:
                guards = [ln[2:] for ln in guard_text.splitlines() if ln.startswith("- ")]
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("amplifier_v2_recall", exc)

        # 2b. ReAct grounding — gather REAL evidence (read repo source spans / recall
        # memory) so generation is conditioned on fact and the verifier has something
        # concrete to check. Merged with any caller-supplied evidence.
        if not request.context.get("skip_evidence"):
            try:
                gathered = await self._with_deadline(
                    self._ensure_evidence().render_pack(
                        problem.objective, task_type=problem.task_type, limit=6
                    ),
                    min(deadline, start + 10.0),
                )
                for span in gathered or []:
                    if span and span not in problem.required_evidence:
                        problem.required_evidence.append(span)
            except (RuntimeError, AttributeError, TypeError, ValueError, asyncio.TimeoutError) as exc:
                record_degradation("amplifier_v2_evidence", exc)

        # 3-8. produce a synthesized, verified answer per mode.
        if mode in (ReasoningMode.DEEP, ReasoningMode.EXTREME, ReasoningMode.PROOF):
            answer, agreement, verdict, n_cand, strategy = await self._deep_path(
                problem, guard_text, sample_budget, deadline, mode, request.context, fallbacks
            )
        else:
            answer, agreement, verdict, n_cand, strategy = await self._shallow_path(
                problem, guard_text, sample_budget, deadline, mode, request.context, fallbacks
            )

        # Verified-answer semantics, fail-closed (July external review): a
        # verdict of None means the verifier CRASHED or never ran — that is
        # not a pass. verifier_ok = "the verifier did not object";
        # verifier_checked = "the verifier actually evaluated something".
        # Only ok AND checked may ever be presented as *verified*: a vacuous
        # pass (prose with nothing to evaluate) is unverified by definition.
        verifier_ok = bool(getattr(verdict, "ok", False)) if verdict is not None else False
        verifier_checked = bool(getattr(verdict, "checked", False))
        verifier_issues = list(getattr(verdict, "issues", []) or []) if verdict is not None else []
        if verdict is None:
            verifier_issues.append("verifier_unavailable")
        verifiers_run = (getattr(verdict, "engine", "") or "").split("+") if verdict is not None else []

        # 8b. Tier escalation — verifier-of-last-resort. If the local pipeline finished
        # verifier-dirty, a stronger generator (72B / api_deep) is wired, and budget
        # remains, spend one bounded pass on the strong tier and adopt a clean candidate.
        # "Go get the answer where it lives" — Wall-2/Wall-3 loophole, never implicit.
        if (
            not verifier_ok
            and self._escalate_generate is not None
            and mode in (ReasoningMode.DEEP, ReasoningMode.EXTREME, ReasoningMode.PROOF)
            and time.monotonic() < deadline
        ):
            esc_prompt = self._build_prompt(problem, guard_text)

            async def _escalate_one(temp: float) -> str:
                try:
                    return str(await self._escalate_generate(esc_prompt, temp) or "").strip()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation("amplifier_v2_escalate_generate", exc)
                    return ""

            try:
                esc_candidates = await self._with_deadline(
                    asyncio.gather(_escalate_one(0.2), _escalate_one(0.5)), deadline
                )
            except (RuntimeError, AttributeError, TypeError, ValueError, asyncio.TimeoutError) as exc:
                record_degradation("amplifier_v2_escalate", exc)
                esc_candidates = []
            for cand in [c for c in (esc_candidates or []) if c]:
                esc_verdict = await self._verify(cand, problem, request.context)
                if bool(getattr(esc_verdict, "ok", False)):
                    answer = cand
                    verdict = esc_verdict
                    verifier_ok = True
                    verifier_issues = list(getattr(esc_verdict, "issues", []) or [])
                    verifiers_run = (getattr(esc_verdict, "engine", "") or "").split("+")
                    fallbacks.append("tier_escalated")
                    logger.info("🧠 [AmplifyV2] tier escalation produced a verifier-clean answer.")
                    break

        # 9. final critique — calibrate confidence to epistemic status. Uses the
        # full evidence pack (caller-supplied + ReAct-gathered source spans/memories).
        evidence = list(problem.required_evidence) + list(request.context.get("evidence", []) or [])
        calibrated_answer, calibration = self._calibrate(answer, verdict, evidence, verifier_ok)

        # PROOF mode refuses to answer unless something ACTUALLY survived
        # verification: a crashed verifier or a vacuous pass (nothing
        # checkable) is not proof. Refusal-on-unverifiable is the feature.
        if mode is ReasoningMode.PROOF and not (verifier_ok and verifier_checked):
            if verdict is None:
                reason = "the verifier was unavailable"
            elif not verifier_checked:
                reason = "nothing in the answer was mechanically checkable"
            else:
                reason = "; ".join(verifier_issues[:2]) or "no verifier-clean candidate"
            calibrated_answer = (
                f"I can't assert an answer here: it did not survive verification ({reason})."
            )
            fallbacks.append("proof_refused_unverified")

        confidence = round(min(calibration.confidence, 0.98 if verifier_ok else 0.55), 4)

        # 9b. learn from the outcome — memoize wins, queue losses for idle retry.
        # CRITICAL soundness gate: only cache/learn when the verifier ACTUALLY checked
        # (verdict.checked). A vacuous pass (ok=True, checked=False — e.g. prose with no
        # arithmetic to evaluate) must never be cached, or a wrong answer gets served as
        # truth forever. The hard bench caught exactly this poisoning.
        if "solved_cache_hit" not in fallbacks:
            if verifier_ok and verifier_checked:
                # Memoize verifier-clean source-independent derivations for instant re-use.
                if _flag_on("AURA_REASONING_CACHE") and not request.context.get("skip_cache"):
                    try:
                        from core.brain.reasoning_solved_cache import get_reasoning_solved_cache

                        get_reasoning_solved_cache().put(
                            problem.objective,
                            problem.task_type,
                            answer=calibrated_answer,
                            confidence=confidence,
                            mode=mode.value,
                            verifiers_run=[v for v in verifiers_run if v],
                            verified=verifier_ok,
                        )
                    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        record_degradation("amplifier_v2_cache_put", exc)
                # Capture as a STaR self-improvement training trace (internal bootstrap).
                try:
                    from core.brain.reasoning_self_improvement import get_reasoning_self_improvement

                    get_reasoning_self_improvement().record_win(
                        problem.objective,
                        problem.task_type,
                        answer=calibrated_answer,
                        confidence=confidence,
                        mode=mode.value,
                        verified=verifier_ok,
                    )
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation("amplifier_v2_self_improve", exc)
            elif not verifier_ok and not request.context.get("skip_precompute_enqueue"):
                # Verifier-dirty under the foreground budget — queue an idle deep retry
                # (off the critical path; the win lands in the cache for next time).
                try:
                    from core.brain.reasoning_precompute import get_precompute_queue

                    get_precompute_queue().enqueue(problem.objective, problem.task_type)
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation("amplifier_v2_precompute_enqueue", exc)

        # 10. record the episode so the next one is wiser.
        try:
            self._ensure_memory().record(
                task_type=problem.task_type,
                objective=problem.objective,
                passed=verifier_ok,
                verifier_issues=verifier_issues,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("amplifier_v2_record", exc)

        receipt = ReasoningReceipt(
            mode=mode.value,
            strategy_used=strategy,
            task_type=problem.task_type,
            num_candidates=n_cand,
            verifiers_run=[v for v in verifiers_run if v],
            valid_candidates=1 if verifier_ok else 0,
            winning_candidate_id=0,
            confidence=confidence,
            agreement=agreement,
            epistemic_status=calibration.overall.value,
            evidence_refs=evidence[:6],
            known_failures=verifier_issues[:6],
            budget_used={
                "samples": n_cand,
                "sample_budget": sample_budget,
                "time_s": round(time.monotonic() - start, 3),
                "time_budget_s": request.time_budget_s,
            },
            fallbacks_used=fallbacks,
            guards_applied=guards,
        )
        logger.info(
            "🧠 [AmplifyV2] mode=%s task=%s cands=%d verifier=%s conf=%.2f status=%s",
            mode.value, problem.task_type, n_cand, "PASS" if verifier_ok else "FAIL",
            confidence, calibration.overall.value,
        )
        return AmplifiedAnswer(
            answer=calibrated_answer,
            confidence=confidence,
            # "verified" is a claim to the user: only ok AND actually-checked
            # earns it. "The verifier had no objection to prose" does not.
            verified=verifier_ok and verifier_checked,
            calibrated=(calibration.downgraded > 0 or calibration.flagged_impossible > 0),
            receipt=receipt,
        )

    # ------------------------------------------------------------------ paths
    async def _shallow_path(self, problem, guard_text, sample_budget, deadline, mode, context, fallbacks):
        """FAST/NORMAL: sample N, verifier-filter, self-consistency."""
        candidates = await self._generate_candidates(problem, guard_text, sample_budget, deadline)
        if not candidates:
            fallbacks.append("no_candidates")
            return "", 0.0, None, 0, "none"
        if mode is ReasoningMode.FAST or len(candidates) == 1:
            answer = candidates[0]
            verdict = await self._verify(answer, problem, context)
            return answer, 1.0, verdict, len(candidates), "direct"
        # NORMAL — verifier-filtered self-consistency over the pool.
        from core.brain.reasoning_amplifier import amplify

        verdicts = await asyncio.gather(*[self._verify(c, problem, context) for c in candidates])
        clean = [c for c, v in zip(candidates, verdicts) if getattr(v, "ok", True)]
        # Harvest the NEGATIVE signal the SFT-only path discards: verified-correct vs
        # verified-wrong candidates for the SAME problem become sound DPO preference pairs
        # (RLVR data, the verifier is the reward). Best-effort; never affects this turn's answer.
        try:
            from core.learning.verifiable_preference_harness import (
                Attempt,
                get_verifiable_preference_harness,
            )

            get_verifiable_preference_harness().ingest(
                problem.objective,
                [
                    Attempt(
                        candidate=c,
                        verified=bool(getattr(v, "ok", False)),
                        checked=bool(getattr(v, "checked", False)),
                        confidence=float(getattr(v, "score", 0.0) or 0.0),
                    )
                    for c, v in zip(candidates, verdicts)
                ],
                domain=problem.task_type,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError) as exc:
            record_degradation("amplifier_v2_preference_harvest", exc, severity="debug")
        pool = clean or candidates
        result = await amplify(pool)
        winner = result.answer or pool[0]
        verdict = await self._verify(winner, problem, context)
        return winner, result.agreement, verdict, len(candidates), "self_consistency"

    async def _deep_path(self, problem, guard_text, sample_budget, deadline, mode, context, fallbacks):
        """DEEP/EXTREME/PROOF: adversarial courtroom, then optional sandbox repair.

        Code/math answers are *structured* (multi-line code, worked arithmetic), which
        the courtroom's single-line judge/simplifier extraction would mangle — and for
        these domains the real verifier is execution, not debate. So they take the
        answer-preserving self-consistency path plus sandbox repair instead.
        """
        if problem.task_type in {"code", "math"}:
            answer, agreement, verdict, n_cand, strategy = await self._shallow_path(
                problem, guard_text, max(3, sample_budget), deadline, ReasoningMode.NORMAL, context, fallbacks
            )
            if answer and time.monotonic() < deadline:
                repaired = await self._sandbox_repair(answer, problem)
                if repaired:
                    answer = repaired
                    fallbacks.append("sandbox_repaired")
                    verdict = await self._verify(answer, problem, context)
            return answer, agreement, verdict, n_cand, f"{strategy}+sandbox"

        evidence = list(problem.required_evidence) + list(context.get("evidence", []) or [])
        try:
            court = self._make_courtroom()
            n_cand = 3 if mode is ReasoningMode.DEEP else min(4, sample_budget)
            verdict_court = await self._with_deadline(
                court.deliberate(problem.objective, evidence=evidence, task_type=problem.task_type, n_candidates=n_cand),
                deadline,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, asyncio.TimeoutError) as exc:
            record_degradation("amplifier_v2_courtroom", exc)
            verdict_court = None

        if verdict_court is None or not getattr(verdict_court, "answer", ""):
            fallbacks.append("courtroom_fallback_to_self_consistency")
            return await self._shallow_path(problem, guard_text, max(3, sample_budget), deadline, ReasoningMode.NORMAL, context, fallbacks)

        answer = verdict_court.answer
        # Code/math: actually run it and self-correct if the courtroom answer is executable.
        if problem.task_type in {"code", "math"} and time.monotonic() < deadline:
            repaired = await self._sandbox_repair(answer, problem)
            if repaired:
                answer = repaired
                fallbacks.append("sandbox_repaired")

        verdict = await self._verify(answer, problem, context)
        agreement = float(getattr(verdict_court, "confidence", 0.7))
        return answer, agreement, verdict, len(getattr(verdict_court, "candidates", []) or [1]), "courtroom"

    # ----------------------------------------------------------- primitives
    async def _generate_candidates(self, problem: ProblemRepresentation, guard_text: str, n: int, deadline: float) -> list[str]:
        sys_block = self._build_prompt(problem, guard_text)

        # Batched lane first: N candidates in ONE decoding pass when the live
        # MLX client supports it. Falls back to serial sampling on any miss.
        if n >= 2:
            try:
                from core.brain.llm.batch_candidates import generate_candidates_batched

                remaining = max(10.0, deadline - time.monotonic())
                batched = await generate_candidates_batched(
                    sys_block, n, timeout_s=remaining
                )
                if batched:
                    return batched
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("amplifier_v2_batch_lane", exc, severity="debug")

        async def _one(i: int) -> str:
            if time.monotonic() >= deadline:
                return ""
            try:
                return str(await self._generate(sys_block, _TEMPS[i % len(_TEMPS)]) or "").strip()
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("amplifier_v2_generate", exc)
                return ""

        results = await asyncio.gather(*[_one(i) for i in range(max(1, n))])
        return [r for r in results if r]

    def _build_prompt(self, problem: ProblemRepresentation, guard_text: str) -> str:
        parts = [problem.objective]
        if problem.constraints:
            parts.append("Constraints: " + "; ".join(problem.constraints))
        if problem.required_evidence:
            parts.append("Use only this material:\n" + "\n".join(f"- {e}" for e in problem.required_evidence[:6]))
        if guard_text:
            parts.append(guard_text)
        if problem.verification_plan:
            parts.append("Your answer will be checked by: " + "; ".join(problem.verification_plan))
        return "\n\n".join(parts)

    async def _verify(self, candidate: str, problem: ProblemRepresentation, context: dict[str, Any]) -> Any:
        try:
            # Pass the objective so domain engines can derive a canonical exact answer
            # from the QUESTION (e.g. math: compute "123456 mod 7" and check it).
            ctx = {
                "required_evidence": problem.required_evidence,
                "objective": problem.objective,
                **(context or {}),
            }
            return await self._ensure_verifier().verify(candidate, task_type=problem.task_type, context=ctx)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("amplifier_v2_verify", exc)
            return None

    async def _sandbox_repair(self, answer: str, problem: ProblemRepresentation) -> str:
        from core.brain.verifiers.code_engine import extract_code_blocks

        blocks = extract_code_blocks(answer)
        if not blocks:
            return ""
        sandbox = self._ensure_sandbox()

        async def repair(code: str, traceback: str) -> str:
            prompt = (
                f"This script failed. Fix it and return only corrected Python.\n\n"
                f"Script:\n{code}\n\nError:\n{traceback}"
            )
            return str(await self._generate(prompt, 0.2) or "")

        result = await sandbox.run_with_self_correction(blocks[0], repair, max_rounds=2)
        if result.ok and result.final_code and result.final_code != blocks[0]:
            return answer.replace(blocks[0], result.final_code)
        return ""

    def _calibrate(self, answer: str, verdict: Any, evidence: list[str], verifier_ok: bool):
        gate = self._ensure_calibration()
        report = gate.assess(
            answer,
            verification=verdict,
            evidence=evidence,
            tool_verified=bool(verifier_ok and getattr(verdict, "checked", False)),
        )
        return report.calibrated_answer or answer, report

    @staticmethod
    async def _with_deadline(coro: Awaitable[Any], deadline: float) -> Any:
        timeout = max(0.5, deadline - time.monotonic())
        return await asyncio.wait_for(coro, timeout=timeout)


def build_amplifier_v2(generate: GenerateFn, **kw: Any) -> ReasoningAmplifierV2:
    """Factory used by the live wiring in reasoning_strategies/cognitive_engine."""
    return ReasoningAmplifierV2(generate, **kw)


# Imperative environment/desktop/tool actions. These are things to *do*, not
# questions to *answer* — the amplifier (which re-derives and verifies an answer)
# must never hijack them away from the tool/action path. Deliberately excludes
# reasoning verbs (write/compute/explain/solve/where/how/what) so "write a function"
# and "compute the total" still amplify.
_ACTION_VERB = (
    r"open|close|click|tap|press|send|email|e-?mail|text|call|dial|message|dm|"
    r"navigate|browse|launch|quit|switch|type|paste|scroll|swipe|drag|drop|"
    r"download|upload|install|uninstall|delete|remove|move|rename|copy|save|"
    r"screenshot|capture|play|pause|skip|mute|unmute|turn|enable|disable|toggle|"
    r"schedule|remind|book|order|buy|purchase|pay|post|tweet|share|like|follow|"
    r"subscribe|unsubscribe|reply|forward|attach|print|scan|record|stream|cast|"
    r"set up|set a|set an|set the|start|stop|restart|reboot|shutdown|lock|unlock"
)
_ACTION_RE = re.compile(rf"^\s*(?:please\s+|can you\s+|could you\s+|go\s+|now\s+|hey,?\s+)*(?:{_ACTION_VERB})\b", re.IGNORECASE)
_GO_TO_RE = re.compile(r"^\s*(?:please\s+)?go to\b", re.IGNORECASE)


def is_action_request(objective: str) -> bool:
    """True if the turn is an imperative action/tool command rather than a question.

    Applied generally: 'open 3 tabs', 'send the email', 'click the button', 'play
    the next song' are actions to dispatch, not problems to reason about — they must
    bypass the reasoning amplifier entirely so the tool/desktop path owns them.
    """
    q = str(objective or "").strip()
    if not q:
        return False
    if _GO_TO_RE.match(q) or _ACTION_RE.match(q):
        return True
    return False


def is_amplifiable(objective: str) -> str | None:
    """Return the task_type to amplify for a turn, or None for ordinary chat/actions.

    Used by the live response lanes to decide whether a turn earns the
    verifier-backed amplifier. Conservative: only verifiable hard *reasoning*
    questions qualify. Imperative actions and casual conversation are excluded so
    the tool path owns commands and chat is never slowed down.
    """
    q = str(objective or "").strip()
    if len(q) < 8:
        return None
    # Actions are dispatched, not answered — never amplify them.
    if is_action_request(q):
        return None
    task_type = classify_task_type(q)
    if task_type in {"code", "math", "repo_audit", "architecture"}:
        return task_type
    return None


async def amplify_turn(
    objective: str,
    generate: GenerateFn,
    *,
    task_type: str | None = None,
    evidence: list[str] | None = None,
    risk_level: str = "normal",
    time_budget_s: float = 30.0,
    extra_context: dict[str, Any] | None = None,
    **kw: Any,
) -> AmplifiedAnswer:
    """One-call entrypoint for the live response path.

    Builds an amplifier wired to the real organs (verifiers, sandbox, calibration,
    memory, evidence provider, substrate) and runs the full pipeline for a turn.
    ``extra_context`` is merged into the request context (e.g. ``skip_cache`` or
    ``skip_precompute_enqueue`` for the idle pre-compute path). ``escalate_generate``
    may be passed through ``**kw`` to enable the deep-tier verifier-of-last-resort.
    """
    tt = task_type or classify_task_type(objective)
    amplifier = ReasoningAmplifierV2(generate, **kw)
    context: dict[str, Any] = {"evidence": list(evidence or [])}
    if extra_context:
        context.update(extra_context)
    request = AmplificationRequest(
        objective=objective,
        task_type=tt,
        risk_level=risk_level,
        time_budget_s=time_budget_s,
        required_evidence=list(evidence or []),
        context=context,
    )
    return await amplifier.amplify(request)
