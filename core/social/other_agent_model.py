"""Other-agent state estimation — a live, first-class theory-of-mind estimate.

The codebase already *stores facts about people* (SocialMemory: milestones, relationship
depth), *reframes topics* (SocialImagination: personal trouble → public issue), and
classifies a single message's subtext (SocialCognitionLayer). What was missing is the thing
the critique's "embodied social cognition" item names directly: a live, continuously-updated
estimate of *what another agent believes, wants, and feels right now* — held as first-class
state with explicit uncertainty, updated from observations, and relaxing toward a prior when
no fresh evidence arrives.

This is not pretending to feel. It is social *state estimation*: a per-agent recursive,
Bayesian-flavored filter over three coupled channels —

    affect   transient feelings (frustration, fatigue, urgency, uncertainty, satisfaction,
             engagement) that fade toward a baseline between observations
    goals    what the agent is currently trying to get (activation decays as it goes stale)
    beliefs  what the agent currently holds true — including beliefs *about Aura* (do they
             think her capable, trustworthy, or suspect she's only role-playing)

Each scalar is a :class:`Signal`: a value, a confidence, and a half-life it relaxes toward a
baseline over. New evidence fuses with the current estimate by precision (confidence) weight,
so a low-confidence estimate moves strongly toward fresh observation while a well-corroborated
one is sticky. Confidence accumulates with corroboration and decays with time, so the model is
honest about going stale: when confidence is low the recommendation is to *ask, not assume*.

The estimate exposes itself to the hierarchical-agency ladder as social signals — a frustrated,
low-trust agent facing an irreversible step raises ``value_conflict`` so GOVERNANCE weighs in;
a clearly active goal feeds STRATEGIC; not knowing the agent well raises ``uncertainty`` so the
ladder is cautious. That is the critique's "what the GOVERNANCE/STRATEGIC tiers consult for
social situations."
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Social.OtherAgentModel")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# ── one scalar belief, with confidence and time-decay toward a baseline ──────


@dataclass
class Signal:
    """A scalar estimate about an agent: a value, our confidence in it, and how it relaxes.

    With no fresh evidence the value drifts toward ``baseline`` and confidence toward 0 on a
    ``half_life_s`` schedule — the estimate forgets gracefully rather than asserting a stale
    feeling forever.
    """

    value: float
    confidence: float
    baseline: float
    half_life_s: float
    updated_at: float = field(default_factory=time.time)

    def decayed(self, now: float) -> tuple[float, float]:
        """Value relaxed toward baseline and confidence faded toward 0, evaluated at ``now``."""
        dt = now - self.updated_at
        if dt <= 0 or self.half_life_s <= 0:
            return self.value, self.confidence
        frac = 0.5 ** (dt / self.half_life_s)
        value = self.baseline + (self.value - self.baseline) * frac
        return value, self.confidence * frac

    def observe(self, observed: float, strength: float, now: float) -> None:
        """Fuse a new observation by precision weight, then stamp the time.

        The current estimate (decayed to ``now``) acts as a prior with precision == its
        confidence; the observation carries precision == ``strength``. The posterior value is
        the precision-weighted mean, and confidence accumulates toward saturation. So a fresh,
        uncertain estimate snaps to the observation, while a well-corroborated one barely moves.
        """
        observed = _clamp(observed)
        strength = _clamp(strength)
        v, c = self.decayed(now)
        total = c + strength
        self.value = observed if total <= 1e-9 else _clamp((v * c + observed * strength) / total)
        self.confidence = _clamp(c + strength * (1.0 - c))
        self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "baseline": self.baseline,
            "half_life_s": self.half_life_s,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, baseline: float, half_life_s: float) -> Signal:
        return cls(
            value=float(d.get("value", baseline)),
            confidence=float(d.get("confidence", 0.0)),
            baseline=float(d.get("baseline", baseline)),
            half_life_s=float(d.get("half_life_s", half_life_s)),
            updated_at=float(d.get("updated_at", time.time())),
        )


# Affect channels: (baseline, half_life_s). Feelings fade; fatigue lingers longest.
_AFFECT_SPEC: dict[str, tuple[float, float]] = {
    "frustration": (0.10, 600.0),
    "fatigue": (0.20, 5400.0),
    "urgency": (0.20, 900.0),
    "uncertainty": (0.30, 1200.0),
    "satisfaction": (0.55, 1800.0),
    "engagement": (0.50, 1500.0),
}

# Beliefs the agent holds *about Aura*: (baseline, half_life_s). These move slowly.
_AURA_BELIEF_SPEC: dict[str, tuple[float, float]] = {
    "aura_capable": (0.50, 604800.0),
    "aura_trustworthy": (0.50, 604800.0),
    "aura_roleplaying": (0.30, 604800.0),
}

_GOAL_HALF_LIFE_S = 1800.0  # an unrefreshed goal decays out of "active" over ~30 min


def _word_re(words: list[str]) -> re.Pattern[str]:
    # Match whole words/phrases, case-insensitive; phrases allow flexible internal whitespace.
    parts = [re.escape(w).replace(r"\ ", r"\s+") for w in words]
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


_CUES: dict[str, re.Pattern[str]] = {
    "frustration": _word_re([
        "frustrated", "frustrating", "annoyed", "annoying", "angry", "ugh", "argh", "wtf",
        "still broken", "still not working", "not working", "doesn't work", "does not work",
        "come on", "seriously", "again", "stuck", "fed up", "ridiculous", "hate this",
    ]),
    "fatigue": _word_re([
        "tired", "exhausted", "exhausting", "long day", "burned out", "burnt out", "sleepy",
        "drained", "worn out", "can't think", "cant think", "no energy", "wiped", "knackered",
    ]),
    "urgency": _word_re([
        "asap", "urgent", "urgently", "immediately", "right now", "hurry", "quickly",
        "deadline", "time-sensitive", "time sensitive", "need this now", "fast", "rush",
    ]),
    "uncertainty": _word_re([
        "not sure", "unsure", "no idea", "don't know", "dont know", "dunno", "confused",
        "don't understand", "dont understand", "maybe", "i think", "possibly", "might be",
        "not certain", "no clue",
    ]),
    "satisfaction_pos": _word_re([
        "thanks", "thank you", "perfect", "great", "awesome", "amazing", "nice", "love it",
        "love this", "exactly", "works now", "it works", "fixed", "beautiful", "brilliant",
    ]),
    "satisfaction_neg": _word_re([
        "wrong", "that's not", "thats not", "not what i", "disappointing", "useless",
        "terrible", "this is bad", "no good", "not helpful",
    ]),
    "trust_pos": _word_re(["i trust you", "you got this", "i believe you", "good call", "well done"]),
    "trust_neg": _word_re([
        "i don't trust", "i dont trust", "are you sure", "prove it", "i don't believe",
        "i dont believe", "doubt", "suspicious", "lying",
    ]),
    "capable_neg": _word_re([
        "can you actually", "you can't", "you cant", "you're not able", "youre not able",
        "you failed", "you keep failing", "incompetent", "you don't get it", "you dont get it",
    ]),
    "roleplay": _word_re([
        "roleplay", "role play", "role-play", "fake", "pretend", "pretending", "just an ai",
        "you're not real", "youre not real", "not really conscious", "make believe",
    ]),
}

_REQUEST_CUES = _word_re([
    "can you", "could you", "would you", "please", "i want", "i need", "i'd like", "id like",
    "let's", "lets", "make", "build", "fix", "add", "create", "write", "implement", "help me",
    "set up", "show me", "give me", "find", "check",
])


@dataclass
class SocialRecommendation:
    """How to act toward an agent, derived from the live estimate — with honesty about doubt."""

    agent_id: str
    should_ask: bool          # confidence too low / agent unsure → clarify, don't assume
    be_concise: bool          # fatigue or urgency high → keep it short
    offer_reassurance: bool   # frustration / low trust / low satisfaction → reassure first
    slow_down: bool           # social rupture risk high → don't barrel ahead
    restraint_level: float    # [0,1] how much to hold back
    tone: str
    confidence: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "should_ask": self.should_ask,
            "be_concise": self.be_concise,
            "offer_reassurance": self.offer_reassurance,
            "slow_down": self.slow_down,
            "restraint_level": round(self.restraint_level, 3),
            "tone": self.tone,
            "confidence": round(self.confidence, 3),
            "reasons": self.reasons,
        }


@dataclass
class AgentStateEstimate:
    """A snapshot of one agent's estimated mental state, decayed to the moment it was taken."""

    agent_id: str
    affect: dict[str, float]
    affect_confidence: dict[str, float]
    goals: list[dict[str, Any]]
    beliefs_about_aura: dict[str, float]
    overall_confidence: float
    social_rupture_risk: float
    observations: int
    at: float
    freshness_s: float = 0.0
    evidence_digest: str = "none"
    identity_verified: bool = False
    inference_limitations: tuple[str, ...] = (
        "affect_is_inferred_not_observed_fact",
        "culture_and_demographics_not_inferred",
        "identity_not_biometrically_verified",
        "clarify_when_confidence_is_low",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "affect": {k: round(v, 3) for k, v in self.affect.items()},
            "affect_confidence": {k: round(v, 3) for k, v in self.affect_confidence.items()},
            "goals": self.goals,
            "beliefs_about_aura": {k: round(v, 3) for k, v in self.beliefs_about_aura.items()},
            "overall_confidence": round(self.overall_confidence, 3),
            "social_rupture_risk": round(self.social_rupture_risk, 3),
            "observations": self.observations,
            "at": self.at,
            "freshness_s": round(self.freshness_s, 3),
            "evidence_digest": self.evidence_digest,
            "identity_verified": self.identity_verified,
            "inference_limitations": list(self.inference_limitations),
        }


class _AgentModel:
    """Mutable per-agent estimate: the affect/belief signals plus a small goal table."""

    def __init__(self) -> None:
        now = time.time()
        self.affect = {n: Signal(b, 0.0, b, hl, now) for n, (b, hl) in _AFFECT_SPEC.items()}
        self.aura_beliefs = {n: Signal(b, 0.0, b, hl, now) for n, (b, hl) in _AURA_BELIEF_SPEC.items()}
        self.goals: dict[str, Signal] = {}
        self.observations = 0
        self.last_seen = now
        self.last_evidence_digest = "none"
        self.last_response_feedback = False

    def to_dict(self, *, include_ephemeral_goals: bool = True) -> dict[str, Any]:
        return {
            "affect": {k: v.to_dict() for k, v in self.affect.items()},
            "aura_beliefs": {k: v.to_dict() for k, v in self.aura_beliefs.items()},
            "goals": (
                {k: v.to_dict() for k, v in self.goals.items()}
                if include_ephemeral_goals
                else {}
            ),
            "observations": self.observations,
            "last_seen": self.last_seen,
            "last_evidence_digest": self.last_evidence_digest,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _AgentModel:
        m = cls()
        for n, (b, hl) in _AFFECT_SPEC.items():
            if n in d.get("affect", {}):
                m.affect[n] = Signal.from_dict(d["affect"][n], baseline=b, half_life_s=hl)
        for n, (b, hl) in _AURA_BELIEF_SPEC.items():
            if n in d.get("aura_beliefs", {}):
                m.aura_beliefs[n] = Signal.from_dict(d["aura_beliefs"][n], baseline=b, half_life_s=hl)
        for text, sig in d.get("goals", {}).items():
            m.goals[text] = Signal.from_dict(sig, baseline=0.0, half_life_s=_GOAL_HALF_LIFE_S)
        m.observations = int(d.get("observations", 0))
        m.last_seen = float(d.get("last_seen", time.time()))
        m.last_evidence_digest = str(d.get("last_evidence_digest") or "none")[:128]
        return m


class OtherAgentStateEstimator:
    """Maintains a live, decaying theory-of-mind estimate for each known agent."""

    def __init__(
        self,
        storage_path: Path | None = None,
        *,
        autosave: bool = True,
        min_save_interval_s: float = 5.0,
        max_goals: int = 8,
        response_feedback_window_s: float = 30 * 60,
    ) -> None:
        if storage_path is None:
            try:
                from core.config import config
                storage_path = config.paths.memory_dir / "other_agent_models.json"
            except (ImportError, AttributeError, RuntimeError):
                storage_path = Path.home() / ".aura" / "data" / "memory" / "other_agent_models.json"
        self._path = Path(storage_path)
        self._autosave = autosave
        self._min_save_interval = min_save_interval_s
        self._max_goals = max_goals
        self._response_feedback_window_s = max(0.0, float(response_feedback_window_s))
        self._lock = threading.RLock()
        self._models: dict[str, _AgentModel] = {}
        self._last_save = 0.0
        self._active_agent_id = ""
        self._pending_responses: dict[str, tuple[str, float, int]] = {}
        self._load()
        logger.info("OtherAgentStateEstimator initialized (%d agents).", len(self._models))

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for agent_id, model in raw.get("agents", {}).items():
                if isinstance(model, dict):
                    self._models[agent_id] = _AgentModel.from_dict(model)
        except (OSError, ValueError) as exc:
            record_degradation("other_agent_model", exc)
            logger.warning("OtherAgentStateEstimator load failed: %s", exc)

    def save(self) -> None:
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with self._lock:
                payload = {
                    "schema_version": 2,
                    "privacy": {
                        "raw_messages_persisted": False,
                        "raw_goals_persisted": False,
                        "content_digests_only": True,
                    },
                    "agents": {
                        aid: model.to_dict(include_ephemeral_goals=False)
                        for aid, model in self._models.items()
                    },
                }
            with local_internal_governed_scope(
                "other_agent_model.save",
                domain="file_write",
            ):
                gateway = get_file_write_gateway()
                gateway.ensure_directory(
                    self._path.parent,
                    source="other_agent_model.save",
                )
                gateway.write_text(
                    self._path,
                    json.dumps(payload, indent=2, sort_keys=True),
                    source="other_agent_model.save",
                )
            self._last_save = time.time()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("other_agent_model", exc)
            logger.error("OtherAgentStateEstimator save failed: %s", exc)

    def _maybe_save(self) -> None:
        # Debounced so 10 Hz telemetry can't thrash the disk; flush at most every interval.
        if self._autosave and (time.time() - self._last_save) >= self._min_save_interval:
            self.save()

    def save_if_due(self) -> bool:
        """Persist a content-free snapshot when the debounce interval elapsed."""
        before = self._last_save
        self._maybe_save()
        return self._last_save > before

    def _model(self, agent_id: str) -> _AgentModel:
        m = self._models.get(agent_id)
        if m is None:
            m = _AgentModel()
            self._models[agent_id] = m
        return m

    # ── observations ──────────────────────────────────────────────────────

    def observe_message(
        self,
        agent_id: str,
        text: str,
        *,
        latency_s: float | None = None,
        hour: int | None = None,
        now: float | None = None,
        persist: bool = True,
        response_context: bool | None = None,
    ) -> AgentStateEstimate:
        """Update the estimate from one message: linguistic cues + timing + time-of-day.

        ``latency_s`` is the gap since this agent's previous message (rapid terse bursts read
        as rising urgency/frustration; a long gap just lets the estimate decay). ``hour`` is
        the local hour 0-23 (late night nudges fatigue).
        """
        text = str(text or "")
        now = time.time() if now is None else now
        lower = text.lower()
        words = len(text.split())

        with self._lock:
            m = self._model(agent_id)
            self._active_agent_id = agent_id
            pending_response = self._pending_responses.pop(agent_id, None)
            if response_context is None:
                response_age_s = (
                    now - pending_response[1]
                    if pending_response is not None
                    else float("inf")
                )
                response_context = (
                    pending_response is not None
                    and 0.0 <= response_age_s <= self._response_feedback_window_s
                )
            m.last_response_feedback = bool(response_context)
            m.observations += 1
            m.last_seen = now
            m.last_evidence_digest = hashlib.sha256(
                f"{agent_id}\n{text}".encode("utf-8", errors="replace")
            ).hexdigest()

            # Linguistic affect cues. Each match group nudges its channel; strength grows with
            # the number of distinct hits, capped so one message can't fully overwrite a belief.
            def _strength(pat: re.Pattern[str]) -> float:
                hits = len(pat.findall(lower))
                return 0.0 if hits == 0 else min(0.6, 0.25 + 0.12 * (hits - 1))

            for chan in ("frustration", "fatigue", "urgency", "uncertainty"):
                st = _strength(_CUES[chan])
                if st > 0:
                    m.affect[chan].observe(0.85, st, now)

            pos, neg = _strength(_CUES["satisfaction_pos"]), _strength(_CUES["satisfaction_neg"])
            if pos > 0:
                m.affect["satisfaction"].observe(0.85, pos, now)
                if response_context:
                    m.aura_beliefs["aura_capable"].observe(0.75, pos * 0.6, now)
            if neg > 0:
                m.affect["satisfaction"].observe(0.15, neg, now)
                m.affect["frustration"].observe(0.7, neg * 0.6, now)

            # Beliefs about Aura herself — the heart of modeling *their* model of *her*.
            for cue, belief, target in (
                ("trust_pos", "aura_trustworthy", 0.85),
                ("trust_neg", "aura_trustworthy", 0.15),
                ("capable_neg", "aura_capable", 0.2),
                ("roleplay", "aura_roleplaying", 0.85),
            ):
                st = _strength(_CUES[cue])
                if st > 0:
                    m.aura_beliefs[belief].observe(target, st, now)

            # Any message is a sign of engagement/presence.
            m.affect["engagement"].observe(0.8, 0.3, now)

            # Timing: rapid, terse follow-ups read as mounting urgency/frustration.
            if latency_s is not None and latency_s < 45 and words <= 12 and m.observations > 1:
                m.affect["urgency"].observe(0.7, 0.2, now)
                m.affect["frustration"].observe(0.55, 0.15, now)

            # Time-of-day: late night raises the fatigue prior.
            if hour is not None and (hour >= 23 or hour < 5):
                m.affect["fatigue"].observe(0.7, 0.2, now)

            # Goal extraction: a request-shaped message activates a goal.
            if _REQUEST_CUES.search(lower):
                goal_text = self._goal_key(text)
                if goal_text:
                    sig = m.goals.get(goal_text)
                    if sig is None:
                        sig = Signal(0.0, 0.0, 0.0, _GOAL_HALF_LIFE_S, now)
                        m.goals[goal_text] = sig
                    sig.observe(0.9, 0.5, now)
                    self._prune_goals(m, now)

            if persist:
                self._maybe_save()
            return self.estimate(agent_id, now)

    def record_response(
        self,
        agent_id: str,
        response_text: str,
        *,
        now: float | None = None,
    ) -> str:
        """Pair the next feedback cue with a response without retaining its text."""
        observed_at = time.time() if now is None else float(now)
        digest = hashlib.sha256(
            f"{agent_id}\n{response_text}".encode("utf-8", errors="replace")
        ).hexdigest()
        with self._lock:
            self._active_agent_id = agent_id
            self._pending_responses[agent_id] = (
                digest,
                observed_at,
                len(str(response_text or "").split()),
            )
        return digest

    def observe_signal(self, agent_id: str, *, now: float | None = None, **signals: float) -> None:
        """Fold perceptual-pump social signals into affect (presence/affiliation/threat/...).

        Accepts the event-mapping signals the phenomenal pump already produces: ``presence``,
        ``affiliation``, ``control_gain``, ``novelty``, ``threat`` — each in [0,1].
        """
        now = time.time() if now is None else now
        with self._lock:
            m = self._model(agent_id)
            if "presence" in signals:
                m.affect["engagement"].observe(_clamp(signals["presence"]), 0.3, now)
            if "affiliation" in signals:
                aff = _clamp(signals["affiliation"])
                m.affect["satisfaction"].observe(aff, 0.25, now)
                m.aura_beliefs["aura_trustworthy"].observe(aff, 0.2, now)
            if "threat" in signals:
                thr = _clamp(signals["threat"])
                m.affect["frustration"].observe(thr, 0.25, now)
                m.affect["urgency"].observe(thr, 0.2, now)
            if "control_gain" in signals:
                # Low control on the agent's side reads as higher uncertainty for them.
                m.affect["uncertainty"].observe(1.0 - _clamp(signals["control_gain"]), 0.2, now)
            if "novelty" in signals:
                m.affect["engagement"].observe(_clamp(signals["novelty"]), 0.2, now)
            self._maybe_save()

    def observe_outcome(self, agent_id: str, *, success: bool, weight: float = 0.4,
                        now: float | None = None) -> None:
        """An action taken for the agent landed (or didn't): adjust satisfaction and trust."""
        now = time.time() if now is None else now
        w = _clamp(weight)
        with self._lock:
            m = self._model(agent_id)
            if success:
                m.affect["satisfaction"].observe(0.85, w, now)
                m.affect["frustration"].observe(0.1, w * 0.6, now)
                m.aura_beliefs["aura_capable"].observe(0.8, w * 0.6, now)
                m.aura_beliefs["aura_trustworthy"].observe(0.75, w * 0.4, now)
            else:
                m.affect["satisfaction"].observe(0.25, w, now)
                m.affect["frustration"].observe(0.7, w * 0.6, now)
                m.aura_beliefs["aura_capable"].observe(0.3, w * 0.5, now)
            self._maybe_save()

    # ── readout ───────────────────────────────────────────────────────────

    def estimate(self, agent_id: str, now: float | None = None) -> AgentStateEstimate:
        """A snapshot of the agent's estimated state, decayed to ``now``."""
        now = time.time() if now is None else now
        with self._lock:
            m = self._models.get(agent_id)
            if m is None:
                return self._empty_estimate(agent_id, now)
            affect, affect_conf = {}, {}
            for name, sig in m.affect.items():
                v, c = sig.decayed(now)
                affect[name], affect_conf[name] = v, c
            beliefs = {name: sig.decayed(now)[0] for name, sig in m.aura_beliefs.items()}
            goals: list[dict[str, str | float]] = []
            for text, sig in m.goals.items():
                act, _ = sig.decayed(now)
                if act >= 0.15:
                    goals.append({"goal": text, "activation": round(act, 3)})
            goals.sort(key=lambda goal: float(goal["activation"]), reverse=True)
            # Confidence reflects the channels we actually have evidence on — averaging in
            # never-observed channels would structurally cap a strong, focused read too low.
            observed = [c for c in affect_conf.values() if c > 0.0]
            overall_conf = sum(observed) / len(observed) if observed else 0.0
            rupture = self._rupture_risk(affect, beliefs)
            return AgentStateEstimate(
                agent_id=agent_id,
                affect=affect,
                affect_confidence=affect_conf,
                goals=goals[:5],
                beliefs_about_aura=beliefs,
                overall_confidence=overall_conf,
                social_rupture_risk=rupture,
                observations=m.observations,
                at=now,
                freshness_s=max(0.0, now - m.last_seen),
                evidence_digest=m.last_evidence_digest,
            )

    def _empty_estimate(self, agent_id: str, now: float) -> AgentStateEstimate:
        affect = {n: b for n, (b, _) in _AFFECT_SPEC.items()}
        beliefs = {n: b for n, (b, _) in _AURA_BELIEF_SPEC.items()}
        return AgentStateEstimate(
            agent_id=agent_id, affect=affect, affect_confidence={n: 0.0 for n in affect},
            goals=[], beliefs_about_aura=beliefs, overall_confidence=0.0,
            social_rupture_risk=self._rupture_risk(affect, beliefs), observations=0, at=now,
            freshness_s=0.0, evidence_digest="none",
        )

    @staticmethod
    def _rupture_risk(affect: dict[str, float], beliefs: dict[str, float]) -> float:
        """How close the relationship is to a social rupture, from frustration/satisfaction/trust."""
        return _clamp(
            0.45 * affect.get("frustration", 0.0)
            + 0.30 * (1.0 - affect.get("satisfaction", 0.5))
            + 0.25 * (1.0 - beliefs.get("aura_trustworthy", 0.5))
        )

    def recommendation(self, agent_id: str, now: float | None = None) -> SocialRecommendation:
        """Translate the estimate into how to act — defaulting to 'ask' when we don't know."""
        est = self.estimate(agent_id, now)
        a, b = est.affect, est.beliefs_about_aura
        reasons: list[str] = []

        should_ask = est.overall_confidence < 0.35 or a["uncertainty"] > 0.6
        if est.overall_confidence < 0.35:
            reasons.append("low confidence in agent state → clarify rather than assume")
        if a["uncertainty"] > 0.6:
            reasons.append("agent seems unsure → confirm intent")

        be_concise = a["fatigue"] > 0.55 or a["urgency"] > 0.6
        if be_concise:
            reasons.append("fatigue/urgency high → keep it short")

        offer_reassurance = a["frustration"] > 0.5 or a["satisfaction"] < 0.4 or b["aura_trustworthy"] < 0.4
        if offer_reassurance:
            reasons.append("frustration/low-trust/low-satisfaction → reassure first")

        slow_down = est.social_rupture_risk > 0.55
        if slow_down:
            reasons.append("social rupture risk → slow down, don't barrel ahead")

        restraint = _clamp(0.35 + 0.4 * a["frustration"] + 0.25 * (1.0 - b["aura_trustworthy"]))

        if est.social_rupture_risk > 0.6:
            tone = "repair"
        elif a["frustration"] > 0.5:
            tone = "calm_direct"
        elif a["satisfaction"] > 0.7 and a["engagement"] > 0.6:
            tone = "collaborative"
        elif a["fatigue"] > 0.55:
            tone = "gentle_brief"
        else:
            tone = "neutral"

        return SocialRecommendation(
            agent_id=agent_id, should_ask=should_ask, be_concise=be_concise,
            offer_reassurance=offer_reassurance, slow_down=slow_down,
            restraint_level=restraint, tone=tone, confidence=est.overall_confidence,
            reasons=reasons,
        )

    @property
    def active_agent_id(self) -> str:
        with self._lock:
            return self._active_agent_id

    def cognitive_snapshot(
        self,
        agent_id: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Privacy-bounded social state for cognition, planning, and repair."""
        resolved_agent = str(agent_id or self.active_agent_id or "unknown")[:160]
        estimate = self.estimate(resolved_agent, now)
        recommendation = self.recommendation(resolved_agent, now)
        constraints: list[str] = []
        if recommendation.should_ask:
            constraints.append("clarify material ambiguity instead of assuming user state")
        if recommendation.be_concise:
            constraints.append("prefer a concise response while urgency or fatigue may be high")
        if recommendation.slow_down:
            constraints.append("slow consequential action and prioritize relationship repair")
        if recommendation.offer_reassurance:
            constraints.append("acknowledge the concrete failure or concern without performative reassurance")
        careful_forecast = self.forecast_social_consequence(
            resolved_agent,
            warmth=0.75,
            directness=0.55,
            reliability=0.9,
            fulfills_expectation=0.6,
            now=now,
        )
        blunt_forecast = self.forecast_social_consequence(
            resolved_agent,
            warmth=0.15,
            directness=0.9,
            reliability=0.35,
            fulfills_expectation=-0.4,
            now=now,
        )
        with self._lock:
            active_model = self._models.get(resolved_agent)
            response_feedback_context = bool(
                active_model and active_model.last_response_feedback
            )
        return {
            "schema_version": 1,
            "agent_id": resolved_agent,
            "identity_verified": estimate.identity_verified,
            "confidence": round(estimate.overall_confidence, 4),
            "freshness_s": round(estimate.freshness_s, 3),
            "observations": estimate.observations,
            "response_feedback_context": response_feedback_context,
            "evidence_digest": estimate.evidence_digest,
            "affect_hypotheses": {
                name: {
                    "value": round(value, 4),
                    "confidence": round(estimate.affect_confidence.get(name, 0.0), 4),
                }
                for name, value in estimate.affect.items()
            },
            "likely_goals": list(estimate.goals[:5]),
            "beliefs_about_aura": {
                key: round(value, 4)
                for key, value in estimate.beliefs_about_aura.items()
            },
            "social_rupture_risk": round(estimate.social_rupture_risk, 4),
            "recommendation": recommendation.to_dict(),
            "planning_constraints": constraints[:8],
            "predicted_impacts": {
                "careful_reliable_response": careful_forecast,
                "blunt_unverified_response": blunt_forecast,
                "assumption": "generic response properties, not a claim about the user's reaction",
            },
            "inference_limitations": list(estimate.inference_limitations),
            "culture": "unknown_not_inferred",
            "power_context": "operator_has_control",
            "privacy": {
                "raw_messages_retained": False,
                "raw_goals_persisted": False,
                "response_text_retained": False,
            },
        }

    # ── social intuition: forward-simulate an action's social consequence ──

    def forecast_social_consequence(
        self,
        agent_id: str,
        *,
        warmth: float = 0.5,
        directness: float = 0.5,
        reliability: float = 0.5,
        fulfills_expectation: float = 0.0,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Forward-simulate how a candidate action would land socially.

        Social *intuition*, not just state estimation: given the agent's current read and an
        action's social properties (warmth, directness, reliability, whether it meets or
        violates what they expect), predict the trust / rupture / satisfaction deltas and a
        recommendation. Grounded in the live estimate — a blunt action lands far worse on an
        already-frustrated agent than a calm one.
        """
        est = self.estimate(agent_id, now)
        frustration = est.affect.get("frustration", 0.3)
        warmth = _clamp(warmth)
        directness = _clamp(directness)
        reliability = _clamp(reliability)
        fulfills = _clamp(fulfills_expectation, -1.0, 1.0)

        trust_delta = _clamp(0.3 * reliability + 0.2 * max(0.0, fulfills) - 0.1 * (1.0 - warmth), -1.0, 1.0)
        if fulfills < 0:
            trust_delta = _clamp(trust_delta + 0.3 * fulfills, -1.0, 1.0)  # breaking expectation costs trust
        # bluntness hurts more when they're already frustrated
        rupture_delta = _clamp(
            (1.0 - warmth) * directness * (0.4 + 0.4 * frustration)
            - 0.2 * reliability
            + 0.3 * max(0.0, -fulfills),
            -1.0, 1.0,
        )
        satisfaction_delta = _clamp(0.3 * fulfills + 0.2 * warmth - 0.15 * (1.0 - reliability), -1.0, 1.0)

        projected_rupture = _clamp(est.social_rupture_risk + rupture_delta)
        if projected_rupture >= 0.6:
            rec = "repair_first" if est.social_rupture_risk >= 0.4 else "soften_before_acting"
        elif trust_delta > 0.15:
            rec = "proceed"
        else:
            rec = "proceed_with_care"
        return {
            "agent_id": agent_id,
            "trust_delta": round(trust_delta, 3),
            "rupture_delta": round(rupture_delta, 3),
            "satisfaction_delta": round(satisfaction_delta, 3),
            "projected_rupture_risk": round(projected_rupture, 3),
            "recommendation": rec,
            "confidence": round(est.overall_confidence, 3),
        }

    # ── agency-ladder seam ────────────────────────────────────────────────

    def social_signals(self, agent_id: str, now: float | None = None) -> dict[str, float]:
        """Estimate projected onto hierarchical-agency :class:`Situation` signal fields.

        - ``value_conflict``: social rupture risk (gated by how sure we are) → GOVERNANCE
        - ``uncertainty``:    the agent's own uncertainty *or* our not-knowing-them → SCIENTIFIC
        - ``goal_horizon``:   strongest active goal's activation → STRATEGIC
        """
        est = self.estimate(agent_id, now)
        conf = est.overall_confidence
        rupture = est.social_rupture_risk * (0.5 + 0.5 * conf)
        their_unc = est.affect.get("uncertainty", 0.3)
        not_knowing = 1.0 - conf
        top_goal = est.goals[0]["activation"] if est.goals else 0.0
        return {
            "value_conflict": _clamp(rupture),
            "uncertainty": _clamp(max(their_unc, 0.7 * not_knowing)),
            "goal_horizon": _clamp(0.6 * top_goal),
        }

    def social_situation(self, agent_id: str, description: str, base: Any = None,
                         now: float | None = None) -> Any:
        """Build (or augment) a :class:`Situation` for the agency ladder from the estimate.

        Returns a ``Situation`` with social signals folded in and the estimate attached in
        ``context`` so the GOVERNANCE/STRATEGIC handlers can consult it. Best-effort: if the
        agency module isn't importable, returns ``base`` unchanged (or ``None``).
        """
        sig = self.social_signals(agent_id, now)
        try:
            from core.agency.hierarchical_agency import Situation
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("other_agent_model", exc, severity="debug")
            return base
        s = base or Situation(description)
        s.value_conflict = max(getattr(s, "value_conflict", 0.0), sig["value_conflict"])
        s.uncertainty = max(getattr(s, "uncertainty", 0.0), sig["uncertainty"])
        s.goal_horizon = max(getattr(s, "goal_horizon", 0.0), sig["goal_horizon"])
        s.context = dict(getattr(s, "context", {}) or {})
        s.context["agent_id"] = agent_id
        s.context["social"] = self.estimate(agent_id, now).to_dict()
        return s

    # ── helpers ───────────────────────────────────────────────────────────

    def _goal_key(self, text: str) -> str:
        cleaned = " ".join(str(text or "").strip().split())
        return cleaned[:120]

    def _prune_goals(self, m: _AgentModel, now: float) -> None:
        if len(m.goals) <= self._max_goals:
            return
        # Drop the least-active goals (decayed to now), keep the most active.
        ranked = sorted(m.goals.items(), key=lambda kv: kv[1].decayed(now)[0], reverse=True)
        m.goals = dict(ranked[: self._max_goals])

    def context_injection(self, agent_id: str, now: float | None = None) -> str:
        """A compact, honest readout for prompt assembly (one of several consumers)."""
        est = self.estimate(agent_id, now)
        if est.observations == 0 or est.overall_confidence < 0.2:
            return ""
        a = est.affect
        top = ", ".join(g["goal"] for g in est.goals[:2]) or "none clearly active"
        salient = sorted(
            (("frustration", a["frustration"]), ("fatigue", a["fatigue"]),
             ("urgency", a["urgency"]), ("uncertainty", a["uncertainty"])),
            key=lambda kv: kv[1], reverse=True,
        )
        salient_str = ", ".join(f"{k} {v:.2f}" for k, v in salient if v > 0.4) or "calm"
        return (
            "## AGENT STATE ESTIMATE (uncertain — verify before relying on it)\n"
            f"- Confidence in this read: {est.overall_confidence:.2f}\n"
            f"- Salient affect: {salient_str}\n"
            f"- Likely active goal(s): {top}\n"
            f"- They seem to view Aura as: capable {est.beliefs_about_aura['aura_capable']:.2f}, "
            f"trustworthy {est.beliefs_about_aura['aura_trustworthy']:.2f}\n"
            f"- Social rupture risk: {est.social_rupture_risk:.2f}"
        )

    def get_health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "module": "OtherAgentStateEstimator",
                "agents": len(self._models),
                "observations": sum(m.observations for m in self._models.values()),
                "status": "online",
                "active_agent_set": bool(self._active_agent_id),
                "pending_response_count": len(self._pending_responses),
                "raw_messages_persisted": False,
                "raw_goals_persisted": False,
            }

    get_status = get_health


_instance: OtherAgentStateEstimator | None = None
_instance_lock = threading.Lock()


def get_other_agent_model() -> OtherAgentStateEstimator:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = OtherAgentStateEstimator()
    return _instance
