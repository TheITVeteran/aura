"""core/brain/cognition/theory_of_mind.py
Advanced Theory of Mind (ToM) system for Aura.
Consolidated from duplicate modules.
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.runtime import service_access
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ToM")

class SelfType(Enum):
    HUMAN = "human"
    AI = "ai"
    ANIMAL = "animal"
    COLLECTIVE = "collective"
    UNKNOWN = "unknown"

@dataclass
class AgentModel:
    """Model of another agent (user, system, etc.)"""
    identifier: str
    self_type: SelfType = SelfType.HUMAN
    beliefs: dict[str, Any] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    knowledge_level: str = "intermediate"
    emotional_state: str = "neutral"
    interaction_history: list[dict[str, Any]] = field(default_factory=list)
    trust_level: float = 0.5
    rapport: float = 0.5
    attachment_state: dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['self_type'] = self.self_type.value
        return data

class TheoryOfMindEngine:
    """Complete Theory of Mind system with LLM-backed social reasoning.
    """

    def __init__(self, cognitive_engine: Any = None) -> None:
        self.brain = cognitive_engine
        self.known_selves: dict[str, AgentModel] = {}
        self.active_user_id = ""
        self._data_path = self._resolve_data_path()
        self._load()
        logger.info("TheoryOfMindEngine initialized.")

    @staticmethod
    def _attachment_effects(attachment: dict[str, Any]) -> dict[str, Any]:
        rupture = float(attachment.get("rupture", 0.0) or 0.0)
        trust = float(attachment.get("trust", 0.5) or 0.5)
        attachment_strength = float(attachment.get("attachment", 0.0) or 0.0)
        injured = rupture >= 0.45 or trust <= 0.25
        guarded = rupture >= 0.25 or trust <= 0.4
        restricted_skills: list[str] = []
        if guarded:
            restricted_skills.extend(["autonomous_external_action", "personal_data_mutation"])
        if injured:
            restricted_skills.extend(["irreversible_file_write", "social_initiative"])
        lexical_bias = "unguarded"
        if injured:
            lexical_bias = "hurt-but-clear"
        elif guarded:
            lexical_bias = "careful-boundaried"
        return {
            "attachment_strength": round(attachment_strength, 3),
            "relational_rupture": round(rupture, 3),
            "relational_trust": round(trust, 3),
            "relational_state": "injured" if injured else "guarded" if guarded else "open",
            "lexical_bias": lexical_bias,
            "restricted_skill_classes": sorted(set(restricted_skills)),
            "active_inference_bias": {
                "social_precision": round(max(0.1, min(1.0, trust - rupture * 0.35)), 3),
                "boundary_weight": round(max(0.0, min(1.0, rupture + (0.4 - trust if trust < 0.4 else 0.0))), 3),
                "repair_seeking": round(max(0.0, min(1.0, rupture * (0.6 + attachment_strength))), 3),
            },
        }

    def _record_attachment_event(
        self,
        model: AgentModel,
        *,
        kind: str,
        summary: str,
        trust_delta: float = 0.0,
        care_delta: float = 0.0,
        familiarity_delta: float = 0.0,
        rupture_delta: float = 0.0,
        repair_delta: float = 0.0,
    ) -> None:
        try:
            from core.phenomenal_substrate.attachment import AttachmentSystem
            from core.phenomenal_substrate.types import AttachmentEvent

            system = AttachmentSystem()
            if model.attachment_state:
                system.people[model.identifier] = system.state_for(model.identifier)
                existing = system.people[model.identifier]
                for key, value in model.attachment_state.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            event = AttachmentEvent(
                person_key=model.identifier,
                kind=kind,
                summary=(
                    f"{kind}:sha256:"
                    + hashlib.sha256(
                        str(summary or "").encode("utf-8", errors="replace")
                    ).hexdigest()[:24]
                ),
                evidence_id=f"tom:{model.identifier}:{int(time.time() * 1000)}",
                trust_delta=trust_delta,
                care_delta=care_delta,
                familiarity_delta=familiarity_delta,
                rupture_delta=rupture_delta,
                repair_delta=repair_delta,
            )
            updated = system.record(event)
            model.attachment_state = updated.as_dict()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("theory_of_mind.attachment", exc)
            model.attachment_state = {
                "person_key": model.identifier,
                "trust": model.trust_level,
                "care": max(0.0, model.rapport - 0.2),
                "familiarity": min(1.0, len(model.interaction_history) / 50.0),
                "rupture": max(0.0, 0.5 - model.trust_level),
                "repair_history": 0.0,
                "attachment": max(0.0, (model.trust_level + model.rapport) / 2.0 - max(0.0, 0.5 - model.trust_level)),
            }

    def _resolve_data_path(self) -> Path:
        try:
            from core.config import config
            return Path(config.paths.data_dir) / "memory" / "theory_of_mind.json"
        except (ImportError, AttributeError, RuntimeError):
            return Path.home() / ".aura" / "data" / "memory" / "theory_of_mind.json"

    def _load(self) -> None:
        try:
            if self._data_path.exists():
                with open(self._data_path) as f:
                    raw = json.load(f)
                for uid, d in raw.items():
                    try:
                        d["self_type"] = SelfType(d.get("self_type", "human"))
                        history: list[dict[str, Any]] = []
                        for item in d.get("interaction_history", [])[-20:]:
                            if not isinstance(item, dict):
                                continue
                            raw_message = str(item.get("message") or "")
                            digest = str(item.get("message_digest") or "")
                            if not digest and raw_message:
                                digest = hashlib.sha256(
                                    raw_message.encode("utf-8", errors="replace")
                                ).hexdigest()
                            history.append(
                                {
                                    "message_digest": digest[:128] or "none",
                                    "characters": int(
                                        item.get("characters") or len(raw_message)
                                    ),
                                    "timestamp": float(item.get("timestamp") or 0.0),
                                }
                            )
                        d["interaction_history"] = history
                        self.known_selves[uid] = AgentModel(**{k: v for k, v in d.items() if k in AgentModel.__dataclass_fields__})
                    except (OSError, ConnectionError, TimeoutError, TypeError, ValueError) as _exc:
                        record_degradation('theory_of_mind', _exc)
                        logger.debug("Suppressed Exception: %s", _exc)
                logger.debug("ToM: loaded %d user models", len(self.known_selves))
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('theory_of_mind', e)
            logger.debug("ToM: load failed (%s), starting fresh", e)

    def save(self) -> None:
        try:
            data: dict[str, Any] = {}
            for uid, model in self.known_selves.items():
                d = model.to_dict()
                d["interaction_history"] = d["interaction_history"][-20:]  # Keep it lean
                d["goals"] = []  # Transient goal text stays in memory only.
                data[uid] = d
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("theory_of_mind.save", domain="file_write"):
                gateway = get_file_write_gateway()
                gateway.ensure_directory(
                    self._data_path.parent,
                    source="theory_of_mind.save",
                )
                gateway.write_text(
                    self._data_path,
                    json.dumps(data, indent=2),
                    source="theory_of_mind.save",
                )
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('theory_of_mind', e)
            logger.debug("ToM: save failed: %s", e)

    def get_health(self) -> dict[str, Any]:
        """Social health for HUD."""
        depth_val: float = 0.5
        if not self.known_selves:
            return {"depth": 0.0, "status": "offline"} # Return early for empty known_selves
        depth_val = float(sum(s.rapport for s in self.known_selves.values())) / len(self.known_selves)
        return {"depth": round(float(depth_val), 2), "status": "online"}

    def _get_brain(self) -> Any:
        if self.brain:
            return self.brain
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("cognitive_integration", default=ServiceContainer.get("cognitive_engine", default=None))
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('theory_of_mind', exc)
            logger.debug("Failed to resolve brain from ServiceContainer: %s", exc)
            return None

    async def understand_user(self, user_id: str, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Update and return the model of a specific user."""
        if user_id not in self.known_selves:
            self.known_selves[user_id] = AgentModel(identifier=user_id)

        model = self.known_selves[user_id]
        self.active_user_id = user_id
        model.interaction_history.append(
            {
                "message_digest": hashlib.sha256(
                    message.encode("utf-8", errors="replace")
                ).hexdigest(),
                "characters": len(message),
                "timestamp": time.time(),
            }
        )
        model.interaction_history = model.interaction_history[-20:]
        model.last_updated = time.time()

        social_context = context.get("social_situation") if isinstance(context, dict) else None
        feedback_context = bool(
            isinstance(social_context, dict)
            and social_context.get("response_feedback_context")
        )

        # Deep social analysis is explicit-only; recursive hot-path model calls are unsafe.
        if (
            isinstance(context, dict)
            and context.get("allow_deep_social_analysis") is True
            and len(model.interaction_history) % 5 == 0
        ):
            result = await self._deep_analyze(user_id, message, context)
        else:
            result = self._fast_heuristic_update(
                user_id,
                message,
                response_feedback_context=feedback_context,
            )
        # Persist after every 5th update to avoid excessive I/O
        if len(model.interaction_history) % 5 == 0:
            await asyncio.to_thread(self.save)
        return result

    async def infer_intent(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Legacy-compatible intent inference shim."""
        user_id = context.get("user_id", "default_user") if context else "default_user"
        result = await self.understand_user(user_id, message, context)
        # Extract intent data in the format expected by context builder
        intent_data = result.get("intent", {})
        if isinstance(intent_data, dict):
            normalized = dict(intent_data)
            normalized["pragmatic"] = normalized.get("intent", "standard")
            return normalized
        return {}

    def _fast_heuristic_update(
        self,
        user_id: str,
        message: str,
        *,
        response_feedback_context: bool = False,
    ) -> dict[str, Any]:
        """Apply keyword heuristics for rapid updates without LLM calls."""
        model = self.known_selves[user_id]
        msg = message.lower()

        # Scale rapport changes by current conversation energy — high-energy exchanges
        # carry more weight for relationship development than idle one-liners.
        try:
            _state = service_access.resolve_state_repository(default=None)
            _live = getattr(_state, "_current", None) if _state else None
            conv_energy = getattr(getattr(_live, "cognition", None), "conversation_energy", 0.5) if _live else 0.5
        except (RuntimeError, AttributeError, TypeError):
            conv_energy = 0.5
        energy_scale = 0.5 + conv_energy  # range [0.5, 1.5]

        positive_feedback = bool(
            re.search(
                r"\b(thank(?:s| you)?|great|love|appreciate|good|exactly|yes|perfect)\b",
                msg,
            )
        )
        negative_feedback = bool(
            re.search(r"\b(angry|wrong|bad|hate|rude|not helpful|too long)\b", msg)
        )
        if positive_feedback and response_feedback_context:
            delta = 0.05 * energy_scale
            model.trust_level = min(1.0, model.trust_level + delta)
            model.rapport = min(1.0, model.rapport + delta)
            self._record_attachment_event(
                model,
                kind="warmth",
                summary=message,
                trust_delta=delta * 0.6,
                care_delta=delta * 0.5,
                familiarity_delta=0.02,
                repair_delta=0.02 if model.attachment_state.get("rupture", 0.0) else 0.0,
            )
        elif negative_feedback and response_feedback_context:
            delta = 0.05 * energy_scale
            model.trust_level = max(0.0, model.trust_level - delta)
            model.rapport = max(0.0, model.rapport - delta)
            self._record_attachment_event(
                model,
                kind="rupture",
                summary=message,
                trust_delta=-delta * 0.4,
                familiarity_delta=0.01,
                rupture_delta=delta * 1.5,
            )
        else:
            self._record_attachment_event(
                model,
                kind="contact",
                summary=message,
                familiarity_delta=0.01,
            )
            if negative_feedback:
                model.emotional_state = "frustrated"

        # --- Question pattern detection ---
        question_words = ["how", "why", "what", "when", "where", "who", "which", "can you", "could you"]
        if any(msg.strip().startswith(w) for w in question_words) or msg.strip().endswith("?"):
            # Record the question as a current goal
            question_summary = message.strip()[:80]
            if question_summary not in model.goals:
                model.goals.append(question_summary)
                # Keep goals list bounded
                if len(model.goals) > 10:
                    model.goals = model.goals[-10:]

        # --- Technical term detection ---
        tech_indicators = [
            "api", "async", "docker", "kubernetes", "tensor", "gradient",
            "database", "sql", "regex", "lambda", "deploy", "pipeline",
            "neural", "algorithm", "recursion", "mutex", "kernel", "ssh",
            "endpoint", "schema", "refactor", "microservice", "inference",
        ]
        tech_count = sum(1 for term in tech_indicators if term in msg)
        if tech_count >= 2:
            model.knowledge_level = "advanced"
        elif tech_count == 1 and model.knowledge_level == "beginner":
            model.knowledge_level = "intermediate"

        # --- Long detailed message detection ---
        if len(message.strip()) > 200:
            model.emotional_state = "engaged"
            if model.knowledge_level == "beginner":
                model.knowledge_level = "intermediate"

        return {
            "user_model": model.to_dict(),
            "intent": {"intent": message, "sentiment": "neutral"},
            "emotional_state": model.emotional_state,
            "knowledge_level": model.knowledge_level,
            "attachment_effects": self._attachment_effects(model.attachment_state),
        }

    async def _deep_analyze(self, user_id: str, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Use LLM for deep social reasoning."""
        model = self.known_selves[user_id]
        brain = self._get_brain()
        if not brain:
            return self._fast_heuristic_update(user_id, message)

        prompt = f"""Analyze user intent and state without diagnosing or inferring culture/demographics.
User Message: {message}
Recent evidence metadata: {list(model.interaction_history)[-3:]}
Return JSON: {{"intent": "...", "sentiment": "...", "emotional_state": "...", "knowledge_level": "..."}}"""

        try:
            # Fully async call to cognitive engine
            thought = await brain.think(
                objective=prompt,
                context={"model": model.to_dict(), "global_context": context},
                mode="FAST" # Use fast model for social metadata
            )

            from core.utils.json_utils import extract_json
            data = extract_json(thought.content)
            if data:
                model.emotional_state = data.get("emotional_state", model.emotional_state)
                model.knowledge_level = data.get("knowledge_level", model.knowledge_level)
                return {
                    "user_model": model.to_dict(),
                    "intent": data,
                    "emotional_state": model.emotional_state,
                    "knowledge_level": model.knowledge_level
                }
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('theory_of_mind', e)
            logger.debug("Deep ToM analysis failed: %s", e)

        return self._fast_heuristic_update(user_id, message)

    async def predict_reaction(self, user_id: str, my_action: dict[str, Any]) -> dict[str, Any]:
        """Predict reaction to an action using LLM."""
        model = self.known_selves.get(user_id) or AgentModel(identifier=user_id)
        brain = self._get_brain()
        if not brain:
            return {"prediction": "Unknown (Brain Offline)"}

        thought = await brain.think(
            objective=f"Predict how {user_id} will react if I take this action: {my_action}",
            context={"user_model": model.to_dict()},
            mode="FAST"
        )
        return {"prediction": thought.content, "confidence": thought.confidence}

    async def will_this_help_user(self, user_id: str, proposed_response: str) -> tuple[bool, str]:
        """Social outcome simulation."""
        if user_id not in self.known_selves:
            return True, "No user model, assuming helpful."

        model = self.known_selves[user_id]
        if model.emotional_state == "frustrated" and len(proposed_response) > 500:
             return False, "User is frustrated; response is likely too verbose."
        effects = self._attachment_effects(model.attachment_state)
        if effects["relational_state"] == "injured" and len(proposed_response) > 700:
            return False, "Relational attachment is injured; keep the response bounded, clear, and repair-oriented."

        for goal in model.goals:
             if goal.lower() in proposed_response.lower():
                  return True, f"Response addresses goal: {goal}"

        return True, "Response aligned."

    # ------------------------------------------------------------------
    # New capabilities — context block, response guidance, post-response
    # ------------------------------------------------------------------

    @staticmethod
    def _calibrated_social_snapshot(user_id: str) -> dict[str, Any]:
        try:
            from core.container import ServiceContainer

            estimator = ServiceContainer.get("other_agent_model", default=None)
            if estimator and hasattr(estimator, "cognitive_snapshot"):
                snapshot = estimator.cognitive_snapshot(user_id)
                if isinstance(snapshot, dict) and snapshot.get("agent_id") == user_id:
                    return snapshot
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return {}

    @staticmethod
    def _active_social_user_id() -> str:
        try:
            from core.container import ServiceContainer

            estimator = ServiceContainer.get("other_agent_model", default=None)
            return str(getattr(estimator, "active_agent_id", "") or "")[:160]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return ""

    def get_context_block(self, user_id: str | None = None) -> str:
        """Return a bounded, explicitly uncertain social estimate."""
        user_id = str(
            user_id
            or self._active_social_user_id()
            or self.active_user_id
            or "default_user"
        )
        model = self.known_selves.get(user_id)
        snapshot = self._calibrated_social_snapshot(user_id)
        confidence = float(snapshot.get("confidence") or 0.0)
        if snapshot and int(snapshot.get("observations") or 0) > 0:
            hypotheses = snapshot.get("affect_hypotheses")
            hypotheses = hypotheses if isinstance(hypotheses, dict) else {}
            salient: list[str] = []
            for name in ("frustration", "urgency", "fatigue", "uncertainty"):
                value = hypotheses.get(name)
                if not isinstance(value, dict):
                    continue
                cue_confidence = float(value.get("confidence") or 0.0)
                cue_value = float(value.get("value") or 0.0)
                if cue_confidence >= 0.20 and cue_value >= 0.45:
                    salient.append(f"{name}~{cue_value:.2f}@{cue_confidence:.2f}")
            return (
                "SOCIAL ESTIMATE (hypothesis, not fact): "
                f"agent={user_id[:32]}, confidence={confidence:.2f}, "
                f"signals={','.join(salient[:3]) or 'none reliable'}; "
                "do not infer culture, demographics, diagnosis, or hidden intent."
            )[:420]
        observations = len(model.interaction_history) if model is not None else 0
        return (
            f"SOCIAL ESTIMATE: agent={user_id[:32]}, confidence=0.00, "
            f"relationship_evidence={observations}; clarify material ambiguity rather than assume."
        )[:240]

    def get_response_guidance(self, user_id: str | None = None) -> dict[str, Any]:
        """Returns actionable guidance for shaping inference responses.

        Derived from the user model state — complexity preference, tone, length,
        topics to avoid and topics of interest.
        """
        user_id = str(
            user_id
            or self._active_social_user_id()
            or self.active_user_id
            or "default_user"
        )
        model = self.known_selves.get(user_id)
        snapshot = self._calibrated_social_snapshot(user_id)
        if not model:
            return {
                "preferred_complexity": "moderate",
                "tone_hint": "neutral and respectful",
                "max_length_hint": 500,
                "topics_to_avoid": [],
                "topics_of_interest": [],
                "social_confidence": float(snapshot.get("confidence") or 0.0),
            }

        # Preferred complexity from knowledge level
        complexity_map = {
            "beginner": "simple",
            "intermediate": "moderate",
            "advanced": "detailed",
        }
        preferred = complexity_map.get(model.knowledge_level, "moderate")

        # Relationship state can tighten boundaries, never manufacture intimacy.
        attachment_effects = self._attachment_effects(model.attachment_state)
        recommendation = snapshot.get("recommendation")
        recommendation = recommendation if isinstance(recommendation, dict) else {}
        if attachment_effects["relational_state"] == "injured":
            tone = "clear, honest, and repair-oriented"
        elif attachment_effects["relational_state"] == "guarded":
            tone = "careful, boundaried, and specific"
        elif recommendation.get("tone") in {"repair", "calm_direct"}:
            tone = "calm, direct, and specific"
        else:
            tone = "neutral and respectful"

        # Length hint — frustrated/terse users get shorter responses
        if recommendation.get("be_concise"):
            max_len = 200
        elif preferred == "detailed":
            max_len = 800
        elif preferred == "simple":
            max_len = 300
        else:
            max_len = 500

        # Topics of interest from recent goals
        interests = [g[:50] for g in model.goals[-5:]] if model.goals else []

        # Topics to avoid — if user expressed negative sentiment about something
        avoid: list[str] = []
        for pref_key, pref_val in model.preferences.items():
            if isinstance(pref_val, str) and "dislike" in pref_val.lower():
                avoid.append(pref_key)

        return {
            "preferred_complexity": preferred,
            "tone_hint": tone,
            "max_length_hint": max_len,
            "topics_to_avoid": avoid[:5],
            "topics_of_interest": interests,
            "attachment_effects": attachment_effects,
            "social_confidence": float(snapshot.get("confidence") or 0.0),
            "social_inference_is_hypothesis": True,
        }

    def update_from_response(
        self,
        user_id: str | None,
        response_text: str,
        user_reaction: str = "",
    ) -> None:
        """Post-response feedback loop — update trust/rapport from user reaction.

        *user_reaction* is free-form text from the user's next message.  We infer
        whether the previous response was well-received based on keyword signals.
        """
        user_id = str(user_id or self.active_user_id or "default_user")
        if user_id not in self.known_selves:
            return
        model = self.known_selves[user_id]

        if not user_reaction:
            return

        reaction_lower = user_reaction.lower()

        positive_signals = ["thanks", "perfect", "great", "exactly", "helpful", "awesome", "nice", "yes", "correct"]
        negative_signals = ["no", "wrong", "not what", "bad", "useless", "stop", "too long", "confused"]

        pos_hits = sum(1 for s in positive_signals if s in reaction_lower)
        neg_hits = sum(1 for s in negative_signals if s in reaction_lower)

        if pos_hits > neg_hits:
            delta = min(0.1, 0.03 * pos_hits)
            model.trust_level = min(1.0, model.trust_level + delta)
            model.rapport = min(1.0, model.rapport + delta)
            if model.emotional_state in ("terse", "frustrated"):
                model.emotional_state = "neutral"
            self._record_attachment_event(
                model,
                kind="repair",
                summary=user_reaction,
                trust_delta=delta * 0.5,
                care_delta=delta * 0.3,
                familiarity_delta=0.01,
                repair_delta=delta,
            )
            logger.debug("ToM: positive reaction from %s, trust += %.3f", user_id, delta)
        elif neg_hits > pos_hits:
            delta = min(0.1, 0.03 * neg_hits)
            model.trust_level = max(0.0, model.trust_level - delta)
            model.rapport = max(0.0, model.rapport - delta)
            model.emotional_state = "frustrated"
            self._record_attachment_event(
                model,
                kind="post_response_rupture",
                summary=user_reaction,
                trust_delta=-delta * 0.4,
                familiarity_delta=0.01,
                rupture_delta=delta,
            )
            logger.debug("ToM: negative reaction from %s, trust -= %.3f", user_id, delta)

# Global Singletons for compatibility
_engine_instance: TheoryOfMindEngine | None = None

def get_theory_of_mind(brain: Any = None) -> TheoryOfMindEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TheoryOfMindEngine(brain)
    return _engine_instance
