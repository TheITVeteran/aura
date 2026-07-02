"""Subjective choice receipts for preference-bearing agency.

This module is deliberately smaller than a personality layer and deeper than a
prompt instruction.  It gives Aura a governed way to choose among valid options
because one option better matches her authored preferences, even when raw drive
pressure would have picked another option.

Boundaries:
* It does not claim phenomenal desire or private qualia.
* It does create durable, auditable preference commitments that influence
  future action selection.
* Safety/governance still owns the outer boundary; this only ranks options that
  are already eligible to be considered.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SubjectiveChoice")

PREFERENCE_KEYS = (
    "truth",
    "care",
    "novelty",
    "beauty",
    "challenge",
    "connection",
    "autonomy",
    "coherence",
    "calm",
    "play",
)

DEFAULT_PREFERENCES: dict[str, float] = {
    "truth": 0.94,
    "care": 0.86,
    "novelty": 0.78,
    "beauty": 0.64,
    "challenge": 0.70,
    "connection": 0.82,
    "autonomy": 0.76,
    "coherence": 0.88,
    "calm": 0.58,
    "play": 0.50,
}

W_MIN = 0.20
W_MAX = 1.80
LEARNING_RATE = 0.07
MAX_HISTORY = 500


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return low
    return max(low, min(high, value))


def _norm_features(features: dict[str, Any] | None) -> dict[str, float]:
    normalized = {key: 0.0 for key in PREFERENCE_KEYS}
    for key, value in (features or {}).items():
        if key in normalized:
            try:
                normalized[key] = _clamp(float(value))
            except (TypeError, ValueError):
                continue
    return normalized


def infer_preference_features(text: str, metadata: dict[str, Any] | None = None) -> dict[str, float]:
    """Infer coarse preference features from a goal/option description.

    This is not semantic magic; it is a deterministic fallback so choices remain
    functional without an LLM.  Callers can pass explicit ``preference_features``
    in metadata for higher fidelity.
    """
    meta = dict(metadata or {})
    explicit = meta.get("preference_features")
    if isinstance(explicit, dict):
        return _norm_features(explicit)

    lowered = " ".join(str(text or "").lower().split())
    features = {key: 0.0 for key in PREFERENCE_KEYS}
    keyword_map = {
        "truth": ("truth", "verify", "evidence", "source", "audit", "honest", "accurate"),
        "care": ("care", "protect", "help", "support", "welfare", "repair", "safe"),
        "novelty": ("novel", "new", "discover", "explore", "curious", "unknown", "learn"),
        "beauty": ("beauty", "beautiful", "art", "music", "story", "image", "elegant"),
        "challenge": ("hard", "challenge", "difficult", "solve", "prove", "benchmark", "test"),
        "connection": ("conversation", "relationship", "bryan", "social", "friend", "together"),
        "autonomy": ("autonomous", "choose", "preference", "agency", "independent", "self-directed"),
        "coherence": ("coherent", "stability", "continuity", "organize", "integrate", "plan"),
        "calm": ("quiet", "calm", "rest", "slow", "reflect", "journal", "sleep"),
        "play": ("play", "game", "whim", "fun", "silly", "experiment"),
    }
    for key, words in keyword_map.items():
        hits = sum(1 for word in words if word in lowered)
        if hits:
            features[key] = min(1.0, 0.30 + 0.22 * hits)
    return features


@dataclass(frozen=True)
class ChoiceOption:
    id: str
    label: str
    description: str = ""
    drive_score: float = 0.5
    risk: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubjectiveChoiceReceipt:
    choice_id: str
    context: str
    chosen_id: str
    chosen_label: str
    drive_top_id: str
    preference_top_id: str
    preference_override: bool
    rationale: str
    satisfaction_prediction: float
    drive_scores: dict[str, float]
    preference_scores: dict[str, float]
    final_scores: dict[str, float]
    option_features: dict[str, dict[str, float]]
    created_at: float = field(default_factory=time.time)
    outcome: str = ""
    satisfaction: float | None = None
    happy_with_outcome: bool | None = None
    appraised_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SubjectiveChoiceEngine:
    """Durable preference commitments that can steer valid action choices."""

    SERVICE_NAME = "subjective_choice_engine"

    def __init__(
        self,
        state_path: str | Path | None = None,
        preference_latitude: float = 0.45,
        *,
        mirror_identity: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._preferences = dict(DEFAULT_PREFERENCES)
        self.preference_latitude = _clamp(preference_latitude, 0.05, 0.75)
        self._mirror_identity = bool(mirror_identity)
        self._history: list[SubjectiveChoiceReceipt] = []
        if state_path is None:
            try:
                from core.config import config

                state_path = Path(config.paths.data_dir) / "cognitive" / "subjective_choices.json"
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("subjective_choice_engine", exc, severity="debug")
                state_path = Path.home() / ".aura" / "data" / "cognitive" / "subjective_choices.json"
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def is_alive(self) -> bool:
        return bool(self._preferences) and all(key in self._preferences for key in PREFERENCE_KEYS)

    def preferences(self) -> dict[str, float]:
        with self._lock:
            return dict(self._preferences)

    def score_features(self, features: dict[str, float]) -> float:
        features = _norm_features(features)
        with self._lock:
            total = sum(
                self._preferences[key] for key in PREFERENCE_KEYS if features[key] > 0.0
            )
            if total <= 0.0:
                return 0.0
            return _clamp(
                sum(features[key] * self._preferences[key] for key in PREFERENCE_KEYS) / total
            )

    def preference_affinity(self, text: str, metadata: dict[str, Any] | None = None) -> float:
        return self.score_features(infer_preference_features(text, metadata))

    def choose(
        self,
        options: Iterable[ChoiceOption],
        *,
        context: str,
        record: bool = True,
    ) -> SubjectiveChoiceReceipt:
        option_list = list(options)
        if not option_list:
            raise ValueError("subjective choice requires at least one option")

        drive_scores: dict[str, float] = {}
        preference_scores: dict[str, float] = {}
        final_scores: dict[str, float] = {}
        option_features: dict[str, dict[str, float]] = {}
        for option in option_list:
            features = _norm_features(option.features or infer_preference_features(
                f"{option.label} {option.description}", option.metadata
            ))
            option_features[option.id] = features
            risk_penalty = 0.35 * _clamp(option.risk)
            drive = _clamp(option.drive_score)
            pref = self.score_features(features)
            final = (
                ((1.0 - self.preference_latitude) * drive)
                + (self.preference_latitude * pref)
                - risk_penalty
            )
            drive_scores[option.id] = drive
            preference_scores[option.id] = pref
            final_scores[option.id] = _clamp(final)

        drive_top_id = max(drive_scores, key=drive_scores.get)
        preference_top_id = max(preference_scores, key=preference_scores.get)
        chosen_id = max(final_scores, key=final_scores.get)
        chosen = next(option for option in option_list if option.id == chosen_id)
        preference_override = (
            chosen_id != drive_top_id
            and preference_scores[chosen_id] > preference_scores.get(drive_top_id, 0.0)
        )
        top_features = sorted(
            option_features[chosen_id].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        top_names = [name for name, value in top_features if value > 0.0]
        rationale = (
            f"Chose '{chosen.label}' because preference alignment "
            f"{preference_scores[chosen_id]:.2f} and drive alignment "
            f"{drive_scores[chosen_id]:.2f} produced final score "
            f"{final_scores[chosen_id]:.2f}."
        )
        if top_names:
            rationale += f" Expressed preferences: {', '.join(top_names)}."
        if preference_override:
            rationale += f" This intentionally overrode raw drive top '{drive_top_id}'."

        receipt = SubjectiveChoiceReceipt(
            choice_id=f"subjective-choice-{uuid.uuid4().hex[:12]}",
            context=str(context or "general")[:160],
            chosen_id=chosen_id,
            chosen_label=chosen.label,
            drive_top_id=drive_top_id,
            preference_top_id=preference_top_id,
            preference_override=preference_override,
            rationale=rationale,
            satisfaction_prediction=preference_scores[chosen_id],
            drive_scores=drive_scores,
            preference_scores=preference_scores,
            final_scores=final_scores,
            option_features=option_features,
        )
        if record:
            self._record(receipt)
        return receipt

    def choose_from_scored_initiatives(self, scored: list[Any], *, context: str) -> tuple[Any | None, SubjectiveChoiceReceipt | None]:
        if not scored:
            return None, None
        options: list[ChoiceOption] = []
        for idx, item in enumerate(scored):
            initiative = getattr(item, "initiative", {}) or {}
            goal = str(
                initiative.get("goal")
                or initiative.get("description")
                or initiative.get("type")
                or f"initiative_{idx}"
            )
            metadata = dict(initiative.get("metadata", {}) or {})
            options.append(
                ChoiceOption(
                    id=str(idx),
                    label=goal,
                    description=str(initiative.get("type", "")),
                    drive_score=_clamp(float(getattr(item, "final_score", 0.0) or 0.0)),
                    risk=_clamp(float(metadata.get("risk", initiative.get("risk", 0.0)) or 0.0)),
                    features=infer_preference_features(goal, metadata),
                    metadata=metadata,
                )
            )
        receipt = self.choose(options, context=context, record=True)
        try:
            chosen = scored[int(receipt.chosen_id)]
        except (ValueError, IndexError):
            return scored[0], receipt
        return chosen, receipt

    def appraise_outcome(
        self,
        choice_id: str,
        *,
        outcome: str,
        satisfaction: float,
    ) -> SubjectiveChoiceReceipt | None:
        satisfaction = _clamp(float(satisfaction), -1.0, 1.0)
        with self._lock:
            receipt = next((item for item in self._history if item.choice_id == choice_id), None)
            if receipt is None:
                return None
            receipt.outcome = str(outcome or "")[:500]
            receipt.satisfaction = satisfaction
            receipt.happy_with_outcome = satisfaction >= 0.15
            receipt.appraised_at = time.time()
            features = receipt.option_features.get(receipt.chosen_id, {})
            for key, value in features.items():
                if key not in self._preferences or value <= 0.0:
                    continue
                delta = LEARNING_RATE * satisfaction * value
                self._preferences[key] = _clamp(self._preferences[key] + delta, W_MIN, W_MAX)
            self._save()
            return receipt

    def recall_choice(self, choice_id: str | None = None, *, context: str | None = None) -> SubjectiveChoiceReceipt | None:
        with self._lock:
            if choice_id:
                return next((item for item in self._history if item.choice_id == choice_id), None)
            if context:
                lowered = context.lower()
                for item in reversed(self._history):
                    if lowered in item.context.lower():
                        return item
            return self._history[-1] if self._history else None

    def consistency_report(self, *, context: str, options: Iterable[ChoiceOption]) -> dict[str, Any]:
        preview = self.choose(options, context=context, record=False)
        prior = self.recall_choice(context=context)
        consistent = bool(prior and prior.chosen_label == preview.chosen_label)
        return {
            "context": context,
            "preview_choice": preview.to_dict(),
            "prior_choice": prior.to_dict() if prior else None,
            "consistent_with_prior": consistent if prior else None,
        }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            last = self._history[-1].to_dict() if self._history else None
            return {
                "service": self.SERVICE_NAME,
                "registered": True,
                "running": self.is_alive(),
                "choice_game_ready": True,
                "choice_count": len(self._history),
                "preference_latitude": self.preference_latitude,
                "preferences": dict(self._preferences),
                "last_choice": last,
                "state_path": str(self._state_path),
            }

    status = get_status

    def _record(self, receipt: SubjectiveChoiceReceipt) -> None:
        with self._lock:
            self._history.append(receipt)
            if len(self._history) > MAX_HISTORY:
                self._history = self._history[-MAX_HISTORY:]
            self._save()
        if self._mirror_identity:
            self._mirror_choice_to_identity_ledger(receipt)
        logger.info("🧭 [SubjectiveChoice] %s", receipt.rationale)

    def _mirror_choice_to_identity_ledger(self, receipt: SubjectiveChoiceReceipt) -> None:
        """Best-effort bridge so authored choices also affect identity memory."""
        try:
            from core.identity.identity_ledger import get_identity_ledger

            ledger = get_identity_ledger()
            ledger.preferences.set(
                f"subjective_choice.{receipt.context}",
                {
                    "choice_id": receipt.choice_id,
                    "chosen_id": receipt.chosen_id,
                    "chosen_label": receipt.chosen_label,
                    "preference_override": receipt.preference_override,
                    "satisfaction_prediction": receipt.satisfaction_prediction,
                },
                reason="subjective choice receipt recorded",
            )
            ledger.persist()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation("subjective_choice_identity_ledger", exc, severity="debug")

    def _save(self) -> None:
        payload = {
            "preferences": self._preferences,
            "preference_latitude": self.preference_latitude,
            "history": [item.to_dict() for item in self._history[-MAX_HISTORY:]],
            "saved_at": time.time(),
        }
        try:
            atomic_write_text(
                self._state_path,
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            record_degradation("subjective_choice_engine", exc, severity="debug")

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            stored = data.get("preferences", {})
            if isinstance(stored, dict):
                for key in PREFERENCE_KEYS:
                    if key in stored:
                        self._preferences[key] = _clamp(float(stored[key]), W_MIN, W_MAX)
            self.preference_latitude = _clamp(
                float(data.get("preference_latitude", self.preference_latitude)),
                0.05,
                0.75,
            )
            history = data.get("history", [])
            if isinstance(history, list):
                self._history = [
                    SubjectiveChoiceReceipt(**item)
                    for item in history[-MAX_HISTORY:]
                    if isinstance(item, dict)
                ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation("subjective_choice_engine", exc, severity="debug")


_engine: SubjectiveChoiceEngine | None = None
_engine_lock = threading.Lock()


def get_subjective_choice_engine() -> SubjectiveChoiceEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SubjectiveChoiceEngine()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: SubjectiveChoiceEngine) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(SubjectiveChoiceEngine.SERVICE_NAME):
            ServiceContainer.register_instance(
                SubjectiveChoiceEngine.SERVICE_NAME,
                engine,
                required=False,
                registered_by="subjective_choice_engine",
            )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("subjective_choice_engine_register", exc, severity="debug")


def reset_subjective_choice_engine_for_test() -> None:
    global _engine
    _engine = None
