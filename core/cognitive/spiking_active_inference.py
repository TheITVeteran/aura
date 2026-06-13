"""Bounded spiking active-inference advisor for live cognition.

This module turns the "Aura brain" chalkboard ideas into runtime machinery:
Poisson-style spike traces, dendritic gating, eligibility traces, Bayesian
belief over cognitive state, and active-inference action tendencies.

It is intentionally advisory. It never executes tools, writes memory, or
mutates policy directly. Instead it produces auditable pressure signals that
the governed CognitiveEngine, AuthorityGateway, routing phases, and inference
sampler can consume.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

EPS = 1e-12

STATE_NAMES: tuple[str, ...] = (
    "overloaded",
    "confused",
    "ready",
    "tool_needed",
    "research_needed",
    "repair_needed",
)

FEATURE_NAMES: tuple[str, ...] = (
    "clarity",
    "energy",
    "urgency",
    "novelty",
    "tool_pressure",
    "error_pressure",
    "social_pressure",
    "memory_pressure",
)

ACTION_NAMES: tuple[str, ...] = (
    "answer_directly",
    "plan_deeply",
    "seek_information",
    "use_governed_tools",
    "ask_clarification",
    "reduce_load",
    "repair_first",
)

_STATE_MEANS = np.array(
    [
        [0.20, 0.15, 0.85, 0.30, 0.25, 0.85, 0.20, 0.30],
        [0.25, 0.62, 0.45, 0.60, 0.25, 0.35, 0.45, 0.35],
        [0.88, 0.78, 0.35, 0.35, 0.20, 0.08, 0.40, 0.25],
        [0.65, 0.72, 0.60, 0.50, 0.92, 0.20, 0.30, 0.30],
        [0.55, 0.72, 0.45, 0.88, 0.55, 0.12, 0.35, 0.25],
        [0.40, 0.48, 0.70, 0.40, 0.45, 0.92, 0.25, 0.45],
    ],
    dtype=np.float64,
)

_REWARD_BY_STATE = np.array(
    [
        [-0.65, -0.15, 1.10, 0.10, 0.05, -0.35],
        [0.20, 0.75, 0.55, 0.40, 0.35, 0.25],
        [-0.25, 0.45, 0.25, 0.20, 1.10, 0.25],
        [-0.55, 0.10, 0.20, 1.25, 0.50, 0.55],
        [0.55, 1.15, -0.15, -0.10, 0.10, 0.20],
        [1.15, 0.35, -0.25, -0.30, -0.15, 0.55],
        [0.35, 0.25, -0.20, 0.15, 0.10, 1.30],
    ],
    dtype=np.float64,
)

_ACTION_COST = np.array([0.08, 0.20, 0.30, 0.34, 0.22, 0.16, 0.28], dtype=np.float64)
_ACTION_RISK = np.array([0.05, 0.08, 0.10, 0.22, 0.04, 0.03, 0.12], dtype=np.float64)
_ACTION_CONTROLLABILITY = np.array([0.10, 0.45, 0.80, 0.75, 0.90, 0.65, 0.85], dtype=np.float64)

_UNCERTAIN_RE = re.compile(
    r"\b(maybe|perhaps|not sure|confused|unclear|unknown|ambiguous|what do you mean|"
    r"which one|either|or|hypothetical|suppose|could|would)\b",
    re.IGNORECASE,
)
_TOOL_RE = re.compile(
    r"\b(open|create|save|export|download|search|browse|click|type|run|execute|"
    r"file|folder|desktop|notes?|chrome|docs?|pdf|app|terminal|website|url|tool)\b",
    re.IGNORECASE,
)
_RESEARCH_RE = re.compile(
    r"\b(research|look up|verify|latest|current|article|source|news|compare|find)\b",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r"\b(error|failed|failure|crash|broken|traceback|exception|stuck|loop|lag|"
    r"memory spike|ram|timeout|cortex warming|unavailable|degraded)\b",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    r"\b(remember|recall|earlier|last time|previous|across sessions|memory|forget)\b",
    re.IGNORECASE,
)
_SOCIAL_RE = re.compile(
    r"\b(hey|hi|hello|thanks|thank you|feel|feeling|lonely|miss|care|relationship|"
    r"who are you|what are you|conscious|sentient|self-aware)\b",
    re.IGNORECASE,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(float(value)):
        return low
    return max(low, min(high, float(value)))


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= EPS:
        return np.ones_like(values, dtype=np.float64) / max(1, values.size)
    return values / total


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(EPS, float(temperature))
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - float(np.max(z))
    return _normalize(np.exp(np.clip(z, -60.0, 60.0)))


def _entropy01(probabilities: np.ndarray) -> float:
    p = _normalize(np.asarray(probabilities, dtype=np.float64))
    if p.size <= 1:
        return 0.0
    entropy = float(-np.sum(p * np.log(p + EPS)))
    return _clamp(entropy / math.log(p.size))


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


@dataclass(frozen=True)
class SpikingActiveInferenceConfig:
    dt: float = 0.01
    tau_psp: float = 0.12
    tau_eligibility: float = 0.65
    learning_rate: float = 0.045
    weight_decay: float = 0.002
    beta_0: float = 4.5
    threshold: float = 0.72
    seed: int = 2719


@dataclass(frozen=True)
class NeurodynamicAdvice:
    advice_id: str
    timestamp: float
    action: str
    belief: dict[str, float]
    probabilities: dict[str, float]
    features: dict[str, float]
    uncertainty: float
    confidence: float
    routing_bias: dict[str, Any]
    sampling_bias: dict[str, float]
    governance: dict[str, Any]
    neural_state: dict[str, float]
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiCompartmentSpikeResponseModel:
    """Small bounded SRM block used as a live cognitive pressure sensor."""

    def __init__(
        self,
        *,
        n_neurons: int = 16,
        compartments: int = 3,
        n_inputs: int = len(FEATURE_NAMES),
        config: SpikingActiveInferenceConfig | None = None,
    ) -> None:
        self.config = config or SpikingActiveInferenceConfig()
        self.n_neurons = int(max(4, min(64, n_neurons)))
        self.compartments = int(max(1, min(8, compartments)))
        self.n_inputs = int(max(1, min(32, n_inputs)))
        self._rng = np.random.default_rng(self.config.seed)
        self.weights = self._rng.normal(
            loc=0.25,
            scale=0.04,
            size=(self.n_neurons, self.compartments, self.n_inputs),
        ).astype(np.float64)
        self.weights = np.clip(self.weights, 0.0, 2.0)
        self.psp = np.zeros(self.n_inputs, dtype=np.float64)
        self.eligibility = np.zeros((self.n_neurons, self.compartments), dtype=np.float64)
        self.dendritic_trace = np.zeros((self.n_neurons, self.compartments), dtype=np.float64)
        self.post_trace = np.zeros(self.n_neurons, dtype=np.float64)
        self.threshold = np.full(self.n_neurons, self.config.threshold, dtype=np.float64)
        self.last_spike_rate = 0.0
        self.last_plateau_rate = 0.0

    def tick(self, input_vector: Sequence[float], *, modulation: float = 1.0) -> dict[str, float]:
        x = np.asarray(input_vector, dtype=np.float64).ravel()
        if x.size != self.n_inputs:
            padded = np.zeros(self.n_inputs, dtype=np.float64)
            n = min(self.n_inputs, x.size)
            padded[:n] = x[:n]
            x = padded
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        modulation = _clamp(modulation, 0.1, 2.0)

        decay_psp = math.exp(-self.config.dt / self.config.tau_psp)
        decay_elig = math.exp(-self.config.dt / self.config.tau_eligibility)
        self.psp = self.psp * decay_psp + x

        dendritic_input = np.einsum("nci,i->nc", self.weights, self.psp)
        plateau = (dendritic_input >= self.threshold[:, None]).astype(np.float64)
        self.last_plateau_rate = float(np.mean(plateau))

        self.eligibility = self.eligibility * decay_elig + plateau
        self.dendritic_trace = (
            self.dendritic_trace * 0.94
            + 0.06 * np.tanh(dendritic_input + self.eligibility)
        )
        membrane = np.sum(dendritic_input + 0.35 * self.dendritic_trace, axis=1)
        logits = np.clip((membrane - self.threshold) * self.config.beta_0, -50.0, 50.0)
        spike_prob = np.clip(1.0 / (1.0 + np.exp(-logits)), 0.0, 1.0)
        spikes = self._rng.random(self.n_neurons) < spike_prob * self.config.dt
        self.last_spike_rate = float(np.mean(spikes))
        self.post_trace = self.post_trace * 0.90 + spikes.astype(np.float64)

        pre = self.psp[None, None, :]
        post = (self.post_trace[:, None, None] + 0.05)
        delta_w = (
            self.config.learning_rate
            * modulation
            * post
            * pre
            * self.config.dt
            - self.config.weight_decay * self.weights * self.config.dt
        )
        self.weights = np.clip(self.weights + delta_w, 0.0, 2.0)
        self.threshold = np.clip(
            self.threshold
            + self.config.dt * (0.02 * (self.post_trace - 0.10) - 0.01 * modulation),
            0.35,
            1.45,
        )

        return self.summary()

    def summary(self) -> dict[str, float]:
        return {
            "spike_rate": _clamp(self.last_spike_rate),
            "plateau_rate": _clamp(self.last_plateau_rate),
            "eligibility_mean": _clamp(float(np.mean(self.eligibility)), 0.0, 3.0),
            "weight_mean": _clamp(float(np.mean(self.weights)), 0.0, 2.0),
            "threshold_mean": _clamp(float(np.mean(self.threshold)), 0.0, 2.0),
        }


class SpikingActiveInferenceAdvisor:
    """General cognitive advisor combining SRM dynamics and active inference."""

    def __init__(self, config: SpikingActiveInferenceConfig | None = None) -> None:
        self.config = config or SpikingActiveInferenceConfig()
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.config.seed + 1)
        self._belief = np.ones(len(STATE_NAMES), dtype=np.float64) / len(STATE_NAMES)
        self._weights = self._rng.normal(0.0, 0.025, size=(len(ACTION_NAMES), len(FEATURE_NAMES)))
        self._bias = np.zeros(len(ACTION_NAMES), dtype=np.float64)
        self._counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
        self._srm = MultiCompartmentSpikeResponseModel(config=self.config)
        self._last_advice: NeurodynamicAdvice | None = None

    def advise(
        self,
        objective: str,
        *,
        context: Mapping[str, Any] | None = None,
        state: Any = None,
        origin: str = "system",
        is_background: bool = False,
    ) -> NeurodynamicAdvice:
        with self._lock:
            features = self._features_from_runtime(objective, context=context, state=state)
            x = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
            likelihood = self._state_likelihood(x)
            self._belief = _normalize(0.72 * self._belief + 0.28 * likelihood)
            uncertainty = _entropy01(self._belief)
            neural_state = self._srm.tick(
                x,
                modulation=0.65
                + 0.50 * features["error_pressure"]
                + 0.35 * features["novelty"]
                + 0.25 * features["tool_pressure"],
            )
            scores = self._score_actions(x, uncertainty, is_background=is_background)
            probabilities = _softmax(scores, self._temperature(features, uncertainty))
            action_index = int(np.argmax(probabilities))
            self._counts[action_index] += 1
            action = ACTION_NAMES[action_index]
            routing_bias = self._routing_bias(action, features, uncertainty, is_background)
            sampling_bias = self._sampling_bias(action, features, uncertainty)
            advice = NeurodynamicAdvice(
                advice_id=self._advice_id(objective, origin),
                timestamp=time.time(),
                action=action,
                belief={
                    name: _clamp(float(value))
                    for name, value in zip(STATE_NAMES, self._belief, strict=True)
                },
                probabilities={
                    name: _clamp(float(value))
                    for name, value in zip(ACTION_NAMES, probabilities, strict=True)
                },
                features={name: _clamp(value) for name, value in features.items()},
                uncertainty=uncertainty,
                confidence=1.0 - uncertainty,
                routing_bias=routing_bias,
                sampling_bias=sampling_bias,
                governance={
                    "consequential_action": False,
                    "executes_tools": False,
                    "writes_memory": False,
                    "authority_gateway_required_for_effects": True,
                    "advisory_only": True,
                    "origin": str(origin or "system")[:64],
                },
                neural_state=neural_state,
                rationale=self._rationale(action, features, uncertainty, routing_bias),
            )
            self._last_advice = advice
            return advice

    def learn_from_feedback(
        self,
        action: str,
        reward: float,
        features: Mapping[str, float] | Sequence[float],
    ) -> dict[str, float]:
        with self._lock:
            if action not in ACTION_NAMES:
                raise ValueError(f"unknown active-inference action: {action!r}")
            idx = ACTION_NAMES.index(action)
            if isinstance(features, Mapping):
                x = np.array([_safe_float(features.get(name), 0.0) for name in FEATURE_NAMES])
            else:
                x = np.asarray(features, dtype=np.float64).ravel()
            if x.size != len(FEATURE_NAMES):
                raise ValueError(f"features must have {len(FEATURE_NAMES)} values")
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            predicted = float(self._weights[idx] @ x + self._bias[idx])
            error = float(np.clip(float(reward) - predicted, -3.0, 3.0))
            self._weights[idx] += self.config.learning_rate * error * x
            self._weights[idx] -= self.config.weight_decay * self._weights[idx]
            self._bias[idx] += self.config.learning_rate * error
            return {
                "action": action,
                "predicted": predicted,
                "reward": float(reward),
                "prediction_error": error,
                "weight_l2": float(np.linalg.norm(self._weights[idx])),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._last_advice is None:
                return {
                    "status": "idle",
                    "belief": {
                        name: float(p)
                        for name, p in zip(STATE_NAMES, self._belief, strict=True)
                    },
                    "neural_state": self._srm.summary(),
                }
            data = self._last_advice.to_dict()
            data["status"] = "active"
            return data

    def _state_likelihood(self, x: np.ndarray) -> np.ndarray:
        std = np.array([0.20, 0.22, 0.24, 0.24, 0.22, 0.20, 0.24, 0.22], dtype=np.float64)
        z = (x[None, :] - _STATE_MEANS) / std[None, :]
        log_likelihood = -0.5 * np.sum(z * z, axis=1)
        log_likelihood -= float(np.max(log_likelihood))
        return _normalize(np.exp(np.clip(log_likelihood, -60.0, 60.0)))

    def _score_actions(self, x: np.ndarray, uncertainty: float, *, is_background: bool) -> np.ndarray:
        learned = self._weights @ x + self._bias
        expected = _REWARD_BY_STATE @ self._belief
        epistemic = _ACTION_CONTROLLABILITY * uncertainty
        optimism = np.array([0.06 / math.sqrt(float(c + 1)) for c in self._counts])
        cost = _ACTION_COST + (0.18 if is_background else 0.0)
        risk = _ACTION_RISK
        scores = learned + expected + 0.38 * epistemic + optimism - cost - 0.85 * risk
        if x[4] > 0.60:
            scores[ACTION_NAMES.index("use_governed_tools")] += 0.40 * x[4]
        if x[5] > 0.55:
            scores[ACTION_NAMES.index("repair_first")] += 0.45 * x[5]
            scores[ACTION_NAMES.index("reduce_load")] += 0.25 * x[5]
        if x[0] < 0.35:
            scores[ACTION_NAMES.index("ask_clarification")] += 0.35 * (1.0 - x[0])
        if x[3] > 0.60 and x[4] < 0.50:
            scores[ACTION_NAMES.index("seek_information")] += 0.30 * x[3]
        return scores

    def _temperature(self, features: Mapping[str, float], uncertainty: float) -> float:
        base = 0.58 + 0.20 * features["novelty"] - 0.18 * features["error_pressure"]
        base += 0.12 * uncertainty
        return max(0.30, min(0.92, base))

    def _routing_bias(
        self,
        action: str,
        features: Mapping[str, float],
        uncertainty: float,
        is_background: bool,
    ) -> dict[str, Any]:
        return {
            "prefer_direct_answer": action == "answer_directly" and uncertainty < 0.55,
            "prefer_deep_reasoning": action == "plan_deeply",
            "seek_information": action == "seek_information",
            "use_tool_gateway": action == "use_governed_tools" or features["tool_pressure"] >= 0.58,
            "ask_clarification": (
                action == "ask_clarification"
                or (features["clarity"] < 0.34 and features["tool_pressure"] < 0.45)
                or (uncertainty >= 0.62 and features["tool_pressure"] < 0.45)
            ),
            "reduce_load": action == "reduce_load" or features["energy"] < 0.22,
            "repair_first": action == "repair_first",
            "metacognition_depth": round(
                _clamp(0.35 + 0.45 * uncertainty + 0.35 * features["error_pressure"], 0.0, 1.0),
                4,
            ),
            "tool_pressure": round(features["tool_pressure"], 4),
            "background_request": bool(is_background),
        }

    def _sampling_bias(
        self,
        action: str,
        features: Mapping[str, float],
        uncertainty: float,
    ) -> dict[str, float]:
        reduce_load = action == "reduce_load" or features["energy"] < 0.22
        concise_clarification = action == "ask_clarification"
        return {
            "temperature_delta": round(
                _clamp(
                    -0.10 * uncertainty
                    - 0.10 * features["error_pressure"]
                    + 0.06 * features["novelty"],
                    -0.20,
                    0.12,
                ),
                4,
            ),
            "top_p_delta": round(_clamp(-0.10 * uncertainty - 0.04 * features["error_pressure"], -0.20, 0.04), 4),
            "max_tokens_factor": round(
                0.62 if reduce_load else (0.75 if concise_clarification else 1.0),
                4,
            ),
            "repetition_penalty_delta": round(
                _clamp(0.04 * features["error_pressure"] + 0.03 * uncertainty, 0.0, 0.08),
                4,
            ),
            "presence_penalty_delta": round(
                _clamp(0.08 * features["novelty"] + 0.04 * features["tool_pressure"], 0.0, 0.12),
                4,
            ),
        }

    def _features_from_runtime(
        self,
        objective: str,
        *,
        context: Mapping[str, Any] | None,
        state: Any,
    ) -> dict[str, float]:
        text = str(objective or "")
        lowered = text.lower()
        length_pressure = _clamp(len(text) / 2400.0)
        question_pressure = _clamp((text.count("?") + lowered.count(" or ")) / 5.0)
        uncertainty = _clamp(
            (0.30 if _UNCERTAIN_RE.search(text) else 0.0)
            + 0.25 * question_pressure
            + 0.18 * length_pressure
        )
        tool_pressure = _clamp(
            (0.62 if _TOOL_RE.search(text) else 0.0)
            + (0.20 if _RESEARCH_RE.search(text) else 0.0)
        )
        research_pressure = 0.38 if _RESEARCH_RE.search(text) else 0.0
        error_pressure = 0.70 if _ERROR_RE.search(text) else 0.0
        memory_pressure = 0.62 if _MEMORY_RE.search(text) else 0.0
        social_pressure = 0.55 if _SOCIAL_RE.search(text) else 0.0
        urgency = 0.72 if any(token in lowered for token in ("now", "urgent", "immediately", "top priority", "fix")) else 0.30

        modifiers = getattr(state, "response_modifiers", {}) if state is not None else {}
        if not isinstance(modifiers, Mapping):
            modifiers = {}
        context = context or {}
        energy = _safe_float(modifiers.get("homeostatic_energy"), 75.0) / 100.0
        energy = _safe_float(context.get("homeostatic_energy"), energy)
        anomaly = _safe_float(modifiers.get("anomaly_threat_level"), 0.0)
        free_energy = _safe_float(modifiers.get("fe", modifiers.get("free_energy", 0.0)), 0.0)
        if bool(context.get("desktop_cognitive_engine_required")):
            tool_pressure = max(tool_pressure, 0.08)
        if modifiers.get("intent_type") == "SKILL" or modifiers.get("matched_skills"):
            tool_pressure = max(tool_pressure, 0.72)
        error_pressure = max(error_pressure, _clamp(anomaly), _clamp(free_energy))
        novelty = _clamp(
            research_pressure
            + (0.28 if any(word in lowered for word in ("new", "novel", "unknown", "idea", "explore")) else 0.0)
            + 0.20 * uncertainty
        )
        clarity = _clamp(1.0 - uncertainty - 0.35 * error_pressure + 0.10 * social_pressure)
        return {
            "clarity": clarity,
            "energy": _clamp(energy),
            "urgency": _clamp(urgency + 0.20 * error_pressure),
            "novelty": novelty,
            "tool_pressure": tool_pressure,
            "error_pressure": error_pressure,
            "social_pressure": _clamp(social_pressure),
            "memory_pressure": _clamp(memory_pressure),
        }

    def _rationale(
        self,
        action: str,
        features: Mapping[str, float],
        uncertainty: float,
        routing_bias: Mapping[str, Any],
    ) -> list[str]:
        rationale = [
            f"selected={action}",
            f"uncertainty={uncertainty:.3f}",
            f"clarity={features['clarity']:.3f}",
            f"tool_pressure={features['tool_pressure']:.3f}",
            f"error_pressure={features['error_pressure']:.3f}",
        ]
        if routing_bias.get("use_tool_gateway"):
            rationale.append("tool effects must go through governed capability path")
        if routing_bias.get("reduce_load"):
            rationale.append("load reduction requested by low energy or overload belief")
        return rationale

    @staticmethod
    def _advice_id(objective: str, origin: str) -> str:
        payload = f"{time.time_ns()}:{origin}:{objective[:240]}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()[:16]


_ADVISOR: SpikingActiveInferenceAdvisor | None = None
_ADVISOR_LOCK = threading.Lock()


def _register_advisor(advisor: SpikingActiveInferenceAdvisor) -> None:
    try:
        from core.container import ServiceContainer

        current = ServiceContainer.get("spiking_active_inference", default=None)
        if current is advisor:
            return
        ServiceContainer.register_instance(
            "spiking_active_inference",
            advisor,
            required=False,
            owner="core/cognitive/spiking_active_inference.py",
            required_for="cognitive advisory routing and substrate sampling",
            failure_policy="degrade_without_effects",
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return


def get_spiking_active_inference_advisor() -> SpikingActiveInferenceAdvisor:
    global _ADVISOR
    with _ADVISOR_LOCK:
        if _ADVISOR is None:
            _ADVISOR = SpikingActiveInferenceAdvisor()
        _register_advisor(_ADVISOR)
        return _ADVISOR


__all__ = [
    "ACTION_NAMES",
    "FEATURE_NAMES",
    "STATE_NAMES",
    "MultiCompartmentSpikeResponseModel",
    "NeurodynamicAdvice",
    "SpikingActiveInferenceAdvisor",
    "SpikingActiveInferenceConfig",
    "get_spiking_active_inference_advisor",
]
