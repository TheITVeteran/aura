"""core/brain/homeostatic_modulator.py
====================================
Computes real-time inference modulation parameters from homeostatic loops.
Maps FitzHugh-Nagumo, FreeEnergy, and LiquidSubstrate states to temperature,
top_p, repetition_penalty, and token-level logit biases.

These values reach the sampler, so an unvalidated NaN or an absent organ does
not merely produce a bad number — it produces a bad *generation*. Two rules
apply throughout:

* A default is labelled a default. When an organ is missing, the snapshot says
  ``measured: false`` for that channel rather than presenting a placeholder as
  a real-time reading (CP126 59c7356b).
* Every reading is validated before it is used, via the shared primitives in
  core/runtime/numeric_safety.py — the same defect class as the tiered-action
  risk inputs (CP126 2bcec133).

CP126 59c7356b / 67d4e9a4 / 2bcec133 / 1233aa4c / d22709dc / 1eb6e7ee.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.numeric_safety import validated_scalar, validated_unit

logger = logging.getLogger("Aura.Brain.HomeostaticModulator")

#: Ceiling on how many learned token weights one foreground inference may
#: score. CP126 1eb6e7ee: the scan was unbounded and ran under the projection
#: lock, so learned-map growth directly raised inference latency.
MAX_SCORED_TOKENS = 4096

#: Bounds on every parameter that reaches the sampler.
PARAM_BOUNDS = {
    "temperature": (0.05, 2.0),
    "top_p": (0.05, 1.0),
    "repetition_penalty": (1.0, 2.0),
    "urgency": (0.0, 1.0),
}

_HOMEOSTATIC_RECOVERABLE_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
    ImportError,
    LookupError,
    TimeoutError,
    json.JSONDecodeError,
)


def _record_homeostatic_degradation(exc: BaseException, *, action: str, severity: str = "warning") -> None:
    record_degradation("homeostatic_modulator", exc, severity=severity, action=action)


@dataclass
class InferenceModulation:
    """Modulation parameters to be injected directly into LLM inference."""

    temperature: float
    top_p: float
    repetition_penalty: float
    logit_bias: dict[int, float]
    head_weights: np.ndarray
    urgency: float
    source_snapshot: dict[str, Any] = field(default_factory=dict)


class SubstrateLogitProjection:
    """Learned sparse mapping from substrate state to token logit biases.

    Uses Hebbian plasticity: when generation succeeds (positive coherence,
    low surprise), strengthen the connection between current substrate state
    and generated tokens. When surprise is high or coherence is negative, weaken.
    """

    def __init__(self, substrate_dim: int = 512, save_path: str | None = None) -> None:
        self.substrate_dim = substrate_dim
        # Map: token_id -> np.ndarray of shape (substrate_dim,) containing association weights
        self.weights: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._last_mismatch: tuple[int, str] | None = None
        #: What the last projection actually did, for telemetry.
        self.last_projection: dict[str, Any] = {}

        if save_path:
            self.save_path = Path(save_path)
        else:
            try:
                from core.config import config as aura_config
                self.save_path = aura_config.paths.data_dir / "substrate_logit_projection.json"
            except _HOMEOSTATIC_RECOVERABLE_ERRORS as exc:
                _record_homeostatic_degradation(
                    exc,
                    action="used local projection path after Aura config lookup failed",
                )
                self.save_path = Path("data/substrate_logit_projection.json")

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def _project(self, substrate_state: Any) -> tuple[np.ndarray, str]:
        """Coerce a substrate vector to the projection's dimension.

        CP126 d22709dc: a mismatched vector was silently truncated or
        zero-padded positionally, so an incompatible substrate LAYOUT was
        treated as compatible and learned weights bound to the wrong
        biological variables. The coercion still happens — there is no channel
        schema to project through yet — but it is now reported, and a caller
        can see from ``self.last_projection`` that its axes are not trusted.
        """
        try:
            state = np.asarray(substrate_state, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return np.zeros(self.substrate_dim, dtype=np.float32), "unreadable"
        if not np.all(np.isfinite(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        if state.size == self.substrate_dim:
            return state, "exact"
        resized = np.zeros(self.substrate_dim, dtype=np.float32)
        copy_len = min(state.size, self.substrate_dim)
        resized[:copy_len] = state[:copy_len]
        kind = "truncated" if state.size > self.substrate_dim else "zero_padded"
        if self._last_mismatch != (state.size, kind):
            self._last_mismatch = (state.size, kind)
            logger.warning(
                "Substrate vector %s to fit the projection (%d -> %d); axis "
                "identity is positional and unverified.",
                kind, state.size, self.substrate_dim,
            )
            _record_homeostatic_degradation(
                ValueError(f"substrate dim {state.size} != projection dim {self.substrate_dim}"),
                action="projected a mismatched substrate vector positionally",
            )
        return resized, kind

    def get_biases(self, substrate_state: np.ndarray) -> dict[int, float]:
        """Compute logit biases for the most strongly associated tokens.

        CP126 1eb6e7ee: this held the projection lock while computing a dot
        product for EVERY persisted token, with no vocabulary bound, index or
        top-k preselection — so learned-map growth directly increased
        foreground inference latency. The scoring is now vectorized, bounded,
        and done outside the lock over a snapshot.
        """
        state, projection_kind = self._project(substrate_state)
        if state.size == 0 or not np.any(state):
            self.last_projection = {"kind": projection_kind, "scored": 0, "biased": 0}
            return {}

        # Snapshot under the lock, score outside it.
        with self._lock:
            if not self.weights:
                self.last_projection = {"kind": projection_kind, "scored": 0, "biased": 0}
                return {}
            token_ids = list(self.weights.keys())
            compatible = [
                (token_id, self.weights[token_id])
                for token_id in token_ids
                if getattr(self.weights[token_id], "size", 0) == self.substrate_dim
            ]
            skipped = len(token_ids) - len(compatible)

        truncated = len(compatible) > MAX_SCORED_TOKENS
        if truncated:
            # Deterministic bound: strongest-norm weights first.
            compatible.sort(key=lambda item: float(np.linalg.norm(item[1])), reverse=True)
            compatible = compatible[:MAX_SCORED_TOKENS]

        if not compatible:
            self.last_projection = {
                "kind": projection_kind, "scored": 0, "biased": 0, "skipped": skipped,
            }
            return {}

        matrix = np.stack([weight for _, weight in compatible])
        scores = np.clip(matrix @ state * 0.5, -2.0, 2.0)
        biases = {
            compatible[index][0]: float(scores[index])
            for index in np.nonzero(np.abs(scores) > 0.05)[0]
        }
        self.last_projection = {
            "kind": projection_kind,
            "scored": len(compatible),
            "biased": len(biases),
            "skipped": skipped,
            "truncated": truncated,
        }
        if skipped:
            logger.debug("Skipped %d incompatible substrate projection weights", skipped)
        return biases

    def learn_step(
        self,
        substrate_state: np.ndarray,
        token_ids: list[int],
        feedback_coherence: float,
        surprise: float,
        lr: float = 0.005
    ) -> None:
        """Update projection weights via reward-modulated Hebbian learning.

        Feedback coherence: positive when output is aligned with substrate goals.
        Surprise: prediction error/surprise of the generated text (perplexity).
        """
        with self._lock:
            if len(substrate_state) == 0 or not token_ids:
                return

            state, _ = self._project(substrate_state)

            # Reward signal: High coherence and low surprise = high reward.
            # CP126 2bcec133: an unvalidated coherence or surprise wrote NaN
            # straight into the persisted weights, poisoning every later bias.
            coherence = float(
                validated_scalar(feedback_coherence, name="feedback_coherence",
                                 low=-1.0, high=1.0, default=0.0)
            )
            surprise_value = float(
                validated_scalar(surprise, name="surprise", low=0.0, high=50.0, default=0.0)
            )
            learning_rate = float(
                validated_scalar(lr, name="lr", low=0.0, high=1.0, default=0.005)
            )
            reward = coherence * math.exp(-surprise_value)

            for token_id in token_ids:
                if token_id not in self.weights:
                    self.weights[token_id] = np.zeros(self.substrate_dim, dtype=np.float32)

                # Hebbian weight update: dW = lr * reward * state
                self.weights[token_id] += learning_rate * reward * state

                # Weight decay/regularization to prevent unbounded growth
                self.weights[token_id] *= 0.98

                # Clip weight vector for stability
                self.weights[token_id] = np.clip(self.weights[token_id], -1.0, 1.0)

            # Prune near-zero weights to keep memory usage sparse
            inactive = [tid for tid, w in self.weights.items() if np.linalg.norm(w) < 1e-4]
            for tid in inactive:
                del self.weights[tid]

    # -- Persistence -----------------------------------------------------------

    def save(self) -> None:
        """Persist weights to disk."""
        with self._lock:
            payload = {
                "substrate_dim": self.substrate_dim,
                "weights": {
                    str(tid): w.tolist()
                    for tid, w in self.weights.items()
                }
            }
            try:
                atomic_write_text(self.save_path, json.dumps(payload, indent=2), encoding="utf-8")
                logger.debug("Persisted SubstrateLogitProjection to: %s", self.save_path)
            except _HOMEOSTATIC_RECOVERABLE_ERRORS as exc:
                _record_homeostatic_degradation(
                    exc,
                    action="continued without persisting substrate logit projection",
                    severity="error",
                )
                logger.error("Failed to save SubstrateLogitProjection: %s", exc)

    def load(self) -> None:
        """Restore weights from disk."""
        if not self.save_path.exists():
            return
        try:
            payload = json.loads(self.save_path.read_text(encoding="utf-8"))
            expected_dim = self.substrate_dim
            persisted_dim = int(payload.get("substrate_dim", expected_dim))
            if persisted_dim != expected_dim:
                logger.warning(
                    "Ignoring SubstrateLogitProjection with incompatible substrate_dim %s at %s; expected %s",
                    persisted_dim,
                    self.save_path,
                    expected_dim,
                )
                return
            weights_raw = payload.get("weights", {})
            weights: dict[int, np.ndarray] = {}
            for tid, raw_weight in weights_raw.items():
                weight = np.asarray(raw_weight, dtype=np.float32)
                if len(weight) != expected_dim:
                    logger.warning(
                        "Skipping incompatible substrate projection weight for token %s at %s: %s != %s",
                        tid,
                        self.save_path,
                        len(weight),
                        expected_dim,
                    )
                    continue
                weights[int(tid)] = weight
            with self._lock:
                self.weights = weights
            logger.info("Loaded SubstrateLogitProjection from: %s", self.save_path)
        except _HOMEOSTATIC_RECOVERABLE_ERRORS as exc:
            _record_homeostatic_degradation(
                exc,
                action="continued with empty substrate logit projection after load failed",
            )
            logger.error("Failed to load SubstrateLogitProjection: %s", exc)


class HomeostaticModulator:
    """Coordinates and converts live states from the FitzHugh-Nagumo oscillator,
    Free Energy engine, and Liquid Substrate into InferenceModulation parameters.
    """

    def __init__(self, substrate_dim: int = 512) -> None:
        self.projection = SubstrateLogitProjection(substrate_dim=substrate_dim)

    def compute_modulation(self) -> InferenceModulation:
        """Read and bundle current state values into an InferenceModulation object."""
        from core.container import ServiceContainer

        # 1. Retrieve engines
        precision_engine = ServiceContainer.get("precision_engine", default=None)
        free_energy_engine = ServiceContainer.get("free_energy_engine", default=None)
        substrate = ServiceContainer.get("liquid_substrate", default=None)

        # CP126 59c7356b: every channel records whether it was MEASURED or
        # defaulted, so telemetry cannot present a placeholder as a live
        # reading of an organ that is not running.
        availability: dict[str, bool] = {}
        faults: list[str] = []

        # 2. Extract raw states
        arousal, fatigue = 0.5, 0.0
        head_weights = np.ones(32, dtype=np.float32)
        head_weight_source = "default"
        if precision_engine is not None:
            arousal, fatigue, precision_faults = self._read_precision(precision_engine)
            faults.extend(precision_faults)
            head_weights, head_weight_source = self._read_head_weights(precision_engine)
        availability["precision_engine"] = precision_engine is not None

        free_energy, urgency = 0.3, 0.5
        if free_energy_engine is not None:
            free_energy, urgency, fe_faults = self._read_free_energy(free_energy_engine)
            faults.extend(fe_faults)
        availability["free_energy_engine"] = free_energy_engine is not None

        frustration = 0.0
        substrate_state = np.zeros(self.projection.substrate_dim, dtype=np.float32)
        if substrate is not None:
            frustration, substrate_state, substrate_faults = self._read_substrate(substrate)
            faults.extend(substrate_faults)
        availability["liquid_substrate"] = substrate is not None

        # 3. Parameter Mapping
        if precision_engine is not None and hasattr(precision_engine, "get_temperature"):
            try:
                temperature = float(
                    validated_scalar(
                        precision_engine.get_temperature(),
                        name="temperature",
                        low=PARAM_BOUNDS["temperature"][0],
                        high=PARAM_BOUNDS["temperature"][1],
                        default=0.7,
                    )
                )
            except _HOMEOSTATIC_RECOVERABLE_ERRORS as exc:
                _record_homeostatic_degradation(
                    exc, action="used the arousal-derived temperature after the precision engine raised"
                )
                temperature = 0.95 - 0.40 * arousal
        else:
            temperature = 0.95 - 0.40 * arousal
        temperature = self._bounded("temperature", temperature, 0.7)

        # Repetition penalty scales up with substrate frustration to avoid looping
        repetition_penalty = self._bounded("repetition_penalty", 1.1 + 0.3 * frustration, 1.1)

        # Top_p scales down with free energy to constrain responses when highly
        # Surprised/Complex.
        top_p = self._bounded("top_p", max(0.6, min(1.0, 0.95 - 0.25 * free_energy)), 0.95)
        urgency = self._bounded("urgency", urgency, 0.5)

        # Logit bias derived from substrate Hebbian projection
        logit_bias = self.projection.get_biases(substrate_state)

        if faults:
            _record_homeostatic_degradation(
                ValueError("; ".join(faults[:4])),
                action="clamped out-of-contract homeostatic readings before they reached the sampler",
            )

        # 4. Snapshot source values for telemetry/debugging
        source_snapshot = {
            "fhn_arousal": arousal,
            "fhn_fatigue": fatigue,
            "free_energy": free_energy,
            "substrate_frustration": frustration,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "logit_bias_count": len(logit_bias),
            # CP126 59c7356b: the honesty fields.
            "availability": availability,
            "measured": {
                "fhn_arousal": availability["precision_engine"],
                "fhn_fatigue": availability["precision_engine"],
                "free_energy": availability["free_energy_engine"],
                "urgency": availability["free_energy_engine"],
                "substrate_frustration": availability["liquid_substrate"],
                "head_weights": head_weight_source == "measured",
            },
            "head_weight_source": head_weight_source,
            "input_faults": faults,
            "fully_measured": all(availability.values()),
            "captured_at": time.time(),
        }

        return InferenceModulation(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            logit_bias=logit_bias,
            head_weights=head_weights,
            urgency=urgency,
            source_snapshot=source_snapshot,
        )

    @staticmethod
    def _bounded(name: str, value: Any, default: float) -> float:
        low, high = PARAM_BOUNDS[name]
        return float(validated_scalar(value, name=name, low=low, high=high, default=default))

    @staticmethod
    def _read_precision(engine: Any) -> tuple[float, float, list[str]]:
        faults: list[str] = []
        try:
            fhn = engine.fhn
            arousal = validated_unit(fhn.arousal, name="fhn_arousal")
            fatigue = validated_unit(fhn.fatigue, name="fhn_fatigue")
        except _HOMEOSTATIC_RECOVERABLE_ERRORS as exc:
            _record_homeostatic_degradation(
                exc, action="used default arousal/fatigue after the precision engine read failed"
            )
            return 0.5, 0.0, [f"precision read failed: {exc}"]
        faults.extend(fault for fault in (arousal.fault, fatigue.fault) if fault)
        return float(arousal), float(fatigue), faults

    def _read_head_weights(self, engine: Any) -> tuple[np.ndarray, str]:
        """A private, shape-checked copy of the engine's head weights.

        CP126 1233aa4c: the engine-owned array was exposed directly in
        InferenceModulation, so a downstream consumer could mutate live engine
        state and a concurrent update could change it after the snapshot.
        """
        default = np.ones(32, dtype=np.float32)
        try:
            raw = engine.get_head_weights()
        except _HOMEOSTATIC_RECOVERABLE_ERRORS as exc:
            _record_homeostatic_degradation(
                exc, action="used uniform head weights after the precision engine read failed"
            )
            return default, "default"
        try:
            weights = np.array(raw, dtype=np.float32, copy=True)
        except (TypeError, ValueError) as exc:
            _record_homeostatic_degradation(
                exc, action="used uniform head weights after receiving a non-array head weight"
            )
            return default, "default"
        if weights.ndim != 1 or weights.size == 0 or not np.all(np.isfinite(weights)):
            _record_homeostatic_degradation(
                ValueError(f"head weights had shape {weights.shape} / non-finite values"),
                action="used uniform head weights after a shape or finiteness check failed",
            )
            return default, "default"
        weights.setflags(write=False)
        return weights, "measured"

    @staticmethod
    def _read_free_energy(engine: Any) -> tuple[float, float, list[str]]:
        faults: list[str] = []
        try:
            free_energy = validated_scalar(
                getattr(engine, "_smoothed_fe", 0.3),
                name="free_energy", low=0.0, high=100.0, default=0.3,
            )
            if hasattr(engine, "get_action_urgency"):
                urgency = validated_unit(engine.get_action_urgency(), name="urgency")
            else:
                urgency = validated_unit(float(free_energy), name="urgency")
        except _HOMEOSTATIC_RECOVERABLE_ERRORS as exc:
            _record_homeostatic_degradation(
                exc, action="used default free-energy/urgency after the engine read failed"
            )
            return 0.3, 0.5, [f"free-energy read failed: {exc}"]
        faults.extend(fault for fault in (free_energy.fault, urgency.fault) if fault)
        return float(free_energy), float(urgency), faults

    def _read_substrate(self, substrate: Any) -> tuple[float, np.ndarray, list[str]]:
        """One coherent snapshot of the substrate.

        CP126 67d4e9a4: frustration was read from ``substrate.x`` BEFORE the
        lock was taken while the activation vector was copied under it, so a
        concurrent update could pair a frustration value from one instant with
        a vector from another.
        """
        faults: list[str] = []
        empty = np.zeros(self.projection.substrate_dim, dtype=np.float32)
        try:
            with substrate.sync_lock:
                vector = np.array(substrate.x, dtype=np.float32, copy=True)
                index = int(substrate.idx_frustration)
        except _HOMEOSTATIC_RECOVERABLE_ERRORS as exc:
            _record_homeostatic_degradation(
                exc, action="used a zero substrate snapshot after the coherent read failed"
            )
            return 0.0, empty, [f"substrate read failed: {exc}"]

        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            faults.append("substrate vector was malformed or non-finite")
            vector = np.nan_to_num(
                vector.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
        raw = vector[index] if 0 <= index < vector.size else 0.0
        frustration = validated_unit(raw, name="substrate_frustration")
        if frustration.fault:
            faults.append(frustration.fault)
        return float(frustration), vector, faults
