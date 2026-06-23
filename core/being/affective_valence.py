from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .aura_now import AffectiveState, BodyState, PredictionState, WorldState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class AffectiveValenceEngine:
    """Turns interoceptive prediction error into control variables."""

    _lesioned: bool = False

    def lesion(self) -> None:
        """Disable affective control for lesion and ablation tests."""
        self._lesioned = True

    def restore(self) -> None:
        """Restore affective control after a lesion test."""
        self._lesioned = False

    @property
    def is_lesioned(self) -> bool:
        return self._lesioned

    def compute(
        self,
        *,
        body: BodyState,
        prediction: PredictionState,
        world: WorldState,
        base_valence: float = 0.0,
        base_arousal: float = 0.5,
        lesion: bool = False,
    ) -> AffectiveState:
        if lesion or self._lesioned:
            return AffectiveState(
                valence=0.0,
                arousal=0.5,
                distress=0.0,
                curiosity=0.0,
                care=0.0,
                boredom=0.8,
                dominance=0.5,
                free_energy=0.0,
                precision={"sensory": 0.5, "policy": 0.5, "interoception": 0.5},
                control_effects={"learning_rate": 0.0, "exploration_temperature": 0.1, "memory_priority": 0.2},
                dominant_drive="lesioned_affect",
            )

        free_energy = _clip(prediction.free_energy)
        controllability = _clip(prediction.controllability)
        pressure = body.total_pressure
        uncertainty = _clip(world.uncertainty + free_energy)
        distress = _clip((pressure * 0.45 + free_energy * 0.45 + world.uncertainty * 0.25) * (1.15 - controllability))
        curiosity = _clip(prediction.expected_information_gain * (0.35 + controllability) * (1.0 - distress * 0.35))
        boredom = _clip((1.0 - curiosity) * (1.0 - pressure) * (0.8 if world.task_active else 1.0))
        care = _clip((world.relationship_count / 5.0) * (1.0 - distress * 0.2))
        valence_raw = (
            float(base_valence)
            + (controllability * 0.55)
            + (curiosity * 0.20)
            + (care * 0.15)
            - (distress * 0.75)
            - (uncertainty * 0.25)
        )
        valence = max(-1.0, min(1.0, math.tanh(valence_raw)))
        arousal = _clip(max(float(base_arousal), pressure * 0.65 + free_energy * 0.55 + curiosity * 0.35))
        dominance = _clip(controllability * 0.7 + (1.0 - distress) * 0.3)
        dominant_drive = "repair" if distress > 0.55 else "curiosity" if curiosity > 0.45 else "coherence"
        precision = {
            "sensory": round(_clip(0.35 + arousal * 0.45 + uncertainty * 0.2), 4),
            "policy": round(_clip(0.35 + dominance * 0.4 + max(0.0, valence) * 0.2), 4),
            "interoception": round(_clip(0.35 + distress * 0.55 + pressure * 0.2), 4),
        }
        control_effects = {
            "learning_rate": round(_clip(0.05 + arousal * 0.18 + prediction.expected_information_gain * 0.12), 4),
            "exploration_temperature": round(_clip(0.15 + curiosity * 0.65 - distress * 0.25), 4),
            "memory_priority": round(_clip(0.25 + abs(valence) * 0.2 + distress * 0.35 + care * 0.2), 4),
            "patience": round(_clip(0.35 + dominance * 0.45 - distress * 0.25), 4),
        }
        return AffectiveState(
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            distress=round(distress, 4),
            curiosity=round(curiosity, 4),
            care=round(care, 4),
            boredom=round(boredom, 4),
            dominance=round(dominance, 4),
            free_energy=round(free_energy, 4),
            precision=precision,
            control_effects=control_effects,
            dominant_drive=dominant_drive,
        )


def generation_controls(affect: AffectiveState) -> dict[str, float]:
    """Direct (non-text) generation-control modulations from the affect substrate.

    #48 — reduce text-mediation. The affect engine's ``control_effects`` are
    derived into concrete numeric modulations of the generation dynamics
    (temperature, token budget, repetition pressure) so a consumer can apply them
    DIRECTLY to sampling, rather than describing affect to the model as a
    ``[Affect: valence=…]`` prompt string. This is the affect engine's direct
    generation-control surface; the live consumed direct pathway today is the
    affective circumplex (see core/brain/inference_gate.py, gated to non-strict
    user-facing generations). It does NOT eliminate text-mediation (full grounding
    is not claimed, CLAIM_BOUNDARIES 4.B) and carries NO phenomenal claim.

    A lesioned affect substrate yields neutral control_effects → neutral
    modulations, so the contribution is fully ablatable.
    """
    ce = dict(getattr(affect, "control_effects", None) or {})
    explore = float(ce.get("exploration_temperature", 0.15))
    patience = float(ce.get("patience", 0.5))
    learning = float(ce.get("learning_rate", 0.0))
    memory_priority = float(ce.get("memory_priority", 0.2))
    return {
        # Curious/exploratory affect raises sampling temperature; distressed lowers it.
        "temperature_delta": round((explore - 0.40) * 0.5, 4),
        # Patient affect is more expansive with the token budget; impatient is terser.
        "max_tokens_scale": round(_clip(0.80 + patience * 0.40, 0.6, 1.3), 4),
        # Low patience (often distress) raises repetition pressure (anti-rumination).
        "repetition_penalty_delta": round((1.0 - patience) * 0.15, 4),
        # Consolidation/plasticity rate and memory write salience flow from affect.
        "learning_rate": round(_clip(learning, 0.0, 1.0), 4),
        "memory_salience": round(_clip(memory_priority, 0.0, 1.0), 4),
    }


def compute_affective_state(**kwargs: Any) -> AffectiveState:
    return AffectiveValenceEngine().compute(**kwargs)
