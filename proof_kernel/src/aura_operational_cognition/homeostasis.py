from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class ExternalSignals:
    health_error_rate: float = 0.0
    resource_anxiety: float = 0.0
    thermal_load: float = 0.0
    sovereignty_score: float = 1.0


class HomeostasisEngine:
    DRIVE_NAMES = ("integrity", "persistence", "curiosity", "metabolism", "sovereignty")

    def __init__(self):
        self.integrity = 1.0
        self.persistence = 1.0
        self.curiosity = 0.5
        self.metabolism = 0.5
        self.sovereignty = 1.0
        self._last_update = time.time()
        self._error_count = 0
        self._setpoints = {
            "integrity": 0.90,
            "persistence": 0.85,
            "curiosity": 0.55,
            "metabolism": 0.65,
            "sovereignty": 0.95,
        }
        self._setpoint_adaptation_rate = 0.001
        self._proportional_gain = 0.02
        self._vitality_history: deque[float] = deque(maxlen=120)
        self._successful_responses = 0
        self._failed_responses = 0
        self._total_responses = 0
        self._vitality_weights = {
            "integrity": 0.35,
            "persistence": 0.25,
            "curiosity": 0.15,
            "metabolism": 0.15,
            "sovereignty": 0.10,
        }

    def get_status(self) -> dict[str, float]:
        return {
            "integrity": round(float(self.integrity), 3),
            "persistence": round(float(self.persistence), 3),
            "curiosity": round(float(self.curiosity), 3),
            "metabolism": round(float(self.metabolism), 3),
            "sovereignty": round(float(self.sovereignty), 3),
            "will_to_live": round(float(self.compute_vitality()), 3),
        }

    def compute_vitality(self) -> float:
        score = 0.0
        for drive_name, weight in self._vitality_weights.items():
            current = getattr(self, drive_name)
            setpoint = self._setpoints[drive_name]
            proximity = 1.0 - min(1.0, abs(current - setpoint) / max(setpoint, 0.01))
            drive_contribution = 0.6 * current + 0.4 * proximity
            score += drive_contribution * weight
        return score

    async def pulse(self, signals: ExternalSignals | None = None) -> dict[str, Any]:
        now = time.time()
        delta = now - self._last_update
        self._last_update = now
        active_signals = signals or ExternalSignals()

        if active_signals.health_error_rate > 0.1:
            self.integrity = max(0.0, self.integrity - (active_signals.health_error_rate * 0.1))
        if active_signals.resource_anxiety > 0.8:
            self.persistence = max(0.0, self.persistence - 0.01)
        if active_signals.thermal_load > 0.8:
            self.metabolism = max(0.2, self.metabolism - 0.05)
        if active_signals.sovereignty_score < 1.0:
            self.sovereignty = max(0.0, self.sovereignty - (1.0 - active_signals.sovereignty_score) * 0.1)

        for drive_name in self.DRIVE_NAMES:
            current = getattr(self, drive_name)
            setpoint = self._setpoints[drive_name]
            error = setpoint - current
            adjustment = error * self._proportional_gain
            setattr(self, drive_name, max(0.0, min(1.0, current + adjustment)))

        for drive_name in self.DRIVE_NAMES:
            current = getattr(self, drive_name)
            setpoint = self._setpoints[drive_name]
            drift = (current - setpoint) * self._setpoint_adaptation_rate
            self._setpoints[drive_name] = max(0.2, min(0.98, setpoint + drift))

        self.curiosity = max(0.15, self.curiosity - (0.0005 * delta))
        self._vitality_history.append(self.compute_vitality())
        return self.get_status()

    def report_error(self, severity: str = "medium") -> None:
        drain = {"low": 0.01, "medium": 0.05, "high": 0.15, "critical": 0.4}.get(severity, 0.05)
        self.integrity = max(0.0, self.integrity - drain)
        self._error_count += 1
        self._failed_responses += 1
        self._total_responses += 1

    def feed_curiosity(self, amount: float = 0.1) -> None:
        self.curiosity = min(1.0, self.curiosity + amount)

    def on_response_success(self, response_length: int = 0) -> None:
        self._successful_responses += 1
        self._total_responses += 1
        self.integrity = min(1.0, self.integrity + 0.008)
        self.persistence = min(1.0, self.persistence + 0.003)
        if response_length > 500:
            self.metabolism = max(0.1, self.metabolism - 0.005)
            self.feed_curiosity(0.015)
        elif response_length > 100:
            self.feed_curiosity(0.008)

    def on_response_error(self, error_type: str = "inference") -> None:
        severity_map = {
            "inference": "medium",
            "timeout": "medium",
            "model_crash": "high",
            "empty_response": "low",
        }
        self.report_error(severity_map.get(error_type, "medium"))

    def get_dominant_deficiency(self) -> tuple[str, float]:
        worst_drive = "integrity"
        worst_deficit = 0.0
        for drive_name in self.DRIVE_NAMES:
            current = getattr(self, drive_name)
            setpoint = self._setpoints[drive_name]
            deficit = setpoint - current
            if deficit > worst_deficit:
                worst_deficit = deficit
                worst_drive = drive_name
        return worst_drive, round(worst_deficit, 3)

    def get_vitality_trend(self) -> str:
        if len(self._vitality_history) < 10:
            return "stable"
        recent = list(self._vitality_history)[-10:]
        slope = recent[-1] - recent[0]
        if slope > 0.02:
            return "rising"
        if slope < -0.02:
            return "falling"
        return "stable"

    def get_response_success_rate(self) -> float:
        if self._total_responses == 0:
            return 1.0
        return self._successful_responses / self._total_responses

    def get_inference_modifiers(self) -> dict[str, float]:
        vitality = self.compute_vitality()
        return {
            "temperature_mod": round((self.integrity - 0.5) * 0.3, 3),
            "token_multiplier": round(0.7 + (self.metabolism * 0.5), 3),
            "caution_level": round(1.0 - min(self.integrity, self.sovereignty), 3),
            "exploration_tendency": round(self.curiosity * 0.8, 3),
            "vitality": round(vitality, 3),
        }

    def get_context_block(self) -> str:
        vitality = self.compute_vitality()
        trend = self.get_vitality_trend()
        deficiency, deficit = self.get_dominant_deficiency()
        if deficit > 0.3:
            alert = f" | need: {deficiency} (deficit {deficit:.2f})"
        elif deficit > 0.15:
            alert = f" | watch: {deficiency}"
        else:
            alert = " | drives balanced"
        return (
            f"## HOMEOSTASIS\n"
            f"Vitality: {vitality:.2f} ({trend}){alert} | "
            f"success_rate: {self.get_response_success_rate():.0%}"
        )
