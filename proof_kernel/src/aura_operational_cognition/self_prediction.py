from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class InternalStatePrediction:
    predicted_affect_valence: float
    predicted_dominant_drive: str
    predicted_focus_source: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PredictionError:
    valence_error: float
    drive_error: float
    focus_error: float
    composite_error: float
    was_surprising: bool
    timestamp: float = field(default_factory=time.time)


class SelfPredictionLoop:
    _HISTORY_SIZE = 60
    _SURPRISE_THRESHOLD = 0.4
    _ERROR_SMOOTHING = 0.3

    def __init__(self, orchestrator: Optional[Any] = None):
        self.orch = orchestrator
        self._lock: Optional[asyncio.Lock] = None
        self._valence_history: deque[float] = deque(maxlen=self._HISTORY_SIZE)
        self._drive_history: deque[str] = deque(maxlen=self._HISTORY_SIZE)
        self._focus_history: deque[str] = deque(maxlen=self._HISTORY_SIZE)
        self._current_prediction: Optional[InternalStatePrediction] = None
        self._error_history: deque[PredictionError] = deque(maxlen=self._HISTORY_SIZE)
        self._smoothed_error = 0.0
        self._surprise_count = 0
        self._valence_error_ema = 0.0
        self._drive_error_ema = 0.0
        self._focus_error_ema = 0.0

    async def tick(self, actual_valence: float, actual_drive: str, actual_focus_source: str) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._current_prediction is not None:
                error = self._compute_error(
                    self._current_prediction,
                    actual_valence,
                    actual_drive,
                    actual_focus_source,
                )
                self._record_error(error)

            self._valence_history.append(actual_valence)
            self._drive_history.append(actual_drive)
            self._focus_history.append(actual_focus_source)
            self._current_prediction = self._predict_next()

    def get_current_prediction(self) -> Optional[InternalStatePrediction]:
        return self._current_prediction

    def get_surprise_signal(self) -> float:
        return self._smoothed_error

    def get_most_unpredictable_dimension(self) -> str:
        errors = {
            "affect_valence": self._valence_error_ema,
            "drive_state": self._drive_error_ema,
            "attentional_focus": self._focus_error_ema,
        }
        return max(errors, key=errors.get)

    def get_snapshot(self) -> dict[str, Any]:
        pred = self._current_prediction
        return {
            "smoothed_error": round(self._smoothed_error, 3),
            "surprise_count": self._surprise_count,
            "valence_error_ema": round(self._valence_error_ema, 3),
            "drive_error_ema": round(self._drive_error_ema, 3),
            "focus_error_ema": round(self._focus_error_ema, 3),
            "most_unpredictable": self.get_most_unpredictable_dimension(),
            "current_prediction": {
                "affect": round(pred.predicted_affect_valence, 2),
                "drive": pred.predicted_dominant_drive,
                "focus": pred.predicted_focus_source,
                "confidence": round(pred.confidence, 2),
            } if pred else None,
        }

    def _predict_next(self) -> InternalStatePrediction:
        if self._valence_history:
            weights = [index + 1 for index in range(len(self._valence_history))]
            total_weight = sum(weights)
            predicted_valence = sum(value * weight for value, weight in zip(self._valence_history, weights)) / total_weight
        else:
            predicted_valence = 0.0

        if self._drive_history:
            predicted_drive = Counter(list(self._drive_history)[-10:]).most_common(1)[0][0]
        else:
            predicted_drive = "curiosity"

        if self._focus_history:
            predicted_focus = Counter(list(self._focus_history)[-10:]).most_common(1)[0][0]
        else:
            predicted_focus = "drive_curiosity"

        confidence = max(0.1, 1.0 - self._smoothed_error)
        return InternalStatePrediction(
            predicted_affect_valence=round(predicted_valence, 3),
            predicted_dominant_drive=predicted_drive,
            predicted_focus_source=predicted_focus,
            confidence=confidence,
        )

    def _compute_error(
        self,
        prediction: InternalStatePrediction,
        actual_valence: float,
        actual_drive: str,
        actual_focus: str,
    ) -> PredictionError:
        valence_err = abs(prediction.predicted_affect_valence - actual_valence)
        drive_err = 0.0 if prediction.predicted_dominant_drive == actual_drive else 1.0
        focus_err = 0.0 if prediction.predicted_focus_source == actual_focus else 1.0
        composite = min(1.0, (valence_err * 0.3) + (drive_err * 0.4) + (focus_err * 0.3))
        return PredictionError(
            valence_error=round(valence_err, 3),
            drive_error=drive_err,
            focus_error=focus_err,
            composite_error=round(composite, 3),
            was_surprising=composite > self._SURPRISE_THRESHOLD,
        )

    def _record_error(self, error: PredictionError) -> None:
        self._error_history.append(error)
        if error.was_surprising:
            self._surprise_count += 1

        self._smoothed_error = self._ERROR_SMOOTHING * error.composite_error + (1 - self._ERROR_SMOOTHING) * self._smoothed_error
        self._valence_error_ema = self._ERROR_SMOOTHING * error.valence_error + (1 - self._ERROR_SMOOTHING) * self._valence_error_ema
        self._drive_error_ema = self._ERROR_SMOOTHING * error.drive_error + (1 - self._ERROR_SMOOTHING) * self._drive_error_ema
        self._focus_error_ema = self._ERROR_SMOOTHING * error.focus_error + (1 - self._ERROR_SMOOTHING) * self._focus_error_ema
