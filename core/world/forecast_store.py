"""core/world/forecast_store.py — Ingestion Forecast Database.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.ForecastStore")


@dataclass
class Forecast:
    forecast_id: str
    target_event: str
    predicted_probability: float
    target_timestamp: float
    created_timestamp: float = field(default_factory=time.time)
    resolution: Optional[str] = None  # "resolved_true", "resolved_false", "stale", "pending"
    resolution_details: Optional[str] = None


class ForecastStore:
    """Stores active forecasts and monitors resolution state."""

    def __init__(self) -> None:
        self.forecasts: Dict[str, Forecast] = {}

    def submit_forecast(self, fc: Forecast) -> None:
        self.forecasts[fc.forecast_id] = fc
        logger.info("🔮 Forecast submitted [%s]: %s (p=%.2f, target_t=%.1f)",
                    fc.forecast_id, fc.target_event, fc.predicted_probability, fc.target_timestamp)

    def resolve_forecast(self, forecast_id: str, success: bool, details: str = "") -> None:
        if forecast_id in self.forecasts:
            fc = self.forecasts[forecast_id]
            fc.resolution = "resolved_true" if success else "resolved_false"
            fc.resolution_details = details
            logger.info("🔮 Forecast resolved [%s]: %s. Details: %s", forecast_id, fc.resolution, details)


# Singleton
_forecast_store_instance: ForecastStore | None = None


def get_forecast_store() -> ForecastStore:
    global _forecast_store_instance
    if _forecast_store_instance is None:
        _forecast_store_instance = ForecastStore()
    return _forecast_store_instance
