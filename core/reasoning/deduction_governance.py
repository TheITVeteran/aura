"""Governance signal for logical inconsistency in Aura's beliefs.

When the natural-deduction prover detects that Aura is holding a proposition and
its negation at high confidence, that is a constitutional concern — an
inconsistent self-model — analogous to the homeostatic-pressure breach in the
plasticity substrate. This module records those events, exposes a governance
signal the constitution/governor can poll, and emits a metric.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.reasoning.belief_consistency import ConsistencyReport
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Reasoning.DeductionGovernance")


class _DeductionGovernance:
    def __init__(self) -> None:
        self._last_report: ConsistencyReport | None = None
        self._inconsistency_events = 0
        self._last_contradictions: list[tuple[str, str]] = []
        self._last_checked_at = 0.0

    def record(self, report: ConsistencyReport) -> None:
        self._last_report = report
        self._last_checked_at = time.time()
        if not report.consistent:
            self._inconsistency_events += 1
            self._last_contradictions = list(report.contradictions)
            for affirm, deny in report.contradictions:
                logger.warning(
                    "🧮 [Deduction] belief inconsistency: \"%s\" contradicts \"%s\" — "
                    "constitutional concern, surfacing for revision.",
                    affirm,
                    deny,
                )
            try:
                from core.observability.metrics import get_metrics

                get_metrics().increment_counter("belief_logical_inconsistency_total")
            except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
                record_degradation("deduction_governance", exc)

    def governance_signal(self) -> dict[str, Any]:
        report = self._last_report
        return {
            "beliefs_consistent": bool(report.consistent) if report else True,
            "contradictions": list(self._last_contradictions),
            "inconsistency_events": self._inconsistency_events,
            "checked": report.checked if report else 0,
            "last_checked_at": self._last_checked_at,
        }


_instance: _DeductionGovernance | None = None


def get_deduction_governance() -> _DeductionGovernance:
    global _instance
    if _instance is None:
        _instance = _DeductionGovernance()
    return _instance


def record_belief_inconsistency(report: ConsistencyReport) -> None:
    """Record a belief-consistency report (no-op safe for consistent reports)."""
    get_deduction_governance().record(report)
