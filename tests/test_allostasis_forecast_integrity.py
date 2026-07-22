"""CP126: the forecast ledger must be able to record a failure.

Four defects combined into a system whose calibration could not go down: an
open band was revised before scoring, a late crossing counted as a hit, an
unrelated escalation relabelled a miss INTERVENED, and intervened forecasts
are excluded from the coverage denominator. Each one alone removes failures;
together they made coverage self-validating.
"""
from __future__ import annotations

import inspect

from core.autonomic.allostasis import (
    ForecastOutcome,
    _VitalCalibration,
)


class TestLateCrossingIsNotAHit:
    def test_miss_late_outcome_exists(self):
        assert ForecastOutcome.MISS_LATE.value == "miss_late"

    def test_resolution_checks_the_upper_deadline(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._resolve_due_forecasts)
        # The crossed branch runs before the expiry branch, so it must reject
        # late crossings itself or they can never be scored as misses.
        assert "ForecastOutcome.MISS_LATE" in source
        assert "now > scored_upper + self._resolution_grace_s" in source

    def test_miss_late_counts_toward_coverage(self):
        book = _VitalCalibration()
        book.hits = 1
        book.miss_late = 1
        assert book.scored == 2, "a late crossing must stay in the denominator"
        assert book.coverage == 0.5


class TestScoringUsesThePreregisteredBand:
    def test_forecast_records_the_band_as_issued(self):
        from core.autonomic.allostasis import Forecast

        fields = Forecast.__dataclass_fields__
        assert "first_eta_lower_unix" in fields
        assert "first_eta_upper_unix" in fields

    def test_resolution_scores_the_issued_band(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._resolve_due_forecasts)
        assert "fc.first_eta_lower_unix or fc.eta_lower_unix" in source
        assert "fc.first_eta_upper_unix or fc.eta_upper_unix" in source

    def test_revision_still_permitted_for_the_operational_band(self):
        import inspect as _inspect

        from core.autonomic import allostasis

        source = _inspect.getsource(allostasis)
        # Revisions remain useful for live ETA display; they just cannot
        # change what the forecast is graded against.
        assert "existing.revisions += 1" in source


class TestInterventionMustNameTheVital:
    def test_signature_accepts_a_vital(self):
        from core.autonomic import allostasis

        sig = inspect.signature(allostasis.AllostasisEngine._intervention_since)
        assert "vital" in sig.parameters

    def test_unrelated_vital_does_not_excuse_the_forecast(self):
        from core.autonomic import allostasis

        engine = allostasis.AllostasisEngine.__new__(allostasis.AllostasisEngine)
        engine._interventions = [
            {"at_unix": 100.0, "action": "throttle", "vital": "disk_percent"},
        ]
        # An escalation on a DIFFERENT vital must not excuse this forecast.
        assert engine._intervention_since(50.0, vital="memory_percent") is None
        # The same vital does.
        assert engine._intervention_since(50.0, vital="disk_percent") is not None

    def test_resolution_passes_the_forecast_vital(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._resolve_due_forecasts)
        assert "vital=fc.vital" in source

    def test_composite_intervention_stays_eligible(self):
        from core.autonomic import allostasis

        engine = allostasis.AllostasisEngine.__new__(allostasis.AllostasisEngine)
        # A load-driven escalation has no single driver; it responds to the
        # whole picture and so remains eligible for any open forecast.
        engine._interventions = [
            {"at_unix": 100.0, "action": "throttle", "vital": None},
        ]
        assert engine._intervention_since(50.0, vital="memory_percent") is not None


class TestInterventionsRecordTheirDriver:
    """Matching on a vital is only sound if the vital is actually recorded.

    Intervention records carried no vital at all, so `vital=` matching would
    have made INTERVENED unreachable rather than accurate. The tier decision
    now names the vital that drove it.
    """

    def test_breach_names_the_breached_vital(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._target_tier)
        assert "self._tier_driver_vital = breach" in source

    def test_forecast_driven_escalation_names_the_forecast_vital(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._target_tier)
        assert source.count('self._tier_driver_vital = getattr(nearest, "vital", None)') == 2

    def test_load_driven_escalation_is_composite(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._target_tier)
        assert "self._tier_driver_vital = None" in source

    def test_intervention_record_carries_the_driver(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._note_tier_change)
        assert '"vital": self._tier_driver_vital' in source


class TestAnticipationIsNotAFault:
    def test_protecting_does_not_record_a_degradation(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._raise_protecting)
        # Entering the designed tier must not synthesize a RuntimeError into
        # the global degradation/resilience systems.
        assert 'RuntimeError(f"allostasis entered PROTECTING' not in source
        assert "designed tier" in source

    def test_publish_failure_is_still_a_degradation(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._raise_protecting)
        assert "existential-threat publish failed" in source

    def test_forecast_publishes_its_confidence_contract(self):
        from core.autonomic import allostasis

        source = inspect.getsource(allostasis.AllostasisEngine._raise_protecting)
        for key in ("p_value", "empirical_coverage", "scored_forecasts", '"observed": False'):
            assert key in source, key
