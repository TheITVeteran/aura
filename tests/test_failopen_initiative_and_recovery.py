"""CP126 fail-open: autonomous initiative and the recovery mutex.

* ``b53bc839`` — energy, thermal and load pressure defaulted to the most
  PERMISSIVE values on a read error (full energy, zero pressure), so an
  unreadable body licensed autonomous initiative. The float conversions also
  caught RuntimeError/AttributeError/TypeError but not ValueError — which is
  exactly what ``float("warm")`` raises — so malformed state killed the
  monitor task instead of being handled.
* ``45d7e755`` — on lock-acquisition failure _set_recovery_in_progress wrote
  the shared flag anyway, defeating the mutex precisely under contention and
  letting overlapping recoveries set or clear each other's state.
"""
from __future__ import annotations

import inspect

import pytest


class TestUnreadableBodyDampsInitiative:
    def test_the_permissive_defaults_are_gone(self):
        from core.autonomy import autonomous_initiative_loop as mod

        source = inspect.getsource(mod.AutonomousInitiativeLoop._evaluate_initiative)
        assert "energy = _UNREADABLE_ENERGY" in source
        assert "thermal_pressure = _UNREADABLE_PRESSURE" in source
        assert "load_pressure = _UNREADABLE_PRESSURE" in source

    def test_unknown_energy_is_low_not_full(self):
        """This loop decides whether to act UNPROMPTED. Not knowing the
        body's state is a reason to hold, not a reason to go."""
        from core.autonomy import autonomous_initiative_loop as mod

        assert 0.0 < mod._UNREADABLE_ENERGY < 0.5

    def test_unknown_pressure_is_elevated_not_zero(self):
        from core.autonomy import autonomous_initiative_loop as mod

        assert 0.5 <= mod._UNREADABLE_PRESSURE <= 1.0

    def test_a_permanently_unreadable_sensor_damps_rather_than_abolishes(self):
        """Not 0/1 — initiative should be restrained, not made impossible."""
        from core.autonomy import autonomous_initiative_loop as mod

        assert mod._UNREADABLE_ENERGY > 0.0
        assert mod._UNREADABLE_PRESSURE < 1.0

    def test_value_error_is_caught(self):
        """float("warm") raises ValueError; without it a malformed reading
        escaped and killed the monitor task."""
        from core.autonomy import autonomous_initiative_loop as mod

        source = inspect.getsource(mod.AutonomousInitiativeLoop._evaluate_initiative)
        assert "except (RuntimeError, AttributeError, TypeError):" not in source

    def test_float_of_a_word_really_does_raise_value_error(self):
        with pytest.raises(ValueError):
            float("warm")


class TestTheRecoveryMutexIsNotDefeated:
    def _source(self):
        from core.brain import cognitive_engine as mod

        return inspect.getsource(mod.CognitiveEngine._set_recovery_in_progress)

    def test_it_reports_whether_the_write_took(self):
        source = self._source()
        assert "-> bool" in source

    def test_clearing_without_the_lock_is_refused(self):
        """Clearing can erase a recovery another task still owns."""
        source = self._source()
        assert "refused to clear the recovery flag without the lock" in source
        assert "return False" in source

    def test_setting_without_the_lock_is_allowed_as_over_marking(self):
        """Failing to acquire usually means someone else holds it — i.e. a
        recovery really is in progress — so the write agrees with reality."""
        source = self._source()
        assert "over-marking is safe" in source

    def test_the_unconditional_unsynchronised_write_is_gone(self):
        source = self._source()
        assert "else:\n            self._recovery_in_progress = value" not in source

    def test_both_unsynchronised_paths_are_recorded(self):
        source = self._source()
        assert source.count("record_degradation") == 2
