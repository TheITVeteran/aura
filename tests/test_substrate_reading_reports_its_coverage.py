"""A dead substrate must not read the same as a calm one.

CP126 (high), core/brain/latent_bridge.py: "Partial and total substrate
failures collapse to healthy defaults. Missing services and many errors
return biologically plausible defaults, anti-trap absence is only debug
logged, and viability lookup failure is silently ignored; the output carries
no degraded flag, missing-channel list, uncertainty, freshness, or receipt."

The defaults are plausible on purpose — a caller needs a number to sample
with. What was wrong is that nothing downstream could tell "she is settled"
from "nothing is reporting", and inference temperature, top_p and token
budget are derived from exactly that difference.

Writing the provenance surfaced a second, larger defect. A single
try/except wrapped every channel read, so the FIRST failing service silently
skipped all the rest: one unavailable homeostasis engine blanked phi, free
energy, neurochemistry, affect, active inference and the causal vector, and
the result was still handed back as an ordinary substrate state. Each
channel now has its own guard.
"""
from __future__ import annotations

import pytest

from core.brain import latent_bridge as lb


class TestOneFailingChannelDoesNotBlankTheRest:
    """The larger defect: a shared try/except made the first failure fatal
    to every later read."""

    def test_a_raising_first_channel_leaves_the_others_readable(self, monkeypatch):
        from core.container import ServiceContainer

        real_get = ServiceContainer.get.__func__

        def _get(cls, name, default="_SENTINEL"):
            if name in ("homeostasis_engine", "homeostatic_engine"):
                raise RuntimeError("homeostasis is down")
            return real_get(cls, name, default)

        monkeypatch.setattr(ServiceContainer, "get", classmethod(_get))
        reading = lb._read_substrate_detailed()
        # It must still return a usable reading rather than aborting.
        assert set(reading.values) == set(lb.DEFAULT_SUBSTRATE_STATE)
        assert "vitality" not in reading.sourced

    def test_each_channel_is_guarded_separately(self):
        import inspect

        source = inspect.getsource(lb._read_substrate_detailed)
        # One guard helper, applied per channel.
        assert source.count("_channel(") >= 8


class TestTheReadingCarriesItsProvenance:
    def test_defaulted_channels_are_named(self):
        reading = lb._read_substrate_detailed()
        assert set(reading.sourced).isdisjoint(reading.defaulted)
        assert set(reading.sourced) | set(reading.defaulted) >= set(lb.DEFAULT_SUBSTRATE_STATE)

    def test_a_fully_unsourced_read_is_degraded(self, monkeypatch):
        from core.container import ServiceContainer

        monkeypatch.setattr(
            ServiceContainer, "get", classmethod(lambda cls, name, default=None: None),
        )
        reading = lb._read_substrate_detailed()
        assert reading.coverage == 0.0
        assert reading.degraded is True

    def test_coverage_is_a_fraction(self):
        assert 0.0 <= lb._read_substrate_detailed().coverage <= 1.0

    def test_the_reading_is_stamped(self):
        assert lb._read_substrate_detailed().read_at > 0

    def test_the_receipt_is_serializable(self):
        payload = lb._read_substrate_detailed().to_dict()
        assert payload["schema"] == "aura.substrate_reading.v1"
        assert set(payload) >= {"coverage", "degraded", "sourced", "defaulted", "read_at"}


class TestValuesStayUsable:
    """Reporting degradation must not stop the caller getting numbers —
    inference still has to pick a temperature."""

    def test_every_channel_has_a_value(self):
        values = lb._read_substrate_detailed().values
        assert set(values) == set(lb.DEFAULT_SUBSTRATE_STATE)
        assert all(isinstance(v, float) for v in values.values())

    def test_the_legacy_accessor_is_unchanged(self):
        """Existing callers take a plain dict and must keep working."""
        plain = lb._read_substrate()
        assert isinstance(plain, dict)
        assert set(plain) == set(lb.DEFAULT_SUBSTRATE_STATE)

    def test_inference_params_still_compute(self):
        params = lb.current_inference_params()
        assert params["temperature"] > 0
        assert params["max_tokens"] > 0
