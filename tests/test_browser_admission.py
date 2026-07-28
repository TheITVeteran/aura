"""Starting a browser is a resource decision, and nobody was making it.

CP126 (medium), core/capabilities/phantom_browser.py: "Resource-lock failure
is explicitly fail-open. Browser startup continues after homeostatic
resource coordination cannot be acquired. There is no admission decision,
resource budget, or later reconciliation, so memory- or latency-sensitive
runtime periods can still start a full browser while status presents normal
readiness."

The lock was the visible half and the smaller one. Continuing without it is
defensible — it is a courtesy signal telling background work to stand down,
and refusing to browse because a coordination helper is missing would be
over-strict. What was not defensible is that status then reported normal
readiness, so a browser running uncoordinated looked exactly like one
running coordinated.

The larger half is the sentence "there is no admission decision". A browser
is several hundred megabytes and a handful of processes, and it was launched
without anyone asking whether the machine could afford one — on a host
already holding a ~20GB resident model, which is precisely where Bryan's
memory-pressure warnings come from.
"""
from __future__ import annotations

import types

import pytest

from core.capabilities.phantom_browser import PhantomBrowser


@pytest.fixture
def browser():
    return PhantomBrowser(visible=False)


def _fake_pressure(monkeypatch, available_gb):
    import core.utils.memory_monitor as mm

    monkeypatch.setattr(
        mm,
        "get_memory_pressure_snapshot",
        lambda: types.SimpleNamespace(available_gb=available_gb),
    )


class TestAdmissionExists:
    def test_a_starved_host_is_refused(self, browser, monkeypatch):
        _fake_pressure(monkeypatch, 0.4)
        verdict = browser._browser_admission()
        assert verdict["can_admit"] is False
        assert "insufficient_memory" in verdict["reason"]

    def test_a_healthy_host_is_admitted(self, browser, monkeypatch):
        _fake_pressure(monkeypatch, 24.0)
        assert browser._browser_admission()["can_admit"] is True

    def test_the_threshold_is_the_boundary(self, browser, monkeypatch):
        limit = PhantomBrowser.MIN_AVAILABLE_GB_FOR_BROWSER
        _fake_pressure(monkeypatch, limit + 0.1)
        assert browser._browser_admission()["can_admit"] is True
        _fake_pressure(monkeypatch, limit - 0.1)
        assert browser._browser_admission()["can_admit"] is False

    def test_the_verdict_carries_the_measurement(self, browser, monkeypatch):
        _fake_pressure(monkeypatch, 0.4)
        assert browser._browser_admission()["available_gb"] == pytest.approx(0.4)


class TestUnmeasurableIsNotRefused:
    """Deliberate asymmetry. This gate exists to catch a MEASURED shortage;
    refusing every browse on a platform without the monitor would break the
    capability wholesale, which is a worse outcome than the bug."""

    def test_a_missing_monitor_admits(self, browser, monkeypatch):
        import core.utils.memory_monitor as mm

        def _boom():
            raise RuntimeError("no monitor on this platform")

        monkeypatch.setattr(mm, "get_memory_pressure_snapshot", _boom)
        verdict = browser._browser_admission()
        assert verdict["can_admit"] is True
        assert verdict["reason"] == "pressure_unmeasured"

    def test_the_unmeasured_case_is_distinguishable(self, browser, monkeypatch):
        """An admit-because-unknown must not look like an admit-because-fine."""
        import core.utils.memory_monitor as mm

        monkeypatch.setattr(
            mm, "get_memory_pressure_snapshot",
            lambda: (_ for _ in ()).throw(RuntimeError("x")),
        )
        assert browser._browser_admission()["available_gb"] is None


class TestStatusReportsCoordination:
    def test_a_fresh_browser_reports_uncoordinated(self, browser):
        assert browser.get_status()["resource_coordinated"] is False

    def test_status_exposes_the_last_admission(self, browser, monkeypatch):
        _fake_pressure(monkeypatch, 0.4)
        browser._last_admission = browser._browser_admission()
        status = browser.get_status()
        assert status["last_admission"]["can_admit"] is False

    def test_active_and_coordinated_are_separate_facts(self, browser):
        """Reporting only "active" made a coordinated and an uncoordinated
        browser indistinguishable."""
        status = browser.get_status()
        assert "active" in status
        assert "resource_coordinated" in status
