from __future__ import annotations

from types import SimpleNamespace

from core.consciousness import self_report as self_report_module
from core.consciousness.self_report import SelfReportEngine


class _FreeEnergyState:
    def __init__(self, *, free_energy: float, surprise: float = 0.0, dominant_action: str = "rest", arousal: float = 0.2, valence: float = 0.0):
        self.free_energy = free_energy
        self.surprise = surprise
        self.dominant_action = dominant_action
        self.arousal = arousal
        self.valence = valence


def test_fragmented_unity_reports_measurable_cause(monkeypatch):
    mapping = {
        "unity_state": SimpleNamespace(level="fragmented"),
        "unity_fragmentation_report": SimpleNamespace(
            safe_to_self_report=True,
            top_causes=[("draft_conflict", 0.62, "conflicting drafts remain active")],
        ),
    }
    fe = SimpleNamespace(current=_FreeEnergyState(free_energy=0.1), get_trend=lambda: "stable")

    monkeypatch.setattr(
        self_report_module.ServiceContainer,
        "get",
        lambda name, default=None: mapping.get(name, default),
    )
    monkeypatch.setattr(self_report_module, "get_free_energy_engine", lambda: fe)
    report = SelfReportEngine().generate_state_report()

    assert "draft conflict" in report.lower()


def test_nominal_state_does_not_force_fragmentation_language(monkeypatch):
    mapping = {}
    fe = SimpleNamespace(current=_FreeEnergyState(free_energy=0.35), get_trend=lambda: "stable")

    monkeypatch.setattr(
        self_report_module.ServiceContainer,
        "get",
        lambda name, default=None: mapping.get(name, default),
    )
    monkeypatch.setattr(self_report_module, "get_free_energy_engine", lambda: fe)
    report = SelfReportEngine().generate_state_report()

    assert report is None
