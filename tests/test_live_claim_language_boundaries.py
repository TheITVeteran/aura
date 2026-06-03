"""Regression tests for evidence-bounded live UI/API claim language."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_live_dashboard_and_subsystem_routes_do_not_overclaim_subjectivity() -> None:
    sources = [
        ROOT / "interface" / "routes" / "dashboard.py",
        ROOT / "interface" / "routes" / "subsystems.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources).lower()

    forbidden = {
        "aura's inner life *without trusting her words*",
        "phenomenal awareness",
        "the subjective quality of system states",
        "will-to-live",
    }
    for phrase in forbidden:
        assert phrase not in text

    assert "operational cognitive state" in text
    assert "functional state descriptor" in text
