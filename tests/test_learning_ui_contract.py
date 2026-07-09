"""UI contract pins for the Learning & Growth panel.

The sidebar's SKILLS tab surfaces the learning stack (weight cycles,
practice accuracy, banked pairs) from /api/system/learning. These pins keep
the HTML ids, the loader, and its call sites from silently drifting apart —
the same failure mode as any cross-file string contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

HTML = Path("interface/static/index.html")
JS = Path("interface/static/aura.js")


def test_learning_panel_ids_exist_in_html():
    html = HTML.read_text(encoding="utf-8")
    for element_id in (
        "learn-generations",
        "learn-promoted",
        "learn-practice-rate",
        "learn-pairs",
        "learn-detail",
    ):
        assert f'id="{element_id}"' in html, f"missing Learning panel element: {element_id}"
    assert "LEARNING &amp; GROWTH" in html


def test_loader_reads_every_panel_id_and_the_endpoint():
    js = JS.read_text(encoding="utf-8")
    assert "async function loadLearningStatus()" in js
    assert "/api/system/learning" in js
    for element_id in (
        "learn-generations",
        "learn-promoted",
        "learn-practice-rate",
        "learn-pairs",
        "learn-detail",
    ):
        assert f"'{element_id}'" in js, f"loader never writes {element_id}"


def test_loader_is_called_on_tab_switch_and_initial_load():
    js = JS.read_text(encoding="utf-8")
    assert js.count("loadLearningStatus()") >= 3  # definition + tab switch + boot


def test_loader_fields_match_the_api_schema():
    """The JS reads exactly the fields /api/system/learning serves."""
    js = JS.read_text(encoding="utf-8")
    for field in ("lineage", "generations", "promoted", "verdict",
                  "last_status", "total_attempts", "total_correct",
                  "total_pairs", "bursts", "preference_store", "selfplay"):
        assert field in js, f"loader no longer references API field {field}"
