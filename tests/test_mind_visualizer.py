"""Tests for the real-time mind visualizer (#39).

The visualizer (interface/static/mind.html) is a thin client over the canonical
/api/inner-state surface. These tests pin the data contract it depends on so the
endpoint can't silently drop a field the page renders, and verify the page ships
and targets the right endpoint.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

_MIND_HTML = Path(__file__).resolve().parents[1] / "interface" / "static" / "mind.html"

# Top-level keys the visualizer reads from /api/inner-state.
_REQUIRED_KEYS = (
    "will",
    "drives",
    "affect",
    "coherence",
    "affective_steering",
    "llm_tiers",
    "cognition_loop",
    "last_initiative",
    "goals",
    "self",
)


def _inner_state_payload() -> dict:
    from interface.routes.inner_state import get_inner_state

    resp = asyncio.run(get_inner_state())
    return json.loads(bytes(resp.body))


def test_inner_state_contract_for_mind_visualizer():
    payload = _inner_state_payload()
    assert isinstance(payload, dict)
    assert "timestamp" in payload
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    assert not missing, f"/api/inner-state dropped keys the mind visualizer needs: {missing}"


def test_mind_page_exists_and_targets_inner_state():
    assert _MIND_HTML.exists(), "interface/static/mind.html is missing"
    html = _MIND_HTML.read_text(encoding="utf-8")
    assert "/api/inner-state" in html
    # Auth convention shared with the other dashboards.
    assert "sessionStorage.getItem('api_token')" in html
    # Core panels the renderer drives are present.
    for marker in ("id=\"core\"", "id=\"valNode\"", "id=\"moodChips\"", "id=\"drives\"",
                   "id=\"awake\"", "id=\"will\"", "id=\"intent\"", "id=\"wholeArc\""):
        assert marker in html, f"mind.html missing panel: {marker}"
    # Plain-language framing for non-experts (no raw jargon as the primary label).
    for human in ("How she feels", "What she wants", "What's awake", "What am I looking at?"):
        assert human in html, f"mind.html missing plain-language section: {human}"
