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

_STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"
_MIND_HTML = _STATIC / "mind.html"
_ACTIVITY_HTML = _STATIC / "activity.html"

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
    # Cross-link into the activity view (product shell nav).
    assert "/activity" in html


def test_receipts_contract_for_activity_view():
    from interface.routes.dashboard import receipts

    resp = asyncio.run(receipts(limit=10, _=None))
    payload = json.loads(bytes(resp.body))
    assert "receipts" in payload
    assert isinstance(payload["receipts"], list)


def test_activity_page_exists_and_is_plain_language():
    assert _ACTIVITY_HTML.exists(), "interface/static/activity.html is missing"
    html = _ACTIVITY_HTML.read_text(encoding="utf-8")
    assert "/api/receipts" in html
    assert "sessionStorage.getItem('api_token')" in html
    assert "/mind" in html  # cross-link back to the mind view
    for human in ("What she's been doing", "What am I looking at?", "worked"):
        assert human in html, f"activity.html missing plain-language framing: {human}"


def test_settings_contract_for_controls_panel():
    from interface.routes.settings import get_all

    resp = asyncio.run(get_all(_=None))
    payload = json.loads(bytes(resp.body))
    assert isinstance(payload.get("schema"), list)
    assert isinstance(payload.get("values"), dict)
    keys = {s["key"] for s in payload["schema"]}
    for control in ("safety.safe_mode", "autonomy.level", "permissions.camera",
                    "permissions.screen", "voice.input_enabled", "notify.enabled"):
        assert control in keys, f"controls panel needs setting {control} in the schema"


def test_controls_page_exists_and_is_plain_language():
    html = (_STATIC / "controls.html").read_text(encoding="utf-8")
    assert "/api/settings" in html
    assert "sessionStorage.getItem('api_token')" in html
    assert "method:'PATCH'" in html  # writes changes back
    for cid in ("sw-safety.safe_mode", "sw-permissions.camera"):
        assert cid in html, f"controls.html missing control: {cid}"

    # No control for autonomy.level, deliberately. It is declared
    # mutable=False — a protected agency invariant read at the authority gate —
    # so rendering a segmented control for it would put a widget on the page
    # that cannot change anything. That is precisely the dead control
    # test_settings_no_dead_controls exists to forbid, and offering it would be
    # worse than omitting it: it would tell the operator they hold a lever they
    # do not.
    assert "seg-autonomy.level" not in html, (
        "autonomy.level is immutable; a control for it would be a dead control"
    )
    # Plain language, and language that is TRUE. "You're in charge" was
    # required here from when autonomy was an operator-selected level. The page
    # now says "Boundaries, not identity — Aura's agency is intrinsic and
    # remains active", which is the honest description of what these controls
    # do: they bound external effects and can trigger emergency containment,
    # and they cannot switch her agency off.
    #
    # Keeping the old phrase would have required the UI to tell the operator
    # they hold authority the system deliberately withholds — a false promise
    # on the most consequential page in the product.
    for human in ("Safe mode", "Boundaries, not identity", "What am I looking at?"):
        assert human in html, f"controls.html missing plain-language framing: {human}"
    assert "agency is intrinsic" in html, (
        "the page must say what these controls do NOT do"
    )
    for link in ("/mind", "/activity"):
        assert link in html  # cross-linked product-shell nav
