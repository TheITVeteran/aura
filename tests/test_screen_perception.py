import pytest

from core.perception.screen_perception import ScreenPerception


@pytest.mark.asyncio
async def test_screen_perception_capture_uses_accessibility_summary_without_screenshot(monkeypatch):
    perception = ScreenPerception()

    async def fake_summary(self):
        return {
            "active_app": "Google Chrome",
            "window_title": "Google Docs - Aura",
            "frontmost_window_bounds": "0,25,1440,900",
            "focused_role": "AXTextArea",
            "focused_name": "Document body",
            "focused_description": "editable text",
            "focused_value": "",
            "accessibility_text": "A visible editable document body",
        }

    monkeypatch.setattr(ScreenPerception, "_frontmost_accessibility_summary", fake_summary)

    snapshot = await perception.capture(save_screenshot=False)

    assert snapshot.active_app == "Google Chrome"
    assert snapshot.window_title == "Google Docs - Aura"
    assert snapshot.frontmost_window_bounds == "0,25,1440,900"
    assert snapshot.focused_role == "AXTextArea"
    assert snapshot.accessibility_text == "A visible editable document body"
    assert snapshot.text_hash
