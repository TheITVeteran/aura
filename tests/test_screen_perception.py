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


def test_screen_perception_ocr_uses_macos_vision_when_tesseract_missing(monkeypatch, tmp_path):
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"not a real png; import fallback test only")

    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "pytesseract":
            raise ImportError("missing pytesseract")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    monkeypatch.setattr(
        ScreenPerception,
        "_ocr_screenshot_with_macos_vision",
        staticmethod(lambda path: "Ask anything\nChatGPT\nAura can read this screen."),
    )

    text = ScreenPerception._ocr_screenshot_sync(str(image_path))

    assert "Ask anything" in text
    assert "Aura can read this screen" in text
