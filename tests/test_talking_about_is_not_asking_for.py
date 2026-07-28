"""Mentioning a capability is not requesting it.

Two live failures on one message. An apology sent to Aura — "Earlier you
described what was on his screen and I decided you had made it up... you have
a continuous vision feed" — came back as:

    I routed this through CognitiveEngine and the governed desktop task lane,
    but it did not complete: Permission denied: Modality 'camera' is
    disabled. Completed 0/0 steps.

Neither gate was reading the sentence. "described ... screen" reads exactly
like "describe my screen", so recounting a past observation was treated as
requesting a new one. And the camera modality matched the bare word
"vision" — a word Aura uses about herself constantly — so a turn that wanted
no camera at all was refused for lack of camera permission.

Both are the overbroad-keyword class: a word that carries a second, ordinary
meaning wired to a consequential gate. The requests below must keep working,
which is the half that makes this a fix rather than a loosening.
"""

import re

import pytest

from core.capabilities.permission_model import _MODALITY_PATTERNS
from interface.routes.chat import _looks_like_desktop_objective

pytestmark = pytest.mark.unit

REQUESTS = [
    "take a screenshot of my screen",
    "read my screen and tell me what is there",
    "what do you see on my screen right now",
    "open chrome and take a screenshot",
]

TALKING_ABOUT_IT = [
    "Earlier you described what was on his screen and I decided you had made it up",
    "you were right about the screen, sorry for doubting you",
    "I thought your screen description was invented",
]


@pytest.mark.parametrize("message", REQUESTS)
def test_a_request_still_routes_to_the_desktop(message: str):
    assert _looks_like_desktop_objective(message)


@pytest.mark.parametrize("message", TALKING_ABOUT_IT)
def test_recounting_is_not_requesting(message: str):
    assert not _looks_like_desktop_objective(message)


def _modalities(text: str) -> set[str]:
    return {
        name
        for name, patterns in _MODALITY_PATTERNS.items()
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    }


class TestTheCameraIsADeviceNotAFaculty:
    def test_the_word_vision_does_not_demand_a_camera(self):
        assert "camera" not in _modalities("you have a continuous vision feed")
        assert "camera" not in _modalities("my vision of the project is broad")

    def test_real_camera_words_still_do(self):
        for text in ("take a photo with the camera", "use the webcam", "turn on computer vision"):
            assert "camera" in _modalities(text), text

    def test_a_screenshot_asks_for_screen_recording_not_the_camera(self):
        found = _modalities("take a screenshot of my screen")
        assert "screen_recording" in found
        assert "camera" not in found
