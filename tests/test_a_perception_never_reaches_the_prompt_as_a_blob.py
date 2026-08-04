"""The injector both fast paths share must never paste a raw capture.

Measured live 2026-08-04, TWICE — the second time after I had reported it
fixed. `_inject_shortcut_results` computed

    summary = str(result.get("summary", result.get("result", result)))

and computer_use returns `{"ok": True, "text": <raw accessibility dump>}`
with no `summary` key. So the fallback stringified the whole dict, the
person's screen contents arrived in the prompt as a "[DIRECT RESULT]" to
synthesize, and the model reproduced the only thing it was given: a wall
of unlabelled text.

The first attempt fixed desktop_task's summary — the OTHER skill — and was
"verified" with AST checks that the wiring existed rather than by calling
this function. It passed while the live turn still dumped. That is the
same defect this codebase keeps producing (absence of a check reported as
a passed check), so every test here EXERCISES the real function against
the real result shapes.
"""
from __future__ import annotations

import pytest

from core.orchestrator.mixins.message_pipeline import MessagePipelineMixin

REQUEST = "Hey, Aura. Can you tell me what you see on the screen?"

# The genuine 2026-08-04 capture, trimmed.
LIVE_CAPTURE = """Д.
Aura
Edit
Window
(9) Kurzgesagt
youngbryan97/aura: A cogniti x
Aura Zenith
Premium
Home
Shorts
RealLifeLore
Talk to Aura...
Liked videos"""


def _inject(result: dict, message: str = REQUEST) -> str:
    return MessagePipelineMixin._inject_shortcut_results(
        MessagePipelineMixin, message, result
    )


# ------------------------------------------ the exact live result shapes


def test_the_computer_use_shape_is_not_dumped_as_a_direct_result():
    """{'ok': True, 'text': ...} — no summary key. THE live shape."""
    out = _inject({"ok": True, "text": LIVE_CAPTURE, "active_app": "Google Chrome"})
    assert "[DIRECT RESULT]" not in out, (
        "the raw capture is still being pasted in as a result to synthesize"
    )
    assert "RAW CAPTURED TEXT" in out
    assert "not your reply" in out


def test_the_whole_result_dict_is_never_stringified():
    """`str(result)` put {'ok': True, 'text': ...} into the prompt verbatim."""
    out = _inject({"ok": True, "text": LIVE_CAPTURE})
    assert "{'ok': True" not in out
    assert '"ok": true' not in out.lower()


def test_the_capture_is_attributed_to_its_source():
    out = _inject({"ok": True, "text": LIVE_CAPTURE, "active_app": "Google Chrome"})
    assert "Google Chrome" in out


def test_the_request_travels_with_the_capture():
    out = _inject({"ok": True, "text": LIVE_CAPTURE, "active_app": "Safari"})
    assert "What was asked:" in out
    assert REQUEST in out.split("What was asked:", 1)[1]


@pytest.mark.parametrize("key", ["text", "screen_text", "accessibility_text"])
def test_every_perception_key_is_framed(key):
    out = _inject({"ok": True, key: LIVE_CAPTURE})
    assert "RAW CAPTURED TEXT" in out, f"{key} reached the prompt unframed"


def test_the_frame_follows_the_question():
    describe = _inject({"ok": True, "text": LIVE_CAPTURE}, "what do you see?")
    locate = _inject({"ok": True, "text": LIVE_CAPTURE}, "is there a video about black holes?")
    quote = _inject({"ok": True, "text": LIVE_CAPTURE}, "read me the exact wording")
    assert "Describe it as a person would" in describe
    assert "one specific thing" in locate
    assert "quoting the relevant part IS the answer" in quote


# ------------------------------------------------- non-perceptions still work


def test_an_ordinary_skill_result_still_gets_its_summary():
    out = _inject({"ok": True, "summary": "Sent the email to Bryan."})
    assert "[DIRECT RESULT]: Sent the email to Bryan." in out
    assert "Synthesize this result" in out


def test_a_desktop_task_summary_is_still_used():
    out = _inject({"ok": True, "summary": "The frontmost app is Google Chrome."})
    assert "[DIRECT RESULT]" in out
    assert "The frontmost app is Google Chrome." in out


def test_a_result_with_no_usable_field_does_not_crash():
    assert _inject({"ok": False}) 
    assert _inject({})


def test_an_empty_capture_is_not_treated_as_a_perception():
    """Nothing was seen; there is no evidence to frame."""
    out = _inject({"ok": True, "text": "   "})
    assert "RAW CAPTURED TEXT" not in out


def test_a_non_string_capture_is_not_treated_as_a_perception():
    out = _inject({"ok": True, "text": {"nested": "value"}})
    assert "RAW CAPTURED TEXT" not in out


# ------------------------------------------------------- both callers use it


def test_both_fast_paths_go_through_this_one_function():
    """response_processing and cognitive_coordinator both call it, so the
    frame cannot be bypassed by whichever path a turn happens to take."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in (
        "core/orchestrator/mixins/response_processing.py",
        "core/coordinators/cognitive_coordinator.py",
    ):
        assert "_inject_shortcut_results(" in (root / rel).read_text("utf-8"), rel


def test_the_perception_is_retained_for_follow_ups():
    from core.perception.observation_evidence import get_observation_memory

    get_observation_memory().clear()
    _inject({"ok": True, "text": LIVE_CAPTURE, "active_app": "Google Chrome"})
    recall = get_observation_memory().recall_for("which repo was open?")
    assert "youngbryan97/aura" in recall, (
        "the perception was framed but not retained, so a follow-up would "
        "force a fresh capture of a screen that has since changed"
    )
