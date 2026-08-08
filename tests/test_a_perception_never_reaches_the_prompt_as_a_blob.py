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


# ----------------------------------------------------- the OTHER live lane
#
# Measured live 2026-08-08, a THIRD time — after the injector above was
# fixed and Aura was restarted onto the new code. The turn never touched the
# injector. It went through core.synthesis._render_tool_results, which
# rendered every result as `repr(result)`:
#
#     chunk = f"[{index}] {result!r}"
#
# so {'ok': True, 'text': <the whole accessibility tree>} arrived at the
# model as dict syntax wrapped around a wall of unlabelled text, and the
# model continued it. Fixing one lane while a second renders the same thing
# its own way is not a fix; it is a coin flip on which lane a turn takes.


def test_the_synthesis_lane_does_not_hand_the_model_a_repr():
    from core.synthesis import _render_tool_results

    rendered, _dropped = _render_tool_results(
        [{"ok": True, "text": LIVE_CAPTURE, "active_app": "Google Chrome"}], REQUEST
    )

    assert "{'ok': True" not in rendered, (
        "the result reached the model as repr(); this is the lane the live "
        "turn used, and it is why fixing the injector changed nothing"
    )
    assert "RAW CAPTURED TEXT" in rendered
    assert "Google Chrome" in rendered


def test_the_synthesis_lane_carries_the_question_that_was_asked():
    from core.synthesis import _render_tool_results

    rendered, _ = _render_tool_results([{"ok": True, "text": LIVE_CAPTURE}], REQUEST)

    assert "What was asked:" in rendered
    assert REQUEST in rendered.split("What was asked:", 1)[1]


def test_the_synthesis_lane_leaves_ordinary_results_alone():
    from core.synthesis import _render_tool_results

    rendered, _ = _render_tool_results(
        [{"ok": True, "summary": "Sent the email to Bryan.", "receipt": "r-1"}], REQUEST
    )

    assert "Sent the email to Bryan." in rendered
    assert "RAW CAPTURED TEXT" not in rendered


def test_a_mixed_batch_frames_only_the_perception():
    from core.synthesis import _render_tool_results

    rendered, _ = _render_tool_results(
        [
            {"ok": True, "summary": "Opened Chrome."},
            {"ok": True, "text": LIVE_CAPTURE},
        ],
        REQUEST,
    )

    assert "Opened Chrome." in rendered
    assert rendered.count("RAW CAPTURED TEXT") == 1


# ------------------------------------------------------- the history lane


def test_conversation_history_does_not_retain_the_raw_tree():
    """History is re-read on every later turn; a blob there poisons all of them."""
    from core.coordinators.context_coordinator import ContextCoordinator

    class _Orch:
        pass

    orch = _Orch()
    orch.conversation_history = []
    coordinator = ContextCoordinator.__new__(ContextCoordinator)
    coordinator.orch = orch

    coordinator.record_action_in_history(
        "computer_use", {"ok": True, "text": LIVE_CAPTURE}
    )

    content = orch.conversation_history[0]["content"]
    assert "{'ok': True" not in content
    assert "computer_use" in content


# ------------------------------------------------------- one recognition point


def test_every_lane_frames_a_capture_the_same_way():
    """Behavioural, not source-text.

    The previous version of this test asserted that the string
    "_inject_shortcut_results(" appeared in two files. It passed while the
    live turn dumped, because a third lane existed that it never looked at.
    A lane is covered when it is CALLED here, with the real result shape.
    """
    from core.coordinators.context_coordinator import ContextCoordinator
    from core.synthesis import _render_tool_results

    result = {"ok": True, "text": LIVE_CAPTURE, "active_app": "Google Chrome"}

    class _Orch:
        pass

    orch = _Orch()
    orch.conversation_history = []
    coordinator = ContextCoordinator.__new__(ContextCoordinator)
    coordinator.orch = orch
    coordinator.record_action_in_history("computer_use", dict(result))

    from core.cognitive.state_machine import _frame_perception

    lanes = {
        "shortcut_injector": _inject(dict(result)),
        "synthesis": _render_tool_results([dict(result)], REQUEST)[0],
        "conversation_history": orch.conversation_history[0]["content"],
        "coordinator_injector": coordinator.inject_shortcut_results(
            REQUEST, dict(result)
        ),
        "state_machine_summary": _frame_perception(dict(result), REQUEST) or "",
    }
    unframed = [name for name, text in lanes.items() if "RAW CAPTURED TEXT" not in text]
    assert not unframed, f"these lanes still paste the capture unframed: {unframed}"


def test_no_lane_renders_a_tool_result_with_a_bare_repr():
    """The structural half: a FIFTH lane must not be able to appear quietly.

    Every module that turns a tool result into prompt text is listed here.
    A bare ``repr``/``str`` of a result dict in one of them is the exact
    construct that produced the live dump, so it is banned outright — and
    when a new lane is written, this list is what makes someone notice.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    lanes = (
        "core/synthesis.py",
        "core/orchestrator/mixins/message_pipeline.py",
        "core/coordinators/context_coordinator.py",
        "core/cognitive/state_machine.py",
    )
    # `{result!r}` or `str(result)` reaching a prompt, outside a guarded
    # else-branch that has already framed perceptions.
    bare = re.compile(r"\{result!r\}")
    offenders = []
    for rel in lanes:
        for line_no, line in enumerate(
            (root / rel).read_text("utf-8").splitlines(), 1
        ):
            if not bare.search(line):
                continue
            if "framed" in line:  # the guarded fallback for non-perceptions
                continue
            offenders.append(f"{rel}:{line_no}")
    assert not offenders, (
        f"these render a tool result with a bare repr into prompt text: {offenders}"
    )


def test_the_perception_is_retained_for_follow_ups():
    from core.perception.observation_evidence import get_observation_memory

    get_observation_memory().clear()
    _inject({"ok": True, "text": LIVE_CAPTURE, "active_app": "Google Chrome"})
    recall = get_observation_memory().recall_for("which repo was open?")
    assert "youngbryan97/aura" in recall, (
        "the perception was framed but not retained, so a follow-up would "
        "force a fresh capture of a screen that has since changed"
    )
