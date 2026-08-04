"""Reading the screen and being able to say what is on it are different acts.

Measured live 2026-08-04. Bryan asked "Hey, Aura. Can you tell me what you
see on the screen?" The governed desktop lane worked — the thought stream
shows ``Step 1/1 read_screen_text: verified. screen_text_returned`` — and
what came back was the raw accessibility dump, verbatim:

    Edit
    Window
    (9) Kurzgesagt
      • In a Nutshel
    • youngbryan97/aura: A cogniti x
    ...
    Show more
    You >

A transcription of the UI tree, not an answer. Only the second thing was
asked for.

The prior form of this defect is in the same family and is what makes it
worth pinning: the summary used to be "Desktop task completed 1/1 governed
desktop steps" — a progress report about the machinery, which reports that
the looking happened without ever saying what was there. Both failures
answer a question about the WORLD with a fact about the SYSTEM.
"""
from __future__ import annotations

from core.skills.desktop_task import DesktopTaskSkill

# The real dump from the live 2026-08-04 turn, trimmed.
LIVE_SCREEN_TEXT = """Edit
Window
(9) Kurzgesagt – In a Nutshell
youngbryan97/aura: A cognitive
Premium
Home
Shorts
Subscriptions
RealLifeLore
Nexpo
fern
Show more
You
Your channel
History
The Reason Why Cancer is so Hard to Beat
The Black Hole That Kills Galaxies - Quasars"""


def _observation(text: str = LIVE_SCREEN_TEXT, app: str = "Google Chrome"):
    return [
        {
            "action": "read_screen_text",
            "result": {"ok": True, "text": text, "active_app": app},
        }
    ]


def _describe(receipts):
    return DesktopTaskSkill._describe_screen_observation(receipts)


# ----------------------------------------------------- it describes, not dumps


def test_the_answer_is_not_the_raw_dump():
    description = _describe(_observation())
    assert description != LIVE_SCREEN_TEXT
    assert description.count("\n") == 0, (
        "the reply is still a line-by-line transcription of the UI tree"
    )


def test_the_answer_names_the_frontmost_app():
    assert "Google Chrome" in _describe(_observation())


def test_the_answer_reports_actual_screen_content():
    description = _describe(_observation())
    assert "Kurzgesagt" in description
    assert "Cancer" in description


def test_window_chrome_is_not_reported_as_content():
    """"Edit", "Window", "Home" describe nothing about what is on screen."""
    description = _describe(_observation())
    for chrome in ("Edit;", "Window;", "Home;", "Shorts;"):
        assert chrome not in description


def test_the_description_is_bounded_not_an_exhaustive_list():
    many = "\n".join(f"Distinct item number {index}" for index in range(60))
    description = _describe(_observation(text=many, app="Finder"))
    assert len(description) < 900
    assert "more items" in description


def test_the_total_count_is_stated_so_nothing_is_silently_dropped():
    description = _describe(_observation())
    assert "distinct text elements in total" in description


# ------------------------------------------------------------- honest edges


def test_an_empty_read_produces_no_description():
    assert _describe(_observation(text="", app="")) == ""


def test_chrome_only_text_says_so_rather_than_inventing_content():
    description = _describe(_observation(text="Edit\nWindow\nFile", app="Finder"))
    assert "window chrome" in description


def test_a_non_observation_receipt_produces_nothing():
    assert _describe([{"action": "click", "result": {"ok": True}}]) == ""


def test_a_malformed_result_is_skipped_not_crashed():
    assert _describe([{"action": "read_screen_text", "result": "not a mapping"}]) == ""
    assert _describe([{"action": "read_screen_text"}]) == ""


def test_an_app_with_no_text_is_still_named():
    assert "Finder" in _describe(_observation(text="", app="Finder"))


def test_inspect_screen_is_treated_as_an_observation_too():
    receipts = [
        {
            "action": "inspect_screen",
            "result": {"ok": True, "text": "Some visible label", "active_app": "Safari"},
        }
    ]
    assert "Safari" in _describe(receipts)


# ------------------------------------------------- it replaces the step count


def test_the_summary_prefers_the_description_over_a_progress_report():
    """The prior form of the same defect: answering with a step count."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "skills" / "desktop_task.py"
    ).read_text("utf-8")
    ast.parse(source)
    assert "_describe_screen_observation(receipts)" in source, (
        "the result summary no longer consults the screen description, so an "
        "observation answers with 'completed N/N governed steps' again"
    )
    index_desc = source.index("_describe_screen_observation(receipts)")
    index_count = source.index("governed \"\n                f\"computer-use steps")
    assert index_desc < index_count, (
        "the step count takes precedence over the description"
    )


# ═══════════════════════════════════════════════════════════════════════
# The reasoning, not the words.
#
# A formatter that writes her sentence for her is the wrong fix — it makes
# ONE phrasing correct and leaves her unable to answer anything else about
# what she saw. These pin that the perception reaches her as EVIDENCE she
# can reason over, shaped by what was actually asked, and that she can
# still refer to it afterwards.
# ═══════════════════════════════════════════════════════════════════════

import ast
from pathlib import Path

from core.perception.observation_evidence import (
    AnswerShape,
    Observation,
    ObservationKind,
    answer_shape_for,
    get_observation_memory,
    remember_observation,
)

ROOT = Path(__file__).resolve().parents[1]


def _obs(request: str = "what do you see on the screen?") -> Observation:
    return Observation(
        kind=ObservationKind.SCREEN_TEXT,
        capture=LIVE_SCREEN_TEXT,
        request=request,
        source="Google Chrome",
    )


# ------------------------------------------- the question shapes the answer


def test_what_do_you_see_wants_a_description():
    assert answer_shape_for("Can you tell me what you see on the screen?") is (
        AnswerShape.DESCRIBE
    )


def test_a_specific_question_wants_a_finding_not_a_tour():
    for question in (
        "Is there a video about black holes?",
        "What's the third video called?",
        "How many tabs are open?",
    ):
        assert answer_shape_for(question) is AnswerShape.LOCATE, question


def test_only_an_explicit_ask_wants_the_literal_text():
    for question in (
        "Read me the exact wording of that error",
        "quote the title verbatim",
        "transcribe what it says",
    ):
        assert answer_shape_for(question) is AnswerShape.TRANSCRIBE, question


def test_describing_is_the_default_not_transcribing():
    """Dumping the buffer is never a reasonable reading of a vague request."""
    assert answer_shape_for("") is AnswerShape.DESCRIBE
    assert answer_shape_for("hey aura") is AnswerShape.DESCRIBE


# --------------------------------------------- evidence, framed for reasoning


def test_the_capture_is_labelled_as_evidence_not_as_speech():
    rendered = _obs().for_reasoning()
    assert "RAW CAPTURED TEXT" in rendered
    assert "not your reply" in rendered


def test_the_capture_is_attributed_to_its_source():
    assert "Google Chrome" in _obs().for_reasoning()


def test_the_request_travels_with_the_evidence():
    rendered = _obs("is there anything about cancer?").for_reasoning()
    assert "is there anything about cancer?" in rendered


def test_the_frame_changes_with_the_question():
    describe = _obs("what do you see?").for_reasoning()
    locate = _obs("is there a video about black holes?").for_reasoning()
    transcribe = _obs("read me the exact wording").for_reasoning()
    assert "Describe it as a person would" in describe
    assert "one specific thing" in locate
    assert "quoting the relevant part IS the answer" in transcribe


def test_the_frame_does_not_write_her_sentence():
    """It supplies material and intent, never phrasing or an example answer."""
    rendered = _obs().for_reasoning()
    assert "The frontmost app is" not in rendered
    assert "Visible on screen:" not in rendered


def test_an_empty_capture_says_so_rather_than_inviting_invention():
    empty = Observation(ObservationKind.SCREEN_TEXT, "", "what do you see?", "Finder")
    rendered = empty.for_reasoning()
    assert "Nothing legible was captured" in rendered
    assert "do not describe a screen that was not read" in rendered


def test_the_evidence_is_bounded():
    huge = Observation(ObservationKind.SCREEN_TEXT, "x" * 50_000, "what do you see?")
    assert len(huge.for_reasoning()) < 6000
    assert "capture truncated" in huge.for_reasoning()


# ------------------------------------------------ she can refer back to it


def test_an_observation_is_retained_so_a_follow_up_can_reference_it():
    memory = get_observation_memory()
    memory.clear()
    remember_observation(_obs())
    recall = memory.recall_for("which one was about cancer?")
    assert "Cancer" in recall
    assert "which one was about cancer?" in recall


def test_a_followup_is_framed_by_the_NEW_question():
    memory = get_observation_memory()
    memory.clear()
    remember_observation(_obs("what do you see?"))
    assert "one specific thing" in memory.recall_for("is there a black hole video?")


def test_recall_states_the_age_rather_than_implying_freshness():
    import time as _time

    memory = get_observation_memory()
    memory.clear()
    stale = _obs()
    stale.at = _time.time() - 900
    remember_observation(stale)
    recall = memory.recall_for("what was on my screen?")
    assert "minute(s) ago" in recall
    assert "may have changed" in recall


def test_nothing_seen_means_nothing_claimed():
    memory = get_observation_memory()
    memory.clear()
    assert memory.recall_for("what did you see?") == ""


def test_retention_is_bounded_because_it_holds_the_persons_screen():
    memory = get_observation_memory()
    memory.clear()
    for index in range(30):
        remember_observation(
            Observation(ObservationKind.SCREEN_TEXT, f"capture {index}", "q")
        )
    assert len(memory.recent(limit=100)) <= 8


def test_telemetry_never_carries_the_capture():
    payload = _obs().to_dict()
    assert "Kurzgesagt" not in str(payload)
    assert payload["capture_chars"] > 0
    assert payload["answer_shape"] == "describe"


# ---------------------------------------------------------- live wiring


def test_the_desktop_result_carries_the_framed_observation():
    source = (ROOT / "core" / "skills" / "desktop_task.py").read_text("utf-8")
    ast.parse(source)
    assert '"observation": (' in source
    assert "observation.for_reasoning()" in source


def test_the_observation_is_retained_on_the_live_path():
    source = (ROOT / "core" / "skills" / "desktop_task.py").read_text("utf-8")
    assert "remember_observation(" in source, (
        "observations are not retained, so she cannot refer to what she saw "
        "after the turn that saw it"
    )


def test_the_reasoning_context_receives_evidence_not_a_finished_sentence():
    source = (
        ROOT / "core" / "orchestrator" / "mixins" / "response_processing.py"
    ).read_text("utf-8")
    ast.parse(source)
    index = source.index('context["skill_result"] = str(')
    window = source[index : index + 400]
    assert 'shortcut_result.get("observation")' in window, (
        "the reasoning context still receives only the pre-written summary, "
        "so there is nothing for her to reason over"
    )
    assert window.index('get("observation")') < window.index('get("summary"')
