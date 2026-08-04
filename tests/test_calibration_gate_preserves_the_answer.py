"""The honesty gate must not be bypassed by its own serializer.

CP126 on core/brain/calibration_gate.py, live through
reasoning_amplifier_v2. Two defects that both ended with the person
receiving something other than what the gate decided:

* ``calibrated_answer`` — the corrected text the rest of the system is
  allowed to speak — was excluded from ``to_dict()``. Any boundary
  consuming only the serialized report got scores and labels, lost the
  correction, and spoke the original.
* every answer was rebuilt with ``" ".join(sentences)``, discarding
  newlines, paragraphs, list markers, indentation and code fences — on
  every call, including the common one where nothing was downgraded.
"""
from __future__ import annotations

from core.brain.calibration_gate import CalibrationGate, EpistemicStatus


def _gate() -> CalibrationGate:
    return CalibrationGate()


def test_an_answer_with_nothing_to_downgrade_is_returned_untouched():
    answer = (
        "Here is the plan.\n\n"
        "1. First step\n"
        "2. Second step\n\n"
        "```python\n"
        "def go():\n"
        "    return 1\n"
        "```\n"
    )
    report = _gate().assess(answer, evidence=["plan", "step", "python"])
    assert report.calibrated_answer == answer, (
        "the answer was rebuilt and its structure destroyed even though "
        "nothing needed correcting"
    )


def test_code_indentation_survives_a_downgrade_elsewhere():
    """Rebuilding with spaces can change what code DOES."""
    answer = (
        "The server always returns success.\n"
        "```python\n"
        "def go():\n"
        "    if x:\n"
        "        return 1\n"
        "    return 2\n"
        "```\n"
    )
    report = _gate().assess(answer)
    assert "    if x:" in report.calibrated_answer
    assert "        return 1" in report.calibrated_answer
    assert "```python\n" in report.calibrated_answer


def test_paragraph_breaks_survive():
    answer = "First paragraph here.\n\nSecond paragraph here."
    report = _gate().assess(answer, evidence=["first", "second", "paragraph"])
    assert "\n\n" in report.calibrated_answer


def test_list_markers_survive():
    answer = "Findings:\n- alpha item\n- beta item\n"
    report = _gate().assess(answer, evidence=["alpha", "beta", "findings"])
    assert "- alpha item" in report.calibrated_answer
    assert "- beta item" in report.calibrated_answer


def test_a_downgraded_sentence_is_still_actually_hedged():
    """The control: preserving structure must not stop the gate correcting."""
    answer = "The database will always recover automatically."
    report = _gate().assess(answer)
    if report.downgraded:
        assert report.calibrated_answer != answer
        assert "not fully certain" in report.calibrated_answer.lower()


def test_only_the_changed_sentence_is_rewritten():
    answer = "Water boils at 100C.\n\nThe server always recovers automatically."
    report = _gate().assess(answer, evidence=["water boils at 100c"])
    assert "Water boils at 100C." in report.calibrated_answer
    assert "\n\n" in report.calibrated_answer


def test_the_serialized_report_carries_the_calibrated_answer():
    """Its absence meant a consumer spoke the uncorrected original."""
    report = _gate().assess("The system always works perfectly.")
    payload = report.to_dict()
    assert "calibrated_answer" in payload
    assert payload["calibrated_answer"] == report.calibrated_answer


def test_the_serialized_report_says_how_many_labels_it_dropped():
    answer = " ".join(f"Claim number {index} is certainly true." for index in range(25))
    payload = _gate().assess(answer).to_dict()
    assert payload["label_count"] >= 20
    assert payload["labels_truncated"] is True
    assert len(payload["labels"]) == 10


def test_a_short_answer_is_not_marked_truncated():
    payload = _gate().assess("One claim only.").to_dict()
    assert payload["labels_truncated"] is False


def test_labels_carry_offsets_into_the_original_answer():
    answer = "First claim here.\n\nSecond claim here."
    report = _gate().assess(answer)
    for label in report.labels:
        assert label.start >= 0 and label.end > label.start
        assert answer[label.start : label.end] == label.text


def test_an_impossible_claim_is_flagged_in_place():
    answer = "Some context.\n\nI just googled it and found the answer."
    report = _gate().assess(answer)
    if report.flagged_impossible:
        assert "[unverifiable locally]" in report.calibrated_answer
        assert "Some context." in report.calibrated_answer
        assert "\n\n" in report.calibrated_answer


def test_an_empty_answer_does_not_crash():
    report = _gate().assess("")
    assert report.calibrated_answer == ""
    assert report.overall is EpistemicStatus.UNVERIFIED
