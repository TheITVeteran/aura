"""Regression contract: self-critique scaffolding never reaches the user.

Live finding (July 8 soak, turn 8): the reasoning critic 'confirmed' a draft
by echoing it behind its own label, and the user received
"Proposed Answer:That's amazing...". The delivery sanitizer owns every shape
the critic's output can take.
"""
from __future__ import annotations

import pytest

from core.brain.reasoning_strategies import ReasoningStrategies

pytestmark = pytest.mark.unit

sanitize = ReasoningStrategies._deliverable_critique_text

ORIGINAL = "That's amazing. Sometimes a clean desk really does reset the whole room."


def test_label_echo_returns_original():
    critique = f"Proposed Answer:{ORIGINAL}"
    assert sanitize(ORIGINAL, critique) == ORIGINAL


def test_label_echo_with_question_preamble_returns_original():
    critique = (
        "Question: I finally cleaned my desk and it feels like a new room.\n"
        f"Proposed Answer:\n{ORIGINAL}"
    )
    assert sanitize(ORIGINAL, critique) == ORIGINAL


def test_truncated_echo_still_detected():
    critique = f"Proposed Answer: {ORIGINAL[:60]}"
    assert sanitize(ORIGINAL, critique) == ORIGINAL


def test_correction_delivers_answer_tag_body_not_narration():
    critique = (
        "Step 1: the original claims electric trains produce smoke — they do not.\n"
        "Step 2: therefore there is no smoke to blow.\n"
        "<answer>There is no smoke: it's an electric train.</answer>"
    )
    assert sanitize("The smoke blows west.", critique) == (
        "There is no smoke: it's an electric train."
    )


def test_tag_formatted_original_keeps_full_critique_for_caller():
    original = "<answer>42</answer>"
    critique = "Recomputing: 6*7 <answer>42</answer>"
    assert sanitize(original, critique) == critique


def test_plain_disagreement_without_scaffold_passes_through():
    critique = "Actually the compressor heats the coolant, not the cabin."
    assert sanitize(ORIGINAL, critique) == critique


def test_label_followed_by_new_content_strips_label_only():
    critique = "Proposed Answer: A completely different corrected explanation."
    assert sanitize(ORIGINAL, critique) == "A completely different corrected explanation."
