"""A screen reading must survive the contract that verifies it.

Asked "what's on my screen right now?" on 2026-08-10 the live runtime answered
"I routed this through CognitiveEngine and the governed desktop task lane, but
it did not complete: expectation incomplete: steps_requested; steps_completed.
Completed 0/0 steps." — while every macOS permission was granted and the
ambient loop had a current observation in hand.

Nothing had failed. The observation path answered correctly in 1ms and
reported ``steps_requested=0, steps_completed=0, receipts=[]``, which is the
truth about an answer that changed nothing. But the desktop task contract
checks those three by TRUTHINESS, so a correct reading presented zero evidence
and was downgraded to a refusal.

Three separate defects sat in that one path, and each gets a test here:

* observing produced no verifiable step, so it could not be claimed;
* the age carried with the answer was computed from an attribute
  ``Observation`` does not have, so it was ~56 years on every turn;
* a FAILED capture was described to the model as an empty one, and came back
  as "There are no windows or applications open on it at this time. It is a
  blank slate." with three windows on screen.
"""

from __future__ import annotations

import time

import pytest

from core.perception.observation_evidence import Observation, ObservationKind


class _StubAmbient:
    def __init__(self, observation):
        self._observation = observation

    def fresh_observation_for(self, question, **_kwargs):
        return self._observation


@pytest.fixture()
def ambient_answer(monkeypatch):
    """Call the real ``_ambient_answer`` against a supplied observation."""

    from core.skills import desktop_task as desktop_task_module

    def _run(observation, objective="what's on my screen right now?"):
        monkeypatch.setattr(
            "core.perception.ambient_presence.get_ambient_presence",
            lambda: _StubAmbient(observation),
            raising=False,
        )
        skill = desktop_task_module.DesktopTaskSkill()
        params = type("P", (), {"steps": []})()
        return skill._ambient_answer(objective, params)

    return _run


def _screen_observation(capture: str, *, age_s: float = 3.0) -> Observation:
    return Observation(
        kind=ObservationKind.SCREEN_TEXT,
        capture=capture,
        request="what's on my screen right now?",
        source="Google Chrome — Inbox",
        at=time.time() - age_s,
    )


def test_observing_is_a_verified_step_not_a_zero_step_non_event(ambient_answer):
    """An answer from observation must satisfy the task contract it is judged by.

    ``steps_requested``/``steps_completed``/``receipts`` are checked for
    truthiness, so 0/0/[] is indistinguishable from "this skill produced no
    evidence at all" and the reading gets thrown away.
    """
    answer = ambient_answer(_screen_observation("Inbox (3)\nCompose\nSettings"))

    assert answer is not None, "a fresh observation must produce an answer"
    assert answer["steps_requested"] >= 1
    assert answer["steps_completed"] == answer["steps_requested"]
    assert answer["receipts"], "the observation itself is the receipt"

    receipt = answer["receipts"][0]
    assert receipt["ok"] is True
    assert receipt["effect_verified"] is True
    assert receipt["effect_evidence"].strip()
    # The chat bridge rejects bare audit ids as effect evidence.
    assert not receipt["effect_evidence"].startswith("receipt_id=")
    # It must be legible as a READ, so it can never be mistaken for a mutation.
    assert receipt["action"] == "observe_screen"


def test_observation_answer_passes_the_chat_bridge_verifier(ambient_answer):
    """The exact gate that produced the live refusal must now accept it."""
    from interface.routes.chat import _verified_desktop_task_result

    answer = ambient_answer(_screen_observation("Inbox (3)\nCompose"))
    verified, reason = _verified_desktop_task_result(answer)

    assert verified is True, f"observation answer still rejected: {reason}"


def test_reported_age_is_the_real_age_not_the_unix_epoch(ambient_answer):
    """``Observation`` stamps ``at``; nothing on it is called ``timestamp``.

    Reading the missing name defaulted to 0.0, making every reported age
    ``time.time()`` — about 56 years — on the one field whose entire purpose is
    to stop a moment-old reading from passing as this instant.
    """
    answer = ambient_answer(_screen_observation("Inbox (3)", age_s=4.0))

    age = answer["observation_age_s"]
    assert age is not None
    assert 0.0 <= age < 60.0, f"age travelled as {age}s"


def test_failed_capture_is_never_described_as_an_empty_screen():
    """"Nothing was read" and "nothing is there" are different facts.

    Only the first is supported by a failed capture. Told merely to "say that
    plainly", the model asserted the second.
    """
    empty = Observation(
        kind=ObservationKind.SCREEN_TEXT,
        capture="",
        request="what do you see right now?",
        source="Google Chrome",
        at=time.time(),
    )
    assert empty.is_empty

    reasoning = empty.for_reasoning().lower()

    # It must say the READING failed.
    assert "capture failed" in reasoning or "could not read" in reasoning

    # And it must forbid the inference the live reply actually made, rather
    # than leaving the model to work out that the two are different.
    assert "must not" in reasoning or "do not say" in reasoning
    assert "blank" in reasoning, "the specific wrong inference must be named"
    assert "empty" in reasoning
