""""No faces detected" in a dark room is not the same as "the room is empty".

`VisionSystem.analyze()` asked the model for prose and then returned

    "objects_detected": [], "text_detected": [], "faces_detected": 0

as CONSTANTS, beside a real `scene_description`. So the model could describe
three people at a whiteboard and a caller reading `faces_detected` got zero.
The no-vision-model fallback returned the same zeros, which made "I could
not look" structurally identical to "I looked and nobody was there".

Three outcomes have to stay distinguishable, and only one of them is a fact
about the room:

  * observed, nothing there            -> 0 / []
  * observed, cannot tell              -> None
  * never observed                     -> None, abstained=True

This is the calibrated-abstention half of CP096's perception list, and the
general-detail half depends on it: a detail the model could not resolve
because of darkness, occlusion, motion, or distance must come back as
unknown, because a confident zero there is what turns a perception gap into
a false statement about the world.
"""
from __future__ import annotations

import pytest

from core.perception.sensory_integration import _parse_vision_reading


# ─────────────────────────────────────── the structured reply is honoured


def test_a_structured_reply_populates_the_fields():
    reading = _parse_vision_reading(
        '{"scene": "a desk with a lamp", "objects": ["lamp", "mug"], '
        '"text": ["INBOX"], "people": 2}'
    )

    assert reading["analyzed"] is True
    assert reading["abstained"] is False
    assert reading["scene_description"] == "a desk with a lamp"
    assert reading["objects_detected"] == ["lamp", "mug"]
    assert reading["text_detected"] == ["INBOX"]
    assert reading["faces_detected"] == 2


def test_json_wrapped_in_prose_and_fences_is_still_read():
    """Models do this constantly; failing to parse would silently downgrade
    every reading to unknown."""
    reading = _parse_vision_reading(
        'Here is the result:\n```json\n{"scene": "a cat", "people": 0}\n```\nHope that helps!'
    )

    assert reading["scene_description"] == "a cat"
    assert reading["faces_detected"] == 0


def test_a_confident_zero_survives():
    """Abstention must not swallow real negative observations.

    If every zero became None, "nobody is in the room" would stop being
    sayable — which is its own failure, in the opposite direction.
    """
    reading = _parse_vision_reading('{"scene": "empty room", "objects": [], "people": 0}')

    assert reading["faces_detected"] == 0
    assert reading["objects_detected"] == []


# ───────────────────────────────────────── cannot-tell stays cannot-tell


def test_null_means_unknown_not_zero():
    """The case the constants destroyed."""
    reading = _parse_vision_reading(
        '{"scene": "too dark to make out", "objects": null, "text": null, "people": null}'
    )

    assert reading["faces_detected"] is None
    assert reading["objects_detected"] is None
    assert reading["text_detected"] is None
    assert reading["scene_description"] == "too dark to make out"


def test_a_prose_only_reply_keeps_the_description_and_admits_unknown():
    """The old code invented empty lists here. The description is real
    evidence and is kept; the counts were never measured and must not be
    reported as zero."""
    reading = _parse_vision_reading("I can see a desk and what might be a person.")

    assert reading["scene_description"].startswith("I can see a desk")
    assert reading["faces_detected"] is None
    assert reading["objects_detected"] is None
    assert reading["reason"] == "unstructured_reply"


def test_malformed_json_does_not_become_an_empty_observation():
    reading = _parse_vision_reading('{"scene": "a desk", "objects": [broken')

    assert reading["faces_detected"] is None
    assert reading["reason"] in {"unparseable_json", "unstructured_reply"}


def test_a_boolean_is_not_a_count():
    """`"people": true` says someone is there, not that exactly one is.

    Coercing it to 1 would be inventing a measurement.
    """
    reading = _parse_vision_reading('{"scene": "someone", "people": true}')

    assert reading["faces_detected"] is None


def test_a_negative_count_is_clamped_not_propagated():
    reading = _parse_vision_reading('{"people": -3}')

    assert reading["faces_detected"] == 0


def test_a_numeric_string_count_is_read():
    reading = _parse_vision_reading('{"people": "3"}')

    assert reading["faces_detected"] == 3


def test_a_single_string_becomes_a_one_item_list():
    reading = _parse_vision_reading('{"objects": "lamp"}')

    assert reading["objects_detected"] == ["lamp"]


def test_an_empty_reply_is_an_abstention():
    reading = _parse_vision_reading("")

    assert reading["analyzed"] is False
    assert reading["abstained"] is True
    assert reading["faces_detected"] is None


def test_an_unexpected_json_shape_abstains():
    reading = _parse_vision_reading('{"scene": ["not", "a", "string"]}')

    # A list where a string was expected must not become the description.
    assert not isinstance(reading["scene_description"], list)


# ────────────────────────────────── the no-model path is an abstention


@pytest.mark.asyncio
async def test_no_vision_model_abstains_rather_than_reporting_an_empty_room(
    monkeypatch,
):
    """The two cases used to be byte-identical in the structured fields."""
    from core.perception import sensory_integration as si

    monkeypatch.setattr(si, "optional_service", lambda name: None)

    result = await si.VisionSystem().analyze({"path": "/tmp/x.jpg"})

    assert result["analyzed"] is False
    assert result["abstained"] is True
    assert result["reason"] == "no_vision_model"
    assert result["faces_detected"] is None, (
        "a runtime with no vision model reported zero faces, which is a "
        "claim about the room rather than about the runtime"
    )
    assert result["objects_detected"] is None
    assert result["text_detected"] is None


@pytest.mark.asyncio
async def test_an_invalid_capture_is_not_an_observation():
    from core.perception.sensory_integration import VisionSystem

    result = await VisionSystem().analyze({"error": "camera_permission_denied"})

    assert result["error"] == "invalid_capture"


def test_the_zero_constants_are_gone_from_the_source():
    """The fix must not survive only in behaviour.

    Re-adding `"faces_detected": 0` next to a real description is a one-line
    regression that no behavioural test would catch if the model happened to
    return nothing during the run.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "perception"
        / "sensory_integration.py"
    ).read_text("utf-8")

    assert '"faces_detected": 0' not in source, (
        "faces_detected is hard-coded to 0 again"
    )
