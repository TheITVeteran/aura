"""An absolute dB threshold made an open microphone behave like a closed one.

LIVE DEFECT, 2026-08-10: "when i talk to my computer nothing happens. no
response or anything."

Her speech was classified ``ambient_speech``, and voice_engine refuses that
outright — it is not in ``_SPEAKER_AT_THE_MACHINE_SOURCES`` — so no utterance
could ever be answered, whatever was said.

The cause was one line::

    near_field = rms_db >= -22.0 and transcript_confidence >= -0.45

-22 dBFS is a property of one input gain and nothing else. A MacBook's built-in
microphone at ordinary speaking distance sits below it, so on that hardware the
owner is permanently reclassified as distant room noise. Nothing a person would
think to try — speaking louder, sitting closer, rephrasing — can fix a number
that does not describe them.

Loudness is now measured against the room's own noise floor, so every
microphone, room and gain setting calibrates itself. The refusals that make an
always-on microphone acceptable are unchanged: media playback and speech down
at the floor are still refused, and audio provenance still has to agree.
"""
from __future__ import annotations

import pytest

from core.senses.audio_attention import (
    _NEAR_FIELD_SNR_DB,
    classify_audio_attention,
    reset_room_calibration,
    room_noise_floor_db,
)

#: The sources voice_engine will actually answer.
ANSWERED = frozenset(
    {"direct_user", "direct_address", "nearby_visible_speaker", "nearby_person"}
)

SENTENCE = "so i was thinking about the deployment pipeline again today"


@pytest.fixture(autouse=True)
def _fresh_room():
    reset_room_calibration()
    yield
    reset_room_calibration()


def _room_tone(level: float, samples: int = 20) -> None:
    for _ in range(samples):
        classify_audio_attention(
            "", rms_db=level, transcript_confidence=-0.2, duration_s=1.0
        )


def _speak(level: float, *, app: str = "", text: str = SENTENCE, duration: float = 8.0):
    return classify_audio_attention(
        text,
        rms_db=level,
        transcript_confidence=-0.2,
        duration_s=duration,
        active_app=app,
    )


@pytest.mark.parametrize(
    ("floor", "speech"),
    [
        (-52.0, -30.0),  # quiet room, laptop mic — the reported case
        (-70.0, -48.0),  # a very low-gain input
        (-40.0, -18.0),  # a hot input
        (-60.0, -35.0),
    ],
)
def test_the_owner_is_heard_on_any_gain_setting(floor, speech):
    """The same voice, four microphones. Only the SNR is constant."""
    _room_tone(floor)

    assert _speak(speech).source in ANSWERED


def test_the_reported_case_was_rejected_by_the_old_rule():
    """Guards the regression directly: -30 dBFS fails an absolute -22."""
    _room_tone(-52.0)

    assert _speak(-30.0).source in ANSWERED
    assert -30.0 < -22.0, "the level that was silently refused"


def test_media_playback_is_still_refused():
    """The documentary must not be able to talk to her."""
    _room_tone(-52.0)

    assert _speak(-30.0, app="Google Chrome").source == "device_media"


def test_speech_down_at_the_noise_floor_is_still_ambient():
    """Someone talking across the room is not talking to the machine."""
    _room_tone(-52.0)

    assert _speak(-46.0).source == "ambient_speech"


def test_the_margin_is_what_separates_them():
    """Just under the margin is ambient; just over it is a person."""
    _room_tone(-50.0)
    floor = room_noise_floor_db()
    assert floor is not None

    assert _speak(floor + _NEAR_FIELD_SNR_DB - 2.0).source not in ANSWERED
    assert _speak(floor + _NEAR_FIELD_SNR_DB + 2.0).source in ANSWERED


def test_an_uncalibrated_room_falls_back_to_the_absolute_level():
    """A fresh process must not decide a whole room is near-field."""
    assert room_noise_floor_db() is None

    assert _speak(-10.0).source in ANSWERED
    reset_room_calibration()
    assert _speak(-60.0).source not in ANSWERED


def test_a_single_dropout_cannot_define_the_floor():
    """A percentile, not a minimum: one -120 sample would rewrite the room."""
    _room_tone(-45.0, samples=40)
    classify_audio_attention(
        "", rms_db=-120.0, transcript_confidence=-0.2, duration_s=0.2
    )

    floor = room_noise_floor_db()
    assert floor is not None and floor > -60.0, floor


def test_the_wake_phrase_still_works_regardless_of_level():
    """Addressing her by name must never depend on gain."""
    _room_tone(-40.0)

    assert _speak(-95.0, text="hey aura are you there").source == "direct_address"


def test_calibration_ignores_unusable_readings():
    from core.senses.audio_attention import observe_room_loudness

    for bad in (float("nan"), float("inf"), float("-inf"), None, "loud"):
        observe_room_loudness(bad)  # type: ignore[arg-type]

    assert room_noise_floor_db() is None
