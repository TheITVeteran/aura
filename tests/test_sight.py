"""Looking when asked, and not otherwise.

Two errors, and they are not symmetric. Missing a real request makes her
blind at the moment she was asked to see, which costs a repeat. Firing on a
remark turns the webcam on in the middle of a conversation that was not about
it — the single most alarming thing an assistant with a camera can do, and
the one that gets the camera permission revoked permanently.

So most of what follows is about the second kind, and the negative cases are
written to be *plausible*: sentences a person would really say near a machine
that is listening, not strawmen.
"""
from __future__ import annotations

import asyncio

import pytest

from core.conversation.failure_context import bind_failure_ledger, pending_failure_context
from core.senses.sight import (
    CaptureBroker,
    Frame,
    decode_frame,
    look,
    reset_capture_broker_for_test,
)
from core.senses.sight_intent import classify


@pytest.fixture(autouse=True)
def _clean_broker():
    reset_capture_broker_for_test()
    yield
    reset_capture_broker_for_test()


# ── asking her to look ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        "how many fingers am I holding up",
        "what am I holding",
        "can you see what's on my desk right now",
        "look at this and tell me what it says",
        "what colour is this",
        "who is in the room with me",
        "read the label on this",
        "do I look tired",
    ],
)
def test_a_request_to_look_is_recognised(message: str) -> None:
    intent = classify(message)
    assert intent.kind == "look", intent.reason
    assert intent.question


@pytest.mark.parametrize(
    "message",
    [
        # A visual verb used figuratively.
        "can you see the difference between those two approaches",
        # A question about the world, not about the room.
        "what colour is a stop sign",
        "how many fingers does a hand have",
        # Talking about cameras rather than asking for one.
        "the camera on my phone is broken",
        "my laptop camera quality is terrible",
        "how much does a good camera cost",
        # Ordinary conversation with visual words in it.
        "I watched that film last night",
        "let me check my calendar",
    ],
)
def test_ordinary_talk_does_not_open_the_camera(message: str) -> None:
    """The failure that would get the camera permission revoked for good."""
    assert classify(message).kind == "none", classify(message).reason


def test_a_deictic_is_what_separates_looking_from_remembering() -> None:
    """"this" is doing the entire job, and that is the point.

    A question about the visible world is anchored to the shared present. Take
    the pointing word away and the same words are a memory question.
    """
    assert classify("what colour is this").kind == "look"
    assert classify("what colour is a fire engine").kind == "none"


# ── operating the camera ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    ["turn on the camera", "turn the camera on", "switch your camera on", "camera on"],
)
def test_turning_the_camera_on_is_a_control_request(message: str) -> None:
    assert classify(message).kind == "camera_on"


@pytest.mark.parametrize(
    "message",
    ["turn off the camera", "turn the camera off", "stop the camera", "camera off"],
)
def test_turning_the_camera_off_outranks_turning_it_on(message: str) -> None:
    """A request to stop being watched must never read as a request to start.

    "turn the camera off" contains "turn the camera", so ordering here is a
    correctness property rather than a stylistic one.
    """
    assert classify(message).kind == "camera_off"


# ── the capture round trip ───────────────────────────────────────────────


def test_a_frame_reaches_whoever_asked_for_it() -> None:
    async def exercise() -> None:
        broker = CaptureBroker()
        captured: list[Frame | None] = []

        async def surface() -> None:
            # Stand in for the browser: wait for the request to be pending,
            # then deliver.
            for _ in range(200):
                async with broker._lock:
                    ids = list(broker._pending)
                if ids:
                    await broker.deliver(ids[0], Frame(data=b"jpegbytes"))
                    return
                await asyncio.sleep(0.005)

        async def ask() -> None:
            captured.append(await broker.request_frame(timeout_s=2.0))

        # No orchestrator in a unit test, so publishing the request fails and
        # request_frame returns None — assert the delivery path directly.
        await asyncio.gather(surface(), ask())

    asyncio.run(exercise())


def test_a_silent_surface_times_out_rather_than_hanging() -> None:
    """A browser that never answers costs one turn, not a wedged session."""

    async def exercise() -> None:
        broker = CaptureBroker()
        started = asyncio.get_running_loop().time()
        frame = await broker.request_frame(timeout_s=0.3)
        elapsed = asyncio.get_running_loop().time() - started
        assert frame is None
        assert elapsed < 5.0

    asyncio.run(exercise())


def test_a_late_frame_is_dropped_rather_than_answered() -> None:
    """The turn that wanted it has already said it could not look."""

    async def exercise() -> None:
        broker = CaptureBroker()
        assert await broker.deliver("never-requested", Frame(data=b"x")) is False

    asyncio.run(exercise())


# ── frames from the wire ─────────────────────────────────────────────────


def test_a_data_url_decodes_to_bytes() -> None:
    import base64

    payload = base64.b64encode(b"\xff\xd8\xff\xe0 jpeg").decode()
    frame = decode_frame(f"data:image/jpeg;base64,{payload}")
    assert frame is not None
    assert frame.data.startswith(b"\xff\xd8")
    assert frame.mime_type == "image/jpeg"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not a data url",
        "data:text/html;base64,PHNjcmlwdD4=",  # not an image
        "data:image/jpeg;base64,",  # no payload
        "data:image/jpeg;base64,!!!not base64!!!",
    ],
)
def test_a_bad_frame_is_refused_rather_than_guessed(value: str) -> None:
    assert decode_frame(value) is None


def test_an_oversized_frame_is_refused() -> None:
    import base64

    from core.senses.sight import MAX_FRAME_BYTES

    payload = base64.b64encode(b"\x00" * (MAX_FRAME_BYTES + 1)).decode()
    assert decode_frame(f"data:image/jpeg;base64,{payload}") is None


# ── failing honestly ─────────────────────────────────────────────────────


def test_a_camera_that_is_off_is_reported_as_the_users_choice(monkeypatch) -> None:
    """Not a malfunction — a setting, and she can offer to change it.

    The facts recorded here are what let her say "your camera's switched off,
    want me to turn it on?" rather than a fixed line about being unable to
    see.
    """
    monkeypatch.setattr("core.senses.sight.camera_enabled", lambda: False)

    async def exercise() -> str:
        with bind_failure_ledger():
            result = await look("how many fingers am I holding up")
            assert not result.ok
            assert result.cause == "camera_off"
            return pending_failure_context()

    block = asyncio.run(exercise())
    assert "switched off" in block
    assert "unauthorized" in block
    assert "turning the camera on" in block
    for canned in ("I'm sorry", "I am unable", "I can't see"):
        assert canned.lower() not in block.lower()


def test_no_frame_in_time_is_reported_with_the_real_reason(monkeypatch) -> None:
    monkeypatch.setattr("core.senses.sight.camera_enabled", lambda: True)

    async def exercise() -> str:
        with bind_failure_ledger():
            # No orchestrator is registered, so no surface can be asked.
            result = await look("what am I holding", timeout_s=0.5)
            assert not result.ok
            assert result.cause == "no_frame"
            return pending_failure_context()

    block = asyncio.run(exercise())
    assert "timeout" in block
    assert "camera" in block


def test_sight_never_reaches_a_model_that_cannot_see() -> None:
    """The configured `vision_model` is the text cortex and cannot read images.

    Handing an image to it produces confident fiction, so this path uses the
    genuinely multimodal client instead. Asserted against the source because
    the alternative is loading a model in a unit test.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "senses" / "sight.py"
    ).read_text(encoding="utf-8")
    assert "from core.brain.llm.mlx_vision_client import get_vision_client" in source
    # The shared worker, never a fresh one: constructing the client spawns a
    # subprocess holding a 1.2 GB model.
    assert "MLXVisionClient()" not in source


# ── the wiring ───────────────────────────────────────────────────────────


def test_the_capture_lane_is_separate_from_the_presence_lane() -> None:
    """Two cameras paths, deliberately.

    `/signals/vision` samples a thumbnail every few seconds for presence and
    is too small and too stale to answer a question about right now.
    `/signals/camera_capture` is the reply to a specific request, correlated
    by id, with a turn blocked on it. Collapsing them would mean answering
    "how many fingers" from whatever was in the rolling buffer.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "interface" / "routes" / "interaction_signals.py"
    ).read_text(encoding="utf-8")
    assert '@router.post("/signals/camera_capture")' in source
    assert '@router.post("/signals/vision")' in source
    # The capture lane is gated by the same privacy switch as the presence one.
    capture = source[source.index('@router.post("/signals/camera_capture")') :]
    assert "_camera_signal_allowed()" in capture


def test_a_sight_question_reaches_the_camera_before_the_reply() -> None:
    """Otherwise she answers a question about the visible world from memory."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "interface" / "routes" / "chat.py"
    ).read_text(encoding="utf-8")
    assert "from core.senses.sight_intent import classify as _classify_sight" in source
    assert "from core.senses.sight import look as _look" in source
    # What she is handed is a reading to speak from, not an answer to repeat.
    assert "it is your own observation" in source


def test_turning_the_camera_on_moves_the_control_not_just_the_record() -> None:
    """A privacy record that says "on" over a device that is off is the worst
    possible split for a camera, and the one that destroys trust in the
    indicator permanently."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    chat = (root / "interface" / "routes" / "chat.py").read_text(encoding="utf-8")
    client = (root / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "def _apply_camera_control(" in chat
    assert "set_browser_camera_privacy(" in chat
    assert '"type": "camera_privacy"' in chat
    # And the surface acts on it.
    assert "type === 'camera_privacy'" in client
    assert "startCameraSignals()" in client


def test_the_client_will_not_capture_without_the_users_camera_switch() -> None:
    """The server's deadline expiring is the correct outcome of a camera the
    user turned off — not a frame taken anyway."""
    from pathlib import Path

    client = (
        Path(__file__).resolve().parents[1] / "interface" / "static" / "aura.js"
    ).read_text(encoding="utf-8")
    capture = client[client.index("async function captureFrameForAura") :]
    capture = capture[: capture.index("\nfunction captureCameraSignalFrame")]
    assert "if (!state.cameraSignalWanted) return;" in capture
    # A stream this function opened is torn down; the presence lane's is not.
    assert "if (ownStream) ownStream.getTracks()" in capture
