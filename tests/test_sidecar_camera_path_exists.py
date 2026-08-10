"""The documented production camera path did not exist.

`core/media/safe_imports.py` says, in two places, that the sensory worker is
"the production process boundary for camera/screen media libraries" and that
"the sidecar sensory worker remains the production camera path". It also
installs an import guard that makes `import cv2` RAISE in Aura's primary
macOS process, because faster-whisper pulls in PyAV and loading OpenCV's
AVFoundation stack beside it registers duplicate Objective-C classes.

The sidecar handled five commands: `init_vision`, `capture_screen`,
`init_audio`, `ping`, `exit`. All screen, no camera.

So on macOS the camera could not be opened anywhere. Not in the main
process, which is banned and raises. Not in the sidecar, which had no
command for it. The "production camera path" was a sentence in a docstring.

These tests hold the two halves together: that the worker really answers the
camera commands (driven against the real loop, not a mock), and that the
authority routes to the sidecar rather than reporting `no_backend` when
in-process capture is banned.
"""
from __future__ import annotations

import queue
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.perception.camera_authority import (
    CameraAuthority,
    CameraDenial,
    _SidecarCapture,
)

ROOT = Path(__file__).resolve().parents[1]


# ────────────────────────────── the worker answers the camera commands


class _Queue:
    """A queue the worker loop can drive without a real process."""

    def __init__(self, items=()):
        self._items = list(items)
        self.sent: list[dict] = []

    def get(self, timeout=None):
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)

    def put(self, item):
        self.sent.append(item)

    def empty(self):
        return not self._items


def _run_worker(commands: list[dict]) -> list[dict]:
    """Drive the real `sensory_worker_loop` over a scripted command list."""
    from core.senses.sensory_worker import sensory_worker_loop

    requests = _Queue([*commands, {"command": "exit"}])
    responses = _Queue()
    sensory_worker_loop(requests, responses)
    return responses.sent


def test_the_worker_knows_the_camera_commands():
    """Before this, all three came back "Unknown command"."""
    replies = _run_worker(
        [
            {"command": "camera_frame"},
            {"command": "camera_close"},
        ]
    )

    messages = [str(r.get("msg", "")) for r in replies]
    assert not any("Unknown command" in m for m in messages), (
        f"the sidecar still has no camera commands: {messages}"
    )


def test_a_frame_request_without_an_open_camera_is_named_not_unknown():
    replies = _run_worker([{"command": "camera_frame"}])

    assert replies[0]["status"] == "error"
    assert replies[0]["msg"] == "camera_not_open"


def test_closing_a_camera_that_was_never_open_is_not_an_error():
    """Release must be safe to call unconditionally, or every caller needs
    its own bookkeeping to avoid a spurious failure on cleanup."""
    replies = _run_worker([{"command": "camera_close"}])

    assert replies[0]["status"] == "ok"


def test_an_unknown_command_is_still_rejected():
    """The catch-all must survive; otherwise a typo silently succeeds."""
    replies = _run_worker([{"command": "camera_teleport"}])

    assert replies[0]["status"] == "error"
    assert "Unknown command" in replies[0]["msg"]


def test_the_worker_releases_the_device_on_exit():
    """A worker that exits holding the camera leaves the light on until the
    OS reclaims the handle at some unrelated moment."""
    source = (ROOT / "core" / "senses" / "sensory_worker.py").read_text("utf-8")

    exit_block = source[source.index('elif cmd == "exit"') :]
    exit_block = exit_block[: exit_block.index("else:")]

    assert "camera.release()" in exit_block, (
        "the sidecar exits without releasing the camera"
    )


def test_frames_cross_the_boundary_encoded():
    """A raw 1080p BGR frame is ~6MB and this is a multiprocessing queue.

    Sending the numpy array would also put numpy in the pickle, which is
    what the process boundary exists to avoid.
    """
    source = (ROOT / "core" / "senses" / "sensory_worker.py").read_text("utf-8")

    frame_block = source[source.index('elif cmd == "camera_frame"') :]
    frame_block = frame_block[: frame_block.index('elif cmd == "camera_close"')]

    assert "imencode" in frame_block


def test_the_camera_handle_outlives_a_single_frame():
    """Reopening per grab strobes the macOS camera indicator and costs
    ~200ms a read."""
    source = (ROOT / "core" / "senses" / "sensory_worker.py").read_text("utf-8")

    frame_block = source[source.index('elif cmd == "camera_frame"') :]
    frame_block = frame_block[: frame_block.index('elif cmd == "camera_close"')]

    assert "VideoCapture" not in frame_block, (
        "camera_frame opens the device itself instead of reusing the handle"
    )


def test_sidecar_jpeg_decodes_when_primary_process_cv2_is_forbidden(monkeypatch):
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    encoded = BytesIO()
    image_module.new("RGB", (2, 1), (255, 0, 0)).save(encoded, format="JPEG")
    calls: list[str] = []

    def request(_self, command, data=None, **_kwargs):
        del data
        calls.append(command)
        if command == "camera_open":
            return {"status": "ok"}
        if command == "camera_frame":
            return {
                "status": "ok",
                "data": encoded.getvalue(),
                "width": 2,
                "height": 1,
            }
        return {"status": "ok"}

    monkeypatch.setattr(_SidecarCapture, "_call", request)
    monkeypatch.setitem(sys.modules, "cv2", None)

    capture = _SidecarCapture(0)
    ok, frame = capture.read()
    capture.release()

    assert ok is True
    assert frame.shape == (1, 2, 3)
    assert frame.dtype == np.uint8
    assert int(frame[0, 0, 2]) > int(frame[0, 0, 0])  # BGR red channel
    assert capture.last_jpeg == encoded.getvalue()
    assert calls == ["camera_open", "camera_frame", "camera_close"]


def test_sidecar_rejects_frame_metadata_that_differs_from_decoded_jpeg(monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    encoded = BytesIO()
    image_module.new("RGB", (2, 1), (0, 255, 0)).save(encoded, format="JPEG")

    def request(_self, command, data=None, **_kwargs):
        del data
        if command == "camera_open":
            return {"status": "ok"}
        return {
            "status": "ok",
            "data": encoded.getvalue(),
            "width": 200,
            "height": 100,
        }

    monkeypatch.setattr(_SidecarCapture, "_call", request)
    capture = _SidecarCapture(0)

    assert capture.read() == (False, None)
    assert capture.last_error == "camera_frame_metadata_mismatch"
    assert capture.last_jpeg is None


def test_worker_releases_camera_when_owner_switch_changes(monkeypatch):
    import core.senses.sensory_worker as worker

    class FakeFrame:
        shape = (1, 1, 3)

    class FakeBuffer:
        @staticmethod
        def tobytes():
            return b"jpeg"

    class FakeCamera:
        released = 0

        @staticmethod
        def isOpened():  # noqa: N802 - mirrors OpenCV's capture contract
            return True

        @staticmethod
        def read():
            return True, FakeFrame()

        def release(self):
            self.released += 1

    camera = FakeCamera()
    fake_cv2 = type(
        "FakeCV2",
        (),
        {
            "VideoCapture": staticmethod(lambda _index: camera),
            "imencode": staticmethod(lambda _suffix, _frame: (True, FakeBuffer())),
        },
    )
    decisions = iter((True, False))
    monkeypatch.setattr(worker, "camera_allowed", lambda: next(decisions))
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    replies = _run_worker(
        [
            {"command": "camera_open", "data": {"index": 0}},
            {"command": "camera_frame"},
        ]
    )

    assert replies[:2] == [
        {"status": "ok"},
        {"status": "error", "msg": "owner_disabled"},
    ]
    assert camera.released == 1


# ──────────────────────────────── the client can carry a payload back


def test_the_client_has_a_payload_returning_call():
    """`_send_command` reduces every reply to a bool.

    A camera frame is a payload, so a bool-only channel cannot carry one —
    which is part of why the sidecar never grew a camera command.
    """
    from core.senses.sensory_client import SensoryLocalClient

    assert hasattr(SensoryLocalClient, "request")
    assert hasattr(SensoryLocalClient, "request_sync")


@pytest.mark.asyncio
async def test_the_client_returns_a_dict_when_the_worker_is_down():
    """Every caller is a capture path that must turn failure into a named
    reason, so this must not raise."""
    from core.senses.sensory_client import SensoryLocalClient

    client = SensoryLocalClient()
    reply = await client.request("camera_frame", auto_restart=False)

    assert isinstance(reply, dict)
    assert reply["status"] == "error"


@pytest.mark.asyncio
async def test_sync_sidecar_request_refuses_to_block_an_event_loop(monkeypatch):
    from core.senses.sensory_client import SensoryLocalClient

    client = SensoryLocalClient()
    monkeypatch.setattr(client, "is_alive", lambda: True)

    reply = client.request_sync("camera_frame")

    assert reply == {"status": "error", "msg": "sync_request_on_event_loop"}


@pytest.mark.asyncio
async def test_async_sidecar_request_runs_the_queue_round_trip_off_loop(monkeypatch):
    import asyncio
    import threading

    from core.senses.sensory_client import SensoryLocalClient

    client = SensoryLocalClient()
    loop_thread = threading.get_ident()
    observed_threads: list[int] = []
    monkeypatch.setattr(client, "is_alive", lambda: True)

    def _blocking(_cmd, _data, _timeout):
        observed_threads.append(threading.get_ident())
        return {"status": "ok", "data": b"jpeg"}

    monkeypatch.setattr(client, "_request_blocking", _blocking)
    reply = await asyncio.wait_for(client.request("camera_frame"), timeout=1.0)

    assert reply["status"] == "ok"
    assert observed_threads and observed_threads[0] != loop_thread


def test_sidecar_capture_uses_the_cross_thread_client_surface(monkeypatch):
    import core.senses.sensory_client as client_mod

    calls: list[str] = []

    class _Client:
        def request_sync(self, command, data=None, **_kwargs):
            del data
            calls.append(command)
            return {"status": "error", "msg": "bounded"}

        async def request(self, *_args, **_kwargs):
            pytest.fail("camera authority nested an async request")

    monkeypatch.setattr(client_mod, "get_sensory_client", lambda: _Client())

    capture = _SidecarCapture(0)

    assert capture.isOpened() is False
    assert calls == ["camera_open"]


# ─────────────────────── the authority routes to the sidecar, not to "no"


def test_a_banned_in_process_backend_does_not_mean_no_camera(monkeypatch):
    """The whole defect in one assertion.

    With in-process cv2 banned, the authority used to report `no_backend` —
    indistinguishable from a machine with no camera at all, and the fix for
    each is completely different.
    """
    import core.perception.camera_authority as mod

    auth = CameraAuthority()
    monkeypatch.setattr(mod, "camera_allowed", lambda: True)
    monkeypatch.setattr(CameraAuthority, "sidecar_required", lambda self: True)
    monkeypatch.setattr(CameraAuthority, "_backend", lambda self: None)

    state = auth.state()

    assert state["backend_available"] is True
    assert state["transport"] == "sidecar"
    assert "no_backend" not in state["blockers"]


def test_no_camera_at_all_still_says_no_backend(monkeypatch):
    """The opposite case must stay distinguishable."""
    import core.perception.camera_authority as mod

    auth = CameraAuthority()
    monkeypatch.setattr(mod, "camera_allowed", lambda: True)
    monkeypatch.setattr(CameraAuthority, "sidecar_required", lambda self: False)
    monkeypatch.setattr(CameraAuthority, "_backend", lambda self: None)

    state = auth.state()

    assert state["backend_available"] is False
    assert state["transport"] == "none"
    assert "no_backend" in state["blockers"]


def test_acquiring_through_the_sidecar_uses_the_sidecar_handle(monkeypatch):
    import core.perception.camera_authority as mod

    auth = CameraAuthority()
    monkeypatch.setattr(mod, "camera_allowed", lambda: True)
    monkeypatch.setattr(CameraAuthority, "sidecar_required", lambda self: True)
    monkeypatch.setattr(CameraAuthority, "_backend", lambda self: None)

    calls: list[tuple[str, object]] = []

    class _FakeSidecar:
        def __init__(self, index):
            calls.append(("open", index))
            self.last_jpeg = None

        def isOpened(self):  # noqa: N802 - mirrors OpenCV's capture contract
            return True

        def read(self):
            calls.append(("read", None))
            return True, object()

        def release(self):
            calls.append(("release", None))

    monkeypatch.setattr(mod, "_SidecarCapture", _FakeSidecar)

    lease = auth.acquire("t", purpose="p")
    assert not isinstance(lease, CameraDenial), lease
    assert auth.read(lease) is not None
    auth.release(lease)

    assert calls == [("open", 0), ("read", None), ("release", None)]


def test_the_privacy_switch_still_wins_over_the_sidecar(monkeypatch):
    """A second transport must not become a second way around the switch."""
    import core.perception.camera_authority as mod

    auth = CameraAuthority()
    monkeypatch.setattr(mod, "camera_allowed", lambda: False)
    monkeypatch.setattr(CameraAuthority, "sidecar_required", lambda self: True)
    monkeypatch.setattr(CameraAuthority, "_backend", lambda self: None)
    monkeypatch.setattr(
        mod,
        "_SidecarCapture",
        lambda index: pytest.fail("the sidecar opened despite the owner's switch"),
    )

    denial = auth.acquire("t", purpose="p")

    assert isinstance(denial, CameraDenial)
    assert denial.reason == "owner_disabled"


def test_safe_imports_claim_now_has_something_behind_it():
    """The docstring promised a production camera path. Check it exists."""
    claim = (ROOT / "core" / "media" / "safe_imports.py").read_text("utf-8")
    worker = (ROOT / "core" / "senses" / "sensory_worker.py").read_text("utf-8")

    assert "production camera path" in claim
    for command in ("camera_open", "camera_frame", "camera_close"):
        assert f'cmd == "{command}"' in worker, (
            f"safe_imports calls the sidecar the production camera path but "
            f"the worker has no {command} command"
        )


# ───────────── every production consumer actually reaches the sidecar


def test_authority_reuses_sidecar_jpeg_without_importing_cv2(monkeypatch):
    np = pytest.importorskip("numpy")
    auth = CameraAuthority()
    payload = b"\xff\xd8already-encoded\xff\xd9"
    capture = SimpleNamespace(last_jpeg=payload)
    lease = SimpleNamespace(active=True, capture=capture)
    monkeypatch.setitem(sys.modules, "cv2", None)

    assert auth.jpeg_bytes(lease, np.zeros((2, 2, 3), dtype=np.uint8)) == payload


def test_authority_selects_the_best_measured_frame_from_a_bounded_burst(monkeypatch):
    np = pytest.importorskip("numpy")
    auth = CameraAuthority()
    dark = np.full((240, 320, 3), 8, dtype=np.uint8)
    sharp = np.random.default_rng(91).integers(
        0, 256, (240, 320, 3), dtype=np.uint8
    )
    frames = iter((dark, sharp, dark))
    lease = SimpleNamespace(active=True, last_error="", capture=object())
    monkeypatch.setattr(auth, "read", lambda _lease: next(frames))
    monkeypatch.setattr(
        auth,
        "jpeg_bytes",
        lambda _lease, frame: b"sharp" if frame is sharp else b"dark",
    )

    selected = auth.capture_best_still(lease, attempts=3, settle_s=0)

    assert selected is not None
    assert selected.frame is sharp
    assert selected.jpeg == b"sharp"
    assert selected.attempt == 2
    assert selected.attempts == 3


@pytest.mark.asyncio
async def test_vision_system_still_capture_reaches_sidecar_when_cv2_is_forbidden(
    monkeypatch,
):
    import core.perception.camera_authority as camera_mod
    from core.perception.sensory_integration import VisionSystem

    np = pytest.importorskip("numpy")
    jpeg = BytesIO()
    pytest.importorskip("PIL.Image").new("RGB", (8, 8), (30, 80, 120)).save(
        jpeg, format="JPEG"
    )
    lease = SimpleNamespace(active=True, last_error="", capture=object())

    class _Authority:
        released = False

        def acquire(self, *_args, **_kwargs):
            return lease

        def read(self, _lease):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def jpeg_bytes(self, _lease, _frame):
            return jpeg.getvalue()

        def capture_best_still(self, _lease):
            from core.perception.frame_quality import assess_frame

            frame = self.read(_lease)
            return SimpleNamespace(
                frame=frame,
                jpeg=self.jpeg_bytes(_lease, frame),
                quality=assess_frame(frame),
                attempt=2,
                attempts=4,
            )

        def release(self, _lease):
            self.released = True

        def sidecar_required(self):
            return True

    authority = _Authority()
    monkeypatch.setattr(camera_mod, "get_camera_authority", lambda: authority)
    monkeypatch.setitem(sys.modules, "cv2", None)
    vision = VisionSystem()

    async def _available():
        return True

    monkeypatch.setattr(vision, "_get_camera_available", _available)
    result = await vision.capture()

    assert result["type"] == "image"
    assert result["data"]
    assert result["frame_quality"]["pixels"] == 64
    assert result["capture_selection"] == {"selected_attempt": 2, "attempts": 4}
    assert authority.released is True


@pytest.mark.asyncio
async def test_vision_system_video_capture_uses_pyav_not_main_process_cv2(
    monkeypatch, tmp_path
):
    import core.perception.camera_authority as camera_mod
    from core.perception.sensory_integration import VisionSystem

    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    lease = SimpleNamespace(active=True, last_error="", capture=object())

    class _Authority:
        released = False

        def acquire(self, *_args, **_kwargs):
            return lease

        def read(self, _lease):
            return np.full((16, 16, 3), 96, dtype=np.uint8)

        def release(self, _lease):
            self.released = True

    authority = _Authority()
    monkeypatch.setattr(camera_mod, "get_camera_authority", lambda: authority)
    monkeypatch.setitem(sys.modules, "cv2", None)
    vision = VisionSystem()

    async def _available():
        return True

    monkeypatch.setattr(vision, "_get_camera_available", _available)
    target = tmp_path / "sidecar.mp4"
    result = await vision.capture(duration=0.08, save_path=str(target))

    assert result["type"] == "video", result
    assert result["frames"] >= 1
    assert target.stat().st_size > 0
    assert authority.released is True


def test_on_demand_camera_provider_captures_without_inprocess_cv2(monkeypatch):
    import core.perception.camera_authority as camera_mod
    from core.perception.sensory_runtime import CameraProvider

    np = pytest.importorskip("numpy")
    lease = SimpleNamespace(active=True, last_error="", capture=object())

    class _Authority:
        def acquire(self, *_args, **_kwargs):
            return lease

        def read(self, _lease):
            return np.zeros((24, 32, 3), dtype=np.uint8)

        def release(self, _lease):
            pass

    monkeypatch.setattr(camera_mod, "get_camera_authority", lambda: _Authority())
    provider = CameraProvider()
    monkeypatch.setattr(provider, "_load", lambda: False)
    monkeypatch.setitem(sys.modules, "cv2", None)

    sight = provider.capture()

    assert sight.captured is True
    assert (sight.width, sight.height) == (32, 24)
    assert sight.detail["faces"] is None, "unmeasured face count became a false zero"


def test_continuous_camera_keeps_sidecar_enabled_when_main_process_is_blocked(
    monkeypatch, tmp_path
):
    import core.config as config_mod
    import core.perception.camera_authority as camera_mod
    import core.senses.continuous_vision as continuous

    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(features=SimpleNamespace(camera_enabled=True)),
    )
    monkeypatch.setattr(
        continuous,
        "main_process_camera_policy",
        lambda _requested: (False, "main process blocked"),
    )
    monkeypatch.setattr(
        camera_mod,
        "get_camera_authority",
        lambda: SimpleNamespace(state=lambda: {"backend_available": True}),
    )

    buffer = continuous.ContinuousSensoryBuffer(tmp_path)

    assert buffer.camera_enabled is False
    assert buffer.camera_capture_enabled is True


def test_continuous_sidecar_calls_are_offloaded_from_the_event_loop():
    source = (ROOT / "core" / "senses" / "continuous_vision.py").read_text("utf-8")
    camera_branch = source[source.index("if self.camera_capture_enabled:") :]
    camera_branch = camera_branch[: camera_branch.index("def get_visual_context_parts")]

    assert "cv2_main_process_blocked" not in camera_branch
    assert "await asyncio.to_thread(" in camera_branch
    assert "authority.read," in camera_branch
    assert "authority.jpeg_bytes," in camera_branch
