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
from pathlib import Path

import pytest

from core.perception.camera_authority import CameraAuthority, CameraDenial

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


# ──────────────────────────────── the client can carry a payload back


def test_the_client_has_a_payload_returning_call():
    """`_send_command` reduces every reply to a bool.

    A camera frame is a payload, so a bool-only channel cannot carry one —
    which is part of why the sidecar never grew a camera command.
    """
    from core.senses.sensory_client import SensoryLocalClient

    assert hasattr(SensoryLocalClient, "request")


@pytest.mark.asyncio
async def test_the_client_returns_a_dict_when_the_worker_is_down():
    """Every caller is a capture path that must turn failure into a named
    reason, so this must not raise."""
    from core.senses.sensory_client import SensoryLocalClient

    client = SensoryLocalClient()
    reply = await client.request("camera_frame", auto_restart=False)

    assert isinstance(reply, dict)
    assert reply["status"] == "error"


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

        def isOpened(self):
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
