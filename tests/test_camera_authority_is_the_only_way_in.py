"""The camera had five owners, one privacy switch, and no truthful state.

Six call sites across four modules opened `cv2.VideoCapture(0)` directly.
Nothing coordinated them, and the consequences were not theoretical:

  * `sensory_runtime.CameraProvider.capture()` never called
    `camera_allowed()`, and `continuous_vision` gated on the
    `features.camera_enabled` BUILD flag instead of the owner's settings
    toggle. Switching the camera off in Aura's settings left both of them
    capturing.
  * `sensory_motor_cortex`'s privacy branch slept while still HOLDING the
    device — camera indicator light on, setting switched off. That is the
    exact state the switch exists to prevent.
  * `sensory_integration._check_camera()` probed availability by OPENING the
    camera, contending for an exclusive device just to ask a question.
  * `core/body/camera_sensor.py` answered every read with a hard-coded
    "disabled_by_policy" having looked at nothing.
  * A holder that crashed left the device open forever, and every later
    acquisition failed with a reason none of them could name.

These test the authority and then assert, structurally, that nothing opens
the camera around it.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from core.perception.camera_authority import (
    STALE_LEASE_S,
    CameraAuthority,
    CameraDenial,
    CameraLease,
)

ROOT = Path(__file__).resolve().parents[1]


# ────────────────────────────────────────────────────── a fake cv2 device


class _FakeCapture:
    def __init__(self, *, opens: bool = True, frames: int | None = None) -> None:
        self._opens = opens
        self._frames = frames
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:  # noqa: N802 - cv2's spelling
        return self._opens and not self.released

    def read(self):
        self.reads += 1
        if self.released:
            return False, None
        if self._frames is not None and self.reads > self._frames:
            return False, None
        return True, object()

    def release(self) -> None:
        self.released = True


class _FakeCv2:
    def __init__(self, *, opens: bool = True, frames: int | None = None) -> None:
        self.opens = opens
        self.frames = frames
        self.created: list[_FakeCapture] = []

    def VideoCapture(self, index):  # noqa: N802 - cv2's spelling
        capture = _FakeCapture(opens=self.opens, frames=self.frames)
        self.created.append(capture)
        return capture


@pytest.fixture
def authority(monkeypatch):
    """An authority with a fake device and the owner's switch on."""

    def _make(*, opens: bool = True, frames: int | None = None, allowed: bool = True):
        auth = CameraAuthority()
        cv2 = _FakeCv2(opens=opens, frames=frames)
        auth._cv2 = cv2
        auth._cv2_checked = True
        import core.perception.camera_authority as mod

        monkeypatch.setattr(mod, "camera_allowed", lambda: allowed)
        # Autonomy is permitted unless a test says otherwise; the directive
        # path has its own tests below.
        monkeypatch.setattr(
            CameraAuthority, "_autonomous_permitted", lambda self, h, p: (True, "")
        )
        return auth, cv2

    return _make


# ─────────────────────────────────────────── the privacy boundary is real


def test_the_owners_switch_stops_acquisition(authority):
    """The defect: two paths captured with this switch off."""
    auth, cv2 = authority(allowed=False)

    result = auth.acquire("test", purpose="p")

    assert isinstance(result, CameraDenial)
    assert result.reason == "owner_disabled"
    assert not cv2.created, "the device was opened despite the owner's switch"


def test_the_switch_is_checked_before_the_device_is_touched(authority):
    """Order matters: opening then closing still lights the camera."""
    auth, cv2 = authority(allowed=False)

    auth.acquire("test", purpose="p")

    assert cv2.created == []


def test_a_denied_os_grant_stops_acquisition(authority):
    auth, cv2 = authority()
    auth.note_os_permission({"granted": False, "status": "denied", "guidance": "System Settings"})

    result = auth.acquire("test", purpose="p")

    assert isinstance(result, CameraDenial)
    assert result.reason == "os_permission"
    assert "System Settings" in result.remedy
    assert not cv2.created


def test_an_unknown_os_grant_does_not_block(authority):
    """Unknown is not denied.

    Failing closed on an unprobed TCC status would disable the camera on
    every path that cannot run an async probe — which is both threads.
    """
    auth, _ = authority()

    result = auth.acquire("test", purpose="p")

    assert isinstance(result, CameraLease)


def test_a_stale_os_grant_is_not_trusted(authority, monkeypatch):
    """The owner can revoke in System Settings at any time."""
    import core.perception.camera_authority as mod

    auth, _ = authority()
    auth.note_os_permission({"granted": False})
    # Age the cache past its TTL.
    recorded, payload = auth._tcc
    auth._tcc = (recorded - (mod.TCC_CACHE_TTL_S + 1), payload)

    assert auth._os_permission() is None


# ─────────────────────────────────────────── one device, one holder


def test_a_second_holder_is_refused_while_the_first_is_live(authority):
    auth, _ = authority()
    first = auth.acquire("streamer", purpose="stream")
    assert isinstance(first, CameraLease)

    second = auth.acquire("prober", purpose="probe")

    assert isinstance(second, CameraDenial)
    assert second.reason == "device_busy"
    assert "streamer" in second.detail


def test_releasing_lets_the_next_holder_in(authority):
    auth, _ = authority()
    first = auth.acquire("a", purpose="p")
    auth.release(first)

    second = auth.acquire("b", purpose="p")

    assert isinstance(second, CameraLease)


def test_release_actually_closes_the_device(authority):
    auth, cv2 = authority()
    lease = auth.acquire("a", purpose="p")

    auth.release(lease)

    assert cv2.created[0].released, "the camera handle was never released"
    assert not lease.active


def test_release_is_idempotent(authority):
    auth, _ = authority()
    lease = auth.acquire("a", purpose="p")

    auth.release(lease)
    auth.release(lease)  # must not raise


# ───────────────────────────────────────────── crash recovery


def test_a_dead_holder_loses_the_device(authority):
    """Before this, a crashed streamer held the camera until restart."""
    auth, _ = authority()
    first = auth.acquire("crashed_streamer", purpose="stream")
    assert isinstance(first, CameraLease)
    # Simulate the holder dying: it stops reading frames.
    first.last_used = time.time() - (STALE_LEASE_S + 1)

    second = auth.acquire("new_holder", purpose="p")

    assert isinstance(second, CameraLease), "the device was never reclaimed"
    assert auth.state()["leases_reclaimed"] == 1


def test_reclaiming_closes_the_dead_holders_handle(authority):
    auth, cv2 = authority()
    first = auth.acquire("crashed", purpose="p")
    first.last_used = time.time() - (STALE_LEASE_S + 1)

    auth.acquire("new", purpose="p")

    assert cv2.created[0].released, "the dead holder's handle leaked"


def test_a_reclaimed_lease_cannot_still_read(authority):
    """Otherwise two holders read the same device and neither knows."""
    auth, _ = authority()
    first = auth.acquire("crashed", purpose="p")
    first.last_used = time.time() - (STALE_LEASE_S + 1)
    auth.acquire("new", purpose="p")

    assert auth.read(first) is None


def test_reading_keeps_a_busy_holder_alive(authority):
    """A slow-but-live consumer must not be evicted mid-stream."""
    auth, _ = authority()
    lease = auth.acquire("streamer", purpose="p")
    lease.last_used = time.time() - (STALE_LEASE_S + 1)

    auth.read(lease)  # proves it is alive

    other = auth.acquire("thief", purpose="p")
    assert isinstance(other, CameraDenial)
    assert other.reason == "device_busy"


# ──────────────────────────────────────── governed autonomous observation


def test_autonomous_observation_asks_before_looking(monkeypatch):
    """Aura pointing a camera at her owner unprompted is a different ask."""
    auth = CameraAuthority()
    auth._cv2 = _FakeCv2()
    auth._cv2_checked = True
    import core.perception.camera_authority as mod

    monkeypatch.setattr(mod, "camera_allowed", lambda: True)
    asked: list[tuple[str, str]] = []

    def _refuse(self, holder, purpose):
        asked.append((holder, purpose))
        return False, "a standing directive forbids it: no watching me"

    monkeypatch.setattr(CameraAuthority, "_autonomous_permitted", _refuse)

    result = auth.acquire("watcher", purpose="monitoring", autonomous=True)

    assert isinstance(result, CameraDenial)
    assert result.reason == "autonomy_refused"
    assert "no watching me" in result.detail
    assert asked == [("watcher", "monitoring")]


def test_owner_requested_capture_does_not_need_the_autonomy_gate(monkeypatch):
    """The owner asking for a photo already carries the owner's authority."""
    auth = CameraAuthority()
    auth._cv2 = _FakeCv2()
    auth._cv2_checked = True
    import core.perception.camera_authority as mod

    monkeypatch.setattr(mod, "camera_allowed", lambda: True)
    called: list[str] = []

    def _spy(self, holder, purpose):
        called.append(holder)
        return True, ""

    monkeypatch.setattr(CameraAuthority, "_autonomous_permitted", _spy)

    auth.acquire("photo", purpose="owner asked", autonomous=False)

    assert called == [], "an owner-requested capture went through the autonomy gate"


def test_an_unreadable_directive_store_refuses_autonomy(monkeypatch):
    """Fail closed. Aura does not watch her owner because a file failed to
    parse."""
    auth = CameraAuthority()
    import core.perception.camera_authority as mod

    monkeypatch.setattr(mod, "camera_allowed", lambda: True)

    class _Broken:
        def check(self, **kw):
            raise RuntimeError("store on fire")

    monkeypatch.setattr(
        "core.governance.standing_directives.get_standing_directives",
        lambda: _Broken(),
    )

    permitted, why = auth._autonomous_permitted("h", "p")

    assert permitted is False
    assert why


# ───────────────────────────────────────────── calibrated abstention


def test_every_denial_names_a_reason_and_a_remedy(authority):
    """"no_frame_or_permission" conflated four different causes and gave
    the owner nothing to act on."""
    auth, _ = authority(allowed=False)

    denial = auth.acquire("t", purpose="p")

    assert isinstance(denial, CameraDenial)
    assert denial.reason
    assert denial.detail
    assert denial.remedy


def test_a_device_that_will_not_open_names_the_likely_cause(authority):
    """On macOS this is almost always an ungranted TCC prompt."""
    auth, _ = authority(opens=False)

    denial = auth.acquire("t", purpose="p")

    assert isinstance(denial, CameraDenial)
    assert denial.reason == "os_permission_suspected"
    assert "Privacy & Security" in denial.remedy


def test_a_failed_open_does_not_leak_the_handle(authority):
    auth, cv2 = authority(opens=False)

    auth.acquire("t", purpose="p")

    assert cv2.created[0].released


# ──────────────────────────────────────────────── truthful state


def test_state_reports_the_live_holder(authority):
    auth, _ = authority()
    auth.acquire("streamer", purpose="watching")

    state = auth.state()

    assert state["in_use"] is True
    assert state["has_optical_feed"] is True
    assert state["holder"]["holder"] == "streamer"
    assert "device_busy" in state["blockers"]


def test_state_reports_the_owner_switch(authority):
    auth, _ = authority(allowed=False)

    state = auth.state()

    assert state["owner_permission"] is False
    assert state["acquirable"] is False
    assert "owner_disabled" in state["blockers"]


def test_state_distinguishes_unknown_from_denied(authority):
    auth, _ = authority()

    assert auth.state()["os_permission"] is None
    assert auth.state()["os_permission_known"] is False

    auth.note_os_permission({"granted": False})
    assert auth.state()["os_permission"] is False
    assert auth.state()["os_permission_known"] is True


@pytest.mark.asyncio
async def test_the_body_sensor_reports_reality_not_a_constant(monkeypatch):
    """It returned "disabled_by_policy" forever, having looked at nothing."""
    from core.body.camera_sensor import CameraSensor
    from core.perception import camera_authority as mod

    auth = CameraAuthority()
    auth._cv2 = _FakeCv2()
    auth._cv2_checked = True
    monkeypatch.setattr(mod, "camera_allowed", lambda: True)
    monkeypatch.setattr(mod, "get_camera_authority", lambda: auth)
    monkeypatch.setattr(CameraAuthority, "_autonomous_permitted", lambda s, h, p: (True, ""))

    async def _no_probe(self):
        return {}

    monkeypatch.setattr(CameraAuthority, "refresh_os_permission", _no_probe)

    idle = await CameraSensor().read()
    assert idle["status"] == "idle_available"

    auth.acquire("streamer", purpose="p")
    streaming = await CameraSensor().read()

    assert streaming["status"] == "streaming"
    assert streaming["has_optical_feed"] is True
    assert streaming["holder"] == "streamer"


@pytest.mark.asyncio
async def test_the_body_sensor_says_unknown_when_it_cannot_tell(monkeypatch):
    """Reporting "off" when the truth is unavailable is how the constant
    survived. Unknown is a different answer and must stay different."""
    from core.body import camera_sensor

    def _boom():
        raise RuntimeError("authority unavailable")

    monkeypatch.setattr(
        "core.perception.camera_authority.get_camera_authority", _boom
    )

    result = await camera_sensor.CameraSensor().read()

    assert result["status"] == "unknown"


# ────────────────────────────── the structural guard on the whole class


_DIRECT_OPEN = re.compile(r"VideoCapture\s*\(\s*(0|camera_index)\s*\)")


def test_nothing_opens_the_camera_around_the_authority():
    """The guard that keeps this from happening a seventh time.

    Opening a video FILE is a different thing and is not matched here; this
    catches opening the live device by index.
    """
    offenders: list[str] = []
    for path in (ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        rel = str(path.relative_to(ROOT))
        if rel.endswith("perception/camera_authority.py"):
            continue
        for line_no, line in enumerate(
            path.read_text("utf-8", errors="ignore").splitlines(), 1
        ):
            if line.lstrip().startswith("#"):
                continue
            if _DIRECT_OPEN.search(line):
                offenders.append(f"{rel}:{line_no}")

    assert not offenders, (
        f"these open the camera device directly instead of through "
        f"camera_authority: {offenders}. Every one of them bypasses the "
        "owner's permission switch, the device lease, and crash recovery."
    )


def test_every_camera_holder_releases_what_it_acquires():
    """An acquire with no matching release is a device held until restart."""
    holders = [
        "core/perception/sensory_runtime.py",
        "core/perception/sensory_integration.py",
        "core/senses/continuous_vision.py",
        "core/somatic/sensory_motor_cortex.py",
    ]
    missing: list[str] = []
    for rel in holders:
        body = (ROOT / rel).read_text("utf-8")
        if ".acquire(" in body and ".release(" not in body:
            missing.append(rel)

    assert not missing, f"these acquire the camera and never release it: {missing}"


def test_the_privacy_switch_reaches_every_holder():
    """The original defect, checked at the seam rather than per module.

    Each holder must reach the authority — which checks `camera_allowed()`
    once, for all of them — rather than each remembering to check it, which
    is what two of them forgot to do.
    """
    holders = [
        "core/perception/sensory_runtime.py",
        "core/perception/sensory_integration.py",
        "core/senses/continuous_vision.py",
        "core/somatic/sensory_motor_cortex.py",
    ]
    missing = [
        rel
        for rel in holders
        if "camera_authority" not in (ROOT / rel).read_text("utf-8")
    ]

    assert not missing, f"these do not go through the camera authority: {missing}"
