from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest


def _denied(reason):
    from core.security.screen_capture_policy import ScreenCaptureAdmission

    return ScreenCaptureAdmission(allowed=False, reason=reason)


def test_runtime_setting_denies_before_foreground_probe(monkeypatch):
    from core.security import screen_capture_policy as policy

    probed = False

    def _probe():
        nonlocal probed
        probed = True
        return "Google Chrome", "Private material - Incognito"

    monkeypatch.setattr(policy, "screen_allowed", lambda: False)
    monkeypatch.setattr("core.senses.screen_context.frontmost_window_hint", _probe)

    admission = policy.evaluate_screen_capture_admission()

    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.RUNTIME_SETTING_DISABLED
    assert probed is False


@pytest.mark.parametrize(
    ("app", "title"),
    [
        ("Google Chrome", "Private material - Incognito"),
        ("Safari", "Private Browsing"),
        ("1Password", "Vault"),
        ("Terminal", "banking credentials"),
    ],
)
def test_private_foreground_is_denied_without_metadata_leak(monkeypatch, app, title):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    admission = policy.evaluate_screen_capture_admission(context=(app, title))

    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.PRIVATE_FOREGROUND
    rendered = str(admission.to_receipt()) + admission.public_error
    assert app not in rendered
    assert title not in rendered


def test_unknown_foreground_fails_closed(monkeypatch):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    admission = policy.evaluate_screen_capture_admission(context=("", ""))
    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.FOREGROUND_UNKNOWN


def test_browser_with_unreadable_title_fails_closed(monkeypatch):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    admission = policy.evaluate_screen_capture_admission(
        context=("Google Chrome", "")
    )
    assert admission.allowed is False
    assert admission.reason is policy.ScreenCaptureDenial.BROWSER_TITLE_UNKNOWN


def test_ordinary_foreground_is_admitted(monkeypatch):
    from core.security import screen_capture_policy as policy

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    admission = policy.evaluate_screen_capture_admission(
        context=("Terminal", "pytest")
    )
    assert admission.allowed is True
    assert admission.to_receipt()["reason"] == "none"


def test_complete_native_foreground_avoids_subprocess(monkeypatch):
    from core.senses import screen_context

    called = False

    class _Gateway:
        def run(self, *_args, **_kwargs):
            nonlocal called
            called = True
            raise AssertionError("subprocess should not run for complete native metadata")

    monkeypatch.setattr(
        screen_context,
        "_native_frontmost_window_hint",
        lambda: ("Terminal", "pytest"),
    )
    monkeypatch.setattr(screen_context, "get_subprocess_gateway", lambda: _Gateway())

    assert screen_context.frontmost_window_hint() == ("Terminal", "pytest")
    assert called is False


def test_subprocess_can_complete_native_app_without_title(monkeypatch):
    from core.senses import screen_context

    class _Gateway:
        @staticmethod
        def run(*_args, **_kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout="Google Chrome|Public documentation",
            )

    monkeypatch.setattr(
        screen_context,
        "_native_frontmost_window_hint",
        lambda: ("Google Chrome", ""),
    )
    monkeypatch.setattr(screen_context, "get_subprocess_gateway", lambda: _Gateway())

    assert screen_context.frontmost_window_hint() == (
        "Google Chrome",
        "Public documentation",
    )


@pytest.mark.asyncio
async def test_host_automation_denies_before_creating_capture_path(monkeypatch):
    from core.capabilities.host_automation import HostAutomationProvider
    from core.security import screen_capture_policy as policy

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _evaluate():
        return denial

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", _evaluate)
    provider = HostAutomationProvider()

    receipt = await provider.take_screenshot()

    assert receipt.success is False
    assert receipt.adapter == "screen_capture_policy"
    assert "private" in receipt.error


@pytest.mark.asyncio
async def test_screen_perception_denies_before_accessibility_or_pixels(monkeypatch):
    from core.perception.screen_perception import ScreenPerception
    from core.security import screen_capture_policy as policy

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _evaluate():
        return denial

    async def _must_not_read(*_args, **_kwargs):
        raise AssertionError("accessibility content was read before privacy admission")

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", _evaluate)
    perception = ScreenPerception()
    monkeypatch.setattr(perception, "_frontmost_accessibility_summary", _must_not_read)
    monkeypatch.setattr(perception, "_take_screenshot", _must_not_read)

    snapshot = await perception.capture(save_screenshot=True)

    assert snapshot.capture_denied is True
    assert snapshot.screen_text == ""
    assert snapshot.accessibility_text == ""
    assert "private" in snapshot.unavailable_reason


@pytest.mark.asyncio
async def test_local_vision_denies_before_screenshot_backend(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.senses.screen_vision import LocalVision

    denial = _denied(policy.ScreenCaptureDenial.FOREGROUND_UNKNOWN)

    async def _evaluate():
        return denial

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", _evaluate)

    assert await LocalVision().capture_screen() is None


@pytest.mark.asyncio
async def test_continuous_vision_does_not_initialize_backend_while_denied(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _evaluate():
        return denial

    class _MSS:
        called = False

        @classmethod
        def mss(cls):
            cls.called = True
            raise AssertionError("capture backend initialized while privacy denied")

    monkeypatch.setattr(
        "core.senses.continuous_vision.evaluate_screen_capture_admission_async",
        _evaluate,
    )
    buffer = ContinuousSensoryBuffer.__new__(ContinuousSensoryBuffer)
    buffer.sct = None
    buffer.monitor = None
    buffer._mss_module = _MSS
    buffer._screen_probe_cooldown_until = 0.0
    buffer._screen_permission_notice_at = 0.0
    buffer._screen_permission_notice_interval_s = 300.0

    assert await buffer._ensure_screen_backend() is False
    assert _MSS.called is False


def test_sensory_sidecar_denies_before_importing_capture_backend(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.senses import sensory_worker

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)
    monkeypatch.setattr(
        sensory_worker,
        "evaluate_screen_capture_admission",
        lambda: denial,
    )
    requests: queue.Queue[dict[str, str]] = queue.Queue()
    responses: queue.Queue[dict[str, object]] = queue.Queue()
    requests.put({"command": "init_vision"})
    requests.put({"command": "exit"})

    sensory_worker.sensory_worker_loop(requests, responses)

    response = responses.get(timeout=0.1)
    assert response["status"] == "error"
    assert response["msg"] == "private_foreground"
    assert "title" not in str(response)


def test_sensory_sidecar_rechecks_foreground_before_each_frame(monkeypatch):
    import sys

    from core.security import screen_capture_policy as policy
    from core.senses import sensory_worker

    admitted = policy.ScreenCaptureAdmission(allowed=True, context_known=True)
    denied = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)
    decisions = iter((admitted, denied))
    captured = False

    class _MSSContext:
        monitors = [{}, {"width": 1, "height": 1}]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def grab(self, _monitor):
            nonlocal captured
            captured = True
            raise AssertionError("private foreground was captured")

    fake_mss = SimpleNamespace(mss=lambda: _MSSContext())
    fake_cv2 = SimpleNamespace(__version__="test")
    monkeypatch.setitem(sys.modules, "mss", fake_mss)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(
        sensory_worker,
        "evaluate_screen_capture_admission",
        lambda: next(decisions),
    )
    monkeypatch.setattr(sensory_worker, "_screen_capture_preflight_allowed", lambda: True)
    requests: queue.Queue[dict[str, str]] = queue.Queue()
    responses: queue.Queue[dict[str, object]] = queue.Queue()
    requests.put({"command": "init_vision"})
    requests.put({"command": "capture_screen"})
    requests.put({"command": "exit"})

    sensory_worker.sensory_worker_loop(requests, responses)

    assert responses.get(timeout=0.1) == {"status": "ok"}
    refusal = responses.get(timeout=0.1)
    assert refusal["status"] == "error"
    assert refusal["msg"] == "private_foreground"
    assert captured is False


def test_native_bridge_refuses_before_transport(monkeypatch):
    from core.security import native_desktop_bridge as bridge
    from core.security import screen_capture_policy as policy

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)
    transported = False

    def _transport(*_args, **_kwargs):
        nonlocal transported
        transported = True
        return {"ok": True}

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission", lambda: denial)
    monkeypatch.setattr(bridge, "_invoke_resident_bridge", _transport)

    result = bridge.invoke_native_desktop_bridge("screenshot", read_only=True)

    assert result["ok"] is False
    assert result["bridge_transport"] == "policy_refusal"
    assert transported is False


@pytest.mark.asyncio
async def test_computer_use_screenshot_denies_before_backend(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.tools.computer_use import ComputerUseSkill

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _require():
        raise policy.ScreenCaptureDeniedError(denial)

    monkeypatch.setattr("core.tools.computer_use.screen_allowed", lambda: True)
    monkeypatch.setattr(policy, "require_screen_capture_admission_async", _require)
    skill = ComputerUseSkill.__new__(ComputerUseSkill)

    with pytest.raises(policy.ScreenCaptureDeniedError):
        await skill._default_screenshot(SimpleNamespace(target="screen", payload={}))


@pytest.mark.asyncio
async def test_screen_sensor_returns_privacy_safe_denial(monkeypatch):
    from core.body import screen_sensor
    from core.body.screen_sensor import ScreenSensor
    from core.security import screen_capture_policy as policy

    denial = _denied(policy.ScreenCaptureDenial.PRIVATE_FOREGROUND)

    async def _evaluate():
        return denial

    monkeypatch.setattr(screen_sensor, "screen_allowed", lambda: True)
    monkeypatch.setattr(
        screen_sensor,
        "evaluate_screen_capture_admission_async",
        _evaluate,
    )

    result = await ScreenSensor.__new__(ScreenSensor).read()

    assert result["available"] is False
    assert result["capture_admission"]["reason"] == "private_foreground"
    assert "title" not in str(result)


def test_computer_use_helper_refuses_unknown_foreground(monkeypatch):
    from core.security import screen_capture_policy as policy
    from core.skills.computer_use import ComputerUseSkill

    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    monkeypatch.setattr(
        "core.senses.screen_context.frontmost_window_hint",
        lambda: ("", ""),
    )
    called = False

    def _read(_self):
        nonlocal called
        called = True
        return "screen content"

    monkeypatch.setattr(ComputerUseSkill, "_read_screen_text_macos", _read)
    result = ComputerUseSkill.__new__(ComputerUseSkill).read_screen_text()

    assert "refused" in result
    assert called is False
