"""tests/test_macos_permission_onboarding.py
================================================
A shippable macOS app must (a) report the *real* mic/camera TCC state — not a
test double that always says "granted" — so a denied permission surfaces in onboarding
instead of failing voice/vision silently, and (b) declare the usage strings and
entitlements the bundle needs (especially Apple Events, without which all
Notes/Mail/Finder/browser automation dies with -1743).

These tests lock in:
  - the AVFoundation mic/camera probe maps every TCC status code correctly and
    degrades gracefully when the framework is missing,
  - "undetermined" (macOS prompts on first use) is NOT treated as an actionable
    missing permission, while "denied" IS,
  - the bundle manifest carries the required usage strings + entitlements and
    serializes to a valid plist.
"""
from __future__ import annotations

import asyncio
import plistlib
import types

import core.security.permission_guard as pg
from core.security.macos_bundle_manifest import (
    HARDENED_RUNTIME_ENTITLEMENTS,
    TCC_USAGE_DESCRIPTIONS,
    info_plist_overrides,
    write_entitlements_plist,
)
from core.security.permission_guard import PermissionGuard, PermissionType
from core.security.permission_setup import PermissionStatus, check_all_permissions, format_report


class _FakeAVCaptureDevice:
    def __init__(self, status: int):
        self._status = status

    def authorizationStatusForMediaType_(self, _media_type):  # noqa: N802 (ObjC name)
        return self._status


class _FakeAVFoundation:
    AVMediaTypeAudio = "soun"
    AVMediaTypeVideo = "vide"

    def __init__(self, status: int):
        self.AVCaptureDevice = _FakeAVCaptureDevice(status)


def _probe(monkeypatch, status: int) -> dict:
    guard = PermissionGuard()
    monkeypatch.setattr(pg.sys, "platform", "darwin")
    monkeypatch.setitem(
        __import__("sys").modules, "AVFoundation", _FakeAVFoundation(status)
    )
    return guard._av_media_authorization_probe("AVMediaTypeAudio", PermissionType.MIC)


def test_av_probe_authorized(monkeypatch):
    out = _probe(monkeypatch, 3)
    assert out["granted"] is True
    assert out["status"] == "active"


def test_av_probe_denied(monkeypatch):
    out = _probe(monkeypatch, 2)
    assert out["granted"] is False
    assert out["status"] == "denied"
    assert "System Settings" in out["guidance"]


def test_av_probe_restricted_is_denied(monkeypatch):
    out = _probe(monkeypatch, 1)
    assert out["granted"] is False
    assert out["status"] == "denied"


def test_av_probe_undetermined_is_soft(monkeypatch):
    out = _probe(monkeypatch, 0)
    assert out["granted"] is False
    assert out["status"] == "undetermined"
    assert "first time" in out["guidance"].lower()


def test_av_probe_missing_framework_assumes_granted(monkeypatch):
    guard = PermissionGuard()
    monkeypatch.setattr(pg.sys, "platform", "darwin")
    # Ensure AVFoundation import fails inside the probe.
    monkeypatch.setitem(__import__("sys").modules, "AVFoundation", None)
    out = guard._av_media_authorization_probe("AVMediaTypeAudio", PermissionType.MIC)
    # Never hard-block a feature on a missing framework.
    assert out["granted"] is True
    assert out["status"] == "assumed"


def test_av_probe_not_applicable_off_macos(monkeypatch):
    guard = PermissionGuard()
    monkeypatch.setattr(pg.sys, "platform", "linux")
    out = guard._av_media_authorization_probe("AVMediaTypeAudio", PermissionType.MIC)
    assert out == {"granted": True, "status": "not_applicable", "guidance": ""}


def test_current_process_identity_reports_tcc_target():
    identity = PermissionGuard().current_process_identity()

    assert identity["pid"] > 0
    assert identity["executable"]
    assert "bundle_identifier" in identity
    assert "parent_pid" in identity


def test_screen_capture_request_updates_cache(monkeypatch):
    guard = PermissionGuard()
    fake_quartz = types.SimpleNamespace(CGRequestScreenCaptureAccess=lambda: True)

    monkeypatch.setattr(pg.sys, "platform", "darwin")
    monkeypatch.setitem(__import__("sys").modules, "Quartz", fake_quartz)

    assert guard.request_screen_capture_access() is True
    cached = guard._cache[PermissionType.SCREEN]
    assert cached["granted"] is True
    assert cached["status"] == "active"


def test_denied_permission_is_actionable_undetermined_is_not():
    denied = PermissionStatus(
        name="MIC", granted=False, available=True, guidance="g", status="denied"
    )
    undetermined = PermissionStatus(
        name="CAMERA", granted=False, available=True, guidance="g", status="undetermined"
    )
    granted = PermissionStatus(
        name="SCREEN", granted=True, available=True, guidance="", status="active"
    )
    assert denied.actionable is True
    assert undetermined.actionable is False
    assert granted.actionable is False


def test_check_all_permissions_separates_denied_from_undetermined(monkeypatch):
    import core.security.permission_setup as ps

    class DemoGuard:
        _RESULTS = {
            "MIC": {"granted": False, "status": "denied"},
            "CAMERA": {"granted": False, "status": "undetermined"},
            "SCREEN": {"granted": True, "status": "active"},
            "ACCESSIBILITY": {"granted": True, "status": "active"},
            "AUTOMATION": {"granted": True, "status": "active"},
        }

        async def check_permission(self, ptype, force=False):
            return dict(self._RESULTS[ptype.name])

        def get_guidance(self, ptype):
            return f"guidance:{ptype.name}"

    monkeypatch.setattr(ps.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ps, "get_permission_guard", lambda: DemoGuard())

    report = asyncio.run(check_all_permissions(refresh=True))
    # Only the denied mic is an actionable "missing" item; undetermined camera is not.
    assert report.missing == ["MIC"]
    assert report.all_granted is False
    text = format_report(report)
    assert "MIC" in text
    assert "Will be requested when first used" in text
    assert "CAMERA" in text  # surfaced informationally, not as missing


def test_bundle_manifest_has_required_keys():
    overrides = info_plist_overrides()
    for key in (
        "NSMicrophoneUsageDescription",
        "NSCameraUsageDescription",
        "NSScreenCaptureUsageDescription",
        "NSAccessibilityUsageDescription",
        "NSAppleEventsUsageDescription",  # the one most often missed
        "NSSpeechRecognitionUsageDescription",
    ):
        assert key in overrides and overrides[key].strip()
    assert overrides is not TCC_USAGE_DESCRIPTIONS  # returns a copy, not the source


def test_entitlements_include_apple_events_and_jit():
    assert HARDENED_RUNTIME_ENTITLEMENTS["com.apple.security.automation.apple-events"] is True
    assert HARDENED_RUNTIME_ENTITLEMENTS["com.apple.security.device.audio-input"] is True
    assert HARDENED_RUNTIME_ENTITLEMENTS["com.apple.security.device.screen-capture"] is True
    assert HARDENED_RUNTIME_ENTITLEMENTS["com.apple.security.network.client"] is True
    assert HARDENED_RUNTIME_ENTITLEMENTS["com.apple.security.network.server"] is True
    assert HARDENED_RUNTIME_ENTITLEMENTS["com.apple.security.cs.allow-jit"] is True
    assert HARDENED_RUNTIME_ENTITLEMENTS[
        "com.apple.security.cs.disable-library-validation"
    ] is True


def test_write_entitlements_plist_is_valid(tmp_path):
    out = write_entitlements_plist(tmp_path / "sub" / "aura.entitlements")
    assert out.exists()
    with out.open("rb") as handle:
        loaded = plistlib.load(handle)
    assert loaded == HARDENED_RUNTIME_ENTITLEMENTS
