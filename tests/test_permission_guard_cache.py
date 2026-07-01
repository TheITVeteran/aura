import os
import time
import unittest
from unittest.mock import patch

import core.security.permission_guard as permission_guard_module
from core.security.permission_guard import PermissionGuard, PermissionType, get_permission_guard


class ScreenProbeRecorder:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        return dict(self.result)


class TemporaryEnv:
    def __init__(self, updates):
        self.updates = updates
        self.previous = {}

    def __enter__(self):
        for key, value in self.updates.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestPermissionGuardCache(unittest.IsolatedAsyncioTestCase):
    async def test_force_refresh_reuses_fresh_cache(self):
        guard = PermissionGuard()
        guard._force_refresh_floor_s = 60.0
        screen_probe = ScreenProbeRecorder({"granted": True, "status": "active", "guidance": ""})
        guard._check_screen_permission = screen_probe

        with patch(
            "core.security.native_desktop_bridge.probe_native_desktop_bridge",
            return_value={"ok": False},
        ):
            with TemporaryEnv({"AURA_ASSUME_SCREEN_PERMISSION": "0"}):
                first = await guard.check_permission(PermissionType.SCREEN, force=True)
                second = await guard.check_permission(PermissionType.SCREEN, force=True)

        self.assertEqual(first, second)
        self.assertEqual(screen_probe.calls, 1)

    async def test_non_force_refreshes_stale_cache_after_ttl(self):
        guard = PermissionGuard()
        guard._cache_ttl_s = 5.0
        guard._cache[PermissionType.SCREEN] = {
            "granted": False,
            "status": "denied",
            "guidance": "stale",
        }
        guard._cache_ts[PermissionType.SCREEN] = time.monotonic() - 10.0
        screen_probe = ScreenProbeRecorder({"granted": True, "status": "active", "guidance": ""})
        guard._check_screen_permission = screen_probe

        with patch(
            "core.security.native_desktop_bridge.probe_native_desktop_bridge",
            return_value={"ok": False},
        ):
            with TemporaryEnv({"AURA_ASSUME_SCREEN_PERMISSION": "0"}):
                refreshed = await guard.check_permission(PermissionType.SCREEN, force=False)

        self.assertTrue(refreshed["granted"])
        self.assertEqual(screen_probe.calls, 1)

    async def test_direct_probe_bypasses_env_permission_assertion(self):
        guard = PermissionGuard()
        guard._screen_preflight_probe = lambda: {
            "granted": False,
            "status": "denied",
            "guidance": "direct denied",
        }

        with patch(
            "core.security.native_desktop_bridge.probe_native_desktop_bridge",
            return_value={"ok": False},
        ):
            with TemporaryEnv({
                "AURA_ASSUME_SCREEN_PERMISSION": "1",
                "AURA_PERMISSION_ASSERTIONS_ALLOWED": "1",
            }):
                asserted = await guard.check_permission(PermissionType.SCREEN)
                direct = await guard.check_permission_direct(PermissionType.SCREEN)

        self.assertTrue(asserted["granted"])
        self.assertEqual(asserted["status"], "asserted_env")
        self.assertFalse(direct["granted"])
        self.assertEqual(direct["status"], "denied")
        self.assertTrue(direct["direct_probe"])

    async def test_env_permission_assertion_is_ignored_without_explicit_test_gate(self):
        guard = PermissionGuard()
        guard._screen_preflight_probe = lambda: {
            "granted": False,
            "status": "denied",
            "guidance": "direct denied",
        }

        with patch(
            "core.security.native_desktop_bridge.probe_native_desktop_bridge",
            return_value={"ok": False},
        ):
            with TemporaryEnv({
                "AURA_ASSUME_SCREEN_PERMISSION": "1",
                "AURA_PERMISSION_ASSERTIONS_ALLOWED": "0",
            }):
                result = await guard.check_permission(PermissionType.SCREEN)

        self.assertFalse(result["granted"])
        self.assertEqual(result["status"], "denied")

    async def test_effective_check_accepts_trusted_native_app_bridge(self):
        guard = PermissionGuard()

        with patch(
            "core.security.native_desktop_bridge.probe_native_desktop_bridge",
            return_value={
                "ok": True,
                "screen_recording": True,
                "accessibility": True,
                "bundle_identifier": "com.aura.desktop",
                "bridge_executable": "/Applications/Aura.app/Contents/MacOS/aura-launcher",
            },
        ):
            result = await guard.check_permission(PermissionType.SCREEN)

        self.assertTrue(result["granted"])
        self.assertEqual(result["status"], "active_native_bridge")
        self.assertTrue(result["native_bridge"])

    async def test_effective_check_accepts_trusted_native_app_automation_bridge(self):
        guard = PermissionGuard()

        with patch(
            "core.security.native_desktop_bridge.probe_native_desktop_bridge",
            return_value={
                "ok": True,
                "screen_recording": True,
                "accessibility": True,
                "automation": True,
                "frontmost_app": "Aura",
                "bundle_identifier": "com.aura.desktop",
                "bridge_executable": "/Applications/Aura.app/Contents/MacOS/aura-launcher",
            },
        ):
            result = await guard.check_permission(PermissionType.AUTOMATION)

        self.assertTrue(result["granted"])
        self.assertEqual(result["status"], "active_native_bridge")
        self.assertTrue(result["native_bridge"])

    async def test_effective_check_reports_denied_native_app_bridge_without_python_fallback(self):
        guard = PermissionGuard()
        guard._screen_preflight_probe = lambda: {
            "granted": True,
            "status": "active",
            "guidance": "python child should not win",
        }

        with patch(
            "core.security.native_desktop_bridge.probe_native_desktop_bridge",
            return_value={
                "ok": True,
                "screen_recording": False,
                "accessibility": False,
                "bundle_identifier": "com.aura.desktop",
                "bridge_executable": "/Applications/Aura.app/Contents/MacOS/aura-launcher",
                "code_signature": {
                    "stable_tcc_identity": False,
                    "tcc_repair_hint": "remove and re-add Aura.app",
                },
            },
        ):
            result = await guard.check_permission(PermissionType.SCREEN)

        self.assertFalse(result["granted"])
        self.assertEqual(result["status"], "denied_native_bridge")
        self.assertTrue(result["native_bridge"])
        self.assertIn("remove and re-add Aura.app", result["guidance"])

    def test_shared_permission_guard_accessor_reuses_singleton(self):
        original = permission_guard_module._SHARED_PERMISSION_GUARD
        permission_guard_module._SHARED_PERMISSION_GUARD = None
        try:
            first = get_permission_guard()
            second = get_permission_guard()
        finally:
            permission_guard_module._SHARED_PERMISSION_GUARD = original

        self.assertIs(first, second)
