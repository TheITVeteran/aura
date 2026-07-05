import json
from types import SimpleNamespace

import pytest


def test_native_bridge_probe_uses_read_only_canonical_subprocess(monkeypatch, tmp_path):
    from core.security import native_desktop_bridge as bridge

    monkeypatch.setenv("AURA_NATIVE_BRIDGE_DIR", str(tmp_path / "no-resident-bridge"))
    executable = tmp_path / "aura-launcher"
    executable.write_text("bridge", encoding="utf-8")
    executable.chmod(0o755)
    calls = []

    class _Gateway:
        def run(self, argv, **kwargs):
            calls.append((argv, kwargs))
            return SimpleNamespace(
                stdout=json.dumps({
                    "ok": True,
                    "screen_recording": True,
                    "accessibility": True,
                    "bundle_identifier": "com.aura.desktop",
                }),
                stderr="",
                returncode=0,
            )

    monkeypatch.setattr(bridge, "bridge_executable", lambda: executable)
    monkeypatch.setattr(bridge, "get_subprocess_gateway", lambda: _Gateway())
    bridge._PROBE_CACHE = (0.0, {})

    result = bridge.probe_native_desktop_bridge(force=True)

    assert result["ok"] is True
    assert result["accessibility"] is True
    assert calls[0][0][0] == str(executable)
    assert calls[0][0][1] == "--native-desktop-bridge"
    assert calls[0][1]["read_only"] is True
    assert calls[0][1]["source"] == "native_desktop_bridge.probe"


def test_native_bridge_does_not_hold_negative_probe_cache_for_ready_ttl(monkeypatch, tmp_path):
    from core.security import native_desktop_bridge as bridge

    monkeypatch.setenv("AURA_NATIVE_BRIDGE_DIR", str(tmp_path / "no-resident-bridge"))
    monkeypatch.setattr(bridge, "_PROBE_READY_TTL_S", 60.0)
    monkeypatch.setattr(bridge, "_PROBE_DEGRADED_TTL_S", 0.001)
    executable = tmp_path / "aura-launcher"
    executable.write_text("bridge", encoding="utf-8")
    executable.chmod(0o755)
    results = [
        {"ok": False, "error": "launch_race"},
        {
            "ok": True,
            "screen_recording": True,
            "accessibility": True,
            "automation": True,
            "bundle_identifier": "com.aura.desktop",
        },
    ]

    class _Gateway:
        def run(self, _argv, **_kwargs):
            if _argv and str(_argv[0]).endswith("codesign"):
                return SimpleNamespace(
                    stdout="",
                    stderr=(
                        "Identifier=com.aura.desktop\n"
                        "Authority=Aura Local Code Signing\n"
                        "TeamIdentifier=not set\n"
                    ),
                    returncode=0,
                )
            return SimpleNamespace(stdout=json.dumps(results.pop(0)), stderr="", returncode=0)

    monkeypatch.setattr(bridge, "bridge_executable", lambda: executable)
    monkeypatch.setattr(bridge, "get_subprocess_gateway", lambda: _Gateway())
    bridge._PROBE_CACHE = (0.0, {})

    first = bridge.probe_native_desktop_bridge(force=False)
    import time

    time.sleep(0.01)
    second = bridge.probe_native_desktop_bridge(force=False)

    assert first["ok"] is False
    assert second["ok"] is True
    assert second["accessibility"] is True
    assert results == []


def test_local_certificate_is_a_stable_tcc_identity_without_team_id(monkeypatch, tmp_path):
    from core.security import native_desktop_bridge as bridge

    executable = tmp_path / "aura-launcher"
    executable.write_text("bridge", encoding="utf-8")

    class _Gateway:
        def run(self, _argv, **_kwargs):
            return SimpleNamespace(
                stdout="",
                stderr=(
                    "Identifier=com.aura.desktop\n"
                    "Authority=Aura Local Code Signing\n"
                    "TeamIdentifier=not set\n"
                    "CDHash=1234abcd\n"
                ),
                returncode=0,
            )

    monkeypatch.setattr(bridge, "get_subprocess_gateway", lambda: _Gateway())

    summary = bridge._code_signature_summary(executable)

    assert summary["adhoc"] is False
    assert summary["authorities"] == ["Aura Local Code Signing"]
    assert summary["stable_tcc_identity"] is True
    assert "tcc_repair_hint" not in summary


def test_native_bridge_permission_requests_do_not_use_short_probe_timeout(monkeypatch):
    from core.security import native_desktop_bridge as bridge

    observed: list[tuple[str, float]] = []

    def _resident(command, *, timeout, **_payload):
        observed.append((command, timeout))
        return {"ok": True, "screen_recording": True}

    monkeypatch.setattr(bridge, "_invoke_resident_bridge", _resident)

    result = bridge.invoke_native_desktop_bridge("request_screen", read_only=True, timeout=45.0)

    assert result["ok"] is True
    assert observed == [("request_screen", 45.0)]


def test_native_bridge_probe_keeps_short_resident_timeout(monkeypatch):
    from core.security import native_desktop_bridge as bridge

    observed: list[tuple[str, float]] = []

    def _resident(command, *, timeout, **_payload):
        observed.append((command, timeout))
        return {
            "ok": True,
            "screen_recording": True,
            "accessibility": True,
            "automation": True,
        }

    monkeypatch.setattr(bridge, "_invoke_resident_bridge", _resident)

    result = bridge.invoke_native_desktop_bridge("probe", read_only=True, timeout=45.0)

    assert result["ok"] is True
    assert observed == [("probe", 3.0)]


def test_native_bridge_keeps_resident_probe_authoritative_over_one_shot(monkeypatch, tmp_path):
    from core.security import native_desktop_bridge as bridge

    monkeypatch.setenv("AURA_NATIVE_BRIDGE_DIR", str(tmp_path / "resident"))
    observed: list[tuple[str, float]] = []
    executable = tmp_path / "aura-launcher"
    executable.write_text("bridge", encoding="utf-8")
    executable.chmod(0o755)

    def _resident(command, *, timeout, **_payload):
        observed.append((command, timeout))
        return {
            "ok": True,
            "screen_recording": True,
            "accessibility": False,
            "automation": True,
            "bridge_transport": "resident_ipc",
        }

    class _Gateway:
        def run(self, _argv, **_kwargs):
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "ok": True,
                        "screen_recording": True,
                        "accessibility": True,
                        "automation": True,
                        "bundle_identifier": "com.aura.desktop",
                    }
                ),
                stderr="",
                returncode=0,
            )

    monkeypatch.setattr(bridge, "bridge_executable", lambda: executable)
    monkeypatch.setattr(bridge, "_invoke_resident_bridge", _resident)
    monkeypatch.setattr(bridge, "get_subprocess_gateway", lambda: _Gateway())

    result = bridge.invoke_native_desktop_bridge("probe", read_only=True, timeout=5.0)

    assert result["ok"] is True
    assert result["accessibility"] is False
    assert result["bridge_transport"] == "resident_ipc"
    assert "resident_reconciled" not in result
    assert observed == [("probe", 3.0)]


def test_native_bridge_does_not_reconcile_one_shot_with_different_bundle(monkeypatch, tmp_path):
    from core.security import native_desktop_bridge as bridge

    monkeypatch.setenv("AURA_NATIVE_BRIDGE_DIR", str(tmp_path / "resident"))
    executable = tmp_path / "aura-launcher"
    executable.write_text("bridge", encoding="utf-8")
    executable.chmod(0o755)

    def _resident(command, *, timeout, **_payload):
        return {
            "ok": True,
            "screen_recording": True,
            "accessibility": False,
            "automation": True,
            "bundle_identifier": "com.aura.desktop",
            "bridge_transport": "resident_ipc",
        }

    class _Gateway:
        def run(self, _argv, **_kwargs):
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "ok": True,
                        "screen_recording": True,
                        "accessibility": True,
                        "automation": True,
                        "bundle_identifier": "com.other.app",
                    }
                ),
                stderr="",
                returncode=0,
            )

    monkeypatch.setattr(bridge, "bridge_executable", lambda: executable)
    monkeypatch.setattr(bridge, "_invoke_resident_bridge", _resident)
    monkeypatch.setattr(bridge, "get_subprocess_gateway", lambda: _Gateway())

    result = bridge.invoke_native_desktop_bridge("probe", read_only=True, timeout=5.0)

    assert result["bridge_transport"] == "resident_ipc"
    assert result["accessibility"] is False
    assert "resident_reconciled" not in result


def test_native_pyautogui_uses_resident_transport_for_effects(monkeypatch):
    from core.security import native_desktop_bridge as bridge

    calls = []

    def _invoke(command, **payload):
        calls.append((command, payload))
        return {"ok": True}

    monkeypatch.setattr(bridge, "invoke_native_desktop_bridge", _invoke)
    native = bridge.NativePyAutoGUI()
    native.PAUSE = 0

    native.write("visible text")

    assert calls == [("write", {"text": "visible text", "interval": 0.0})]


def test_resident_bridge_skips_stale_ipc_dirs_when_launcher_is_not_alive(monkeypatch, tmp_path):
    from core.security import native_desktop_bridge as bridge

    bridge_root = tmp_path / "native_bridge"
    request_dir = bridge_root / "requests"
    response_dir = bridge_root / "responses"
    request_dir.mkdir(parents=True)
    response_dir.mkdir(parents=True)

    monkeypatch.setenv("AURA_NATIVE_BRIDGE_DIR", str(bridge_root))
    monkeypatch.setattr(bridge, "_resident_bridge_process_running", lambda *_args, **_kwargs: False)

    result = bridge._invoke_resident_bridge("probe", timeout=0.05)

    assert result is None
    assert list(request_dir.iterdir()) == []
    assert list(response_dir.iterdir()) == []


def test_native_pyautogui_translates_general_input_primitives(monkeypatch):
    from core.security import native_desktop_bridge as bridge

    calls = []

    def _invoke(command, **payload):
        calls.append((command, payload))
        return {"ok": True}

    monkeypatch.setattr(bridge, "invoke_native_desktop_bridge", _invoke)
    native = bridge.NativePyAutoGUI()
    native.PAUSE = 0

    native.moveTo(120, 240, duration=0.5)
    native.click(120, 240, clicks=2, button="left")
    native.write("general text", interval=0.01)
    native.hotkey("command", "l")
    native.press("escape")
    native.scroll(-4)

    assert [command for command, _ in calls] == [
        "move", "click", "write", "hotkey", "press", "scroll",
    ]
    assert calls[1][1]["clicks"] == 2
    assert calls[3][1]["keys"] == ["command", "l"]


@pytest.mark.asyncio
async def test_permission_guard_accepts_granted_native_app_identity(monkeypatch):
    from core.security import native_desktop_bridge as bridge
    from core.security.permission_guard import PermissionGuard, PermissionType

    monkeypatch.setattr(
        bridge,
        "probe_native_desktop_bridge",
        lambda force=False, prefer_one_shot=False: {
            "ok": True,
            "screen_recording": True,
            "accessibility": True,
            "bundle_identifier": "com.aura.desktop",
            "bridge_executable": "/Applications/Aura.app/Contents/MacOS/aura-launcher",
        },
    )
    guard = PermissionGuard()

    screen = await guard.check_permission_direct(PermissionType.SCREEN)
    accessibility = await guard.check_permission_direct(PermissionType.ACCESSIBILITY)

    assert screen["granted"] is True
    assert accessibility["granted"] is True
    assert screen["status"] == "active_native_bridge"
    assert accessibility["bundle_identifier"] == "com.aura.desktop"
