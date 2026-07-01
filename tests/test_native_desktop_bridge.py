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
        lambda: {
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
