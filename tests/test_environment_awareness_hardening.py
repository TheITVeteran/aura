from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core import environment_awareness as env_module
from core.environment_awareness import UserIdentityManager


def test_environment_command_probe_blocks_non_allowlisted_command(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(
        env_module,
        "record_degradation",
        lambda module, exc, **kwargs: recorded.append((module, type(exc).__name__, kwargs)),
    )

    result = asyncio.run(env_module._run_command(["python", "--version"]))

    assert result == ""
    assert recorded
    assert recorded[0][0] == "environment_awareness"
    assert recorded[0][1] == "PermissionError"
    assert recorded[0][2]["receipt_required"] is True
    assert "non-allowlisted" in str(recorded[0][2]["action"])


def test_environment_command_probe_uses_subprocess_gateway(monkeypatch):
    calls: list[tuple[tuple[str, ...], float, bool, bool, str]] = []

    class Gateway:
        async def run_async(
            self,
            cmd,
            **_,
        ):
            calls.append(
                (
                    tuple(cmd),
                    _["timeout"],
                    _["read_only"],
                    _["capture_output"],
                    _["source"],
                )
            )
            return SimpleNamespace(returncode=0, stdout="123\n", stderr="")

    monkeypatch.setattr(env_module, "get_subprocess_gateway", lambda: Gateway())

    result = asyncio.run(env_module._run_command(["sysctl", "-n", "hw.memsize"], timeout_s=2))

    assert result == "123"
    assert calls == [
        (
            ("sysctl", "-n", "hw.memsize"),
            2.0,
            True,
            True,
            "core.environment_awareness.run_command",
        )
    ]


def test_user_identity_manager_corrupt_store_fails_soft(monkeypatch, tmp_path):
    recorded: list[tuple[str, str, dict[str, object]]] = []
    data_path = tmp_path / "user_sessions.json"
    data_path.write_text("{bad-json", encoding="utf-8")

    monkeypatch.setattr(env_module, "_data_path", lambda _filename: data_path)
    monkeypatch.setattr(
        env_module,
        "record_degradation",
        lambda module, exc, **kwargs: recorded.append((module, type(exc).__name__, kwargs)),
    )

    manager = UserIdentityManager()

    assert manager._known_fingerprints == {}
    assert recorded
    assert recorded[0][0] == "environment_awareness"
    assert recorded[0][2]["receipt_required"] is True
    assert "fingerprint store" in str(recorded[0][2]["action"])


def test_user_identity_manager_persists_under_configured_data_path(monkeypatch, tmp_path):
    data_path = tmp_path / "user_sessions.json"
    writes: list[tuple[str, str, str]] = []

    class FileGateway:
        def write_text(self, path, text, *, encoding, source):
            writes.append((str(path), encoding, source))
            path.write_text(text, encoding=encoding)

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def write_text_async(self, *args, **kwargs):
            return self.write_text(*args, **kwargs)

    monkeypatch.setattr(env_module, "_data_path", lambda _filename: data_path)
    monkeypatch.setattr(env_module, "get_file_write_gateway", lambda: FileGateway())

    manager = UserIdentityManager()
    manager.register_identity("abc123", "Bryan")

    assert data_path.exists()
    assert "Bryan" in data_path.read_text(encoding="utf-8")
    assert writes == [
        (
            str(data_path),
            "utf-8",
            "core.environment_awareness.save_known_fingerprints",
        )
    ]
