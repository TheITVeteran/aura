from __future__ import annotations

from pathlib import Path

import pytest

import core.self_modification.shadow_runtime as shadow_runtime_mod
from core.self_modification.shadow_runtime import ShadowRuntime


@pytest.mark.asyncio
async def test_shadow_runtime_boot_uses_file_and_subprocess_gateways(tmp_path, monkeypatch) -> None:
    file_write_calls: list[tuple[str, str]] = []
    spawn_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class FakeFileWriteGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown"):
            target = Path(path)
            file_write_calls.append((target.name, source))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding=encoding)

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"SHADOW_OK: Import successful\n", b""

    class FakeSubprocessGateway:
        async def spawn_async(self, argv, **kwargs):
            spawn_calls.append((tuple(argv), kwargs))
            return FakeProcess()

    monkeypatch.setattr(
        shadow_runtime_mod,
        "get_file_write_gateway",
        lambda: FakeFileWriteGateway(),
    )
    monkeypatch.setattr(
        shadow_runtime_mod,
        "get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )

    result = await ShadowRuntime(str(tmp_path))._run_in_subprocess(
        "print('shadow boot')",
        tmp_path,
        timeout_seconds=1,
    )

    assert result == {
        "exit_code": 0,
        "stdout": "SHADOW_OK: Import successful\n",
        "stderr": "",
    }
    assert file_write_calls == [
        ("_shadow_boot.py", "core.self_modification.shadow_runtime.shadow_boot_script")
    ]
    assert spawn_calls
    _argv, kwargs = spawn_calls[0]
    assert kwargs["source"] == "core.self_modification.shadow_runtime.shadow_boot"
    assert kwargs["cwd"] == str(tmp_path)
