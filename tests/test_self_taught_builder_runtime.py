from __future__ import annotations

import pytest

from core.capabilities import self_taught_builder


@pytest.mark.asyncio
async def test_functional_test_uses_attributed_read_only_subprocess_gateway(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    tester = tmp_path / "test_game.js"
    tester.write_text("console.log('{}')", encoding="utf-8")
    monkeypatch.setattr(self_taught_builder, "_TESTER", tester)
    calls: list[tuple[list[str], dict[str, object]]] = []

    class _Process:
        returncode = 0

        async def communicate(self):
            return b'{"playable": true}', b""

    class _Gateway:
        async def spawn_async(self, argv, **kwargs):
            calls.append((list(argv), dict(kwargs)))
            return _Process()

    monkeypatch.setattr(self_taught_builder, "get_subprocess_gateway", lambda: _Gateway())

    result = await self_taught_builder._functional_test(str(tmp_path / "app.html"))

    assert result["playable"] is True
    assert calls[0][0][0] == "node"
    assert calls[0][1]["read_only"] is True
    assert calls[0][1]["source"] == "self_taught_builder.functional_test"


@pytest.mark.asyncio
async def test_functional_test_reaps_process_when_communication_times_out(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    tester = tmp_path / "test_game.js"
    tester.write_text("while (true) {}", encoding="utf-8")
    monkeypatch.setattr(self_taught_builder, "_TESTER", tester)

    class _Process:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False
            self.killed = False

        async def communicate(self):
            raise TimeoutError("functional test timed out")

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    process = _Process()

    class _Gateway:
        async def spawn_async(self, _argv, **_kwargs):
            return process

    monkeypatch.setattr(self_taught_builder, "get_subprocess_gateway", lambda: _Gateway())

    result = await self_taught_builder._functional_test(str(tmp_path / "app.html"))

    assert result["playable"] is None
    assert "timed out" in result["reason"]
    assert process.terminated is True
    assert process.killed is False
