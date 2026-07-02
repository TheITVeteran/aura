from __future__ import annotations

from pathlib import Path

import pytest

import core.skills.code_repl as code_repl_mod
from core.skills.code_repl import CodeREPLSkill


@pytest.mark.asyncio
async def test_code_repl_subprocess_fallback_uses_file_gateway_and_action_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_calls: list[tuple[str, str]] = []
    action_calls: list[dict[str, object]] = []

    class FakeFileWriteGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown") -> None:
            target = Path(path)
            file_calls.append((target.name, source))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding=encoding)

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def write_text_async(self, *args, **kwargs):
            return self.write_text(*args, **kwargs)

    class FakeActionExecutor:
        @classmethod
        async def execute(cls, **kwargs):
            action_calls.append(kwargs)
            return {"ok": True, "stdout": "ok\n", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(
        code_repl_mod,
        "get_file_write_gateway",
        lambda: FakeFileWriteGateway(),
    )
    monkeypatch.setattr(code_repl_mod, "ActionExecutor", FakeActionExecutor)

    result = await CodeREPLSkill()._execute_via_subprocess(
        "print('ok')",
        timeout_s=2,
        cwd=tmp_path,
    )

    assert result == {
        "ok": True,
        "stdout": "ok\n",
        "stderr": "",
        "returncode": 0,
        "engine": "subprocess",
        "summary": "Code executed via ActionExecutor.",
    }
    assert file_calls
    temp_name, source = file_calls[0]
    assert source == "core.skills.code_repl.temp_script"
    assert not (tmp_path / temp_name).exists()
    assert action_calls
    params = action_calls[0]["params"]
    assert isinstance(params, dict)
    assert params["cwd"] == str(tmp_path)
    assert params["timeout"] == 2.0
