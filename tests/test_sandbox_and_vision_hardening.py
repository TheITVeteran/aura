import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from security import code_sandbox, sandbox


def test_secure_sandbox_rejects_same_named_absolute_binary(tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    runtime_sandbox = sandbox.SecureSandbox(workdir=tmp_path / "work")

    with pytest.raises(sandbox.SecurityViolationError):
        runtime_sandbox._validate_command([str(fake_python), "-V"])


def test_secure_sandbox_allows_current_runtime_python(tmp_path: Path) -> None:
    runtime_sandbox = sandbox.SecureSandbox(workdir=tmp_path / "work")

    assert runtime_sandbox._validate_command([sys.executable, "-V"]) == [
        sys.executable,
        "-V",
    ]


def test_secure_sandbox_launch_failure_is_degraded_result(monkeypatch, tmp_path: Path) -> None:
    runtime_sandbox = sandbox.SecureSandbox(workdir=tmp_path / "work")

    def fail_popen(*_args, **_kwargs):
        fail_popen.calls = getattr(fail_popen, "calls", 0) + 1
        raise OSError("process launch unavailable")

    monkeypatch.setattr(sandbox.subprocess, "Popen", fail_popen)

    result = runtime_sandbox.execute_command(["python", "-V"])

    assert result.success is False
    assert result.exit_code == -1
    assert "process launch unavailable" in result.stderr
    assert result.security_violations == ["process launch unavailable"]


def test_secure_sandbox_launches_through_subprocess_gateway(monkeypatch, tmp_path: Path) -> None:
    spawn_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class FakeProcess:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return "Python 3.x\n", ""

    class FakeSubprocessGateway:
        def spawn(self, argv, **kwargs):
            spawn_calls.append((tuple(argv), kwargs))
            return FakeProcess()

    runtime_sandbox = sandbox.SecureSandbox(workdir=tmp_path / "work")
    monkeypatch.setattr(
        sandbox,
        "get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )

    result = runtime_sandbox.execute_command(["python", "-V"])

    assert result.success is True
    assert spawn_calls
    _argv, kwargs = spawn_calls[0]
    assert kwargs["source"] == "security.sandbox.execute_command"
    assert kwargs["cwd"] == str(runtime_sandbox.workdir)


def test_code_repair_sandbox_does_not_swallow_programmer_fault(monkeypatch, tmp_path: Path) -> None:
    repair_sandbox = code_sandbox.CodeRepairSandbox()

    def fail_parse(_source: str):
        fail_parse.calls = getattr(fail_parse, "calls", 0) + 1
        raise AssertionError("programmer fault")

    monkeypatch.setattr(code_sandbox.ast, "parse", fail_parse)

    try:
        with pytest.raises(AssertionError, match="programmer fault"):
            repair_sandbox.verify_patch(tmp_path / "candidate.py", "print('ok')")
    finally:
        repair_sandbox.sandbox.cleanup()


def test_sandbox_and_vision_sources_have_typed_runtime_boundaries() -> None:
    source_paths = [
        Path("security/sandbox.py"),
        Path("security/code_sandbox.py"),
        Path("senses/vision_service.py"),
    ]

    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        assert "except Exception" not in source
    assert "while True" not in Path("senses/vision_service.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_vision_service_writes_bounded_capture(monkeypatch, tmp_path: Path) -> None:
    from senses import vision_service

    class FakeScreen:
        rgb = b"rgb"
        size = (1, 1)

    class FakeCapture:
        monitors = [None, object()]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def grab(self, _monitor):
            return FakeScreen()

    fake_mss = SimpleNamespace(
        mss=lambda: FakeCapture(),
        tools=SimpleNamespace(to_png=lambda _rgb, _size: b"png"),
    )

    monkeypatch.setattr(vision_service, "mss_available", True)
    monkeypatch.setattr(vision_service, "mss", fake_mss)

    await vision_service.run_vision_loop(output_dir=tmp_path, interval_s=0.01, max_frames=1)

    output = tmp_path / "sensory_vision.json"
    assert output.exists()
    assert '"status": "active"' in output.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_vision_service_records_capture_degradation(monkeypatch, tmp_path: Path) -> None:
    from senses import vision_service

    recorded: list[tuple[str, str]] = []

    class FailingCapture:
        monitors = [None, object()]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def grab(self, _monitor):
            self.grab_calls = getattr(self, "grab_calls", 0) + 1
            raise OSError("screen locked")

    fake_mss = SimpleNamespace(
        mss=lambda: FailingCapture(),
        tools=SimpleNamespace(to_png=lambda _rgb, _size: b"png"),
    )

    monkeypatch.setattr(vision_service, "mss_available", True)
    monkeypatch.setattr(vision_service, "mss", fake_mss)
    monkeypatch.setattr(
        vision_service,
        "record_degradation",
        lambda component, error: recorded.append((component, str(error))),
    )

    await vision_service.run_vision_loop(output_dir=tmp_path, interval_s=0.01, max_frames=1)

    assert recorded == [("vision_service.capture", "screen locked")]
    assert not (tmp_path / "sensory_vision.json").exists()
