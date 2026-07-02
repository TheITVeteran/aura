"""Tests for the authoritative self-modification safety harness."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import core.self_modification.safe_modification_harness as harness_mod
from core.self_modification.distributed_sandbox_gateway import DistributedSandboxGateway
from core.self_modification.safe_modification_harness import SafeModificationHarness


def test_harness_rejects_patch_without_related_pytest(tmp_path: Path) -> None:
    source = tmp_path / "core" / "uncovered_runtime_patch.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")

    result = asyncio.run(SafeModificationHarness(tmp_path).run(["core/uncovered_runtime_patch.py"]))

    assert result.passed is False
    assert result.checks["pytest"] is False
    assert any("no related pytest files" in error for error in result.errors)


def test_harness_runs_related_pytest_before_passing(tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "calculator.py"
    tests = tmp_path / "tests" / "test_calculator.py"
    source.parent.mkdir(parents=True)
    tests.parent.mkdir(parents=True)
    source.write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    tests.write_text(
        "from pkg.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    result = asyncio.run(SafeModificationHarness(tmp_path).run(["pkg/calculator.py"]))

    assert result.passed is True
    assert result.checks["pytest"] is True
    assert result.checks["candidate_overlay"] is True
    assert result.checks["source_immutable"] is True


def test_harness_runs_requested_scale_out_gate_against_candidate(tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "calculator.py"
    tests = tmp_path / "tests" / "test_calculator.py"
    source.parent.mkdir(parents=True)
    tests.parent.mkdir(parents=True)
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tests.write_text(
        "from pkg.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        SafeModificationHarness(tmp_path).run(
            ["pkg/calculator.py"],
            require_distributed_sandbox=True,
            distributed_gateway=DistributedSandboxGateway(provider="local", max_workers=1),
        )
    )
    assert result.passed is True
    assert result.checks["distributed_sandbox"] is True


def test_harness_runs_pytest_against_candidate_bytes(tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "calculator.py"
    tests = tmp_path / "tests" / "test_calculator.py"
    source.parent.mkdir(parents=True)
    tests.parent.mkdir(parents=True)
    source.write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    tests.write_text(
        "from pkg.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    staged = "def add(a: int, b: int) -> int:\n    return a - b\n"
    result = asyncio.run(
        SafeModificationHarness(tmp_path).run(
            ["pkg/calculator.py"],
            patch_content={"pkg/calculator.py": staged},
        )
    )

    assert result.passed is False
    assert result.checks["candidate_overlay"] is True
    assert result.checks["pytest"] is False
    assert result.checks["source_immutable"] is True
    assert source.read_text(encoding="utf-8") == (
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )


def test_harness_treats_changed_test_file_as_coverage(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_self_check.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_self_check():\n    assert True\n", encoding="utf-8")

    result = asyncio.run(SafeModificationHarness(tmp_path).run(["tests/test_self_check.py"]))

    assert result.passed is True
    assert result.checks["pytest"] is True


def test_harness_routes_temp_compile_and_pytest_through_gateways(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "pkg" / "calculator.py"
    tests = tmp_path / "tests" / "test_calculator.py"
    source.parent.mkdir(parents=True)
    tests.parent.mkdir(parents=True)
    source.write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    tests.write_text(
        "from pkg.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    file_write_sources: list[str] = []
    subprocess_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class FakeFileWriteGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown"):
            file_write_sources.append(source)
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding=encoding)

        def write_bytes(self, path, payload, *, source="unknown"):
            file_write_sources.append(source)
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def write_text_async(self, *args, **kwargs):
            return self.write_text(*args, **kwargs)
        async def write_bytes_async(self, *args, **kwargs):
            return self.write_bytes(*args, **kwargs)

    class FakeSubprocessGateway:
        async def run_async(self, argv, **kwargs):
            subprocess_calls.append((tuple(argv), kwargs))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        harness_mod,
        "get_file_write_gateway",
        lambda: FakeFileWriteGateway(),
    )
    monkeypatch.setattr(
        harness_mod,
        "get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )

    result = asyncio.run(SafeModificationHarness(tmp_path).run(["pkg/calculator.py"]))

    assert result.passed is True
    assert "core.self_modification.safe_modification_harness.compile_temp" in file_write_sources
    assert "core.self_modification.safe_modification_harness.rollback_backup" in file_write_sources
    assert subprocess_calls
    argv, kwargs = subprocess_calls[0]
    assert argv[:6] == (
        harness_mod.sys.executable,
        "-m",
        "pytest",
        "-p",
        "pytest_asyncio.plugin",
        "-x",
    )
    assert kwargs["source"] == "core.self_modification.safe_modification_harness.pytest"
    assert kwargs["cwd"] != tmp_path
    assert kwargs["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
