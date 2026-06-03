"""Tests for the authoritative self-modification safety harness."""
from __future__ import annotations

import asyncio
from pathlib import Path

from core.self_modification.safe_modification_harness import SafeModificationHarness


def test_harness_rejects_patch_without_related_pytest(tmp_path: Path) -> None:
    source = tmp_path / "core" / "uncovered_runtime_patch.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")

    result = asyncio.run(
        SafeModificationHarness(tmp_path).run(["core/uncovered_runtime_patch.py"])
    )

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
        "from pkg.calculator import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    result = asyncio.run(SafeModificationHarness(tmp_path).run(["pkg/calculator.py"]))

    assert result.passed is True
    assert result.checks["pytest"] is True


def test_harness_treats_changed_test_file_as_coverage(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_self_check.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_self_check():\n    assert True\n", encoding="utf-8")

    result = asyncio.run(SafeModificationHarness(tmp_path).run(["tests/test_self_check.py"]))

    assert result.passed is True
    assert result.checks["pytest"] is True
