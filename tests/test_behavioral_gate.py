"""Behavioral gate — self-modification patches must prove behavior.

Pins: deterministic impacted-test selection (stem + bounded import scan),
clone-and-run execution against a toy repo (good patch passes, logic-bug
patch that COMPILES fails), fail-closed on zero coverage, and the
code_repair wiring (no more pass-by-default).
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from core.self_modification.behavioral_gate import (
    run_behavioral_gate,
    select_impacted_tests,
)


def _toy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n"
    )
    (repo / "tests" / "test_calc.py").write_text(
        "from pkg.calc import add\n\n\n"
        "def test_add_commutes():\n"
        "    assert add(2, 3) == 5\n"
        "    assert add(3, 2) == 5\n"
    )
    (repo / "tests" / "test_via_import.py").write_text(
        "import pkg.calc\n\n\n"
        "def test_add_zero_identity():\n"
        "    assert pkg.calc.add(7, 0) == 7\n"
    )
    (repo / "tests" / "test_unrelated.py").write_text(
        "def test_unrelated():\n    assert True\n"
    )
    return repo


class TestImpactedSelection:
    def test_stem_and_import_matches_ranked_first(self, tmp_path):
        repo = _toy_repo(tmp_path)
        impacted = select_impacted_tests("pkg/calc.py", repo)
        names = [p.name for p in impacted]
        assert names[0] == "test_calc.py"          # stem match strongest
        assert "test_via_import.py" in names       # direct import found
        assert "test_unrelated.py" not in names    # unrelated excluded

    def test_no_tests_dir_returns_empty(self, tmp_path):
        assert select_impacted_tests("pkg/calc.py", tmp_path) == []


class TestGateExecution:
    def test_correct_patch_passes(self, tmp_path):
        repo = _toy_repo(tmp_path)
        verdict = asyncio.run(run_behavioral_gate(
            "pkg/calc.py",
            "def add(a, b):\n    return b + a\n",  # equivalent behavior
            repo_root=repo,
        ))
        assert verdict.covered is True
        assert verdict.passed is True, verdict.detail
        assert "test_calc.py" in " ".join(verdict.tests)

    def test_compiling_logic_bug_is_caught(self, tmp_path):
        """The review's exact scenario: parses perfectly, poisons logic."""
        repo = _toy_repo(tmp_path)
        verdict = asyncio.run(run_behavioral_gate(
            "pkg/calc.py",
            "def add(a, b):\n    return a - b\n",  # compiles fine; wrong
            repo_root=repo,
        ))
        assert verdict.covered is True
        assert verdict.passed is False

    def test_zero_coverage_fails_closed(self, tmp_path):
        repo = _toy_repo(tmp_path)
        verdict = asyncio.run(run_behavioral_gate(
            "pkg/uncovered_module.py",
            "VALUE = 1\n",
            repo_root=repo,
        ))
        assert verdict.covered is False
        assert verdict.passed is False
        assert "not auto-promotable" in verdict.detail

    def test_live_tree_is_never_touched(self, tmp_path):
        repo = _toy_repo(tmp_path)
        original = (repo / "pkg" / "calc.py").read_text()
        asyncio.run(run_behavioral_gate(
            "pkg/calc.py",
            "def add(a, b):\n    return a - b\n",
            repo_root=repo,
        ))
        assert (repo / "pkg" / "calc.py").read_text() == original


class TestCodeRepairWiring:
    def test_pass_by_default_hole_is_gone(self):
        from core.self_modification import code_repair

        src = inspect.getsource(code_repair)
        assert "run_behavioral_gate" in src
        assert 'results["unit_tests"] = True # "N/A"' not in src
        # Core patches without coverage fail closed, visibly.
        assert "not auto-promotable" in inspect.getsource(
            __import__("core.self_modification.behavioral_gate", fromlist=["x"])
        )
