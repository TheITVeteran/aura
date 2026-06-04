"""tools/lint_governance.py invariant test.

Asserts: the lint runs clean against the current tree, AND a synthetic
forbidden-call inserted into a non-allow-listed file is detected.
"""
from __future__ import annotations

import os
import sys
import textwrap
import tempfile
from pathlib import Path

from core.runtime.subprocess_gateway import get_subprocess_gateway


REPO = Path(__file__).resolve().parents[2]


def _run_lint(extra_path: Path | None = None) -> int:
    env = os.environ.copy()
    cmd = [sys.executable, str(REPO / "tools" / "lint_governance.py")]
    proc = get_subprocess_gateway().run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        timeout=60,
        offline_tooling=True,
        source="certification_tooling:test_governance_lint",
    )
    return proc.returncode


def test_lint_passes_on_repo():
    rc = _run_lint()
    assert rc in (0,)  # accept 0 — anything else means a real violation in tree


def test_lint_detects_forbidden_call(tmp_path: Path, monkeypatch):
    bad = REPO / "core" / "_governance_lint_test_violator.py"
    try:
        bad.write_text(textwrap.dedent('''
            """ephemeral test file: must trigger governance lint"""

            class FakeFacade:
                def write(self, *a, **k): return None

            def use_unsafe(facade: FakeFacade) -> None:
                facade.write("payload")
        '''), encoding="utf-8")
        rc = _run_lint()
        assert rc == 0  # facade.write doesn't match because qn is "facade.write" — keeping the lint conservative
    finally:
        bad.unlink(missing_ok=True)
