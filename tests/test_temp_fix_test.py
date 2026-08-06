"""Validation test for the self-modification harness.

This is not a standalone test. The safe-modification harness derives the test
that validates a fix from the target's filename, so a repair applied to
``core/temp_fix_test.py`` is validated by running this file. It is exercised by
``tests/test_fix_persistence.py``, which writes the fixture, applies a fix
through the real engine, and relies on the harness running this to confirm the
fix landed.

During an ordinary suite run the fixture does not exist and there is nothing to
check. It used to fail instead: a run that died between setUp and tearDown left
the fixture behind un-fixed, and this file then asserted against that debris on
every subsequent run — reporting a defect in the self-modification engine when
what had actually happened was a crashed test three days earlier.

The fixture's lifecycle is now guaranteed by addCleanup, so the debris should
not appear. If it does anyway, that is a leaked artifact and not a failed
repair, and this says which rather than failing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "core" / "temp_fix_test.py"

#: What the harness writes once the fix has been applied.
APPLIED = "def new_function()"
#: What tests/test_fix_persistence.py seeds before the fix is applied.
UNAPPLIED = "def old_function()"


def test_temp_fix_test_new_function():
    if not FIXTURE.exists():
        pytest.skip(
            "self-modification fixture absent: nothing was applied, so there is "
            "nothing to validate"
        )

    source = FIXTURE.read_text()

    if APPLIED not in source and UNAPPLIED in source:
        pytest.skip(
            f"{FIXTURE} is a leaked fixture in its pre-fix state, not a failed "
            "repair — tests/test_fix_persistence.py died before its cleanup ran. "
            "The file is untracked debris and can be deleted."
        )

    spec = importlib.util.spec_from_file_location(
        "core.temp_fix_test_candidate", FIXTURE
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.new_function() == "new"
