"""Is this process running the code that is on disk?

`SourceBodyAwareness` watches the GIT DIRTY STATE, so committing an edit leaves
the tree clean while the running process stays exactly as stale. Nothing
measured the other quantity, so "are you running my latest fix?" had no
instrument behind it — and three separate defects on 2026-08-10 each cost a long
detour for want of it.
"""

from __future__ import annotations

import pathlib

from core.runtime.loaded_source_drift import scan_drift


def test_a_freshly_imported_tree_reads_clean():
    import core.conversation.screen_reading_claim  # noqa: F401

    report = scan_drift()
    assert report.checked > 0
    # Test modules are legitimately unmeasurable: pytest's assertion rewriting
    # compiles them in memory, so there is no bytecode cache to compare a
    # source file against. They are reported as `unknown` rather than counted
    # as clean, which is the contract — an unmeasurable module is not a fresh
    # one. Everything the runtime actually imports must be measurable.
    assert all(
        name.startswith("tests.") for name in report.unknown
    ), f"unmeasurable runtime modules: {report.unknown}"
    assert not report.is_stale, report.narrative()


def test_a_real_edit_is_detected_and_clears():
    import core.conversation.screen_reading_claim as module

    path = pathlib.Path(module.__file__)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n_DRIFT_PROBE = 1\n")
        report = scan_drift()
        assert "core.conversation.screen_reading_claim" in report.stale_modules
    finally:
        path.write_bytes(original)
    assert not scan_drift().is_stale


def test_cosmetic_rewrites_do_not_cry_stale():
    """A touch, a byte-identical restore, and a comment must all read clean.

    An instrument that fires on `git checkout` is one nobody trusts by the
    third time, so a timestamp mismatch is confirmed against the compiled code
    objects rather than believed.
    """
    import core.conversation.screen_reading_claim as module

    path = pathlib.Path(module.__file__)
    original = path.read_bytes()
    try:
        for rewrite in (original, original + b"\n\n", original + b"\n# a comment\n"):
            path.write_bytes(rewrite)
            assert not scan_drift().is_stale, rewrite[-20:]
    finally:
        path.write_bytes(original)


def test_dependencies_are_not_project_source():
    """The virtualenv lives INSIDE the repo root here.

    Scanning it swept in every installed package — 792 modules — and this
    instrument answers "am I running the code I just edited". Nobody edits
    their dependencies in place.
    """
    import torch  # noqa: F401  (ships modules under the in-repo .venv)

    report = scan_drift()
    assert not any("site-packages" in drift.path for drift in report.stale)
    assert all(not module.startswith("torch.") for module in report.stale_modules)


def test_a_relative_dunder_file_is_not_mistaken_for_project_source():
    """`torch.classes.__file__` is the bare string "_classes.py".

    `.resolve()` joins that to the current working directory — this repository —
    so two synthetic modules with no source file anywhere were reported as
    project files missing from disk.
    """
    import sys

    import torch  # noqa: F401

    synthetic = [
        name
        for name in ("torch.classes", "torch.ops")
        if name in sys.modules
        and isinstance(getattr(sys.modules[name], "__file__", None), str)
        and not pathlib.Path(sys.modules[name].__file__).is_absolute()
    ]
    if not synthetic:
        return
    report = scan_drift()
    for name in synthetic:
        assert name not in report.stale_modules
        assert name not in report.unknown
