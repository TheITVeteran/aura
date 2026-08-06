"""tests/test_spawned_children_are_reapable.py — every long-lived child Aura
spawns must be reachable by something that can kill it.

Measured on 2026-08-06: ``core/reaper.py`` implements an identity-verified
manifest — it checks create_time and cwd before signalling, so it cannot kill a
stranger that inherited a recycled PID — and exactly two call sites in the whole
tree registered anything with it. The two children that matter most were not
among them. The external memory sentinel and the external liveness sentinel are
both spawned with ``start_new_session=True``, deliberately, so that a watchdog
outlives the process it watches; the consequence nobody had followed through was
that no process group, no atexit hook and no manifest entry covered them. Every
boot that did not exit cleanly left two orphans holding a log fd.

The reaper being present, healthy and correct told us nothing about whether it
covered anything — which is the same shape as a module reporting "online" while
its callback was never wired. So this file asserts coverage, not existence.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modules that spawn children intended to outlive the call that made them.
_SPAWNING_MODULES = ("aura_main.py",)

# Spawns whose child is bounded and awaited inside the same call — a reaper
# entry for a process that is already gone by the next line is noise. Keyed by
# module path to the set of function names allowed to skip registration.
_BOUNDED_SPAWN_FUNCTIONS: dict[str, frozenset[str]] = {
    "aura_main.py": frozenset(),
}


def _enclosing_function(tree: ast.AST, target: ast.AST) -> str:
    """Name of the innermost function containing ``target``."""
    best = ""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None)
        if end is None:
            continue
        if node.lineno <= target.lineno <= end:
            best = node.name
    return best


def _popen_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        if name == "Popen":
            calls.append(node)
    return calls


@pytest.mark.parametrize("relative_path", _SPAWNING_MODULES)
def test_every_detached_spawn_is_registered_with_the_reaper(relative_path):
    """A child spawned into its own session must be registered for reaping.

    ``start_new_session=True`` is the marker that matters: it is exactly the
    flag that removes the child from the parent's process group, so it is
    exactly the case where nothing implicit will clean up after it.
    """
    source_path = PROJECT_ROOT / relative_path
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    allowed = _BOUNDED_SPAWN_FUNCTIONS.get(relative_path, frozenset())

    unregistered: list[tuple[int, str]] = []
    for call in _popen_calls(tree):
        detached = any(
            kw.arg == "start_new_session"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in call.keywords
        )
        if not detached:
            continue
        function_name = _enclosing_function(tree, call)
        if function_name in allowed:
            continue
        # Registration must happen in the same function that spawned, so the
        # child cannot escape between the spawn and a caller that forgets.
        enclosing = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        registers = False
        if enclosing is not None:
            for node in ast.walk(enclosing):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                    if name in {"_register_spawned_child", "register_reaper_pid"}:
                        registers = True
                        break
        if not registers:
            unregistered.append((call.lineno, function_name or "<module>"))

    assert not unregistered, (
        f"{relative_path} spawns detached children without registering them for "
        f"reaping at {unregistered}. Call _register_spawned_child(proc, role=...) "
        "in the spawning function — start_new_session=True means nothing else will."
    )


def test_reaper_registration_records_identity_not_just_a_pid():
    """A bare PID is not enough evidence to signal, and must not be stored as if it were.

    PIDs are recycled. The manifest carries create_time and cwd so cleanup can
    prove the process it is about to kill is the one it registered; a record
    without that metadata is refused rather than acted on.
    """
    import os

    from core.reaper import ReaperManifest, _pid_cleanup_authorized

    # This process, not PID 1: launchd is not introspectable, so asking about
    # it would test the observer's error path rather than the record's shape.
    # _pid_record only builds the record — nothing is written to the manifest.
    record = ReaperManifest._pid_record(os.getpid())
    assert record.get("create_time"), (
        "manifest PID records carry no create_time, so cleanup cannot tell the "
        "registered process from a stranger that inherited its PID"
    )

    authorized, _pid, reason = _pid_cleanup_authorized({"pid": 1})
    assert authorized is False
    assert reason == "legacy_pid_without_identity"


def test_registering_a_child_is_survivable_when_the_manifest_is_unavailable(monkeypatch):
    """A failed registration costs a leaked child; a raised one costs the runtime."""
    import aura_main

    def _explode(_pid):
        raise OSError("manifest unavailable")

    monkeypatch.setattr("core.reaper.register_reaper_pid", _explode)

    class _Proc:
        pid = 4242

        def poll(self):
            return 0  # already exited; nothing for the reaper to signal

    aura_main._register_spawned_child(_Proc(), role="test_child")

    # Registration must have happened despite the manifest failing, so the
    # graceful-shutdown reap still knows about the child the manifest lost.
    assert 4242 in aura_main._SPAWNED_CHILDREN
    aura_main.reap_spawned_children()
    assert 4242 not in aura_main._SPAWNED_CHILDREN


def test_graceful_shutdown_reaps_a_live_detached_child():
    """The reap path must actually terminate a running child, not just forget it.

    The reaper manifest covers unexpected death. Graceful shutdown had no
    equivalent, so a clean exit left both external sentinels alive and watching
    a PID that no longer existed.
    """
    import subprocess
    import sys

    import aura_main

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        start_new_session=True,
    )
    try:
        aura_main._register_spawned_child(proc, role="test_sleeper")
        assert proc.poll() is None, "child exited before the test could reap it"

        reaped = aura_main.reap_spawned_children(timeout=5.0)

        assert "test_sleeper" in reaped
        assert proc.poll() is not None, "reap_spawned_children left the child running"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
