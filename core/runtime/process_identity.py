"""Exact process identity helpers for lifecycle and cleanup tooling."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_PYTHON_EXECUTABLE = re.compile(r"^(?:python|pypy)(?:\d+(?:\.\d+)*)?$", re.IGNORECASE)
_PYTHON_OPTIONS_WITH_VALUE = frozenset({"-W", "-X", "--check-hash-based-pycs"})


def python_script_argument(cmdline: Sequence[Any]) -> str | None:
    """Return the script Python executes, excluding text and module invocations."""

    arguments = [str(item) for item in cmdline if str(item)]
    if not arguments:
        return None
    executable = Path(arguments[0]).name
    if _PYTHON_EXECUTABLE.fullmatch(executable) is None:
        return None

    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            index += 1
            return arguments[index] if index < len(arguments) else None
        if argument in {"-c", "-m"}:
            return None
        if argument in _PYTHON_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if argument.startswith("-W") or argument.startswith("-X"):
            index += 1
            continue
        if argument.startswith("-"):
            index += 1
            continue
        return argument
    return None


def command_invokes_python_script(
    cmdline: Sequence[Any],
    *,
    expected_script: str | Path,
    cwd: str | Path = "",
) -> bool:
    """Match one exact script path, never a substring elsewhere in argv."""

    script_argument = python_script_argument(cmdline)
    if not script_argument:
        return False
    expected = Path(expected_script).expanduser().resolve(strict=False)
    observed = Path(script_argument).expanduser()
    if not observed.is_absolute():
        if not str(cwd or "").strip():
            return False
        observed = Path(cwd).expanduser() / observed
    return observed.resolve(strict=False) == expected


def process_invokes_python_script(
    process: Any,
    *,
    expected_script: str | Path,
) -> bool:
    return command_invokes_python_script(
        getattr(process, "cmdline", ()) or (),
        expected_script=expected_script,
        cwd=getattr(process, "cwd", "") or "",
    )


def select_script_process_tree(
    processes: Iterable[Any],
    *,
    expected_scripts: Iterable[str | Path],
    protected_pids: Iterable[int] = (),
) -> tuple[Any, ...]:
    """Select exact script roots and their observed descendants.

    Returned observations are ordered root-first for graceful termination.
    Callers must still revalidate PID creation time immediately before sending
    a signal so PID reuse cannot target an unrelated process.
    """

    observations = tuple(processes)
    protected = {int(pid) for pid in protected_pids if int(pid) > 0}
    scripts = tuple(Path(path).expanduser().resolve(strict=False) for path in expected_scripts)
    roots = {
        int(getattr(process, "pid", 0) or 0)
        for process in observations
        if int(getattr(process, "pid", 0) or 0) not in protected
        and any(
            process_invokes_python_script(process, expected_script=script)
            for script in scripts
        )
    }
    if not roots:
        return ()

    selected = []
    for process in observations:
        pid = int(getattr(process, "pid", 0) or 0)
        if pid <= 0 or pid in protected:
            continue
        ancestors = {
            int(parent)
            for parent in (getattr(process, "ancestor_pids", ()) or ())
            if int(parent) > 0
        }
        if pid in roots or ancestors & roots:
            selected.append(process)
    selected.sort(
        key=lambda process: (
            int(getattr(process, "pid", 0) or 0) not in roots,
            len(getattr(process, "ancestor_pids", ()) or ()),
            int(getattr(process, "pid", 0) or 0),
        )
    )
    return tuple(selected)


__all__ = [
    "command_invokes_python_script",
    "process_invokes_python_script",
    "python_script_argument",
    "select_script_process_tree",
]
