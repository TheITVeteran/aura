"""A log call in a module with no logger is a NameError with a delay fuse.

Two landed on the desktop-task research and focus-hold paths. Neither fires
until the exact condition it describes occurs — "fewer sources than asked for"
and "hold_focus was refused" — which is to say, both fire only when something
has already gone wrong. ruff's F821 caught them; this keeps them caught.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ROOTS = ("core", "interface")


def _modules_that_log_without_a_logger() -> list[str]:
    offenders: list[str] = []
    for root in _ROOTS:
        for path in pathlib.Path(root).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError:
                continue
            names: set[str] = set()
            uses_logger = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Store):
                        names.add(node.id)
                    elif node.id in {"logger", "log", "_logger", "LOGGER"}:
                        uses_logger = True
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        names.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.arg):
                    names.add(node.arg)
                elif isinstance(node, ast.Attribute):
                    pass
            if uses_logger and not (names & {"logger", "log", "_logger", "LOGGER"}):
                offenders.append(str(path))
    return offenders


def test_every_module_that_logs_owns_a_logger() -> None:
    offenders = _modules_that_log_without_a_logger()
    assert not offenders, (
        "These modules call a logger they never bound — the call raises "
        "NameError the first time its condition is true:\n  "
        + "\n  ".join(sorted(offenders))
    )
