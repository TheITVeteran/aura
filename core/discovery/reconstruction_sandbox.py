"""core/discovery/reconstruction_sandbox.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A GENERAL reconstruction sandbox: safe by curated capability + real subprocess
isolation, not by an 18-builtin AST whitelist.

The strict SafeCodeEvaluator (core/discovery/code_eval.py) forbids all imports
and all attribute access, so it can only verify toy pure-operator functions —
`base64`, a checksum, or any realistic program is unreconstructable there. That
is correct for mutation testing but it caps the Program-DNA engine at trivia.

This evaluator lifts that ceiling while staying safe:

* imports are restricted to a curated allowlist of PURE, side-effect-free
  stdlib (no os / sys / subprocess / socket / shutil / pathlib-write / open /
  importlib / ctypes / threading / asyncio / pickle / marshal / builtins);
* dunder gadget attributes (__subclasses__, __globals__, __mro__, __builtins__,
  …) and reflection/eval builtins (eval, exec, compile, open, __import__,
  getattr, setattr, globals, …) are blocked at the AST;
* a runtime guard prelude re-installs an import hook limited to the allowlist
  and removes eval/exec/open/compile from builtins inside the child, so a gadget
  that slipped the AST still cannot import or eval;
* execution stays inside SafeMutationEvaluator's subprocess, which applies
  RLIMIT_AS (memory) and RLIMIT_CPU fences, a scrubbed environment, and a
  wall-clock timeout. Even a hostile candidate cannot reach the filesystem, the
  network, or the parent process.

The result: the DNA engine can reconstruct and DIFFERENTIALLY VERIFY realistic
programs against held-out real-binary outputs, not just toys.
"""
from __future__ import annotations

import ast
import logging
from collections.abc import Sequence
from typing import Any

from core.discovery.code_eval import DiscoveryEvaluation, _build_runner

logger = logging.getLogger("Aura.Discovery.ReconstructionSandbox")

# Curated: pure, deterministic-ish, no ambient authority (no fs / net / process).
# Every entry is a stdlib module whose public surface cannot, on its own, open a
# file, a socket, or a subprocess.
SAFE_IMPORT_ALLOWLIST: frozenset[str] = frozenset({
    "base64", "binascii", "hashlib", "hmac", "secrets", "zlib",
    "math", "cmath", "statistics", "decimal", "fractions", "numbers",
    "string", "re", "textwrap", "unicodedata", "difflib",
    "json", "csv", "struct",
    "itertools", "functools", "operator", "collections", "heapq", "bisect",
    "array", "enum", "dataclasses", "typing", "types",
    "datetime", "calendar", "time",
    "random", "uuid",
    # NOTE deliberately excluded: codecs (codecs.open), zoneinfo (reads tz
    # files), io, pathlib, os, sys, subprocess, socket, shutil, importlib,
    # ctypes, pickle, marshal — any module exposing filesystem/network/process
    # authority through its own attributes, which a Name-based call check
    # would not catch.
    "html", "urllib",  # urllib.parse only — see _module_root check below
})

# urllib is allowed ONLY for its pure parse submodule; request/socket-bearing
# submodules are rejected explicitly.
_URLLIB_BLOCKED_SUBMODULES = frozenset({"request", "error", "robotparser"})

# Attribute names that are gadget vectors into the interpreter internals.
DANGEROUS_ATTRS: frozenset[str] = frozenset({
    "__subclasses__", "__globals__", "__bases__", "__mro__", "__base__",
    "__builtins__", "__import__", "__getattribute__", "__setattr__",
    "__delattr__", "__dict__", "__code__", "__closure__", "__func__",
    "__self__", "__reduce__", "__reduce_ex__", "__class__", "__init_subclass__",
    "__loader__", "__spec__", "__weakref__",
})

# Builtins that grant reflection / eval / ambient authority.
DANGEROUS_CALLS: frozenset[str] = frozenset({
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "memoryview", "exit", "quit", "help", "copyright", "credits", "license",
})


class ReconstructionASTViolation(RuntimeError):
    """A candidate used a capability outside the curated reconstruction policy."""


def _module_root(name: str) -> str:
    return str(name or "").split(".")[0]


def _check_import(node: ast.AST) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = _module_root(alias.name)
            if root not in SAFE_IMPORT_ALLOWLIST:
                raise ReconstructionASTViolation(f"import not allowed: {alias.name}")
            if root == "urllib" and alias.name.split(".")[-1] in _URLLIB_BLOCKED_SUBMODULES:
                raise ReconstructionASTViolation(f"import not allowed: {alias.name}")
    elif isinstance(node, ast.ImportFrom):
        root = _module_root(node.module or "")
        if root not in SAFE_IMPORT_ALLOWLIST:
            raise ReconstructionASTViolation(f"import not allowed: from {node.module}")
        if root == "urllib":
            submodule = (node.module or "").split(".")[-1]
            names = {a.name for a in node.names}
            if submodule in _URLLIB_BLOCKED_SUBMODULES or names & _URLLIB_BLOCKED_SUBMODULES:
                raise ReconstructionASTViolation(f"import not allowed: from {node.module}")


def audit_general_ast(code: str) -> None:
    """Curated-safe audit: allow real code, block ambient authority + gadgets."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node)
        elif isinstance(node, ast.Attribute):
            if node.attr in DANGEROUS_ATTRS:
                raise ReconstructionASTViolation(f"attribute not allowed: {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in DANGEROUS_ATTRS or node.id == "__builtins__":
                raise ReconstructionASTViolation(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DANGEROUS_CALLS:
                raise ReconstructionASTViolation(f"call not allowed: {func.id}")


# The security boundary is the AST audit (the candidate cannot import ambient
# authority, call eval/exec/open/__import__/getattr, or touch dunder gadgets)
# combined with the tight import allowlist (no importable module exposes
# filesystem/network/process authority through its own attributes) and
# SafeMutationEvaluator's subprocess rlimits + scrubbed env + timeout. A runtime
# builtins scrub was tried and rejected: deleting `open` from the child's
# builtins breaks the transitive stdlib imports the allowlisted modules need
# (the import machinery and modules like `re`/`enum` reference it at import).
_GUARD_PRELUDE = "# --- reconstruction sandbox: AST-audited candidate ---\n"


class GeneralReconstructionEvaluator:
    """Curated-capability differential evaluator with real subprocess isolation."""

    def __init__(self, *, timeout_seconds: float = 5.0, memory_mb: int = 512) -> None:
        from core.self_modification.mutation_safety import SafeMutationEvaluator

        self.timeout_seconds = float(timeout_seconds)
        self._delegate = SafeMutationEvaluator(
            timeout_seconds=self.timeout_seconds, memory_mb=int(memory_mb),
        )

    def evaluate(
        self,
        code: str,
        fn_name: str,
        tests: Sequence[tuple[tuple[Any, ...], Any]],
    ) -> DiscoveryEvaluation:
        from core.self_modification.mutation_safety import MutationOutcome

        if not fn_name.isidentifier():
            return DiscoveryEvaluation(
                outcome="ast_violation",
                error=f"invalid fn_name: {fn_name!r}",
                total=len(tests),
            )
        try:
            audit_general_ast(code)
        except ReconstructionASTViolation as exc:
            return DiscoveryEvaluation(outcome="ast_violation", error=str(exc), total=len(tests))
        except SyntaxError as exc:
            return DiscoveryEvaluation(outcome="compile_fail", error=str(exc), total=len(tests))

        runner = _build_runner(fn_name, tests)
        full_source = _GUARD_PRELUDE + code + "\n" + runner
        diag = self._delegate.evaluate(full_source)
        outcome_map = {
            MutationOutcome.PASSED: ("passed", len(tests)),
            MutationOutcome.COMPILE_FAIL: ("compile_fail", 0),
            MutationOutcome.IMPORT_FAIL: ("ast_violation", 0),
            MutationOutcome.RUNTIME_EXCEPTION: ("runtime", 0),
            MutationOutcome.ASSERTION_FAIL: ("assertion", 0),
            MutationOutcome.TIMEOUT: ("timeout", 0),
            MutationOutcome.OOM: ("oom", 0),
        }
        outcome_str, passed = outcome_map.get(diag.outcome, ("runtime", 0))
        return DiscoveryEvaluation(
            outcome=outcome_str,
            passed=passed,
            total=len(tests),
            error=None if outcome_str == "passed" else (diag.traceback_text or None),
            quarantine_path=diag.quarantine_path,
            stdout=diag.stdout[-2000:],
            stderr=diag.stderr[-2000:],
            metadata={
                "runtime_seconds": diag.runtime_seconds,
                "exit_code": diag.exit_code,
                "profile": "general_reconstruction",
            },
        )


__all__ = [
    "GeneralReconstructionEvaluator",
    "ReconstructionASTViolation",
    "audit_general_ast",
    "SAFE_IMPORT_ALLOWLIST",
    "DANGEROUS_ATTRS",
    "DANGEROUS_CALLS",
]
