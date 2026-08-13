"""Deterministically bind the local Python import closure of an entry point."""

from __future__ import annotations

import ast
import hashlib
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Final

MAX_SOURCE_FILES: Final = 10_000
MAX_SOURCE_BYTES: Final = 256 * 1024 * 1024
_CACHE_LOCK = threading.RLock()
_CLOSURE_CACHE: dict[
    tuple[str, tuple[str, ...]],
    tuple[tuple[str, ...], dict[str, tuple[int, int, int, int, int]]],
] = {}


class PythonSourceClosureError(RuntimeError):
    """A local source closure could not be measured without ambiguity."""


def _relative_source(root: Path, candidate: Path) -> str | None:
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or resolved.suffix != ".py":
        return None
    return relative.as_posix()


def _resolve_module(root: Path, module: str) -> str | None:
    if not module or any(not part.isidentifier() for part in module.split(".")):
        return None
    base = root.joinpath(*module.split("."))
    return _relative_source(root, base.with_suffix(".py")) or _relative_source(
        root, base / "__init__.py"
    )


def _package_for(relative: str) -> tuple[str, ...]:
    path = Path(relative)
    if path.name == "__init__.py":
        return path.parts[:-1]
    return path.with_suffix("").parts[:-1]


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_payload(root: Path, relative: str) -> tuple[bytes, tuple[int, int, int, int, int]]:
    lexical = root / relative
    try:
        resolved = lexical.resolve(strict=True)
        if resolved.relative_to(root).as_posix() != relative:
            raise ValueError("source path identity changed")
        before = _identity(resolved)
        payload = resolved.read_bytes()
        after = _identity(resolved)
    except (OSError, ValueError) as exc:
        raise PythonSourceClosureError(
            f"Python source changed while reading: {relative}"
        ) from exc
    if before != after or len(payload) != before[2]:
        raise PythonSourceClosureError(f"Python source changed while reading: {relative}")
    return payload, after


def _cached_hashes(
    root: Path,
    files: tuple[str, ...],
    identities: dict[str, tuple[int, int, int, int, int]],
) -> dict[str, str] | None:
    try:
        if any(_identity(root / relative) != identities[relative] for relative in files):
            return None
    except (KeyError, OSError):
        return None
    measured: dict[str, str] = {}
    for relative in files:
        payload, identity = _stable_payload(root, relative)
        if identity != identities[relative]:
            return None
        measured[relative] = hashlib.sha256(payload).hexdigest()
    return measured


def _imported_modules(tree: ast.AST, *, package: tuple[str, ...]) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            trim = node.level - 1
            if trim > len(package):
                continue
            prefix = package[: len(package) - trim]
            base_parts = (*prefix, *(node.module or "").split("."))
        else:
            base_parts = tuple((node.module or "").split("."))
        base = ".".join(part for part in base_parts if part)
        if base:
            modules.add(base)
            modules.update(f"{base}.{alias.name}" for alias in node.names)
    return modules


def local_python_source_sha256s(
    source_root: Path,
    entry_files: Iterable[str],
) -> dict[str, str]:
    """Hash every statically imported local module reachable from ``entry_files``.

    Parsing and hashing use the same immutable byte read for each file. Imports
    outside ``source_root`` are deliberately excluded; standard-library and
    third-party package identities belong to the separate runtime receipt.
    """

    root = source_root.expanduser().resolve(strict=True)
    pending: list[str] = []
    for entry in entry_files:
        normalized = str(entry).strip().replace("\\", "/").lstrip("/")
        relative = _relative_source(root, root / normalized)
        if relative is None:
            raise PythonSourceClosureError(f"Python source entry is unavailable: {entry}")
        pending.append(relative)

    cache_key = (str(root), tuple(sorted(set(pending))))
    with _CACHE_LOCK:
        cached = _CLOSURE_CACHE.get(cache_key)
    if cached is not None:
        cached_hashes = _cached_hashes(root, *cached)
        if cached_hashes is not None:
            return cached_hashes

    measured: dict[str, str] = {}
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    total_bytes = 0
    while pending:
        relative = pending.pop()
        if relative in measured:
            continue
        if len(measured) >= MAX_SOURCE_FILES:
            raise PythonSourceClosureError("Python source closure exceeds file limit")
        try:
            payload, identity = _stable_payload(root, relative)
            tree = ast.parse(payload, filename=relative)
        except (PythonSourceClosureError, SyntaxError, ValueError) as exc:
            raise PythonSourceClosureError(
                f"Python source closure cannot parse {relative}"
            ) from exc
        total_bytes += len(payload)
        if total_bytes > MAX_SOURCE_BYTES:
            raise PythonSourceClosureError("Python source closure exceeds byte limit")
        measured[relative] = hashlib.sha256(payload).hexdigest()
        identities[relative] = identity
        for module in _imported_modules(tree, package=_package_for(relative)):
            dependency = _resolve_module(root, module)
            if dependency is not None and dependency not in measured:
                pending.append(dependency)
    files = tuple(sorted(measured))
    with _CACHE_LOCK:
        _CLOSURE_CACHE[cache_key] = (files, identities)
    return {relative: measured[relative] for relative in files}


__all__ = [
    "PythonSourceClosureError",
    "local_python_source_sha256s",
]
