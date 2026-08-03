"""The deprecated self-modification engine is gone, and must stay gone.

``core/self_modification_engine.py`` was a duplicate of
``core/self_modification/self_modification_engine.py`` — one character
apart in import path, 1493 lines and a 37-module package apart in
substance. It had:

* zero importers anywhere in the repo;
* its own docstring declaring it superseded;
* direct application already disabled behind
  ``_LEGACY_DIRECT_APPLICATION_ERROR``;
* a ``verify_changes`` whose sandbox import (``..security.code_sandbox``,
  which resolves ABOVE the ``core`` package) could never succeed, so
  verification always failed closed and nothing it certified was real;
* a ``_rollback`` that copied every file under any supplied directory into
  the code base, with no manifest, no expected hashes, no binding to a
  proposal and no symlink rejection.

This file used to hold four tests proving that engine could not mutate
source. Deleting the module makes that guarantee absolute rather than
maintained: there is no engine left to keep disabled.

What remains is the guarantee itself, in its strongest form — a
plausible-looking self-modification API must not reappear at the old
import path, where a future wiring could reach it by typo.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

LEGACY_MODULE = "core.self_modification_engine"
LEGACY_PATH = ROOT / "core" / "self_modification_engine.py"
LIVE_MODULE = "core.self_modification.self_modification_engine"


def test_the_legacy_module_file_is_gone():
    assert not LEGACY_PATH.exists(), (
        "core/self_modification_engine.py is back. It is a near-namesake of the "
        "live engine with a rollback that wrote anything handed to it; if this "
        "is intentional, it needs the review the original never had."
    )


def test_the_legacy_import_path_does_not_resolve():
    """Import path, not just the file: a package or shim would also resolve."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(LEGACY_MODULE)


def test_nothing_in_the_tree_imports_the_legacy_path():
    offenders: list[str] = []
    for directory in ("core", "interface", "tools", "tests"):
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts or path == Path(__file__):
                continue
            text = path.read_text("utf-8", errors="ignore")
            for needle in (
                "from core.self_modification_engine import",
                "import core.self_modification_engine",
            ):
                if needle in text:
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"legacy self-modification engine imported by: {offenders}"


def test_the_live_engine_is_still_there():
    """The control: deleting the duplicate must not have taken the real one."""
    module = importlib.import_module(LIVE_MODULE)
    assert hasattr(module, "AutonomousSelfModificationEngine")


def test_the_service_name_resolves_to_the_live_engine():
    """`self_modification_engine` is a SERVICE name; it must point at the package.

    The two shared this name, which is how a typo could have become a
    wiring. Pinned so the registration cannot drift back.
    """
    provider = (ROOT / "core" / "providers" / "ops_provider.py").read_text("utf-8")
    assert "from core.self_modification.self_modification_engine import" in provider
    assert "from core.self_modification_engine import" not in provider
