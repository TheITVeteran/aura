"""Every name aura_main imports at boot must actually exist.

Boot wires subsystems inside try/except blocks that record a degradation and
carry on. That is the right shape for a subsystem that fails — and it is also
how a subsystem that was NEVER WIRED hides in plain sight.

Measured live 2026-08-03:

    Verifier Foundry boot failed: cannot import name 'boot_verifier_foundry'
    from 'core.brain.verifiers.foundry'

The module defined get_verifier_foundry; nothing ever defined the boot_ entry
point aura_main imported. Every launch logged the failure, degraded honestly,
and ran without the foundry. An ImportError for a name in your own repository
is not a runtime condition to degrade over — it is a wiring defect, and it
cannot be caught by the except clause that swallows it.

This resolves names statically: it parses the target module rather than
importing it, so it stays fast and cannot be fooled by import side effects.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOT_MODULE = PROJECT_ROOT / "aura_main.py"

#: Only first-party packages. Third-party names move with the environment and
#: are legitimately guarded by try/except ImportError.
FIRST_PARTY_ROOTS = ("core", "interface", "skills", "infrastructure")


def _module_source(module_name: str) -> str | None:
    """Locate a module's source without importing it."""

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    try:
        return Path(spec.origin).read_text(encoding="utf-8")
    except OSError:
        return None


def _module_level_names(source: str) -> set[str]:
    """Names a module binds at module level."""

    names: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.If | ast.Try):
            # Conditional definitions still bind at module level.
            for sub in ast.walk(node):
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
                elif isinstance(sub, ast.Import | ast.ImportFrom):
                    for alias in sub.names:
                        names.add(alias.asname or alias.name.split(".")[0])
    return names


def _first_party_imports(path: Path) -> list[tuple[int, str, str]]:
    """(line, module, name) for every first-party from-import in a file."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        module = node.module or ""
        if not module.startswith(FIRST_PARTY_ROOTS):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            found.append((node.lineno, module, alias.name))
    return found


BOOT_IMPORTS = _first_party_imports(BOOT_MODULE)


def test_the_scan_found_imports_to_check():
    """A scan that silently matches nothing would pass vacuously."""
    assert len(BOOT_IMPORTS) > 50, f"only {len(BOOT_IMPORTS)} first-party imports found"


@pytest.mark.parametrize(
    ("line", "module", "name"),
    BOOT_IMPORTS,
    ids=[f"{module}:{name}" for _, module, name in BOOT_IMPORTS],
)
def test_every_boot_import_resolves(line, module, name):
    source = _module_source(module)
    assert source is not None, f"aura_main.py:{line} imports from missing module {module!r}"

    defined = _module_level_names(source)
    # A submodule import (`from core.pkg import mod`) resolves as a module.
    if name not in defined and importlib.util.find_spec(f"{module}.{name}") is not None:
        return

    assert name in defined, (
        f"aura_main.py:{line} imports {name!r} from {module!r}, which does not "
        f"define it. Boot swallows this as a degradation, so the subsystem "
        f"never runs and the failure looks like a runtime condition."
    )
