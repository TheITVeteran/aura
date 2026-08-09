"""The authority's integration list must name real call sites, and only those.

``substrate_authority.py`` described itself as "the MANDATORY gate through
which ALL significant actions must pass", and listed five integration
points. Two of them — memory writes and tool execution — had no call site
anywhere: neither ``core/memory`` nor ``core/agency`` referenced the module.

Prose cannot be verified, which is exactly why it drifts. This checks the
list against the code in both directions: every documented site really calls
``authorize()``, and every site that calls it is documented.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AUTHORITY = ROOT / "core" / "consciousness" / "substrate_authority.py"
SEARCH_ROOTS = ("core", "interface", "skills")
SKIP_DIR_PARTS = {".git", ".venv", "__pycache__", "archive", "node_modules", "tests"}


def _documented_sites() -> set[str]:
    """Paths bulleted under 'Integration points' in the module docstring."""
    docstring = ast.get_docstring(ast.parse(AUTHORITY.read_text(encoding="utf-8")))
    assert docstring, "the authority lost its docstring"
    section = docstring.split("Integration points", 1)
    assert len(section) == 2, "the authority no longer lists its integration points"
    # Stop at the paragraph that deliberately names what is NOT integrated.
    body = section[1].split("Not integrated", 1)[0]
    return set(re.findall(r"^\s*-\s+(\S+\.py)", body, flags=re.MULTILINE))


def _actual_sites() -> set[str]:
    """Files containing a real call to ``.authorize(`` on the authority.

    Parsed rather than grepped: the module's own docstring contains a usage
    example, and a docstring is not a call site.
    """
    found: set[str] = set()
    for top in SEARCH_ROOTS:
        base = ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            relative = path.relative_to(ROOT)
            if any(part in SKIP_DIR_PARTS for part in relative.parts):
                continue
            if path == AUTHORITY:
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "authorize" not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "authorize"
                    and _is_substrate_authority(node, source)
                ):
                    found.add(str(relative))
                    break
    return found


def _is_substrate_authority(node: ast.Call, source: str) -> bool:
    """Distinguish the substrate authority from other things named authorize().

    ExecutiveInhibitor.authorize and confirmations.authorize are different
    gates with the same verb; a file only counts if it also names this
    module or resolves it from the container under the canonical key.
    """
    return (
        "substrate_authority" in source
        or "SubstrateAuthority" in source
    )


def test_every_documented_integration_point_really_calls_the_gate():
    documented = _documented_sites()
    actual = _actual_sites()
    assert documented, "the integration list is empty"
    phantom = sorted(documented - actual)
    assert not phantom, (
        "the authority documents integration points that do not call it: "
        f"{phantom}. This is how 'implemented' comes to be read as 'live' — "
        "wire it or remove it from the list."
    )


def test_every_real_call_site_is_documented():
    documented = _documented_sites()
    actual = _actual_sites()
    undocumented = sorted(actual - documented)
    assert not undocumented, (
        "these call the substrate authority and are not in its integration "
        f"list: {undocumented}. Add them, so the reach of the gate can be "
        "read off the module rather than reconstructed by grep."
    )


def test_memory_and_tools_are_still_honestly_marked_as_ungated():
    """They were claimed for a long time. If that changes, change the prose."""
    docstring = ast.get_docstring(ast.parse(AUTHORITY.read_text(encoding="utf-8")))
    assert "Not integrated" in docstring
    memory_gated = any(
        "substrate_authority" in p.read_text(encoding="utf-8", errors="ignore")
        for p in (ROOT / "core" / "memory").rglob("*.py")
    )
    agency_gated = any(
        "substrate_authority" in p.read_text(encoding="utf-8", errors="ignore")
        for p in (ROOT / "core" / "agency").rglob("*.py")
    )
    if memory_gated or agency_gated:
        pytest.fail(
            "memory writes and/or tool execution now consult the substrate "
            "authority — good, but the docstring still lists them under "
            "'Not integrated'. Move them into the integration list."
        )


def test_the_docstring_does_not_claim_to_be_unbypassable():
    """Cognitive and effectful code share one process and one set of OS
    privileges. Convention and the governance ratchets are what hold, not
    the kernel, and the module must not imply otherwise."""
    text = AUTHORITY.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(text)) or ""
    # Matched as assertions, not as substrings: the docstring now quotes both
    # phrases in order to disown them, and a naive `not in` would fail on the
    # correction itself.
    assert not re.search(r"constraints that\s+cannot be bypassed", docstring)
    assert not re.search(
        r"gate through which ALL significant\s+actions\s+must pass", docstring
    )
    assert "Not unbypassable" in docstring
    assert "privilege separation" in docstring
