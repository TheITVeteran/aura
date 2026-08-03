"""The Aegis lock protects rebinding, and says so.

CP126 1947a19c. It was documented as "Singleton True-Lock (Memory Protection)",
which reads as a guarantee that the live topology cannot be altered. It never
was that and cannot be — routing learns, so the contents of ``pathways`` and
``hyphae`` are mutated constantly and by design. Meanwhile internal code
side-stepped the guard with ``object.__setattr__`` in five places, so even the
rebinding guarantee had five undocumented doors.

The guarantee is now stated accurately and has one door.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from core.mycelium import MycelialNetwork

pytestmark = pytest.mark.unit

_SOURCE = pathlib.Path("core/mycelium.py").read_text(encoding="utf-8")

#: Attributes that may still be set with a raw ``object.__setattr__``. All are
#: bootstrap or session-local state, none is a protected container. This
#: allowlist only shrinks.
_PERMITTED_RAW_REBINDS = {"_aegis_locked", "_session_confidence"}


def _raw_rebind_targets() -> list[str]:
    """Attribute names passed to a bare ``object.__setattr__`` call.

    ``_aegis_replace`` is skipped: its single dynamic call IS the door, and
    counting it would make the ratchet measure itself.
    """
    tree = ast.parse(_SOURCE)
    door = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_aegis_replace"
        ),
        None,
    )
    assert door is not None, "the sanctioned replacement path must exist"
    inside_door = set(map(id, ast.walk(door)))

    targets: list[str] = []
    for node in ast.walk(tree):
        if id(node) in inside_door:
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "__setattr__"
            and isinstance(func.value, ast.Name)
            and func.value.id == "object"
        ):
            continue
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            targets.append(str(node.args[1].value))
        else:
            targets.append("<dynamic>")
    return targets


def test_no_protected_container_is_rebound_behind_the_guard():
    offenders = [
        name
        for name in _raw_rebind_targets()
        if name in MycelialNetwork._AEGIS_PROTECTED_ATTRS
    ]

    assert not offenders, (
        "Protected containers must be replaced via _aegis_replace, not a raw "
        f"object.__setattr__: {sorted(set(offenders))}"
    )


def test_the_raw_rebind_allowlist_only_shrinks():
    found = set(_raw_rebind_targets())

    assert found <= _PERMITTED_RAW_REBINDS, (
        "New raw object.__setattr__ target(s); route them through "
        f"_aegis_replace instead: {sorted(found - _PERMITTED_RAW_REBINDS)}"
    )


def test_rebinding_a_protected_container_is_still_refused():
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False
    net = MycelialNetwork()
    try:
        object.__setattr__(net, "_aegis_locked", True)

        for attr in sorted(MycelialNetwork._AEGIS_PROTECTED_ATTRS):
            with pytest.raises(PermissionError):
                setattr(net, attr, {})
    finally:
        MycelialNetwork._instance = None
        MycelialNetwork._initialized = False


def test_the_contents_are_deliberately_mutable():
    """Stating the actual scope. A route that could not learn would be a worse
    bug than the one this guard prevents."""
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False
    net = MycelialNetwork()
    try:
        net.register_pathway(pathway_id="p", pattern=r"^go", skill_name="s")
        object.__setattr__(net, "_aegis_locked", True)

        before = net.pathways["p"].confidence
        net.pathways["p"].reinforce(True, verified=True)

        assert net.pathways["p"].confidence != before
    finally:
        MycelialNetwork._instance = None
        MycelialNetwork._initialized = False


def test_the_docstring_does_not_promise_memory_protection():
    tree = ast.parse(_SOURCE)
    setattr_doc = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__setattr__":
            setattr_doc = ast.get_docstring(node) or ""
            break

    header = setattr_doc.split("CP126", 1)[0]
    assert header
    assert "Memory Protection" not in header
    assert "rebind" in setattr_doc.lower()
