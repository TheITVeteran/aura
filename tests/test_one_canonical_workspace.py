"""There is exactly one Global Workspace.

Aura shipped four modules named ``global_workspace``. Two of them defined a
class called ``GlobalWorkspace`` with a ``WorkItem`` and a ``publish`` —
genuinely competing implementations of the same idea. A system cannot claim a
unified workspace while running several, and "consciousness/global_workspace vs
global_workspace" is exactly the kind of ambiguity that makes a unity claim
unfalsifiable.

Resolution: core/consciousness/global_workspace.py is canonical (1167 lines, 6
production importers, the coalition competition / ignition / gate receipts the
runtime depends on). core/global_workspace.py — which had zero production
importers — is now a facade over it, following the same pattern core/will.py
uses over core/governance/will.py.

The other two are not buses despite the name and are left alone; these tests pin
that they stay that way.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANONICAL = ROOT / "core" / "consciousness" / "global_workspace.py"
FACADE = ROOT / "core" / "global_workspace.py"

# Named "global_workspace" but a different concern entirely. Neither is a
# broadcast bus; both are single-consumer helpers.
NOT_A_BUS = (
    ROOT / "core" / "phenomenal_substrate" / "global_workspace.py",
    ROOT / "core" / "workspace" / "global_workspace.py",
)


def test_the_facade_is_the_canonical_class():
    """Importing the legacy path must yield the canonical object, not a twin."""
    from core.consciousness.global_workspace import GlobalWorkspace as Canonical
    from core.global_workspace import GlobalWorkspace as Legacy

    assert Legacy is Canonical, (
        "core/global_workspace.py defines a second GlobalWorkspace — there must "
        "be exactly one"
    )


def test_the_facade_shares_the_canonical_workitem():
    from core.consciousness.global_workspace import WorkItem as Canonical
    from core.global_workspace import WorkItem as Legacy

    assert Legacy is Canonical


def test_the_facade_defines_no_implementation_of_its_own():
    """A facade that quietly grows a class body is a duplicate again."""
    tree = ast.parse(FACADE.read_text(encoding="utf-8"), filename=str(FACADE))
    classes = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and not n.name.startswith("_")
    ]
    assert not classes, (
        f"core/global_workspace.py should re-export only, but defines: {classes}"
    )


def test_only_one_module_implements_a_broadcast_bus():
    """The structural check: one competition/broadcast implementation.

    A broadcast bus here means a module that both defines ``GlobalWorkspace``
    and implements ``publish`` on it. Exactly one module in the tree may.
    """
    implementers: list[str] = []

    for path in ROOT.joinpath("core").rglob("global_workspace.py"):
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "GlobalWorkspace":
                continue
            methods = {
                b.name
                for b in node.body
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "publish" in methods:
                implementers.append(rel)

    assert implementers == ["core/consciousness/global_workspace.py"], (
        f"expected exactly one broadcast workspace, found: {implementers}"
    )


def test_the_other_workspace_modules_are_not_buses():
    """They share a name, not a job — keep it that way."""
    for path in NOT_A_BUS:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GlobalWorkspace":
                methods = {
                    b.name
                    for b in node.body
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                assert "publish" not in methods, (
                    f"{path.relative_to(ROOT)} has grown a broadcast bus — it is "
                    "now competing with the canonical workspace"
                )


# ---------------------------------------------------------------------------
# The merge must not have lost anything
# ---------------------------------------------------------------------------


def test_canonical_honours_the_history_retention_policy(monkeypatch):
    """The retired module's one real advantage had to survive the merge.

    core/global_workspace.py read AURA_GLOBAL_WORKSPACE_HISTORY_MAX through
    working_history_retention_policy; the canonical hardcoded a 100-record cap.
    Retiring the module without porting this would have been a silent regression.
    """
    monkeypatch.setenv("AURA_GLOBAL_WORKSPACE_HISTORY_MAX", "1500")

    from core.consciousness.global_workspace import GlobalWorkspace

    assert GlobalWorkspace().max_history == 1500


def test_canonical_history_stays_bounded():
    """Whatever the policy says, the buffer must not grow without limit."""
    from core.consciousness.global_workspace import GlobalWorkspace

    ws = GlobalWorkspace()
    assert isinstance(ws.max_history, int)
    assert 0 < ws.max_history <= 100_000, (
        f"workspace history bound is not sane: {ws.max_history}"
    )
