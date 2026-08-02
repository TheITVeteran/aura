"""A capability needs a call site, not a class.

Two subsystems were found defined, registered, tested in isolation, and
never invoked by anything that runs:

* ``core/consciousness/latent_bridge.py`` — the backward path from model
  hidden state into the substrate. ``attach_latent_bridge()`` has no caller
  anywhere. The consciousness layer status said ``deferred``, which reads as
  "will happen shortly" and had said so since it was written.
* ``core/being/closed_loop_controller.py`` — ``build_main15_closed_loop()``
  is called from its own docstring and its own tests.

Neither is a bug in the code they contain. Both are correct in isolation.
The defect is the CLAIM: substantial, tested and uninvoked reads exactly
like working from the outside.

This file pins the honest statements so they cannot silently drift back into
implied liveness — and, more usefully, fails the moment someone wires one
up without updating what the system says about itself.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Directories that are the live system. A call from tests/, tools/ or an
#: archive is not a production call site.
_PRODUCTION_ROOTS = ("core", "interface")

_SKIP_PARTS = frozenset(
    {".git", ".venv", "__pycache__", "node_modules", ".claude", "artifacts", "archive"}
)


def _production_call_sites(function_name: str, defining_file: str) -> list[str]:
    """Files under core/ or interface/ that CALL ``function_name``.

    The defining file is excluded: a function referenced only inside the
    module that defines it has no caller in any sense that matters.
    """
    hits: list[str] = []
    for root in _PRODUCTION_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(ROOT)
            if _SKIP_PARTS.intersection(rel.parts):
                continue
            if str(rel) == defining_file:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else ""
                )
                if name == function_name:
                    hits.append(str(rel))
                    break
    return sorted(set(hits))


def test_latent_bridge_status_matches_its_wiring():
    """The status must track reality in whichever direction reality moves."""
    callers = _production_call_sites(
        "attach_latent_bridge", "core/consciousness/latent_bridge.py"
    )
    system = (ROOT / "core" / "consciousness" / "system.py").read_text(encoding="utf-8")

    if callers:
        assert '"latent_bridge"] = "unwired"' not in system, (
            "attach_latent_bridge() now has production callers "
            f"({callers}) — the consciousness layer status still reports it "
            "as unwired. Update the status to match."
        )
    else:
        assert '"latent_bridge"] = "unwired"' in system, (
            "attach_latent_bridge() has no production caller, so the backward "
            "hidden-state path is not live. The layer status must say so — it "
            "previously said 'deferred', which reads as a promise that "
            "something will attach it."
        )
        assert '"latent_bridge"] = "deferred"' not in system, (
            "'deferred' is a claim about a future event. Nothing redeems it "
            "here, so it must not be used."
        )


def test_main15_controller_declares_that_it_is_not_wired():
    callers = _production_call_sites(
        "build_main15_closed_loop", "core/being/closed_loop_controller.py"
    )
    module = (ROOT / "core" / "being" / "closed_loop_controller.py").read_text(
        encoding="utf-8"
    )

    if callers:
        assert "NOT WIRED INTO THE LIVE RESPONSE PATH" not in module, (
            f"build_main15_closed_loop() now has production callers ({callers}); "
            "the module still declares itself unwired."
        )
    else:
        assert "NOT WIRED INTO THE LIVE RESPONSE PATH" in module, (
            "build_main15_closed_loop() has no production caller. The module "
            "must say so: substantial, tested and uninvoked reads exactly like "
            "working from the outside."
        )


def test_the_call_site_scanner_actually_finds_calls():
    """A scanner that finds nothing would pass both tests above vacuously."""
    # record_degradation is called all over core/; if this comes back empty the
    # scanner is broken and the assertions above prove nothing.
    hits = _production_call_sites("record_degradation", "core/runtime/errors.py")
    assert len(hits) > 20, f"scanner found only {len(hits)} callers; it is broken"
