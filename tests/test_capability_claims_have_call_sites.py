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

#: The boot entrypoint lives at the repo root, not under a package. Excluding
#: it made a capability wired ONLY from boot look uncalled — which is the exact
#: mistake this file exists to catch, in the direction that deletes live code.
_PRODUCTION_ENTRYPOINTS = ("aura_main.py",)

_SKIP_PARTS = frozenset(
    {".git", ".venv", "__pycache__", "node_modules", ".claude", "artifacts", "archive"}
)


def _production_call_sites(function_name: str, defining_file: str) -> list[str]:
    """Production files that CALL ``function_name``.

    The defining file is excluded: a function referenced only inside the
    module that defines it has no caller in any sense that matters.
    """
    hits: list[str] = []
    candidates: list[Path] = [
        ROOT / name for name in _PRODUCTION_ENTRYPOINTS if (ROOT / name).is_file()
    ]
    for root in _PRODUCTION_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        candidates.extend(base.rglob("*.py"))
    for path in candidates:
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

def test_verifier_curriculum_declares_that_it_is_not_wired():
    """Found by the residue sweep, same shape as the two above."""
    callers = _production_call_sites(
        "boot_verifier_curriculum", "core/brain/verifier_curriculum.py"
    ) + _production_call_sites(
        "get_verifier_curriculum", "core/brain/verifier_curriculum.py"
    )
    module = (ROOT / "core" / "brain" / "verifier_curriculum.py").read_text(
        encoding="utf-8"
    )
    if callers:
        assert "NOT WIRED INTO THE LIVE RUNTIME" not in module, (
            f"verifier_curriculum now has production callers ({callers}); the "
            "module still declares itself unwired."
        )
    else:
        assert "NOT WIRED INTO THE LIVE RUNTIME" in module, (
            "Neither boot_verifier_curriculum() nor get_verifier_curriculum() "
            "has a production caller and nothing reads the ServiceContainer "
            "key it registers. The module must say so."
        )


def _key_readers(key: str) -> list[str]:
    """Production files that resolve a ServiceContainer key by name."""
    hits: list[str] = []
    for path in (ROOT / "core").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if f'"{key}"' in text and (
            "optional_service(" in text or "ServiceContainer.get(" in text
        ):
            hits.append(str(path))
    return hits


def test_the_verifier_foundry_is_live_and_not_mislabelled():
    """The contrast case: the foundry IS wired, so it must not be declared dead.

    latent_cortex_service calls get_verifier_foundry() directly, so
    'delete the uncalled thing' applied bluntly would have taken a live
    capability with it.

    This test used to also assert that ``boot_verifier_foundry`` stayed
    deleted, on the grounds that nothing read the ``verifier_foundry``
    container key. Both halves of that premise have since become false:
    aura_main imports the wrapper (and logged an ImportError on every boot
    while it was missing), and procedural_memory and verifier_curriculum both
    resolve the key. So the assertion now runs the other way — the wrapper must
    exist BECAUSE it is imported, and the key must have readers.
    """
    callers = _production_call_sites(
        "get_verifier_foundry", "core/brain/verifiers/foundry.py"
    )
    assert callers, "get_verifier_foundry lost its production callers"
    module = (ROOT / "core" / "brain" / "verifiers" / "foundry.py").read_text(
        encoding="utf-8"
    )
    assert "NOT WIRED" not in module

    boot_callers = _production_call_sites(
        "boot_verifier_foundry", "core/brain/verifiers/foundry.py"
    )
    if "def boot_verifier_foundry" in module:
        assert boot_callers, (
            "boot_verifier_foundry exists with no production caller — either "
            "wire it or delete it"
        )
        assert _key_readers("verifier_foundry"), (
            "the wrapper registers a container key nothing reads"
        )
    else:
        assert not boot_callers, (
            "boot_verifier_foundry is imported but not defined; every boot "
            "will log an ImportError and run without the foundry"
        )


def test_cross_tier_verifier_declares_that_it_is_not_wired():
    """Found by the CP126 pass, same shape as the three above.

    Its docstring claimed it "wires to the live Solver tier in production".
    That was a claim about a call site that does not exist.
    """
    callers = _production_call_sites(
        "get_cross_tier_verifier", "core/brain/cross_tier_verifier.py"
    ) + _production_call_sites(
        "CrossTierVerifier", "core/brain/cross_tier_verifier.py"
    )
    module = (ROOT / "core" / "brain" / "cross_tier_verifier.py").read_text("utf-8")
    if callers:
        assert "NOT WIRED INTO THE LIVE RESPONSE PATH" not in module, (
            f"cross-tier verification now has production callers ({callers}); "
            "the module still declares itself unwired."
        )
    else:
        assert "NOT WIRED INTO THE LIVE RESPONSE PATH" in module, (
            "neither get_cross_tier_verifier() nor CrossTierVerifier has a "
            "production caller, so cross-tier verification is not live."
        )


def test_compute_router_declares_that_it_is_not_wired():
    """It handles API keys and cloud spend, so an implied liveness is worse here."""
    callers = _production_call_sites("ComputeRouter", "core/brain/compute_router.py")
    module = (ROOT / "core" / "brain" / "compute_router.py").read_text("utf-8")
    if callers:
        assert "NOT WIRED INTO THE LIVE RUNTIME" not in module, (
            f"ComputeRouter now has production callers ({callers}); the module "
            "still declares itself unwired."
        )
    else:
        assert "NOT WIRED INTO THE LIVE RUNTIME" in module, (
            "ComputeRouter has no production caller; live cloud fallback goes "
            "through core/brain/llm_health_router.py. An unwired module that "
            "looks live is how one gets adopted without review."
        )
